"""Structural findings that justify the FarmSmart advisory product.

These are robust, model-free results computed directly from the WFP Ghana
price data:
  1. Spatial arbitrage  - how much prices differ across markets (sell-where).
  2. Seasonal swing      - harvest-low to lean-season-high amplitude (sell-when).

Run:  python -m src.insights
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import data as data_mod
from src.features import to_monthly

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RECENT_YEARS = 5
MIN_SERIES_MONTHS = 36


def _reliable(monthly: pd.DataFrame, min_months: int = MIN_SERIES_MONTHS) -> pd.DataFrame:
    """Keep only (market, commodity, pricetype) series with enough history."""
    counts = monthly.groupby(["market", "commodity", "pricetype"], observed=True)["month"].transform("size")
    return monthly[counts >= min_months]


def spatial_spread(monthly: pd.DataFrame, recent_years: int = RECENT_YEARS) -> pd.DataFrame:
    """Robust cross-market per-kg price gap per commodity in recent years.

    For each commodity-month-pricetype we take the P90/P10 ratio across markets
    (robust to a single bad market). A farmer who can choose where to sell
    captures up to this gap.
    """
    cutoff = monthly["month"].max() - pd.DateOffset(years=recent_years)
    recent = monthly[monthly["month"] >= cutoff]

    rows = []
    for (commodity, _ptype, _month), g in recent.groupby(
        ["commodity", "pricetype", "month"], observed=True
    ):
        if g["market"].nunique() < 3:
            continue
        p10, p90 = g["price"].quantile([0.10, 0.90])
        if p10 > 0:
            rows.append({"commodity": commodity, "spread_pct": (p90 - p10) / p10 * 100})

    spreads = pd.DataFrame(rows)
    out = (
        spreads.groupby("commodity")["spread_pct"]
        .median()
        .round(1)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"spread_pct": "median_cross_market_gap_%"})
    )
    return out


def seasonal_swing(monthly: pd.DataFrame, recent_years: int = RECENT_YEARS) -> pd.DataFrame:
    """Median within-year per-kg price swing per commodity (lean vs harvest)."""
    cutoff = monthly["month"].max() - pd.DateOffset(years=recent_years)
    recent = monthly[monthly["month"] >= cutoff].copy()
    recent["year"] = recent["month"].dt.year

    rows = []
    for (commodity, _ptype, market, year), g in recent.groupby(
        ["commodity", "pricetype", "market", "year"], observed=True
    ):
        if len(g) < 8:  # need most of the year present
            continue
        p10, p90 = g["price"].quantile([0.10, 0.90])
        if p10 > 0:
            rows.append({"commodity": commodity, "swing_pct": (p90 - p10) / p10 * 100})

    swings = pd.DataFrame(rows)
    out = (
        swings.groupby("commodity")["swing_pct"]
        .median()
        .round(1)
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"swing_pct": "median_seasonal_swing_%"})
    )
    return out


def run() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = data_mod.clean(data_mod.load(data_mod.download()))
    monthly = _reliable(to_monthly(df, staples_only=True))

    spatial = spatial_spread(monthly)
    seasonal = seasonal_swing(monthly)

    spatial.to_csv(RESULTS_DIR / "insight_spatial_spread.csv", index=False)
    seasonal.to_csv(RESULTS_DIR / "insight_seasonal_swing.csv", index=False)

    print(f"Staples covered: {monthly['commodity'].nunique()} | "
          f"markets: {monthly['market'].nunique()} | "
          f"period: {monthly['month'].min().date()} to {monthly['month'].max().date()}")
    print(f"\n=== Cross-market price gap (last {RECENT_YEARS}y, top staples) ===")
    print(spatial.head(10).to_string(index=False))
    print(f"  Overall median cross-market gap: {spatial['median_cross_market_gap_%'].median():.1f}%")
    print(f"\n=== Seasonal within-year price swing (last {RECENT_YEARS}y) ===")
    print(seasonal.head(10).to_string(index=False))
    print(f"  Overall median seasonal swing: {seasonal['median_seasonal_swing_%'].median():.1f}%")


if __name__ == "__main__":
    run()
