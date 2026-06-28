"""Turn raw WFP price records into a monthly, model-ready feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd

SERIES_KEYS = ["market", "commodity", "pricetype"]
RETURN_LAGS = [1, 2, 3, 4, 5]
ROLL_WINDOWS = [3, 6]
HORIZONS = [1, 3, 6]

# Storable staple crops: forecasting sell-timing only adds value for crops a
# farmer can actually store. Perishables (tomatoes, onions, peppers, fish,
# meat, eggs) are excluded - their prices are near-random month to month.
STAPLE_KEYWORDS = (
    "maize",
    "rice",
    "millet",
    "sorghum",
    "cassava",
    "yam",
    "gari",
    "soybean",
    "cowpea",
    "groundnut",
    "plantain",
    "cocoyam",
    "beans",
)


def is_staple(commodity: str) -> bool:
    name = commodity.lower()
    return any(k in name for k in STAPLE_KEYWORDS)


def to_monthly(df: pd.DataFrame, staples_only: bool = True) -> pd.DataFrame:
    """Aggregate cleaned records to one median GHS-per-kg price per series-month."""
    work = df.copy()
    if staples_only:
        work = work[work["commodity"].map(is_staple)].copy()
    work["month"] = work["date"].dt.to_period("M").dt.to_timestamp()
    grouped = (
        work.groupby(SERIES_KEYS + ["admin1", "month"], observed=True)["price_per_kg"]
        .median()
        .reset_index()
        .rename(columns={"price_per_kg": "price"})
    )
    return grouped


def build_features(
    monthly: pd.DataFrame, min_history: int = 36, horizons: tuple[int, ...] = tuple(HORIZONS)
) -> pd.DataFrame:
    """Create per-series features (known at time t) plus future-return targets.

    All predictors use only information available at month t. Targets are the
    log price change from t to t+h for each horizon h in ``horizons``, so the
    model forecasts where the price is headed.
    """
    frames: list[pd.DataFrame] = []
    for keys, grp in monthly.groupby(SERIES_KEYS, observed=True):
        grp = grp.sort_values("month")
        # Reindex to a continuous monthly grid so lags are calendar-correct.
        full_idx = pd.date_range(grp["month"].min(), grp["month"].max(), freq="MS")
        grp = grp.set_index("month").reindex(full_idx)
        grp.index.name = "month"
        grp[SERIES_KEYS] = keys
        grp["admin1"] = grp["admin1"].ffill().bfill()
        grp["price"] = grp["price"].interpolate(limit=2)
        grp = grp.dropna(subset=["price"])
        if len(grp) < min_history:
            continue

        grp["log_price"] = np.log(grp["price"])
        grp["cur_ret"] = grp["log_price"].diff()  # log(price[t]/price[t-1]), known at t
        for r in RETURN_LAGS:
            grp[f"ret_{r}"] = grp["cur_ret"].shift(r)
        grp["ann_ret"] = grp["log_price"] - grp["log_price"].shift(12)  # YoY, known at t
        for win in ROLL_WINDOWS:
            grp[f"retmean_{win}"] = grp["cur_ret"].rolling(win).mean()
            grp[f"retstd_{win}"] = grp["cur_ret"].rolling(win).std()

        # Future-return targets and the seasonal-naive reference per horizon.
        for h in horizons:
            grp[f"target_{h}"] = grp["log_price"].shift(-h) - grp["log_price"]
            # Seasonal naive: price at target month last year (known at t for h<=12).
            grp[f"seasonal_{h}"] = grp["price"].shift(12 - h)
            target_month = grp.index.to_period("M").shift(h).to_timestamp()
            grp[f"tmonth_sin_{h}"] = np.sin(2 * np.pi * target_month.month / 12)
            grp[f"tmonth_cos_{h}"] = np.cos(2 * np.pi * target_month.month / 12)
        frames.append(grp.reset_index())

    feat = pd.concat(frames, ignore_index=True)

    # Cross-sectional signal: how the whole commodity / whole market moved this
    # month. cur_ret is known at t, so grouping introduces no leakage.
    feat["comm_ret"] = feat.groupby(["commodity", "month"])["cur_ret"].transform("mean")
    feat["market_ret"] = feat.groupby(["market", "month"])["cur_ret"].transform("mean")

    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.dropna(subset=[f"ret_{r}" for r in RETURN_LAGS] + ["ann_ret", "cur_ret"])
    return feat


BASE_FEATURES = (
    ["cur_ret"]
    + [f"ret_{r}" for r in RETURN_LAGS]
    + [f"retmean_{w}" for w in ROLL_WINDOWS]
    + [f"retstd_{w}" for w in ROLL_WINDOWS]
    + ["ann_ret", "log_price", "comm_ret", "market_ret"]
)
CATEGORICAL_COLUMNS = ["commodity", "market", "admin1", "pricetype"]


def feature_columns(horizon: int) -> list[str]:
    """Predictors for a given horizon (adds that horizon's target-month season)."""
    return BASE_FEATURES + [f"tmonth_sin_{horizon}", f"tmonth_cos_{horizon}"]
