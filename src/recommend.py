"""Turn price data into a plain-language farmer advisory (demo of the product).

For a chosen staple it answers the two questions a smallholder cares about:
  - WHERE should I sell?  (markets ranked by recent GHS/kg)
  - WHEN should I sell?   (calendar month with the historically highest price)

Run:  python -m src.recommend --commodity Maize
"""

from __future__ import annotations

import argparse
import calendar

import pandas as pd

from src import data as data_mod
from src.features import to_monthly

RECENT_YEARS = 3


def advise(commodity: str, recent_years: int = RECENT_YEARS) -> str:
    monthly = to_monthly(data_mod.clean(data_mod.load(data_mod.download())))
    sub = monthly[monthly["commodity"].str.lower() == commodity.lower()]
    if sub.empty:
        options = sorted(monthly["commodity"].unique())
        return f"No data for '{commodity}'. Available staples: {', '.join(options)}"

    cutoff = sub["month"].max() - pd.DateOffset(years=recent_years)
    recent = sub[sub["month"] >= cutoff]
    retail = recent[recent["pricetype"].str.lower() == "retail"]
    recent = retail if not retail.empty else recent

    # WHERE: rank markets by recent median price per kg.
    by_market = (
        recent.groupby("market", observed=True)["price"].median().round(2).sort_values(ascending=False)
    )
    best_market, best_price = by_market.index[0], by_market.iloc[0]
    worst_market, worst_price = by_market.index[-1], by_market.iloc[-1]
    gap = (best_price - worst_price) / worst_price * 100 if worst_price else 0

    # WHEN: average price by calendar month (seasonality).
    recent = recent.copy()
    recent["mnum"] = recent["month"].dt.month
    by_month = recent.groupby("mnum")["price"].median()
    hi_m, lo_m = by_month.idxmax(), by_month.idxmin()
    swing = (by_month.max() - by_month.min()) / by_month.min() * 100 if by_month.min() else 0

    lines = [
        f"FarmSmart advisory for {commodity} (last {recent_years} years, GHS/kg)",
        "-" * 60,
        f"WHERE to sell: prices are highest in {best_market} (~GHS {best_price}/kg) "
        f"and lowest in {worst_market} (~GHS {worst_price}/kg).",
        f"  -> Selling in {best_market} instead of {worst_market} is about "
        f"{gap:.0f}% more per kg.",
        f"WHEN to sell: prices peak around {calendar.month_name[hi_m]} and bottom "
        f"around {calendar.month_name[lo_m]}.",
        f"  -> Storing from {calendar.month_name[lo_m]} to {calendar.month_name[hi_m]} "
        f"historically gains about {swing:.0f}%.",
        "",
        "Top markets by recent price (GHS/kg):",
        by_market.head(6).to_string(),
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="FarmSmart farmer advisory demo")
    parser.add_argument("--commodity", default="Maize", help="e.g. Maize, Rice (local), Sorghum")
    args = parser.parse_args()
    print(advise(args.commodity))


if __name__ == "__main__":
    main()
