"""Quick EDA to understand price-series noise and structure."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import data as data_mod
from src.features import SERIES_KEYS, to_monthly

df = data_mod.load(data_mod.download())
print("pricetype:\n", df["pricetype"].value_counts())
print("\nunit (top 15):\n", df["unit"].value_counts().head(15))
print("\ncommodity (top 25):\n", df["commodity"].value_counts().head(25))

monthly = to_monthly(df)

# Per-series month-over-month absolute % change (naive error proxy).
rows = []
for keys, grp in monthly.groupby(SERIES_KEYS, observed=True):
    grp = grp.sort_values("month")
    if len(grp) < 24:
        continue
    pct = grp["price"].pct_change().abs()
    rows.append((keys[1], keys[0], keys[2], keys[3], len(grp),
                 round(pct.median() * 100, 1), round(pct.mean() * 100, 1)))

vol = pd.DataFrame(rows, columns=["commodity", "market", "unit", "pricetype",
                                  "months", "median_mom_%", "mean_mom_%"])
print(f"\nseries with >=24 months: {len(vol)}")
print("\nMost volatile series (mean MoM %):")
print(vol.sort_values("mean_mom_%", ascending=False).head(12).to_string(index=False))
print("\nLeast volatile series (mean MoM %):")
print(vol.sort_values("mean_mom_%").head(12).to_string(index=False))
print(f"\nMedian of per-series median MoM%: {vol['median_mom_%'].median():.1f}")
