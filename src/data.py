"""Download and load the WFP Ghana food prices dataset (primary Ghanaian source).

Data source: World Food Programme Price Database, published on the Humanitarian
Data Exchange (HDX) at https://data.humdata.org/dataset/wfp-food-prices-for-ghana

The HDX CSV ships with a second header row of HXL hashtags (e.g. "#date"),
which we strip on load.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import requests

HDX_PACKAGE_API = (
    "https://data.humdata.org/api/3/action/package_show?id=wfp-food-prices-for-ghana"
)
RAW_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "wfp_food_prices_ghana.csv"

# Columns we rely on downstream.
EXPECTED_COLUMNS = {
    "date",
    "admin1",
    "market",
    "commodity",
    "unit",
    "pricetype",
    "currency",
    "price",
}


def _find_csv_resource_url() -> str:
    """Query the HDX CKAN API and return the download URL of the prices CSV."""
    resp = requests.get(HDX_PACKAGE_API, timeout=60)
    resp.raise_for_status()
    resources = resp.json()["result"]["resources"]

    csv_resources = [r for r in resources if (r.get("format") or "").lower() == "csv"]
    if not csv_resources:
        raise RuntimeError("No CSV resource found in the HDX package.")

    # Prefer the prices file over the small "markets" lookup file.
    def score(resource: dict) -> tuple[int, int]:
        name = (resource.get("name") or "").lower()
        size = resource.get("size") or 0
        has_price = 1 if "price" in name else 0
        return (has_price, size)

    best = max(csv_resources, key=score)
    return best["download_url"]


def download(dest: Path = RAW_CSV_PATH, force: bool = False) -> Path:
    """Download the prices CSV to ``dest`` (cached unless ``force``)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest

    url = _find_csv_resource_url()
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def load(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Load the WFP CSV, dropping the HXL hashtag row and coercing types."""
    raw = pd.read_csv(path, dtype=str, keep_default_na=False)

    # The first data row in WFP HDX exports is HXL tags like "#date".
    if len(raw) and str(raw.iloc[0]["date"]).startswith("#"):
        raw = raw.iloc[1:].reset_index(drop=True)

    missing = EXPECTED_COLUMNS - set(raw.columns)
    if missing:
        raise RuntimeError(f"Dataset is missing expected columns: {sorted(missing)}")

    df = raw.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "price", "commodity", "market"])
    df = df[df["price"] > 0]
    return df.reset_index(drop=True)


_KG_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)?\s*kg\s*$", re.IGNORECASE)


def _kg_weight(unit: str) -> float:
    """Return the kg weight encoded in a unit string, or NaN if not weight-based."""
    m = _KG_RE.match(str(unit))
    if not m:
        return float("nan")
    return float(m.group(1)) if m.group(1) else 1.0


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise to GHS-per-kg and drop non-weight units and per-commodity outliers.

    Markets report different pack sizes (e.g. 'KG', '100 KG'), so raw prices are
    not comparable across rows. We convert to price-per-kg, then remove extreme
    outliers per commodity (likely unit/data-entry errors) using a wide
    1st-99th percentile clip.
    """
    out = df.copy()
    out["kg"] = out["unit"].map(_kg_weight)
    out = out.dropna(subset=["kg"])
    out = out[out["kg"] > 0]
    out["price_per_kg"] = out["price"] / out["kg"]

    keep = []
    for _commodity, g in out.groupby("commodity", observed=True):
        lo, hi = g["price_per_kg"].quantile([0.01, 0.99])
        keep.append(g[(g["price_per_kg"] >= lo) & (g["price_per_kg"] <= hi)])
    out = pd.concat(keep, ignore_index=True)
    return out


if __name__ == "__main__":
    p = download()
    frame = load(p)
    print(f"Downloaded -> {p}")
    print(f"Rows: {len(frame):,}")
    print(f"Date range: {frame['date'].min().date()} to {frame['date'].max().date()}")
    print(f"Markets: {frame['market'].nunique()} | Commodities: {frame['commodity'].nunique()}")
