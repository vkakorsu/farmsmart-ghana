"""Forward price forecasting for the demo.

Trains one LightGBM per horizon on all available history, then projects the
national median price for a chosen staple over the next few months. Predictors
use only information known at the last observed month, so this is a genuine
out-of-sample projection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src import data as data_mod
from src.features import (
    CATEGORICAL_COLUMNS,
    build_features,
    feature_columns,
    to_monthly,
)

FORECAST_HORIZONS = (1, 2, 3, 4, 5, 6)
RANDOM_STATE = 42


def build_all() -> pd.DataFrame:
    """Cleaned per-kg features for all staples, with targets for horizons 1-6."""
    df = data_mod.clean(data_mod.load(data_mod.download()))
    feat = build_features(to_monthly(df, staples_only=True), horizons=FORECAST_HORIZONS)
    for col in CATEGORICAL_COLUMNS:
        feat[col] = feat[col].astype("category")
    return feat


def train_models(feat: pd.DataFrame) -> dict[int, LGBMRegressor]:
    """Train one gradient-boosted model per horizon on all history."""
    models: dict[int, LGBMRegressor] = {}
    for h in FORECAST_HORIZONS:
        target = f"target_{h}"
        train = feat.dropna(subset=[target])
        features = feature_columns(h) + CATEGORICAL_COLUMNS
        lo, hi = train[target].quantile([0.01, 0.99])
        model = LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1,
        )
        model.fit(train[features], train[target].clip(lo, hi),
                  categorical_feature=CATEGORICAL_COLUMNS)
        models[h] = model
    return models


def forecast_commodity(
    feat: pd.DataFrame, models: dict[int, LGBMRegressor], commodity: str
) -> pd.DataFrame:
    """Return national median history and a forward forecast for a commodity.

    Columns: month, price, kind ('history' or 'forecast').
    """
    sub = feat[feat["commodity"] == commodity]
    if sub.empty:
        return pd.DataFrame(columns=["month", "price", "kind"])

    history = (
        sub.groupby("month", observed=True)["price"].median().reset_index().assign(kind="history")
    )

    last_month = sub["month"].max()
    latest = sub[sub["month"] == last_month]

    rows = []
    for h, model in models.items():
        features = feature_columns(h) + CATEGORICAL_COLUMNS
        x = latest[features]
        if x.isna().any(axis=None):
            continue
        pred_price = latest["price"].to_numpy() * np.exp(model.predict(x))
        fmonth = (last_month.to_period("M") + h).to_timestamp()
        rows.append({"month": fmonth, "price": float(np.median(pred_price)), "kind": "forecast"})

    forecast = pd.DataFrame(rows)
    # Anchor the forecast line to the last history point for a continuous chart.
    if not forecast.empty:
        anchor = history.iloc[[-1]].assign(kind="forecast")
        forecast = pd.concat([anchor[["month", "price", "kind"]], forecast], ignore_index=True)

    return pd.concat([history, forecast], ignore_index=True)


if __name__ == "__main__":
    feats = build_all()
    mdls = train_models(feats)
    out = forecast_commodity(feats, mdls, "Maize")
    print(out.tail(10).to_string(index=False))
