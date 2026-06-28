"""Multi-horizon staple-price forecasting with a time-based backtest.

Compares three forecasters at 1, 3 and 6 month horizons:
  - Naive carry-forward (random walk): price[t+h] = price[t]
  - Seasonal naive: price[t+h] = price at the same month last year
  - LightGBM (FarmSmart): learns the future log-return from lags + seasonality

Run:  python -m src.forecast
Outputs results/metrics.csv and charts in results/.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src import data as data_mod
from src.features import (
    CATEGORICAL_COLUMNS,
    HORIZONS,
    build_features,
    feature_columns,
    to_monthly,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
TEST_MONTHS = 18
RANDOM_STATE = 42


def ape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100


def evaluate(y_true, y_pred) -> dict[str, float]:
    errs = ape(y_true, y_pred)
    return {
        "MdAPE_%": round(float(np.median(errs)), 2),
        "MAPE_%": round(float(np.mean(errs)), 2),
        "MAE_GHS": round(mean_absolute_error(y_true, y_pred), 2),
        "RMSE_GHS": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 2),
    }


def _fit_predict(train, test, horizon):
    features = feature_columns(horizon) + CATEGORICAL_COLUMNS
    train = train.copy()
    test = test.copy()
    for col in CATEGORICAL_COLUMNS:
        train[col] = train[col].astype("category")
        test[col] = pd.Categorical(test[col], categories=train[col].cat.categories)

    target = f"target_{horizon}"
    lo, hi = train[target].quantile([0.01, 0.99])
    y_train = train[target].clip(lo, hi)

    model = LGBMRegressor(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(train[features], y_train, categorical_feature=CATEGORICAL_COLUMNS)
    pred_ret = model.predict(test[features])
    pred_price = test["price"].to_numpy() * np.exp(pred_ret)
    return pred_price, model, features


def run() -> pd.DataFrame:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    df = data_mod.clean(data_mod.load(data_mod.download()))
    print(f"  cleaned per-kg records: {len(df):,}")

    feat = build_features(to_monthly(df, staples_only=True))
    print(f"  staple rows: {len(feat):,} across {feat['commodity'].nunique()} commodities, "
          f"{feat['market'].nunique()} markets, "
          f"{feat['month'].min().date()} to {feat['month'].max().date()}")

    months = feat["month"].drop_duplicates().sort_values()
    cutoff = months.iloc[-TEST_MONTHS]

    all_rows = []
    artifacts = {}
    for h in HORIZONS:
        target = f"target_{h}"
        sub = feat.dropna(subset=[target, f"seasonal_{h}"])
        train = sub[sub["month"] < cutoff]
        test = sub[sub["month"] >= cutoff]
        if test.empty:
            continue

        actual = test["price"].to_numpy() * np.exp(test[target].to_numpy())

        naive_pred = test["price"].to_numpy()  # carry forward
        seasonal_pred = test[f"seasonal_{h}"].to_numpy()
        model_pred, model, features = _fit_predict(train, test, h)

        naive_err = np.abs(actual - naive_pred)
        model_err = np.abs(actual - model_pred)
        win = float(np.mean(model_err < naive_err) * 100)

        for name, pred in [
            ("Naive carry-forward", naive_pred),
            ("Seasonal naive", seasonal_pred),
            ("LightGBM (FarmSmart)", model_pred),
        ]:
            m = evaluate(actual, pred)
            m["beats_naive_%"] = round(win, 1) if name == "LightGBM (FarmSmart)" else np.nan
            m.update({"horizon_months": h, "model": name})
            all_rows.append(m)

        print(f"\nHorizon {h}m: FarmSmart MdAPE {evaluate(actual, model_pred)['MdAPE_%']}% "
              f"vs naive {evaluate(actual, naive_pred)['MdAPE_%']}% | "
              f"MAPE {evaluate(actual, model_pred)['MAPE_%']}% vs {evaluate(actual, naive_pred)['MAPE_%']}% | "
              f"beats naive on {win:.1f}% of forecasts")
        if not artifacts or h == max(HORIZONS):
            artifacts = {"test": test, "pred": model_pred, "model": model,
                         "features": features, "actual": actual}

    metrics = pd.DataFrame(all_rows)[
        ["horizon_months", "model", "MdAPE_%", "MAPE_%", "MAE_GHS", "RMSE_GHS", "beats_naive_%"]
    ]
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False)
    print("\n=== Backtest by horizon (lower is better) ===")
    print(metrics.to_string(index=False))

    if artifacts:
        _save_plots(metrics, artifacts)
    return metrics


def _save_plots(metrics, artifacts) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # MdAPE by horizon and model.
    pivot = metrics.pivot(index="horizon_months", columns="model", values="MdAPE_%")
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot.bar(ax=ax)
    ax.set_ylabel("Median APE (%)")
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_title("FarmSmart Ghana - forecast accuracy by horizon")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "accuracy_by_horizon.png", dpi=120)

    # Actual vs forecast for a few series at the longest horizon.
    eval_df = artifacts["test"].copy()
    eval_df["actual"] = artifacts["actual"]
    eval_df["pred"] = artifacts["pred"]
    eval_df["series"] = eval_df["market"].astype(str) + " | " + eval_df["commodity"].astype(str)
    top = eval_df["series"].value_counts().index[:4]
    fig2, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, name in zip(axes.ravel(), top):
        s = eval_df[eval_df["series"] == name].sort_values("month")
        ax.plot(s["month"], s["actual"], "o-", label="Actual")
        ax.plot(s["month"], s["pred"], "s--", label="Forecast")
        ax.set_title(name, fontsize=9)
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        ax.legend(fontsize=8)
    fig2.suptitle(f"Actual vs {max(HORIZONS)}-month forecast (held-out window)")
    fig2.tight_layout()
    fig2.savefig(RESULTS_DIR / "forecast_eval.png", dpi=120)

    imp = pd.Series(artifacts["model"].feature_importances_,
                    index=artifacts["features"]).sort_values()
    fig3, ax3 = plt.subplots(figsize=(8, 6))
    imp.plot.barh(ax=ax3)
    ax3.set_title("Feature importance (LightGBM, 6-month horizon)")
    fig3.tight_layout()
    fig3.savefig(RESULTS_DIR / "feature_importance.png", dpi=120)


if __name__ == "__main__":
    run()
