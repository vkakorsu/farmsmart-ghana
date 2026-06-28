"""FarmSmart Ghana - interactive demo.

Run locally:   streamlit run app.py
Deploy free:   push to GitHub, then create an app at https://share.streamlit.io
"""

from __future__ import annotations

import calendar

import pandas as pd
import streamlit as st

from src import data as data_mod
from src import predict
from src.features import to_monthly

st.set_page_config(page_title="FarmSmart Ghana", page_icon="🌾", layout="wide")


@st.cache_data(show_spinner="Loading Ghanaian market data...")
def load_monthly() -> pd.DataFrame:
    df = data_mod.clean(data_mod.load(data_mod.download()))
    return to_monthly(df, staples_only=True)


@st.cache_data(show_spinner="Preparing features...")
def load_features() -> pd.DataFrame:
    return predict.build_all()


@st.cache_resource(show_spinner="Training forecast models (first load only)...")
def get_models(_feat: pd.DataFrame):
    return predict.train_models(_feat)


def main() -> None:
    st.title("FarmSmart Ghana")
    st.caption(
        "Price intelligence and sell-timing advice for smallholder farmers, "
        "built on real Ghanaian market data (WFP via HDX)."
    )

    monthly = load_monthly()
    commodities = sorted(monthly["commodity"].unique())
    default_idx = commodities.index("Maize") if "Maize" in commodities else 0

    col_a, col_b = st.columns(2)
    commodity = col_a.selectbox("Crop", commodities, index=default_idx)
    years = col_b.slider("Years of recent data to use", 1, 8, 3)

    sub = monthly[monthly["commodity"] == commodity]
    cutoff = sub["month"].max() - pd.DateOffset(years=years)
    recent = sub[sub["month"] >= cutoff]
    retail = recent[recent["pricetype"].str.lower() == "retail"]
    recent = retail if not retail.empty else recent

    if recent.empty:
        st.warning("No data for this selection.")
        return

    # WHERE to sell.
    by_market = (
        recent.groupby("market", observed=True)["price"].median().round(2).sort_values(ascending=False)
    )
    best_market, best_price = by_market.index[0], by_market.iloc[0]
    worst_market, worst_price = by_market.index[-1], by_market.iloc[-1]
    gap = (best_price - worst_price) / worst_price * 100 if worst_price else 0

    # WHEN to sell.
    tmp = recent.copy()
    tmp["mnum"] = tmp["month"].dt.month
    by_month = tmp.groupby("mnum")["price"].median()
    hi_m, lo_m = int(by_month.idxmax()), int(by_month.idxmin())
    swing = (by_month.max() - by_month.min()) / by_month.min() * 100 if by_month.min() else 0

    st.subheader("Advice")
    c1, c2 = st.columns(2)
    c1.metric("Best market to sell", best_market, f"+{gap:.0f}% vs {worst_market}")
    c1.write(f"Prices are highest in **{best_market}** (about GHS {best_price}/kg) "
             f"and lowest in **{worst_market}** (about GHS {worst_price}/kg).")
    c2.metric("Best month to sell", calendar.month_name[hi_m], f"+{swing:.0f}% vs {calendar.month_name[lo_m]}")
    c2.write(f"Prices usually peak around **{calendar.month_name[hi_m]}** and bottom "
             f"around **{calendar.month_name[lo_m]}**. Storing across that window has "
             f"historically gained about **{swing:.0f}%**.")

    st.subheader("Seasonality (median GHS/kg by month)")
    season_df = by_month.rename(index=lambda m: calendar.month_abbr[m]).rename("GHS/kg")
    st.bar_chart(season_df)

    st.subheader("Where prices are highest (median GHS/kg by market)")
    st.bar_chart(by_market.rename("GHS/kg"))

    st.subheader("Price forecast (national median GHS/kg, 6 months ahead of latest data)")
    try:
        feat = load_features()
        models = get_models(feat)
        fc = predict.forecast_commodity(feat, models, commodity)
        if fc.empty or "forecast" not in set(fc["kind"]):
            st.info("Not enough history to forecast this crop. Showing price history.")
            st.line_chart(sub.groupby("month", observed=True)["price"].median().rename("GHS/kg"))
        else:
            wide = (
                fc.pivot_table(index="month", columns="kind", values="price")
                .rename(columns={"history": "History (GHS/kg)", "forecast": "Forecast (GHS/kg)"})
            )
            recent = wide[wide.index >= (wide.index.max() - pd.DateOffset(years=4))]
            st.line_chart(recent)
            last_hist = fc[fc["kind"] == "history"].iloc[-1]
            last_fc = fc[fc["kind"] == "forecast"].iloc[-1]
            change = (last_fc["price"] - last_hist["price"]) / last_hist["price"] * 100
            arrow = "rise" if change >= 0 else "fall"
            st.write(
                f"From the latest available data ({last_hist['month'].strftime('%B %Y')}), "
                f"the model projects {commodity} prices to **{arrow} about "
                f"{abs(change):.0f}%** over the following six months "
                f"(to about GHS {last_fc['price']:.2f}/kg by {last_fc['month'].strftime('%B %Y')}). "
                "This projects six months beyond the latest official WFP observation. In "
                "production the same model runs on a live feed such as MoFA market prices."
            )
    except Exception as exc:  # keep the demo resilient
        st.warning(f"Forecast unavailable right now ({exc}). Showing price history.")
        st.line_chart(sub.groupby("month", observed=True)["price"].median().rename("GHS/kg"))

    st.caption(
        "Forecasts and advice are estimates, not guarantees. Source: World Food "
        "Programme Ghana Food Prices (HDX), normalised to price per kilogram. "
        "The official WFP series spans 2006 to mid-2023 (latest available); in "
        "production the same pipeline runs on a live feed such as MoFA market prices."
    )


if __name__ == "__main__":
    main()
