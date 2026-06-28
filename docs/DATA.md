# Data Sources and Disclosure

FarmSmart Ghana uses Ghanaian market data. All sources are public and documented
here for the reproducibility and ethics review.

## Primary source (used by the model)

### WFP Ghana Food Prices (Humanitarian Data Exchange)
- **Link:** https://data.humdata.org/dataset/wfp-food-prices-for-ghana
- **Publisher:** World Food Programme (WFP) Price Database.
- **Content:** monthly retail and wholesale prices of food commodities across
  Ghanaian markets, with market name, admin1/admin2, latitude/longitude,
  commodity, unit, price-type, currency (GHS) and USD price.
- **Coverage:** Ghanaian markets (Accra, Kumasi, Techiman, Tamale, Cape Coast and
  more); staples include maize, rice, millet, sorghum, cassava, yam, gari,
  soybeans, cowpeas. Monthly observations span **2006 to July 2023**, which is the
  latest the WFP series currently provides. The HDX file was re-published in May
  2026, but the coverage window is unchanged.
- **Access:** downloaded automatically via the HDX CKAN API in `src/data.py`. No
  special permission required.
- **Licence:** per the HDX dataset page.

## Complementary Ghanaian sources

- **MoFA SRID** (https://srid.mofa.gov.gh/datasets): agricultural commodity prices
  and production estimates (yield, area, production), and FSNMS food-security
  tables. Historical mirror: GitHub `DavidQuartey/Weekly-Agric-Market-Prices`
  (MoFA weekly prices 2009-2014, MIT licence).
- **Ghana Statistical Service** (https://microdata.statsghana.gov.gh): census,
  AHIES, DHS 2022 for socio-economic context.
- **Ghana Open Data Initiative**: national open-data portal across ministries.

## Data currency and the production plan

The WFP Ghana series ends in July 2023, so the public demo learns seasonality and
spatial structure from 2006-2023 and forecasts forward from that latest month. The
modelling pipeline is source-agnostic: it runs on any monthly market-price series.
For a real deployment we would connect a current Ghanaian feed (for example MoFA
SRID weekly market prices, or an aggregator such as Esoko), at which point the same
model forecasts from the newest available month with no code changes.

## Cleaning and handling (reproducibility + ethics)

- Prices are normalised to **GHS per kg** (`src/data.py`), because markets report
  different pack sizes. Non-weight units are dropped.
- Per-commodity outliers are removed with a 1st-99th percentile clip.
- Analysis is restricted to storable staples and to market-commodity series with
  at least 36 months of history.
- The data is aggregate market-price data with no personal information, so privacy
  risk is minimal. Each source licence is respected.
