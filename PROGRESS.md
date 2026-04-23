# DS4FE — Weekly Progress Log

RA work for Professor Schafer | Limit Order Book Feature Engineering Series

---

## Week 1

**Started the series from scratch.**

- Initial commit: Parts 1 and 2 notebooks (data tour + feature engineering on daily OHLCV data)
- Rewrote all markdown prose in textbook style
- Restructured into a clean 3-part series; removed duplicated content
- Added Part 3: single-stock return prediction (Ridge, walk-forward CV, OOS R²)
- Part 3: added look-ahead bias demo, feature intuition, shortened prose
- Added `.gitignore` to exclude data files and DSII course materials
- Added `DS4FE_Series_Notes.md`

**Deliverables:** Parts 1, 2, 3 — daily OHLCV pipeline complete.

---

## Week 2

**Built the LOB series (Part 4) and completed the full series audit.**

- Added Part 4: LOB feature engineering (Databento mbp-1 data, OFI, IC analysis, calm vs stress comparison, cross-asset SPY signal)
- Added Part 5: Multifactor models and cross-sectional return prediction
- Part 4: added prediction benchmark (Ridge walk-forward, OOS R² by hour)
- Part 4: added XGBoost benchmark with look-ahead audit note
- Part 4: restructured with Part I / Part II divider; expanded IC vs OOS R² explanation
- Part 3: added OOS R² metric, AAPL LOB vs daily cross-frequency comparison
- Parts 3 + 5: added cross-sectional vs time-series momentum insight
- Series audit: fixed all cross-references, Series Notes, feature counts

**Deliverables:** Full 5-part series complete (Parts 1–5).

---

## Week 3

**Split monolithic Part 4 into four focused sub-notebooks.**

- Part 4a: LOB Data Demo — Databento mbp-1 schema, event types, calm (Oct 2023) vs stress (Aug 5 2024 BOJ shock) period comparison
- Part 4b: Features & IC — OFI construction, daily IC analysis, t-stats, calm vs stress IC, cross-asset SPY signal
- Part 4c: Model Importance — Ridge vs XGBoost benchmark, walk-forward CV, hyperparameter grid search, OOS R² by hour
- Part 4d: Multi-Stock — pooled 5-symbol model (NVDA, AAPL, TSLA, MSFT, SPY), within-symbol rolling z-score normalization, cross-sectional OBI deviation

**Deliverables:** Parts 4a–4d with executed outputs.

---

## Week 4

**Added trade tape data and Part 4e.**

- Downloaded Databento `trades` schema for all 5 symbols × 2 periods (calm Oct 2023 + stress Aug 2024); 10 parquet files, ~560 MB total
- Added `download_trades.py` script
- Built Part 4e: Trade Tape Features
  - Educational framing: "The Illusion of the Order Book — Intent vs. Action"
  - Feature 1: `signed_volume_ratio` (SVR) — buyer vs seller aggressor volume, range [−1, 1]
  - Feature 2: `mid_vwap_bias` (MVB) — rolling 60s VWAP vs mid-price in basis points
  - IC analysis: SVR IC @ 1s = −0.065 (strongest feature), MVB IC @ 10s = +0.016
  - SVR ↔ OBI correlation is low → genuinely orthogonal to the order book
  - Walk-forward R² comparison: XGBoost LOB-only = −0.00115 → LOB + SVR + MVB = +0.00017
- Cleaned all source-attribution references from Parts 4b, 4c, 4d, 4e; replaced with neutral microstructure literature framing

**Deliverables:** Part 4e executed, all notebooks clean, pushed to `main`.

---

## Week 5

**Revised and polished Part 4a for presentation.**

- Deleted old monolithic `DS4FE_Part4_LOB_Features.ipynb` (superseded by 4a–4e)
- Fixed uint32 overflow bug in calm vs stress OBI computation (bid/ask sizes cast to int64)
- Replaced NVDA with SPY for the calm vs stress comparison — SPY shows the textbook spread spike and OBI instability on Aug 5; NVDA's shock was overnight so intraday looked misleading
- Fixed order book snapshot: switched from row 500 (09:30:02, book not yet built) to 11:30 ET mid-session; added int64 cast to prevent size overflow in bar chart
- Added calm vs stress event-type grouped bar chart (normalized to %, Fill partial filtered out)
- Added section transition markdown cells throughout: "Spread and Mid-Price Dynamics", "Spread Across Stocks", "Order Book Imbalance Over Time", "Depth Profile"
- Added concluding Summary cell with handoff to Part 4b
- Rewrote all em-dash heavy prose to cleaner phrasing throughout

**Deliverables:** Part 4a fully polished with correct outputs; pushed to `main`.
