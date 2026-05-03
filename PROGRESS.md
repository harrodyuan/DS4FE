# DS4FE — Weekly Progress Log

Limit Order Book Feature Engineering Series

Start date: February 25, 2026 | Today: May 3, 2026 | Total: **10 weeks**

---

## Weeks 1–4 (Feb 25 – Mar 24, 2026)

**Background reading and onboarding.**

- Reviewed LOB microstructure literature (order flow imbalance, depth profiles, Kyle's Lambda)
- Familiarized with Databento API and mbp-1/mbp-10 schemas
- Scoped the series structure and teaching angles
- No commits — planning and reading phase

---

## Week 5 (Mar 25–31, 2026)

**Built the complete 5-part series from scratch.**

- Initial commit: Parts 1 and 2 notebooks (data tour + feature engineering on daily OHLCV data)
- Rewrote all markdown prose in textbook style; restructured into a clean 3-part series
- Added Part 3: single-stock return prediction (Ridge, walk-forward CV, OOS R²)
  - Added look-ahead bias demo, feature intuition, shortened prose
- Added Part 4: LOB feature engineering (Databento mbp-1 data, OFI, IC analysis, calm vs stress comparison, cross-asset SPY signal)
  - Added prediction benchmark (Ridge walk-forward, OOS R² by hour)
  - Added XGBoost benchmark with look-ahead audit note
  - Restructured with Part I / Part II divider; expanded IC vs OOS R² explanation
- Added Part 5: Multifactor models and cross-sectional return prediction
- Part 3: added OOS R² metric, AAPL LOB vs daily cross-frequency comparison
- Parts 3 + 5: added cross-sectional vs time-series momentum insight
- Series audit: fixed all cross-references, Series Notes, feature counts
- Added `.gitignore` to exclude data files and local working materials
- Added `DS4FE_Series_Notes.md`

**Deliverables:** Full 5-part series (Parts 1–5) — first complete version committed.

---

## Weeks 6–7 (Apr 1–14, 2026)

**Consolidation and structural refactoring.**

- Split monolithic Part 4 into four focused sub-notebooks:
  - Part 4a: LOB Data Demo — Databento mbp-1 schema, event types, calm (Oct 2023) vs stress (Aug 5 2024 BOJ shock) period comparison
  - Part 4b: Features & IC — OFI construction, daily IC analysis, t-stats, calm vs stress IC, cross-asset SPY signal
  - Part 4c: Model Importance — Ridge vs XGBoost benchmark, walk-forward CV, hyperparameter grid search, OOS R² by hour
  - Part 4d: Multi-Stock — pooled 5-symbol model (NVDA, AAPL, TSLA, MSFT, SPY), within-symbol rolling z-score normalization, cross-sectional OBI deviation
- Removed API key from source; moved to `.env`

**Deliverables:** Parts 4a–4d structured and committed.

---

## Week 8 (Apr 15–21, 2026)

**Added trade tape data and Part 4e; executed all notebooks.**

- Downloaded Databento `trades` schema for all 5 symbols × 2 periods (calm Oct 2023 + stress Aug 2024); 10 parquet files, ~560 MB total
- Added `download_trades.py` script
- Built Part 4e: Trade Tape Features
  - Educational framing: "The Illusion of the Order Book — Intent vs. Action"
  - Feature 1: `signed_volume_ratio` (SVR) — buyer vs seller aggressor volume, range [−1, 1]
  - Feature 2: `mid_vwap_bias` (MVB) — rolling 60s VWAP vs mid-price in basis points
  - IC analysis: SVR IC @ 1s = −0.065 (strongest feature), MVB IC @ 10s = +0.016
  - SVR ↔ OBI correlation is low → genuinely orthogonal to the order book
  - Walk-forward R² comparison: XGBoost LOB-only = −0.00115 → LOB + SVR + MVB = +0.00017
- Executed Part 4d with per-symbol OOS R² bar chart

**Deliverables:** Part 4e executed; all 4a–4e notebooks clean and pushed to `main`.

---

## Week 9 (Apr 22–28, 2026)

**Revised and polished Part 4a for presentation.**

- Deleted old monolithic `DS4FE_Part4_LOB_Features.ipynb` (superseded by 4a–4e)
- Fixed uint32 overflow bug in calm vs stress OBI computation (bid/ask sizes cast to int64)
- Replaced NVDA with SPY for the calm vs stress comparison — SPY shows the textbook spread spike and OBI instability on Aug 5; NVDA's shock was overnight so intraday looked misleading
- Fixed order book snapshot: switched from row 500 (09:30:02, book not yet built) to 11:30 ET mid-session
- Added calm vs stress event-type grouped bar chart (normalized to %, Fill partial filtered out)
- Added section transition markdown cells throughout
- Added concluding Summary cell with handoff to Part 4b
- Rewrote all em-dash heavy prose to cleaner phrasing

**Deliverables:** Part 4a fully polished with correct outputs; pushed to `main`.

---

## Week 10 (Apr 29–May 5, 2026)

**Built Part 4f (ISOMAP), Part 4g (multi-symbol DR comparison), and Part 4h (joint ISOMAP). Professor check-in on May 1.**

### Part 4f — Unsupervised LOB Feature Engineering: ISOMAP

- Downloaded full calm-period mbp-10 data for NVDA (Oct 2–31, 22 trading days, ~50M ticks) via Databento
- Built Part 4f notebook end-to-end:
  - Animated order book visualization (9:30–10:30 opening hour, 30-second steps, interactive jshtml player)
  - Constructed 10D OBI feature matrix: OBI at each of 10 depth levels, aggregated to 1-minute bars
  - Rolling L0–L9 correlation chart: banded structure (adjacent levels 0.76–0.86; L0 vs L9 = 0.07) — direct evidence that the book state lives on a low-dimensional manifold
  - ISOMAP fit (n_components=2, n_neighbors=15): reconstruction error = 0.028, **97.2% of geodesic manifold structure preserved in 2D**
  - PCA comparison: 78.6% in 2D — 18.6 pp gap confirms manifold is curved, not flat
  - Scree plot: elbow at 2 components (3rd adds < 1 pp), confirming intrinsic dimensionality ≈ 2
  - Depth profile bar chart: Z₁ peaks at L6 (book-wide consensus), Z₂ has sign-flip at L4–L5 (HFT-vs-institutional contrast axis)
  - Improved 2×5 all-levels scatter grid: shared colorbar, gradient-direction arrows, spine color-coding by dominant axis
  - OOS projection via Nyström extension: Oct 10–31 states land cleanly inside training manifold region
  - XGBoost IC comparison: Raw 10D OBI IC ≈ 0; ISOMAP 2D IC = +0.048 (t=1.66, marginal); ISOMAP distills correlated inputs into a more useful representation

**Key finding:** The 10-level OBI vector has intrinsic dimensionality ≈ 2. ISOMAP recovers this structure with 97.2% fidelity. Two axes emerge: Z₁ (book-wide consensus, peaks at L5–L6) and Z₂ (HFT-vs-institutional contrast, sign-flip at L4–L5).

### Part 4g — Multi-Symbol DR Comparison (NVDA, AAPL, MSFT)

- Downloaded full Oct 2023 mbp-10 data for AAPL and MSFT (split requests to avoid timeout)
- Ran identical pipeline per symbol (PCA, ISOMAP, t-SNE, UMAP, K-Means, OOS IC)
- Cross-symbol comparison table:

  | Metric | NVDA | AAPL | MSFT |
  |--------|------|------|------|
  | ISOMAP 2D preserved | 97.2% | 96.7% | 96.7% |
  | PCA 2D variance | 78.6% | 82.2% | 87.0% |
  | ISOMAP gap over PCA | +18.6 pp | +14.5 pp | +9.7 pp |
  | Z₁ peak level | L6 | L7 | L4 |
  | Z₂ sign-flip (crossover) | ~L4–L5 | ~L5–L6 | ~L4 |
  | Best K (clusters) | 2 | 2 | 2 |
  | OOS IC (ISOMAP 2D) | +0.004 | +0.011 | +0.020 |

**Key finding:** Z₁/Z₂ structure is consistent across all three large-cap Nasdaq names. K=2 (open/close vs mid-day) is universal. ISOMAP consistently outperforms PCA by 10–19 pp.

### Part 4h — Joint ISOMAP: Unified LOB Manifold

- Pooled all three symbols' 1-min OBI bars (no scaler — OBI already in [−1, 1]) into one ~19,300-bar training set
- Fit ONE Isomap (n_components=2, n_neighbors=30) on the pooled data
- Joint reconstruction error = 0.027 → **97.3% preserved** — matches per-symbol quality
- Symbols **overlap** on the shared manifold — no clean stock-identity separation; regimes are market-wide
- Z₁/Z₂ interpretation **survives** joint training: Z₁=consensus (peaks L5), Z₂=contrast (sign-flip L4–L5)
- Joint UMAP: K=2 optimal (silhouette = 0.538, highest of all models); cluster membership is mixed-symbol
- OOS IC near zero in pooled cross-symbol prediction — per-symbol calibration needed for predictive use

**Key finding:** The LOB manifold is approximately universal across large-cap Nasdaq names in the same market window. Stock identity is a second-order effect; intraday time-of-day regimes dominate.

### Next milestone

- Part 4i — project Aug 2024 BOJ shock week onto the calm manifold (unsupervised regime detection)

**Deliverables:** Parts 4f, 4g, 4h notebooks and all scripts committed; figures in `figures/`.

---
