# DS4FE — Weekly Progress Log

Limit Order Book Feature Engineering Series

Start date: February 25, 2026 | Today: May 11, 2026 | Total: **12 weeks**

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

### Part 4i — Stress Period Projection: BOJ Shock Week on the Calm Manifold

- Downloaded NVDA mbp-10 data for Aug 5–9, 2024 (BOJ shock week, 31.6M rows)
- Built `run_4i_stress_projection.py` comparing two feature representations:
  - **Model A (10 features — OBI means only):** stress bars land inside calm cloud; only 3.6% outside calm 95th percentile
  - **Model B (20 features — OBI means + within-minute OBI std):** 16.8% outside 95th; stress mean distance = 2× calm mean
- Key insight: 1-minute mean OBI washes out directional crashes (every update within the minute points the same way → mean is moderate). The within-minute standard deviation is the stress signal — lower during a directional move, not higher.
- Additional findings: book depth 11× higher during stress (includes 10:1 June 2024 split); manifold path length 0.1–1.4 SD shorter than calm (book locked into fewer states despite larger price moves)
- Built `DS4FE_Part4i_Stress_Projection.ipynb` with 6 figures including Aug 5 minute-by-minute trajectory and manifold distance through the week

**Key finding:** Mean OBI alone is insufficient to detect regime stress. Adding the second moment (within-minute OBI std) makes the structural abnormality detectable — 4× more stress bars flagged — without labels, price data, or model retraining.

**Deliverables:** Parts 4f, 4g, 4h, 4i notebooks and all scripts committed; figures in `figures/`.

---

## Week 11 (May 5–11, 2026)

**Standalone demo notebook polished based on detailed external review. Meeting preparation.**

### DS4FE_ISOMAP_Demo.ipynb — Full rebuild

Received a cell-by-cell critique of the demo notebook and implemented all corrections:

**Structural changes (56 cells → 42 cells, 8 sections):**
- Rebuilt notebook from scratch using `build_demo_notebook.py` for reproducibility
- Replaced mislabeled `4f_isomap_vs_pca.png` (it showed OBI₀ coloring, not a PCA comparison) with `4g_NVDA_scree.png` (full October data)
- Trimmed per-symbol deep dive (15 figures) to summary table + joint embedding + joint depth profile
- Added Part 4i stress projection as Section 8 (3 figures: model comparison, Aug 5 path, manifold distance through week)

**12 factual and framing fixes:**

| Fix | Detail |
|-----|--------|
| Raw data scale | Removed "50M ticks/day" headline; states "8,580 bars after aggregation" |
| Snapshot description | "Noisy and uneven — that's why we aggregate" instead of "balanced" |
| k inconsistency | Explicitly states k=15 per-symbol, k=30 for joint/stress; resolves figure label mismatch |
| "97% fidelity" language | Reframed as geometry-fidelity measure; note it is not directly comparable to PCA explained variance |
| Z₂ sign | Fixed to match depth-profile figure: negative near top-of-book, positive deeper; sign arbitrariness noted |
| Time-of-day regime claim | "Shared book-state space" rather than "time of day is stronger than stock identity" |
| OOS IC overclaim | p≈0.20; "main contribution is state representation, not standalone prediction" |
| Joint embedding | Weakened unsupported "stronger organizing dimension" claim |
| BOJ chronology | Rate hike July 31 (effective Aug 1); selloff Aug 5 — not conflated |
| Stress-distance contradiction | "Shifts upward, more extreme tail" — not "persistently above 95th" (only 16.8% exceed it) |
| Conclusion | Rewritten around state representation + regime detection; no return-prediction overclaim |
| Narrative order | Section 2 reordered: snapshot → heatmap → correlation → ISOMAP motivation |

### New figure: neighbor sensitivity

- Generated `4g_NVDA_neighbor_sensitivity.png`: ISOMAP 2D fidelity for k = 5, 10, 15, 20, 30, 50
- Result: (1 − residual) stays 0.951–0.980 across the full range — the 2D structure is stable and not sensitive to the choice of k
- Directly answers the expected question: "How sensitive is this to n_neighbors?"

**Deliverables:** `DS4FE_ISOMAP_Demo.ipynb` (42 cells, 18 embedded figures), `build_demo_notebook.py`, `gen_neighbor_sensitivity.py` — all committed and pushed.

---

## Week 12 (May 11, 2026)

**Pre-meeting: ISOMAP vs PCA cross-over test reveals ISOMAP does not significantly outperform PCA.**

### Cross-over verification (DS4FE_ISOMAP_crossover_verification.ipynb)

Identified a methodological flaw in earlier notebooks: the scree plot compared ISOMAP's geodesic reconstruction error against PCA's cumulative Euclidean variance — two different loss functions — creating a false impression that "ISOMAP needs 2D while PCA needs 7D."

Correct cross-over test (both methods, same metric, k = 1…10, NVDA Oct 2023):

| k | PCA R² | ISOMAP R² | PCA geo ρ | ISOMAP geo ρ |
|---|--------|-----------|-----------|--------------|
| 2 | 0.769 | 0.761 | 0.944 | 0.950 |
| 7 | 0.960 | 0.876 | 0.970 | 0.979 |

- At k=2, PCA and ISOMAP tied on Euclidean variance (~77% each)
- ISOMAP geo ρ consistently ~0.007 higher — real but small advantage
- ISOMAP R² plateaus at ~0.876 because it optimises geodesic structure, not Euclidean variance

**Key finding: ISOMAP does not meaningfully outperform PCA on this dataset. The OBI manifold is mildly curved; PCA's flat approximation captures the dominant structure equally well at 2D.**

### Stress detection comparison (DS4FE_ISOMAP_stress_comparison.ipynb)

Tested PCA vs ISOMAP vs UMAP for separating Aug 2024 BOJ shock data from Oct 2023 calm manifold (KS test, Cohen's d, % above 95th percentile):

| Method | KS stat | % flagged | Cohen's d |
|--------|---------|-----------|-----------|
| PCA    | 0.026 n.s. | 6.3% | +0.034 |
| ISOMAP | 0.022 n.s. | 5.7% | +0.010 |
| UMAP   | 0.048 *    | 6.7% | +0.071 |

All three weak — consistent with Part 4i: adding within-minute OBI std is the key feature engineering step, not the DR method choice.

**Key finding: ISOMAP does not outperform PCA for stress detection. Method choice is secondary to feature engineering.**

### New files added

- `DS4FE_LOB_ISOMAP_v2.ipynb` — consolidated ISOMAP notebook (NVDA+AAPL+MSFT, full Oct 2023)
- `DS4FE_ISOMAP_crossover_verification.ipynb` — cross-over test
- `DS4FE_ISOMAP_stress_comparison.ipynb` — PCA vs ISOMAP vs UMAP stress detection

---
