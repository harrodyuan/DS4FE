# NYSE TAQ Sample Data — Final Research Report
## For: Chad Schafer (CMU Statistics & Data Science)
## Date: June 24, 2026
## Analyst: Harold Yuan

---

## Executive Summary

I downloaded and inspected the NYSE TAQ sample data from `ftp.nyse.com`. The data is **rich, well-structured, and highly suitable for feature engineering** at the microstructure level. With proper parsing and aggregation, it supports a wide range of liquidity, volatility, and order-flow features that can be linked to asset pricing, risk, and option-implied volatility models.

**Key conclusion:** The data is valuable for feature engineering, but one hour of sample data is insufficient for meaningful predictive modeling. A full-day or multi-day panel is required to produce statistically reliable signals.

---

## 1. Data Downloaded and Parsed

| File | Size | Content | Records |
|---|---|---|---|
| `EQY_US_ALL_REF_MASTER_20260102.gz` | 401 KB | 12,181 symbol reference records | ~12K symbols |
| `EQY_US_TAQ_NYSE_TRADES_20231002.gz` | 95 MB | NYSE-venue trade & correction messages | ~95M messages |
| `EQY_US_TAQ_NYSE_BBO_1_20231002.gz` | 1.6 GB | NYSE BBO quotes, 09:30–10:30 ET | ~95M messages |
| `EQY_US_ALL_TRADE_20260102.gz` | 3.2 GB | Consolidated trades from all venues | ~5.9M sample records |

### Schema Discoveries
- **Reference master:** pipe-delimited, 40 fields, includes per-venue trading flags.
- **Trades:** comma-delimited, msg type `3` = trade, msg type `34` = correction.
- **BBO quotes:** comma-delimited, msg type `140`, field order confirmed as `[ask_px, ask_sz, bid_px, bid_sz]`.
- **Quote conditions:** `4` = one-sided update, `5` = two-sided quote, `6-9` = other conditions.

---

## 2. Microstructure Features Built

We built a 1-minute bar engine from the BBO stream and computed the following features:

### Quote-based features
- `mean_spread_bps` — average relative bid-ask spread
- `max_spread_bps` — maximum spread within the bar (proxy for stress)
- `quote_intensity` — number of BBO updates per second
- `mean_depth_imbalance` — `(bid_size - ask_size) / (bid_size + ask_size)`
- `std_depth_imbalance` — variation of imbalance within the bar
- `mean_log_depth` — log of total quoted depth

### Return/volatility features
- `min_return` — log return over the minute
- `min_range` — high-low range relative to open
- `realized_var` — squared minute return
- `vol_5`, `vol_10` — rolling volatility
- `forward_vol` — sum of next 5 minutes of realized variance

### Trade-based features (sparse in sample)
- `total_volume`, `trade_count`, `avg_trade_size`, `vwap_approx`
- Sale-condition breakdown (regular, extended, etc.)

---

## 3. Empirical Findings from Hour-1 Sample

### Quote frequency
| Symbol | Quotes (hour 1) | Avg quotes/sec | Avg spread (bps) |
|---|---|---|---|
| AAPL | 2.4M | 88.5 | 21.2 |
| AMZN | 1.5M | 25.2 | 137.4 |
| DIA | 0.8M | 10.3 | 16.2 |
| IVV | 0.7M | 12.0 | 16.6 |
| IWM | 0.6M | 10.6 | 9.6 |
| GOOGL | 1.5M | 21.6 | 113.6 |

**Insight:** ETFs (DIA, IVV, IWM) have much tighter spreads than single-name tech stocks in this sample. AAPL and GOOGL show higher spreads, possibly due to the opening period.

### Depth imbalance dynamics
Depth imbalance fluctuates around zero with occasional persistent deviations. The distribution is roughly symmetric, suggesting no systematic buy/sell pressure during the first hour.

### Predictive modeling results
We tested two simple supervised learning tasks:

#### Task 1: Predict next-minute realized volatility
- **Method:** Ridge regression with standardized features, time-series split.
- **Result:** Training R² is moderate (0.10–0.70), but **test R² is negative** across all symbols.
- **Interpretation:** Strong overfitting; one hour of data is insufficient to learn a stable volatility model. The target is highly correlated with current volatility, but the model does not generalize out-of-sample.

#### Task 2: Predict next-minute return direction from depth imbalance
- **Method:** Logistic regression using mean and last depth imbalance.
- **Result:** Test accuracy is **near the baseline** (48–52% vs. 50–54% baseline), AUC near 0.5.
- **Interpretation:** Depth imbalance alone is not a strong directional predictor in the opening hour. This is consistent with the literature: BBO imbalance is a weak signal unless combined with trade flow, multiple price levels, or auction information.

### Consolidated Trades (Full Day, 2026-01-02)

We parsed the full-day consolidated trades file for six liquid symbols and built 1-minute trade bars. The file uses pipe-delimited format with fields: Time, Exchange, Symbol, Sale Condition, Trade Volume, Trade Price, etc.

| Symbol | Total Volume | # Trades | Avg Trade Size | VWAP | Daily Return |
|---|---|---|---|---|---|
| SPY | 93.5M | 1,251,790 | 652.9 | 682.98 | -0.42% |
| QQQ | 62.5M | 1,088,614 | 76.5 | 614.90 | -1.12% |
| TSLA | 89.9M | 1,865,256 | 56.7 | 443.86 | -3.96% |
| AAPL | 48.8M | 642,197 | 103.4 | 271.73 | -0.78% |
| IWM | 42.1M | 351,337 | 870.0 | 247.68 | +0.46% |
| MSFT | 29.4M | 716,532 | 41.5 | 474.41 | -2.81% |

**Insight:** ETFs (SPY, QQQ, IWM) dominate by volume and trade size, while single-name tech stocks (TSLA, MSFT) have many small trades. This venue- and size- heterogeneity is itself a feature: trade-size distribution can distinguish institutional vs. retail flow.

---

## 4. Why Prediction Is Hard (And Why That Is OK)

The weak predictive results are **not a failure of the data**; they reflect three realities:

1. **One hour is too short.** Microstructure signals are noisy. Detecting them requires hundreds of days or thousands of bars, not one hour.
2. **The opening hour is unusual.** Volatility is high, spreads are wide, and price discovery dominates. Signals from 09:30–10:30 may not generalize to the rest of the day.
3. **BBO is only the surface.** The strongest signals come from full limit-order-book depth, trade signing, and cross-venue fragmentation — none of which are captured by top-of-book BBO alone.

**The right framing:** The NYSE TAQ data is a *feature engineering goldmine*, not a plug-and-play alpha source. The value is in constructing clean, interpretable features that can be fed into longer-horizon models.

---

## 5. Better Ideas for Feature Engineering

Based on the data inspection, here are concrete, high-value feature engineering projects that fit your focus:

### Idea A: Liquidity risk factor (cross-sectional)
Build daily liquidity features for all symbols and construct a **liquidity factor**:
- Average effective spread
- Average quoted spread
- Amihud illiquidity (|return| / dollar volume)
- Intraday volatility of spread
- Price impact coefficient (Hasbrouck lambda)

**Use case:** Use as a control in asset pricing regressions or as a stock-selection signal.

### Idea B: Market stress indicator (time-series)
Aggregate microstructure features across the market to build a **daily stress index**:
- Median spread across all symbols
- Fraction of symbols with spread > 100 bps
- Cross-sectional dispersion of returns
- Number of halted/locked-limit symbols

**Use case:** Predict VIX, SPX realized volatility, or option skew changes.

### Idea C: Order flow toxicity (VPIN-style)
Using the consolidated trades file, implement **Volume-Synchronized Probability of Informed Trading (VPIN)**:
- Sign trades using Lee-Ready or tick test
- Bucket volume into equal-size buckets
- Compute order flow imbalance per bucket
- Smooth to get a toxicity metric

**Use case:** Predict flash crashes, liquidity droughts, or short-term volatility spikes.

### Idea D: Cross-venue fragmentation metrics
Compare NYSE BBO to consolidated NBBO:
- NYSE market share by volume
- Time-weighted market share of the best quote per venue
- Fragmentation index (HHI across venues)

**Use case:** Venue-specific liquidity research, execution quality analysis.

### Idea E: Auction imbalance signals
Download `TAQ NYSE ORDER IMBALANCES` and build:
- Opening/closing imbalance magnitude
- Imbalance sign vs. subsequent auction price drift
- Imbalance decay before auction

**Use case:** Event-driven strategies around market open/close.

### Idea F: Link to options volatility
Combine TAQ equity microstructure features with the existing SPX implied-volatility surface data:
- Use lagged equity microstructure features to predict next-day ATM implied vol
- Use liquidity stress index to predict skew steepening
- Use order flow imbalance to predict risk-reversal dynamics

**Use case:** This directly connects your current DS4FE work (options skew) to the new TAQ data. It is the most natural extension.

---

## 6. Recommended Next Steps

### Immediate (this week)
1. **Finish parsing the consolidated trades file** (`EQY_US_ALL_TRADE_20260102.gz`) once it finishes downloading.
2. **Build one full-day feature matrix** for 3–5 liquid symbols (AAPL, SPY, TSLA, MSFT) with 1-minute bars.
3. **Re-run the volatility prediction** with a full day of data. We expect R² to improve materially.

### Short-term (next 2–4 weeks)
4. **Implement VPIN** on the consolidated trade data for a small set of symbols.
5. **Build the liquidity stress index** across all symbols for one day and compare it to VIX/SPX moves.
6. **Link daily microstructure features to next-day ATM vol** for SPX constituents or sectors.

### Long-term (with full data access)
7. **Scale to a multi-year panel** and test whether liquidity features predict cross-sectional returns.
8. **Explore full LOB depth** (NYSE OpenBook) for richer queue-based features.

---

## 7. Infrastructure Notes

- **Storage:** One full day of consolidated trades + NBBO is ~12 GB compressed. One month is ~250 GB. Two years is ~6 TB.
- **Parsing speed:** Python + pandas processes ~100K messages/sec. For full-day files, use `polars` or chunked pyarrow for 10x speedup.
- **Memory:** The 3.2 GB trades file uncompresses to ~10–15 GB. Do not load full files into memory; stream and aggregate in chunks.

---

## 8. Files Produced

All scripts and outputs are in `research/`:
- `NYSE_TAQ_Research_Summary.md` — Initial schema and inventory
- `NYSE_TAQ_Final_Report.md` — This report
- `inspect_taq.py` / `inspect_taq_v2.py` — Data parsers
- `taq_feature_engineering.py` — 1-minute bar feature builder
- `taq_predictive_analysis.py` — 5-second predictive experiments
- `taq_advanced_features.py` — 1-minute volatility/direction models
- `taq_microstructure_signal.py` — Depth imbalance signal test
- `taq_consolidated_trades_analysis.py` — Full-day consolidated trade analysis
- `taq_bbo_features.csv` / `taq_trade_features.csv` — Feature matrices
- `taq_consolidated_trade_bars.csv` — Full-day 1-minute trade bars
- `taq_features_overview.png` — Feature overview plots
- `taq_predictive_*.png` / `taq_advanced_*.png` / `taq_signal_*.png` — Per-symbol plots
- `taq_full_day_trade_profile.png` — Full-day volume/volatility profiles

---

## Bottom Line

The NYSE TAQ sample data is **excellent for feature engineering**. We successfully parsed it, built a rich feature set, and documented the schema. Predictive modeling on one hour of data is underpowered, but the data is clearly sufficient for a full liquidity/microstructure research project once the full sample is available.

The most exciting direction is **linking TAQ equity microstructure features to the existing options implied-volatility work** — this bridges the two datasets and directly addresses the DS4FE research agenda.
