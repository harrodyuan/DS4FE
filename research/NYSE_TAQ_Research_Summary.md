# NYSE TAQ Sample Data — Research Summary
## For: Chad Schafer (CMU Statistics & Data Science)
## Date: June 24, 2026
## Analyst: Harold Yuan

---

## 1. Data Inventory

Three sample files downloaded from `https://ftp.nyse.com/Historical%20Data%20Samples/`:

| File | Size | Content | Records |
|---|---|---|---|
| `EQY_US_ALL_REF_MASTER_20260102.gz` | 401 KB | Symbol reference master | 12,181 symbols |
| `EQY_US_TAQ_NYSE_TRADES_20231002.gz` | 95 MB | NYSE venue trade prints | ~950K messages |
| `EQY_US_TAQ_NYSE_BBO_1_20231002.gz` | 1.6 GB | NYSE venue BBO quotes (hour 1) | ~95M messages |

The BBO file is **hour 1 only** (09:30–10:30 ET). Extrapolating to a full trading day (6.5 hrs) and all ~20 NYSE hours gives ~1.2 TB/day raw compressed. This is serious data volume.

---

## 2. Schema Discovered

### Reference Master (pipe-delimited, `|`)
40 fields including: Symbol, CUSIP, Security_Type, Listed_Exchange, Round_Lot, Tick_Pilot_Indicator, and per-venue trading flags (TradedOnNYSE, TradedOnArca, TradedOnNasdaq, etc.).

**Key finding:** 8,936 symbols are NYSE-listed, 11,715 Arca-listed, 10,981 Nasdaq-listed. Security types: A (common stock) = 5,161, ETF = 4,622, B (bond) = 439, C (ADRs) = 404.

### NYSE Trades (comma-delimited)
Message types observed:
- `3` — Trade execution (~1.2% of messages)
- `34` — Trade correction/cancel (~99% of messages in this sample — unusual, may be a data artifact of the sample)

Trade fields: msg_type, sequence_num, symbol, ?, ?, ?, sale_condition, size, price, ...

Sale conditions: C (regular), E (extended), W (average price), P (prior reference), O (opening).

### NYSE BBO (comma-delimited)
Message type `140` — BBO Quote.

**Field order (confirmed empirically):**
```
140, seq, timestamp, symbol, quote_cond, ask_px, ask_sz, bid_px, bid_sz, flag, ...
```

Quote condition codes:
- `4` — One-sided update (either bid or ask is 0)
- `5` — Two-sided quote
- `6, 7, 8, 9` — Other conditions (opening, closing, halted, etc.)

---

## 3. Microstructure Statistics (Hour 1 Sample)

### Quote Frequency
| Symbol | Quote Count (1 hr) | Likely Type |
|---|---|---|
| DIA | 30,845 | ETF (Dow Jones) |
| BLDP | 30,790 | Equity |
| IVV | 26,279 | ETF (S&P 500) |
| IWM | 13,848 | ETF (Russell 2000) |
| AAPL | 12,466 | Large-cap equity |

Top symbols see **~8-9 quotes per second** in the first hour. This is typical for liquid names.

### Spread Distribution
- **Valid two-sided quotes parsed:** 427,408 (out of 470,080 sampled)
- **Mean spread:** 314.75 bps
- **Median spread:** 40.53 bps
- **25th percentile:** 14.94 bps
- **75th percentile:** 237.31 bps
- **Max:** 20,000 bps (likely stale or bad quote)

**Note:** The mean is inflated by illiquid symbols and wide ETF spreads. Median of ~40 bps is more representative.

---

## 4. What This Enables for Feature Engineering

The data is **significantly richer** than the MBP-1 (top-of-book only) data used in the prior LOB manifold study. Here is what we can now build:

### A. Quote-based microstructure features
From the BBO feed alone:
- **Quoted spread** — time-series of `ask - bid`
- **Relative spread** — spread / mid_price (what we computed above)
- **Quote intensity** — number of BBO updates per unit time
- **Bid-ask bounce** — autocorrelation of mid-price changes
- **Depth imbalance** — `bid_sz / (bid_sz + ask_sz)` (available because we have sizes!)

### B. Trade-based features
From the trades feed:
- **Trade sign** — classify each trade as buyer-initiated vs. seller-initiated (tick test or Lee-Ready)
- **Effective spread** — `2 * |trade_price - mid_price| / mid_price`
- **Volume-weighted average price (VWAP)** — deviation of trade price from VWAP
- **Trade size distribution** — large trades vs. small trades

### C. Combined trade+quote features
- **Realized variance** — sum of squared log-returns over 5-min buckets
- **Price impact** — regression of signed volume on subsequent return (Hasbrouck)
- **Order flow imbalance (OFI)** — measure of buying vs. selling pressure
- **Amihud illiquidity** — `|return| / dollar_volume`

### D. Cross-sectional features (unique to this dataset)
Because the reference master links all symbols, we can build:
- **Sector-relative spreads** — is this stock's spread wider than its sector median?
- **ETF vs. equity comparison** — do ETFs have tighter spreads? (Yes, based on preliminary look)
- **Cross-venue arbitrage signals** — compare NYSE BBO vs. consolidated NBBO (if we download it)

---

## 5. Comparison to Existing DS4FE Data

| Feature | Databento OPRA (Options) | NYSE TAQ (Equities) |
|---|---|---|
| Product | SPX options | All NYSE-listed equities |
| Granularity | EOD snapshot | Tick-by-tick, nanosecond timestamps |
| Depth | Surface grid (7×9) | BBO + sizes, can reconstruct LOB |
| Best use case | Volatility modeling, skew dynamics | Microstructure, liquidity, execution |
| Data volume | ~10 MB/day | ~1.2 TB/day |

**Synergy opportunity:** Link the two datasets. Use TAQ microstructure features from the equity underlying to predict next-day ATM implied volatility from the options surface. This is a genuine cross-market signal.

---

## 6. Open Questions / Next Steps

1. **Full-day sample:** Hour 1 is the most active. Do spreads widen in the afternoon? Do quote intensities decay?
2. **Consolidated vs. venue-specific:** The `EQY_US_ALL_NBBO` file gives the national best bid/offer. Comparing NYSE BBO to NBBO reveals venue-specific liquidity.
3. **Trade signing:** The Lee-Ready algorithm requires quote timestamps at trade time. We have nanosecond precision — this is feasible.
4. **Bad data filtering:** The max spread of 20,000 bps suggests stale quotes or test messages. We need a filter (e.g., drop quotes older than 1 second, or spreads > 500 bps).
5. **Cross-asset link:** Can we map these equities to the SPX components and build an "implied vol predictor" from microstructure features?

---

## 7. Infrastructure Recommendation

For a full study:
- **Storage:** ~1.5 TB/month compressed. A 2-year panel = ~36 TB. This requires server-grade storage, not a laptop.
- **Compute:** Parsing 95M messages per hour requires vectorized pandas or polars. Consider `polars` over `pandas` for 10x speedup.
- **Sampling strategy:** If full history is too large, sample 1 day per month (24 days = ~36 TB → ~1.4 TB). Or sample 1 symbol per hour for all hours.

---

*Prepared for Chad Schafer. Raw data and parsing scripts available in `data/nyse_taq_samples/`.*
