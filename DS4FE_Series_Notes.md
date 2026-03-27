# DS4FE Lecture Series — Key Points Summary

**Data Science for Feature Engineering**
Dataset: 50 large-cap US stocks across 5 sectors, 2010–2024, daily OHLCV + macro (SPY, VIX, Treasury yields, USD index). Downloaded once via `download_data.py`, stored in Parquet for reuse across all notebooks.

---

## Part 1: Data Tour (`DS4FE_Part1_Data_Tour.ipynb`)

**What it covers:** Explore the dataset before touching any model.

**Key points:**

- **Coverage heatmap** — confirms all 50 tickers have consistent daily data across the full period. Missing data or stale prices here would silently corrupt every downstream feature.

- **Cumulative returns by sector** — Tech dramatically outperforms all other sectors over 2010–2024; Energy/Industrials lag. This matters for any cross-sectional model: sector membership is a strong confound. A model that "predicts returns well" may just be overweighting Tech.

- **Annual return heatmap** — shows that returns cluster in time (2020 crash, 2022 drawdown hit everything simultaneously). Individual stock returns are far less independent than cross-sectional sample sizes suggest.

- **Correlation heatmap** — within-sector correlations are high (~0.6–0.8); cross-sector correlations lower (~0.3–0.5). This has direct implications for diversification and for model evaluation: you cannot treat 50 stocks × 3,500 days as 175,000 independent observations.

- **Macro time series (4-panel)** — VIX spikes mark the 2020 COVID crash and 2022 rate-hike regime. The yield spread inverted in 2022–2023 (classic recession signal). These regimes affect every feature built on top of returns.

- **Beta and R² per stock** — most large-cap stocks have beta near 1.0 and R² of 0.3–0.6 against SPY. High R² means a large fraction of individual return variance is explained by the market factor alone. Any predictive model needs to account for this, otherwise it is just forecasting the market.

---

## Part 2: Feature Engineering (`DS4FE_Part2_Feature_Engineering.ipynb`)

**What it covers:** Build 12 signals across all 50 stocks and test whether any of them predict future returns.

**Key points:**

- **Single stock first (AAPL)** — raw daily returns have near-zero autocorrelation (ACF plot). Squared returns have strong autocorrelation. Implication: *direction* is hard to predict; *volatility* is predictable. This motivates features beyond just lagged returns.

- **Feature families:**
  - *Momentum* (1d, 5d, 21d, 63d): captures slow information diffusion and institutional flow momentum
  - *Realized volatility* (21d, 63d): captures risk regime
  - *Volume ratio and ILLIQ*: captures liquidity and unusual activity
  - *Macro* (SPY, VIX, yield spread, USD index): places each stock in its market context

- **Look-ahead bias prevention** — every feature uses `.shift(1)` so the most recent observation is from the previous close. The forward return target uses `.shift(-1)`. This is the most common error in backtesting: forgetting to shift means you are using today's information to predict today's return.

- **IC (Information Coefficient)** — Spearman rank correlation between a feature and the forward return, computed cross-sectionally each day. Mean IC for the best features is around 0.02–0.04. Small, but nonzero. A mean IC of 0.02 across 50 stocks and 3,500 days is statistically significant even if it looks trivially small.

- **Quintile portfolios** — sort stocks into 5 buckets by feature value each day; compare average returns across buckets. If the feature has predictive content, there should be a monotone spread from Q1 to Q5 (or Q5 to Q1). This is the standard industry tool for validating factors before building a model.

- **Cross-sectional vs time-series** — working with 50 stocks simultaneously multiplies the number of observations by 50, dramatically improving statistical power compared to a single stock.

---

## Part 3: Single Stock Prediction (`DS4FE_Part3_Single_Stock_Prediction.ipynb`)

**What it covers:** Combine features into a predictive ML model on AAPL. Walk-forward backtest + trading simulation + cross-frequency OOS R² comparison vs LOB.

**Key points:**

- **The look-ahead bias trap (demonstrated live)** — WRONG: use today's return as a feature to predict today's return → Win rate ~100%, R² ≈ 1.0. CORRECT: use yesterday's return to predict tomorrow's → Win rate ~51%, R² ≈ 0.0001. The difference is literally one `.shift(1)`. This is the most important practical lesson in quant research.

- **Walk-forward evaluation** — train on all data through day $t$, predict day $t+1$, expand window, repeat. Never shuffle a time series. Random splitting leaks future data into the training set. The first 504 days (~2 years) are consumed as the seed window and produce no out-of-sample predictions.

- **15 features (12 from Part 2 + 3 new)** — adds `mom_252d` (12-month momentum), `vol_ratio` (short/long vol ratio), `vix_chg` (VIX change). All 15 are time-series features for AAPL alone — not cross-sectional rankings.

- **Time-series vs cross-sectional momentum** — the momentum features here measure AAPL's own past return, NOT how AAPL ranks relative to other stocks. The Jegadeesh-Titman momentum anomaly is cross-sectional: stocks that beat *other stocks* tend to keep beating *other stocks*. Time-series autocorrelation in a single large-cap is near zero. Cross-sectional momentum is what Part 5 exploits.

- **OOS R² comparison — same stock, different horizons:**
  | Model | OOS R² |
  |---|---|
  | Daily AAPL \| Ridge | −0.015 |
  | Daily AAPL \| Random Forest | −0.094 |
  | LOB 10s AAPL \| Ridge | −0.004 |
  | LOB 10s AAPL \| XGBoost | −0.001 |
  | LOB 10s NVDA \| XGBoost (Part 4) | +0.00029 |

  Daily models are more negative because (1) daily momentum signals are arbitraged away by institutions; (2) 1 prediction/day gives ~252 OOS observations/year vs ~10,000/day for LOB — much harder to detect any true edge.

- **Fundamentals (PE/PB/PS) limitation** — `yfinance` only returns today's snapshot. Historical point-in-time fundamentals require a paid source (Compustat, Sharadar, SimFin). Currently replaced with `mom_252d` as a directional proxy.

---

## Part 4: LOB Feature Engineering (`DS4FE_Part4_LOB_Features.ipynb`)

**What it covers:** Limit order book data from NASDAQ ITCH feed via Databento. Two parts: (I) data understanding + feature engineering + IC signal check; (II) walk-forward prediction with Ridge and XGBoost.

**Key points:**

- **Data** — Databento mbp-1 (level 1) and mbp-10 (10-level). Calm: Oct 2–12 2023 (10 days). Stress: Aug 5–9 2024 (Japan carry trade unwind). Stocks: NVDA, TSLA, AAPL, MSFT, SPY.

- **Core features** — spread, mid-price, OBI (order book imbalance = (bid_sz − ask_sz)/(bid_sz + ask_sz)), OFI (order flow imbalance, Cont et al. 2014 = Δbid_queue − Δask_queue). Resampled to 1-second bars.

- **IC vs OOS R²** — IC is model-free and measures average signal. OOS R² measures whether a model can exploit the signal on unseen data. Positive IC does not guarantee positive OOS R². The IC measures existence of signal; OOS R² measures exploitability.

- **Walk-forward (6 train days, 1 OOS day)** — Ridge OOS R²: −0.00196. XGBoost OOS R²: +0.00029. XGBoost captures the spread-filter nonlinearity (OFI predicts only when spread is tight). Look-ahead audit: all features use past data only; scaler fit on training set only.

- **Cross-asset signal** — SPY OBI lagged 1 second predicts NVDA return. Market-wide order flow leads individual stock moves.

- **Calm vs stress** — stress period (Aug 5 2024) has 3.5× more book events/second. Spread widens, OFI becomes more volatile. IC of OBI and OFI both change in stressed markets.

---

## Part 5: Multifactor Models (`DS4FE_Part5_Multifactor.ipynb`)

**What it covers:** Cross-sectional multifactor investing on the 50-stock daily panel. Fama-MacBeth regression, IC-weighted composite, long-short portfolio.

**Key points:**

- **Cross-sectional frame** — rank all 50 stocks by factor exposure each day. Top quintile long, bottom quintile short. Market-neutral by construction. This is the correct context for momentum and other factors — NOT predicting a single stock from its own history.

- **Factors (all price-based):** `mom_1m` (reversal), `mom_12m` (momentum, IC t-stat ~2.6), `vol_63d` (low-vol anomaly), `illiq` (Amihud liquidity premium), `size` (log market cap proxy).

- **Fama-MacBeth two-pass regression** — characteristics-based approach: standardized factor exposure IS the loading. Second pass: cross-sectional OLS each day → time series of premia λ_t. t-stat = mean(λ_t) / std(λ_t) × √T. Robust to cross-sectional correlation.

- **IC-weighted composite** — trailing 252-day IC used as factor weights. Strictly out-of-sample (IC window shifted forward). Composite Sharpe ratio typically higher than any individual factor due to regime diversification.

- **Data limitations** — no point-in-time fundamentals. `shares_out` is a current snapshot. HML and EP factors require Compustat or Sharadar. Framework is correct; better data would make it publication-quality.

---

## Open Questions

- **Historical fundamentals**: point-in-time PE/PB/PS for value factor requires Compustat (WRDS) or Sharadar (Nasdaq Data Link).
- **LOB multi-day OOS**: Part 4 XGBoost evaluated on 1 OOS day only. A robust estimate requires 20–30 days across different market regimes.
- **LOB at 100ms**: internship model used 100ms data with 40+ features. Running Part 4 at 100ms would give a direct comparison.
- **LOB multi-stock**: Part 4 uses NVDA only. Averaging OOS R² across NVDA, TSLA, AAPL, MSFT would give a more reliable signal estimate.
