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

**What it covers:** Combine features into a predictive ML model on AAPL. Walk-forward backtest + trading simulation.

**Key points:**

- **The look-ahead bias trap (demonstrated live)** — WRONG: use today's return as a feature to predict today's return → Win rate ~100%, R² ≈ 1.0. CORRECT: use yesterday's return to predict tomorrow's → Win rate ~51%, R² ≈ 0.0001. The difference is literally one `.shift(1)`. This is the most important practical lesson in quant research.

- **Walk-forward evaluation** — train on all data through day $t$, predict day $t+1$, expand window, repeat. Never shuffle a time series. Random splitting leaks future data into the training set. The first 504 days (~2 years) are consumed as the seed window and produce no out-of-sample predictions.

- **Model comparison (Linear, Ridge, Random Forest)** — all three produce win rates of ~50–52%. Random forest offers no consistent advantage over linear models. This tells you that the predictive relationship between features and next-day returns for a single large-cap is approximately linear and small. There is not much non-linearity for a tree ensemble to exploit.

- **Fundamentals (PE/PB/PS) problem** — `yfinance` only returns today's snapshot. Using it as a historical feature labels every 2012 training observation with 2025's valuation ratio. Historical point-in-time fundamentals require a paid source (Compustat, Sharadar, SimFin). Currently replaced with `mom_252d` as a directional proxy. **Open question: whether to add proper historical PE/PB from yfinance quarterly financials** (`income_stmt`, `balance_sheet`) in `download_data.py`.

- **Trading simulation** — hold AAPL when model predicts positive return, cash otherwise. A 51–52% win rate can compound to a positive equity curve over thousands of days. However, round-trip transaction costs of 5–10 bps largely eliminate the edge at daily frequency.

- **Why single-stock prediction is hard** — ~3,500 observations over 14 years is not much data for a low signal-to-noise problem. Signal-to-noise in daily returns is notoriously low. This motivates the panel approach: the same IC of 0.03 applied consistently across 50 stocks is a much larger information advantage than for one stock alone.

---

## Open Questions / Next Steps

- **PE/PB/PS data**: add proper historical fundamentals via yfinance quarterly financials, or leave as `mom_252d` proxy?
- **Part 4**: cross-sectional ML model on all 50 stocks simultaneously → long-short portfolio construction → Sharpe ratio analysis
- **Part 5 (tentative)**: LOB/HFT features (order book imbalance, trade sign, intraday VWAP deviation)
