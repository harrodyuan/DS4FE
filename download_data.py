"""
DS4FE Data Download Script
===========================
Downloads daily OHLCV data for 50 large-cap US stocks (2010-2024),
SPY market index, and macro data (VIX, Treasury yields, USD index).
Saves to Parquet for use across all DS4FE lecture notebooks.

Run once:
    python download_data.py

Output files:
    data/ds4fe_panel.parquet          -- long panel: date, ticker, OHLCV, ret  (main)
    data/ds4fe_daily_prices.parquet   -- wide close prices
    data/ds4fe_market.parquet         -- SPY daily: close, volume, market_ret, vix, yield_10y, yield_3m
    data/ds4fe_info.csv               -- ticker metadata (sector, name, market cap)
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import time

# ── Ticker universe ────────────────────────────────────────────────────────────
TICKERS = {
    'Technology'   : ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'META', 'ADBE', 'CRM', 'INTC', 'AMD', 'QCOM'],
    'Financials'   : ['JPM', 'BAC', 'GS', 'MS', 'WFC', 'BLK', 'AXP', 'C', 'USB', 'PNC'],
    'Healthcare'   : ['JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'TMO', 'ABT', 'LLY', 'BMY', 'AMGN'],
    'Consumer'     : ['AMZN', 'HD', 'MCD', 'NKE', 'SBUX', 'TGT', 'WMT', 'COST', 'LOW', 'TJX'],
    'Energy/Indus' : ['XOM', 'CVX', 'CAT', 'GE', 'BA', 'MMM', 'UPS', 'HON', 'LMT', 'RTX'],
}
ALL_TICKERS = [t for group in TICKERS.values() for t in group]
TICKER_SECTOR = {t: s for s, tickers in TICKERS.items() for t in tickers}

START = '2010-01-01'
END   = '2024-12-31'

os.makedirs('data', exist_ok=True)

# ── Step 1: batch download OHLCV ───────────────────────────────────────────────
print(f'Downloading daily OHLCV for {len(ALL_TICKERS)} tickers ({START} to {END})...')
print('Using yf.download() batch call — avoids rate limits.\n')

raw = yf.download(
    ALL_TICKERS,
    start=START,
    end=END,
    auto_adjust=True,   # adjusts for splits & dividends
    progress=True,
)

close  = raw['Close'].copy()
volume = raw['Volume'].copy()
high   = raw['High'].copy()
low    = raw['Low'].copy()
open_  = raw['Open'].copy()

print(f'\nRaw shape: {close.shape}  (dates x tickers)')
print(f'Date range: {close.index[0].date()} to {close.index[-1].date()}')
print(f'Missing values: {close.isna().sum().sum()} cells ({close.isna().mean().mean():.1%})')

# ── Step 2: save wide-format prices ───────────────────────────────────────────
wide_out = 'data/ds4fe_daily_prices.parquet'
close.to_parquet(wide_out)
print(f'\nSaved wide close prices -> {wide_out}')

# ── Step 3: build long-format panel ───────────────────────────────────────────
print('\nBuilding long-format panel...')

log_ret = np.log(close / close.shift(1))

# Stack each field to long format
def to_long(df, col_name):
    return (
        df.stack()
        .reset_index()
        .rename(columns={'Date': 'date', 'Ticker': 'ticker', 0: col_name})
    )

panel = (
    to_long(close,   'close')
    .merge(to_long(volume,  'volume'),  on=['date', 'ticker'])
    .merge(to_long(high,    'high'),    on=['date', 'ticker'])
    .merge(to_long(low,     'low'),     on=['date', 'ticker'])
    .merge(to_long(open_,   'open'),    on=['date', 'ticker'])
    .merge(to_long(log_ret, 'ret'),     on=['date', 'ticker'])
)

panel['sector'] = panel['ticker'].map(TICKER_SECTOR)
panel = panel.dropna(subset=['ret']).reset_index(drop=True)
panel['date'] = pd.to_datetime(panel['date'])

# Add forward return (target variable for prediction)
panel = panel.sort_values(['ticker', 'date'])
panel['ret_fwd'] = panel.groupby('ticker')['ret'].shift(-1)

panel_out = 'data/ds4fe_panel.parquet'
panel.to_parquet(panel_out, index=False)

print(f'Panel rows  : {len(panel):,}')
print(f'Saved panel -> {panel_out}')

# ── Step 4: fetch ticker metadata ─────────────────────────────────────────────
print('\nFetching ticker metadata (name, market cap, etc.)...')
print('Note: this makes one request per ticker — may take ~2 min.\n')

info_rows = []
for i, ticker in enumerate(ALL_TICKERS):
    try:
        info = yf.Ticker(ticker).info
        info_rows.append({
            'ticker'       : ticker,
            'sector'       : TICKER_SECTOR[ticker],
            'short_name'   : info.get('shortName', ticker),
            'long_name'    : info.get('longName', ticker),
            'industry'     : info.get('industry', ''),
            'market_cap'   : info.get('marketCap', np.nan),
            'shares_out'   : info.get('sharesOutstanding', np.nan),
            'country'      : info.get('country', ''),
            'exchange'     : info.get('exchange', ''),
        })
        print(f'  [{i+1:02d}/{len(ALL_TICKERS)}] {ticker}: {info.get("shortName", "")}')
        time.sleep(0.3)   # small delay to be polite
    except Exception as e:
        print(f'  [{i+1:02d}/{len(ALL_TICKERS)}] {ticker}: ERROR - {e}')
        info_rows.append({'ticker': ticker, 'sector': TICKER_SECTOR[ticker]})

info_df = pd.DataFrame(info_rows)
info_out = 'data/ds4fe_info.csv'
info_df.to_csv(info_out, index=False)
print(f'\nSaved metadata -> {info_out}')

# ── Step 5: download market index + macro data ────────────────────────────────
print('\nDownloading market index + macro data...')

MACRO_TICKERS = {
    'SPY'      : 'SPY',       # S&P 500 ETF — market return proxy
    '^VIX'     : 'vix',       # CBOE implied volatility index
    '^TNX'     : 'yield_10y', # 10-year Treasury yield (annualized %)
    '^IRX'     : 'yield_3m',  # 3-month Treasury bill yield (annualized %)
    'DX-Y.NYB' : 'usd_index', # US Dollar index
}

macro_raw = yf.download(
    list(MACRO_TICKERS.keys()),
    start=START,
    end=END,
    auto_adjust=True,
    progress=False,
)

market = pd.DataFrame(index=macro_raw['Close'].index)
for yf_ticker, col_name in MACRO_TICKERS.items():
    if yf_ticker in macro_raw['Close'].columns:
        market[col_name] = macro_raw['Close'][yf_ticker]

# SPY volume separately
if 'SPY' in macro_raw['Volume'].columns:
    market['spy_volume'] = macro_raw['Volume']['SPY']

# Compute SPY log return
market['market_ret'] = np.log(market['SPY'] / market['SPY'].shift(1))

# Rename SPY close for clarity
market = market.rename(columns={'SPY': 'spy_close'})

# Forward-fill yields (they don't trade every day — weekends/holidays)
for col in ['yield_10y', 'yield_3m', 'usd_index']:
    if col in market.columns:
        market[col] = market[col].ffill()

# Yield spread: 10y minus 3m (inverted = recession signal)
if 'yield_10y' in market.columns and 'yield_3m' in market.columns:
    market['yield_spread'] = market['yield_10y'] - market['yield_3m']

market.index.name = 'date'
market = market.dropna(subset=['market_ret']).reset_index()
market['date'] = pd.to_datetime(market['date'])

market_out = 'data/ds4fe_market.parquet'
market.to_parquet(market_out, index=False)

print(f'Market/macro rows : {len(market):,}')
print(f'Columns           : {list(market.columns)}')
print(f'Saved             -> {market_out}')

# ── Summary ───────────────────────────────────────────────────────────────────
print('\n' + '='*55)
print('DOWNLOAD COMPLETE')
print('='*55)
print(f'  ds4fe_panel.parquet         — long panel  (main dataset)')
print(f'  ds4fe_daily_prices.parquet  — wide close prices')
print(f'  ds4fe_market.parquet        — SPY + macro (VIX, yields, USD)')
print(f'  ds4fe_info.csv              — ticker metadata')
print()
print('Panel columns  :', list(panel.columns))
print('Market columns :', list(market.columns))
print()
print('Load in any notebook with:')
print('  panel  = pd.read_parquet("data/ds4fe_panel.parquet")')
print('  market = pd.read_parquet("data/ds4fe_market.parquet")')
print('  info   = pd.read_csv("data/ds4fe_info.csv")')
