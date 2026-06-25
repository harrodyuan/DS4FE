import gzip
import pandas as pd
import numpy as np

# ============================================================
# File 1: Reference Master
# ============================================================
print("=" * 60)
print("FILE 1: REFERENCE MASTER (2026-01-02)")
print("=" * 60)

with gzip.open('EQY_US_ALL_REF_MASTER_20260102.gz', 'rt', encoding='latin-1') as f:
    header = f.readline().strip().split('|')
    rows = [line.strip().split('|') for line in f]

df_ref = pd.DataFrame(rows, columns=header)
print(f"Total symbols: {len(df_ref)}")
print(f"\nSecurity types:")
print(df_ref['Security_Type'].value_counts().head())
print(f"\nNYSE-listed: {(df_ref['TradedOnNYSE'] == '1').sum()}")
print(f"Arca-listed: {(df_ref['TradedOnArca'] == '1').sum()}")
print(f"Nasdaq-listed: {(df_ref['TradedOnNasdaq'] == '1').sum()}")

# ============================================================
# File 2: NYSE Trades (sample)
# ============================================================
print("\n" + "=" * 60)
print("FILE 2: NYSE TRADES (2023-10-02)")
print("=" * 60)

trade_rows = []
correction_rows = []
count = 0
max_rows = 300000

with gzip.open('EQY_US_TAQ_NYSE_TRADES_20231002.gz', 'rt', encoding='latin-1') as f:
    for line in f:
        parts = line.strip().split(',')
        msg_type = parts[0]
        if msg_type == '3':
            trade_rows.append({
                'msg_type': msg_type,
                'seq': parts[1],
                'symbol': parts[2],
                'sale_cond': parts[6] if len(parts) > 6 else '',
                'size': int(parts[7]) if len(parts) > 7 and parts[7] else 0,
                'price': float(parts[8]) if len(parts) > 8 and parts[8] else 0.0,
            })
        elif msg_type == '34':
            correction_rows.append({'msg_type': msg_type, 'symbol': parts[3] if len(parts) > 3 else ''})
        count += 1
        if count >= max_rows:
            break

df_trades = pd.DataFrame(trade_rows)
print(f"Total rows scanned: {count}")
print(f"Trade messages: {len(df_trades)}")
print(f"Correction messages: {len(correction_rows)}")

vol = df_trades.groupby('symbol').agg({'size': 'sum', 'price': 'count'}).rename(columns={'price': 'trades'}).sort_values('size', ascending=False)
print(f"\nTop 10 symbols by volume (sample):")
print(vol.head(10))

print(f"\nSale conditions:")
print(df_trades['sale_cond'].value_counts().head())

print(f"\nPrice stats:")
print(df_trades['price'].describe())

# ============================================================
# File 3: NYSE BBO (sample)
# ============================================================
print("\n" + "=" * 60)
print("FILE 3: NYSE BBO (2023-10-02, hour 1)")
print("=" * 60)

quote_rows = []
count = 0
max_rows = 500000

with gzip.open('EQY_US_TAQ_NYSE_BBO_1_20231002.gz', 'rt', encoding='latin-1') as f:
    for line in f:
        parts = line.strip().split(',')
        msg_type = parts[0]
        if msg_type == '140':
            if len(parts) >= 10:
                quote_rows.append({
                    'seq': parts[1],
                    'time': parts[2],
                    'symbol': parts[3],
                    'quote_cond': parts[4],
                    'bid_px': float(parts[5]) if parts[5] else 0,
                    'bid_sz': int(parts[6]) if parts[6] else 0,
                    'ask_px': float(parts[7]) if parts[7] else 0,
                    'ask_sz': int(parts[8]) if parts[8] else 0,
                    'flag': parts[9] if len(parts) > 9 else '',
                })
        count += 1
        if count >= max_rows:
            break

df_quotes = pd.DataFrame(quote_rows)
print(f"Total rows scanned: {count}")
print(f"Quote messages: {len(df_quotes)}")

valid = (df_quotes['bid_px'] > 0) & (df_quotes['ask_px'] > 0) & (df_quotes['ask_px'] >= df_quotes['bid_px'])
df_valid = df_quotes[valid].copy()
df_valid['spread'] = df_valid['ask_px'] - df_valid['bid_px']
df_valid['mid'] = (df_valid['bid_px'] + df_valid['ask_px']) / 2
df_valid['spread_bps'] = df_valid['spread'] / df_valid['mid'] * 10000

print(f"Valid two-sided quotes: {len(df_valid)}")
print(f"\nSpread stats (bps):")
print(df_valid['spread_bps'].describe())

print(f"\nTop symbols by quote frequency:")
print(df_quotes['symbol'].value_counts().head(10))

print(f"\nQuote condition codes:")
print(df_quotes['quote_cond'].value_counts().head())

# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Reference master: {len(df_ref)} symbols")
print(f"Trades sample: {len(df_trades)} trades from {df_trades['symbol'].nunique()} symbols")
print(f"BBO quotes sample: {len(df_quotes)} quotes from {df_quotes['symbol'].nunique()} symbols")
print(f"Avg spread: {df_valid['spread_bps'].mean():.2f} bps")
print(f"Median spread: {df_valid['spread_bps'].median():.2f} bps")
