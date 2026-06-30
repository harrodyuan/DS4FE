"""
NYSE Consolidated Trades Analysis
==================================
Parses the full-day consolidated trades file (EQY_US_ALL_TRADE_20260102.gz)
and builds trade-based microstructure features and visualizations.
"""

import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, time
import os

DATA_DIR = '../data/nyse_taq_samples'


def parse_consolidated_trades(filepath, symbols, max_rows=5000000):
    """Parse consolidated trades for selected symbols."""
    rows = []
    with gzip.open(filepath, 'rt', encoding='latin-1') as f:
        header = f.readline().strip().split('|')
        for i, line in enumerate(f):
            parts = line.strip().split('|')
            if len(parts) < 6:
                continue
            symbol = parts[2].strip()
            if symbol not in symbols:
                continue
            try:
                time_str = parts[0]
                # Format appears to be HHMMSSNNNNNNNNN (nanoseconds)
                hour = int(time_str[0:2])
                minute = int(time_str[2:4])
                second = int(time_str[4:6])
                ns = int(time_str[6:]) if len(time_str) > 6 else 0
                
                rows.append({
                    'symbol': symbol,
                    'exchange': parts[1].strip(),
                    'sale_cond': parts[3].strip(),
                    'volume': int(parts[4]) if parts[4] else 0,
                    'price': float(parts[5]) if parts[5] else 0.0,
                    'hour': hour,
                    'minute': minute,
                    'second': second,
                    'time_bin': hour * 60 + minute,
                })
            except (ValueError, IndexError):
                continue
            
            if len(rows) >= max_rows:
                break
    
    return pd.DataFrame(rows)


def build_trade_bars(df, symbol):
    """Build 1-minute trade bars for a symbol."""
    bars = df.groupby('time_bin').agg(
        total_volume=('volume', 'sum'),
        trade_count=('price', 'count'),
        avg_price=('price', 'mean'),
        vwap=('price', lambda x: np.average(x, weights=df.loc[x.index, 'volume'])),
        open_price=('price', 'first'),
        high_price=('price', 'max'),
        low_price=('price', 'min'),
        close_price=('price', 'last'),
        avg_volume=('volume', 'mean'),
        max_volume=('volume', 'max'),
    ).reset_index()
    
    bars['symbol'] = symbol
    bars['minute_return'] = np.log(bars['close_price'] / bars['open_price'])
    bars['minute_range'] = (bars['high_price'] - bars['low_price']) / bars['open_price'] * 10000
    bars['realized_var'] = bars['minute_return'] ** 2
    bars['dollar_volume'] = bars['total_volume'] * bars['vwap']
    bars['time_of_day'] = bars['time_bin'].apply(lambda x: f"{x // 60:02d}:{x % 60:02d}")
    
    return bars


def plot_full_day_profile(bars_all, symbols):
    """Create full-day trading profile visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Volume profile by time of day
    ax = axes[0, 0]
    for sym in symbols:
        df = bars_all[bars_all['symbol'] == sym]
        ax.plot(df['time_bin'], df['total_volume'].rolling(5).mean(), label=sym, alpha=0.7)
    ax.set_xlabel('Minute of day')
    ax.set_ylabel('Total volume (5-min MA)')
    ax.set_title('Volume Profile by Time of Day')
    ax.legend(fontsize=8)
    
    # Plot 2: Trade count profile
    ax = axes[0, 1]
    for sym in symbols:
        df = bars_all[bars_all['symbol'] == sym]
        ax.plot(df['time_bin'], df['trade_count'].rolling(5).mean(), label=sym, alpha=0.7)
    ax.set_xlabel('Minute of day')
    ax.set_ylabel('Trade count (5-min MA)')
    ax.set_title('Trade Count Profile')
    ax.legend(fontsize=8)
    
    # Plot 3: Volatility signature (time-of-day effect)
    ax = axes[1, 0]
    for sym in symbols:
        df = bars_all[bars_all['symbol'] == sym]
        ax.plot(df['time_bin'], df['realized_var'].rolling(5).mean() * 10000, label=sym, alpha=0.7)
    ax.set_xlabel('Minute of day')
    ax.set_ylabel('Realized variance (x10,000)')
    ax.set_title('Volatility Signature')
    ax.legend(fontsize=8)
    
    # Plot 4: Average trade size profile
    ax = axes[1, 1]
    for sym in symbols:
        df = bars_all[bars_all['symbol'] == sym]
        ax.plot(df['time_bin'], df['avg_volume'].rolling(5).mean(), label=sym, alpha=0.7)
    ax.set_xlabel('Minute of day')
    ax.set_ylabel('Average trade size')
    ax.set_title('Average Trade Size Profile')
    ax.legend(fontsize=8)
    
    plt.tight_layout()
    plt.savefig('taq_full_day_trade_profile.png', dpi=150, bbox_inches='tight')
    print("Saved plot: taq_full_day_trade_profile.png")


def main():
    filepath = os.path.join(DATA_DIR, 'EQY_US_ALL_TRADE_20260102.gz')
    symbols = ['AAPL', 'TSLA', 'MSFT', 'SPY', 'QQQ', 'IWM']
    
    print("Parsing consolidated trades...")
    df = parse_consolidated_trades(filepath, set(symbols), max_rows=10000000)
    print(f"Parsed {len(df)} trade records")
    print(f"Symbols found: {df['symbol'].unique()}")
    
    if len(df) == 0:
        print("No data found. Exiting.")
        return
    
    all_bars = []
    for symbol in df['symbol'].unique():
        print(f"\nBuilding bars for {symbol}...")
        sym_df = df[df['symbol'] == symbol]
        bars = build_trade_bars(sym_df, symbol)
        all_bars.append(bars)
        print(f"  Bars: {len(bars)}")
        print(f"  Total volume: {bars['total_volume'].sum():,}")
        print(f"  Total trades: {bars['trade_count'].sum():,}")
        print(f"  VWAP: {bars['dollar_volume'].sum() / bars['total_volume'].sum():.2f}")
    
    bars_all = pd.concat(all_bars, ignore_index=True)
    bars_all.to_csv('taq_consolidated_trade_bars.csv', index=False)
    print("\nSaved trade bars to taq_consolidated_trade_bars.csv")
    
    # Plot full-day profiles
    plot_full_day_profile(bars_all, df['symbol'].unique()[:6])
    
    # Print summary statistics
    print("\n=== FULL-DAY SUMMARY ===")
    summary = bars_all.groupby('symbol').agg(
        total_volume=('total_volume', 'sum'),
        total_trades=('trade_count', 'sum'),
        avg_trade_size=('avg_volume', 'mean'),
        vwap=('vwap', lambda x: np.average(x, weights=bars_all.loc[x.index, 'total_volume'])),
        open_price=('open_price', 'first'),
        close_price=('close_price', 'last'),
        daily_return=('close_price', lambda x: np.log(x.iloc[-1] / x.iloc[0])),
        total_realized_var=('realized_var', 'sum'),
    ).reset_index()
    print(summary.round(4))


if __name__ == '__main__':
    main()
