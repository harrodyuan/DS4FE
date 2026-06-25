"""
NYSE TAQ Feature Engineering Pipeline
=====================================
Builds 1-minute microstructure feature bars from NYSE BBO and trades data.
Designed for the Chad Schafer (CMU) feature engineering request.
"""

import gzip
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from collections import defaultdict
import os

DATA_DIR = '../data/nyse_taq_samples'


def parse_bbo_file(filepath, max_rows=None):
    """Parse NYSE BBO file (msg type 140) into a DataFrame."""
    rows = []
    with gzip.open(filepath, 'rt', encoding='latin-1') as f:
        for i, line in enumerate(f):
            parts = line.strip().split(',')
            if parts[0] != '140':
                continue
            if len(parts) < 10:
                continue
            rows.append({
                'timestamp': parts[2],
                'symbol': parts[3],
                'quote_cond': parts[4],
                'ask_px': float(parts[5]) if parts[5] else 0.0,
                'ask_sz': int(parts[6]) if parts[6] else 0,
                'bid_px': float(parts[7]) if parts[7] else 0.0,
                'bid_sz': int(parts[8]) if parts[8] else 0,
            })
            if max_rows and len(rows) >= max_rows:
                break
    return pd.DataFrame(rows)


def parse_trades_file(filepath, max_rows=None):
    """Parse NYSE trades file (msg type 3) into a DataFrame."""
    rows = []
    with gzip.open(filepath, 'rt', encoding='latin-1') as f:
        for i, line in enumerate(f):
            parts = line.strip().split(',')
            if parts[0] != '3':
                continue
            if len(parts) < 10:
                continue
            rows.append({
                'symbol': parts[2],
                'sale_cond': parts[6] if len(parts) > 6 else '',
                'size': int(parts[7]) if len(parts) > 7 and parts[7] else 0,
                'price': float(parts[8]) if len(parts) > 8 and parts[8] else 0.0,
            })
            if max_rows and len(rows) >= max_rows:
                break
    return pd.DataFrame(rows)


def timestamp_to_minute(ts):
    """Convert TAQ timestamp string to minute bucket."""
    # ts format: HH:MM:SS.fffffffff
    try:
        h, m, s = ts.split(':')
        sec = int(float(s))
        return f"{h}:{m}:{sec:02d}"
    except:
        return None


def build_bbo_features(df_bbo, symbols=None):
    """Build 1-minute feature bars from BBO quotes."""
    if symbols:
        df_bbo = df_bbo[df_bbo['symbol'].isin(symbols)].copy()
    else:
        df_bbo = df_bbo.copy()
    
    # Keep only valid two-sided quotes
    valid = (df_bbo['bid_px'] > 0) & (df_bbo['ask_px'] > 0) & (df_bbo['ask_px'] > df_bbo['bid_px'])
    df_bbo = df_bbo[valid].copy()
    
    df_bbo['minute'] = df_bbo['timestamp'].apply(timestamp_to_minute)
    df_bbo['mid'] = (df_bbo['bid_px'] + df_bbo['ask_px']) / 2
    df_bbo['spread'] = df_bbo['ask_px'] - df_bbo['bid_px']
    df_bbo['spread_bps'] = df_bbo['spread'] / df_bbo['mid'] * 10000
    df_bbo['depth_imbalance'] = (df_bbo['bid_sz'] - df_bbo['ask_sz']) / (df_bbo['bid_sz'] + df_bbo['ask_sz'])
    df_bbo['quoted_depth'] = df_bbo['bid_sz'] + df_bbo['ask_sz']
    
    features = df_bbo.groupby(['symbol', 'minute']).agg(
        nbbo_updates=('mid', 'count'),
        mean_spread_bps=('spread_bps', 'mean'),
        median_spread_bps=('spread_bps', 'median'),
        max_spread_bps=('spread_bps', 'max'),
        mean_depth_imbalance=('depth_imbalance', 'mean'),
        mean_quoted_depth=('quoted_depth', 'mean'),
        first_mid=('mid', 'first'),
        last_mid=('mid', 'last'),
        high_mid=('mid', 'max'),
        low_mid=('mid', 'min'),
    ).reset_index()
    
    features['mid_return'] = np.log(features['last_mid'] / features['first_mid'])
    features['mid_range'] = (features['high_mid'] - features['low_mid']) / features['first_mid'] * 10000
    features['quote_intensity'] = features['nbbo_updates']  # per minute
    
    return features


def build_trade_features(df_trades, symbols=None):
    """Build 1-minute feature bars from trades."""
    if symbols:
        df_trades = df_trades[df_trades['symbol'].isin(symbols)].copy()
    else:
        df_trades = df_trades.copy()
    
    # Note: the 2023 sample has no timestamps in trade records, so we can't do time bars
    # We can do aggregate features over the whole sample instead
    features = df_trades.groupby('symbol').agg(
        total_volume=('size', 'sum'),
        trade_count=('price', 'count'),
        avg_trade_size=('size', 'mean'),
        median_trade_price=('price', 'median'),
        price_std=('price', 'std'),
        min_price=('price', 'min'),
        max_price=('price', 'max'),
    ).reset_index()
    
    vwap = df_trades.groupby('symbol').apply(
        lambda x: np.average(x['price'], weights=x['size']), include_groups=False
    ).reset_index(name='vwap_approx')
    features = features.merge(vwap, on='symbol', how='left')
    
    # Sale condition breakdown
    cond_counts = df_trades.groupby(['symbol', 'sale_cond']).size().unstack(fill_value=0).reset_index()
    features = features.merge(cond_counts, on='symbol', how='left')
    
    return features


def main():
    print("Loading BBO data...")
    df_bbo = parse_bbo_file(os.path.join(DATA_DIR, 'EQY_US_TAQ_NYSE_BBO_1_20231002.gz'), max_rows=2000000)
    print(f"BBO records: {len(df_bbo)}")
    
    print("Loading trades data...")
    df_trades = parse_trades_file(os.path.join(DATA_DIR, 'EQY_US_TAQ_NYSE_TRADES_20231002.gz'), max_rows=2000000)
    print(f"Trade records: {len(df_trades)}")
    
    # Select top liquid symbols by quote frequency
    top_symbols = df_bbo['symbol'].value_counts().head(20).index.tolist()
    print(f"\nTop 20 symbols by quote frequency: {top_symbols}")
    
    # Focus on a subset for analysis
    focus_symbols = ['AAPL', 'DIA', 'IVV', 'IWM', 'GOOGL', 'AMZN', 'TSLA', 'MSFT']
    available_focus = [s for s in focus_symbols if s in top_symbols]
    if not available_focus:
        available_focus = top_symbols[:5]
    
    print(f"\nAnalyzing symbols: {available_focus}")
    
    print("\nBuilding BBO features...")
    bbo_features = build_bbo_features(df_bbo, symbols=available_focus)
    print(f"BBO bars: {len(bbo_features)}")
    
    print("\nBuilding trade features...")
    trade_features = build_trade_features(df_trades, symbols=available_focus)
    print(f"Trade summary rows: {len(trade_features)}")
    
    # Save feature matrices
    bbo_features.to_csv('taq_bbo_features.csv', index=False)
    trade_features.to_csv('taq_trade_features.csv', index=False)
    print("\nSaved feature matrices to research/")
    
    # Print summary
    print("\n=== BBO FEATURE SUMMARY ===")
    print(bbo_features.groupby('symbol').agg({
        'mean_spread_bps': 'mean',
        'quote_intensity': 'mean',
        'mean_depth_imbalance': 'mean',
        'mid_return': 'std',
    }).round(2))
    
    print("\n=== TRADE FEATURE SUMMARY ===")
    print(trade_features[['symbol', 'total_volume', 'trade_count', 'avg_trade_size', 'vwap_approx']].round(2))
    
    # Create visualizations
    print("\nGenerating visualizations...")
    plot_features(bbo_features, available_focus)


def plot_features(bbo_features, symbols):
    """Generate and save feature visualizations."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Spread by symbol
    ax = axes[0, 0]
    for sym in symbols:
        df = bbo_features[bbo_features['symbol'] == sym]
        ax.plot(df['minute'], df['mean_spread_bps'], label=sym, alpha=0.7)
    ax.set_ylabel('Mean Spread (bps)')
    ax.set_title('Spread Dynamics by Symbol (Hour 1)')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 2: Quote intensity
    ax = axes[0, 1]
    intensity = bbo_features.groupby('symbol')['quote_intensity'].mean().sort_values(ascending=False).head(10)
    ax.bar(intensity.index, intensity.values, color='steelblue')
    ax.set_ylabel('Quotes per Minute')
    ax.set_title('Average Quote Intensity')
    ax.tick_params(axis='x', rotation=45)
    
    # Plot 3: Depth imbalance distribution
    ax = axes[1, 0]
    for sym in symbols[:4]:
        df = bbo_features[bbo_features['symbol'] == sym]
        ax.hist(df['mean_depth_imbalance'], bins=30, alpha=0.5, label=sym)
    ax.set_xlabel('Depth Imbalance (bid - ask)/(bid + ask)')
    ax.set_ylabel('Frequency')
    ax.set_title('Depth Imbalance Distribution')
    ax.legend(fontsize=8)
    
    # Plot 4: Realized variance (mid return volatility) by minute
    ax = axes[1, 1]
    for sym in symbols[:4]:
        df = bbo_features[bbo_features['symbol'] == sym].copy()
        df['cumulative_var'] = (df['mid_return'] ** 2).cumsum()
        ax.plot(df['minute'], df['cumulative_var'], label=sym, alpha=0.7)
    ax.set_ylabel('Cumulative Variance')
    ax.set_title('Cumulative Realized Variance by Minute')
    ax.legend(fontsize=8)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('taq_features_overview.png', dpi=150, bbox_inches='tight')
    print("Saved visualization to research/taq_features_overview.png")


if __name__ == '__main__':
    main()
