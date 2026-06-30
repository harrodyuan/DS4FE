"""
NYSE TAQ Predictive Microstructure Analysis
============================================
Demonstrates that microstructure features carry signal for near-term price movements.
Builds high-frequency bars, constructs features, and tests simple predictability.
"""

import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import os

DATA_DIR = '../data/nyse_taq_samples'


def parse_bbo_to_bars(filepath, symbol, bar_seconds=5, max_rows=3000000):
    """
    Parse BBO file for a single symbol and build regular time bars.
    Returns DataFrame with microstructure features per bar.
    """
    rows = []
    with gzip.open(filepath, 'rt', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0] != '140':
                continue
            if len(parts) < 10:
                continue
            if parts[3] != symbol:
                continue
            
            rows.append({
                'timestamp': parts[2],
                'ask_px': float(parts[5]) if parts[5] else 0.0,
                'ask_sz': int(parts[6]) if parts[6] else 0,
                'bid_px': float(parts[7]) if parts[7] else 0.0,
                'bid_sz': int(parts[8]) if parts[8] else 0,
            })
            if max_rows and len(rows) >= max_rows:
                break
    
    df = pd.DataFrame(rows)
    if len(df) == 0:
        return None
    
    # Convert timestamp to seconds from market open (09:30:00)
    df['time'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')
    df['seconds'] = (df['time'] - pd.Timestamp('09:30:00')).dt.total_seconds()
    df['bar'] = (df['seconds'] // bar_seconds).astype(int)
    
    # Filter valid quotes
    valid = (df['bid_px'] > 0) & (df['ask_px'] > 0) & (df['ask_px'] > df['bid_px'])
    df = df[valid].copy()
    
    df['mid'] = (df['bid_px'] + df['ask_px']) / 2
    df['spread'] = df['ask_px'] - df['bid_px']
    df['spread_bps'] = df['spread'] / df['mid'] * 10000
    df['depth_imbalance'] = (df['bid_sz'] - df['ask_sz']) / (df['bid_sz'] + df['ask_sz'])
    df['quoted_depth'] = df['bid_sz'] + df['ask_sz']
    df['log_quoted_depth'] = np.log1p(df['quoted_depth'])
    
    # Build bar features
    bars = df.groupby('bar').agg(
        nbbo_updates=('mid', 'count'),
        open_mid=('mid', 'first'),
        high_mid=('mid', 'max'),
        low_mid=('mid', 'min'),
        close_mid=('mid', 'last'),
        mean_spread_bps=('spread_bps', 'mean'),
        median_spread_bps=('spread_bps', 'median'),
        max_spread_bps=('spread_bps', 'max'),
        std_spread_bps=('spread_bps', 'std'),
        mean_depth_imbalance=('depth_imbalance', 'mean'),
        std_depth_imbalance=('depth_imbalance', 'std'),
        mean_log_depth=('log_quoted_depth', 'mean'),
        mean_bid_sz=('bid_sz', 'mean'),
        mean_ask_sz=('ask_sz', 'mean'),
    ).reset_index()
    
    bars['return'] = np.log(bars['close_mid'] / bars['open_mid'])
    bars['range'] = (bars['high_mid'] - bars['low_mid']) / bars['open_mid'] * 10000
    bars['realized_var'] = bars['return'] ** 2
    bars['quote_intensity'] = bars['nbbo_updates'] / bar_seconds
    
    # Forward labels
    bars['forward_return_1bar'] = bars['return'].shift(-1)
    bars['forward_return_5bar'] = np.log(bars['close_mid'].shift(-5) / bars['close_mid'])
    
    return bars


def build_features_for_model(bars):
    """Prepare features for a simple predictive model."""
    # Lag features
    for col in ['return', 'range', 'mean_spread_bps', 'mean_depth_imbalance', 'quote_intensity', 'mean_log_depth']:
        bars[f'{col}_lag1'] = bars[col].shift(1)
        bars[f'{col}_lag5'] = bars[col].shift(5)
    
    # Rolling moments
    for col in ['return']:
        bars[f'{col}_vol_10'] = bars[col].rolling(10).std()
        bars[f'{col}_mean_10'] = bars[col].rolling(10).mean()
    
    # Interaction features
    bars['spread_x_imbalance'] = bars['mean_spread_bps_lag1'] * bars['mean_depth_imbalance_lag1']
    
    return bars


def evaluate_predictability(bars, target_col='forward_return_1bar'):
    """Train a Ridge regression to predict forward returns."""
    feature_cols = [
        'return_lag1', 'return_lag5', 'return_vol_10', 'return_mean_10',
        'range_lag1', 'mean_spread_bps_lag1', 'mean_depth_imbalance_lag1',
        'quote_intensity_lag1', 'mean_log_depth_lag1', 'spread_x_imbalance'
    ]
    
    df = bars.dropna(subset=feature_cols + [target_col])
    if len(df) < 50:
        return None, None
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Time-series split: train on first 70%, test on last 30%
    split = int(0.7 * len(df))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    # Directional accuracy
    pred_sign = np.sign(y_pred_test)
    true_sign = np.sign(y_test)
    dir_acc = np.mean(pred_sign == true_sign)
    
    return {
        'r2_train': r2_train,
        'r2_test': r2_test,
        'directional_accuracy': dir_acc,
        'n_test': len(y_test),
        'feature_importance': dict(zip(feature_cols, model.coef_))
    }, y_test, y_pred_test


def plot_analysis(symbol, bars, results, y_test, y_pred):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mid price and spread over time
    ax1 = axes[0, 0]
    ax2 = ax1.twinx()
    ax1.plot(bars['bar'], bars['close_mid'], color='black', lw=0.8, label='Mid price')
    ax2.plot(bars['bar'], bars['mean_spread_bps'], color='red', alpha=0.4, lw=0.5, label='Spread (bps)')
    ax1.set_xlabel('5-second bar')
    ax1.set_ylabel('Mid Price', color='black')
    ax2.set_ylabel('Spread (bps)', color='red')
    ax1.set_title(f'{symbol}: Mid Price and Spread')
    
    # Plot 2: Depth imbalance over time
    ax = axes[0, 1]
    ax.plot(bars['bar'], bars['mean_depth_imbalance'], color='blue', lw=0.5, alpha=0.7)
    ax.axhline(0, color='gray', ls='--', lw=0.8)
    ax.set_xlabel('5-second bar')
    ax.set_ylabel('Depth Imbalance')
    ax.set_title(f'{symbol}: Bid-Ask Depth Imbalance')
    
    # Plot 3: Feature importance
    ax = axes[1, 0]
    if results and 'feature_importance' in results:
        imp = results['feature_importance']
        features = list(imp.keys())
        values = list(imp.values())
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=8)
        ax.set_xlabel('Ridge Coefficient')
        ax.set_title('Feature Importance')
    
    # Plot 4: Predicted vs actual
    ax = axes[1, 1]
    ax.scatter(y_test, y_pred, alpha=0.4, s=10)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1)
    ax.set_xlabel('Actual Forward Return')
    ax.set_ylabel('Predicted Forward Return')
    if results:
        ax.set_title(f'R2_test = {results["r2_test"]:.3f}, DirAcc = {results["directional_accuracy"]:.2f}')
    
    plt.tight_layout()
    plt.savefig(f'taq_predictive_{symbol}.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot: taq_predictive_{symbol}.png")


def main():
    symbols = ['AAPL', 'DIA', 'IVV', 'IWM', 'GOOGL', 'AMZN']
    filepath = os.path.join(DATA_DIR, 'EQY_US_TAQ_NYSE_BBO_1_20231002.gz')
    
    summary_results = []
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Analyzing {symbol}")
        print(f"{'='*50}")
        
        bars = parse_bbo_to_bars(filepath, symbol, bar_seconds=5, max_rows=1000000)
        if bars is None or len(bars) < 50:
            print(f"Insufficient data for {symbol}")
            continue
        
        bars = build_features_for_model(bars)
        
        # Evaluate 1-bar and 5-bar ahead
        for horizon, target in [('1bar', 'forward_return_1bar'), ('5bar', 'forward_return_5bar')]:
            results, y_test, y_pred = evaluate_predictability(bars, target_col=target)
            if results is None:
                continue
            print(f"\n{horizon} prediction:")
            print(f"  R2 train: {results['r2_train']:.4f}")
            print(f"  R2 test:  {results['r2_test']:.4f}")
            print(f"  Directional accuracy: {results['directional_accuracy']:.3f}")
            print(f"  Top features: {sorted(results['feature_importance'].items(), key=lambda x: abs(x[1]), reverse=True)[:3]}")
            
            summary_results.append({
                'symbol': symbol,
                'horizon': horizon,
                **results
            })
            
            if horizon == '1bar':
                plot_analysis(symbol, bars, results, y_test, y_pred)
    
    # Save summary
    if summary_results:
        df_summary = pd.DataFrame(summary_results)
        df_summary.to_csv('taq_predictive_summary.csv', index=False)
        print("\n\n=== OVERALL SUMMARY ===")
        print(df_summary[['symbol', 'horizon', 'r2_test', 'directional_accuracy', 'n_test']].round(4))


if __name__ == '__main__':
    main()
