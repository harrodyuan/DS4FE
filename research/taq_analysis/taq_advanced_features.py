"""
NYSE TAQ Advanced Feature Engineering
======================================
Builds robust microstructure features and tests their predictive power
for volatility and trade direction. Uses proper standardization and
realistic prediction targets.
"""

import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score, roc_auc_score
import os

DATA_DIR = '../data/nyse_taq_samples'


def parse_bbo_for_symbol(filepath, symbol, max_rows=5000000):
    """Parse full BBO stream for a single symbol."""
    rows = []
    with gzip.open(filepath, 'rt', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split(',')
            if parts[0] != '140' or parts[3] != symbol:
                continue
            if len(parts) < 10:
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
    return pd.DataFrame(rows)


def build_minute_bars(df, symbol):
    """Build 1-minute microstructure bars for a symbol."""
    # Keep valid quotes
    valid = (df['bid_px'] > 0) & (df['ask_px'] > 0) & (df['ask_px'] > df['bid_px'])
    df = df[valid].copy()
    
    df['time'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')
    df['minute'] = df['time'].dt.floor('min')
    df['mid'] = (df['bid_px'] + df['ask_px']) / 2
    df['spread'] = df['ask_px'] - df['bid_px']
    df['spread_bps'] = df['spread'] / df['mid'] * 10000
    df['depth_imbalance'] = (df['bid_sz'] - df['ask_sz']) / (df['bid_sz'] + df['ask_sz'])
    df['quoted_depth'] = df['bid_sz'] + df['ask_sz']
    df['log_depth'] = np.log1p(df['quoted_depth'])
    df['mid_return'] = np.log(df['mid'] / df['mid'].shift(1))
    
    # Aggregate to 1-minute bars
    bars = df.groupby('minute').agg(
        open_mid=('mid', 'first'),
        high_mid=('mid', 'max'),
        low_mid=('mid', 'min'),
        close_mid=('mid', 'last'),
        n_updates=('mid', 'count'),
        mean_spread_bps=('spread_bps', 'mean'),
        median_spread_bps=('spread_bps', 'median'),
        max_spread_bps=('spread_bps', 'max'),
        std_spread_bps=('spread_bps', 'std'),
        mean_depth_imbalance=('depth_imbalance', 'mean'),
        std_depth_imbalance=('depth_imbalance', 'std'),
        mean_log_depth=('log_depth', 'mean'),
    ).reset_index()
    
    # Minute-level returns
    bars['min_return'] = np.log(bars['close_mid'] / bars['open_mid'])
    bars['min_range'] = (bars['high_mid'] - bars['low_mid']) / bars['open_mid'] * 10000
    bars['realized_var'] = bars['min_return'] ** 2
    bars['quote_intensity'] = bars['n_updates'] / 60.0
    
    return bars


def add_features_and_labels(bars):
    """Add lag features and forward labels."""
    # Lags
    for col in ['min_return', 'min_range', 'realized_var', 'mean_spread_bps', 
                'mean_depth_imbalance', 'quote_intensity', 'max_spread_bps', 'mean_log_depth']:
        bars[f'{col}_lag1'] = bars[col].shift(1)
        bars[f'{col}_lag5'] = bars[col].shift(5)
    
    # Rolling volatility
    bars['vol_5'] = bars['min_return'].rolling(5).std()
    bars['vol_10'] = bars['min_return'].rolling(10).std()
    bars['mean_return_5'] = bars['min_return'].rolling(5).mean()
    
    # Forward labels: sum of next 5 minutes of realized variance
    bars['forward_vol'] = sum(bars['realized_var'].shift(-k) for k in range(1, 6))
    bars['forward_return'] = bars['min_return'].shift(-1)
    bars['forward_abs_return'] = bars['forward_return'].abs()
    
    return bars


def predict_volatility(bars, symbol):
    """Predict forward realized volatility using microstructure features."""
    feature_cols = [
        'realized_var_lag1', 'realized_var_lag5', 'vol_5', 'vol_10',
        'mean_spread_bps_lag1', 'max_spread_bps_lag1', 'mean_depth_imbalance_lag1',
        'quote_intensity_lag1', 'mean_log_depth_lag1', 'min_range_lag1'
    ]
    
    df = bars.dropna(subset=feature_cols + ['forward_vol'])
    if len(df) < 30:
        return None
    
    X = df[feature_cols].values
    y = df['forward_vol'].values
    
    # Log transform target for stability
    y_log = np.log1p(y)
    
    split = int(0.7 * len(df))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_log[:split], y_log[split:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = Ridge(alpha=10.0)
    model.fit(X_train_s, y_train)
    
    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)
    
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    
    return {
        'symbol': symbol,
        'task': 'volatility',
        'r2_train': r2_train,
        'r2_test': r2_test,
        'n_test': len(y_test),
        'importance': dict(zip(feature_cols, model.coef_))
    }, y_test, y_pred_test


def predict_direction(bars, symbol):
    """Predict direction of next-minute return using microstructure features."""
    feature_cols = [
        'min_return_lag1', 'mean_return_5', 'vol_5',
        'mean_depth_imbalance_lag1', 'mean_depth_imbalance_lag5',
        'mean_spread_bps_lag1', 'quote_intensity_lag1'
    ]
    
    df = bars.dropna(subset=feature_cols + ['forward_return'])
    if len(df) < 30:
        return None
    
    df['target'] = (df['forward_return'] > 0).astype(int)
    X = df[feature_cols].values
    y = df['target'].values
    
    split = int(0.7 * len(df))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_s, y_train)
    
    y_pred_train = model.predict(X_train_s)
    y_pred_test = model.predict(X_test_s)
    y_prob_test = model.predict_proba(X_test_s)[:, 1]
    
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    
    try:
        auc = roc_auc_score(y_test, y_prob_test)
    except:
        auc = np.nan
    
    return {
        'symbol': symbol,
        'task': 'direction',
        'train_acc': train_acc,
        'test_acc': test_acc,
        'auc': auc,
        'baseline': max(np.mean(y_test), 1 - np.mean(y_test)),
        'n_test': len(y_test),
        'importance': dict(zip(feature_cols, model.coef_[0]))
    }


def plot_results(symbol, bars, vol_results, y_test, y_pred):
    """Create comprehensive visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Price and spread
    ax1 = axes[0, 0]
    ax2 = ax1.twinx()
    ax1.plot(bars['minute'], bars['close_mid'], color='black', lw=0.8)
    ax2.plot(bars['minute'], bars['mean_spread_bps'], color='red', alpha=0.4, lw=0.5)
    ax1.set_ylabel('Mid Price', color='black')
    ax2.set_ylabel('Spread (bps)', color='red')
    ax1.set_title(f'{symbol}: Price and Spread Dynamics')
    
    # Plot 2: Volatility over time
    ax = axes[0, 1]
    ax.plot(bars['minute'], bars['realized_var'].rolling(5).sum(), color='blue', lw=0.8)
    ax.set_ylabel('5-min Realized Variance')
    ax.set_title(f'{symbol}: Realized Volatility')
    
    # Plot 3: Volatility prediction
    ax = axes[1, 0]
    ax.scatter(y_test, y_pred, alpha=0.4, s=15)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=1)
    ax.set_xlabel('Actual log(1+forward_vol)')
    ax.set_ylabel('Predicted log(1+forward_vol)')
    if vol_results:
        ax.set_title(f'Volatility Prediction: R2_test = {vol_results["r2_test"]:.3f}')
    
    # Plot 4: Feature importance
    ax = axes[1, 1]
    if vol_results and 'importance' in vol_results:
        imp = vol_results['importance']
        features = list(imp.keys())
        values = list(imp.values())
        y_pos = np.arange(len(features))
        ax.barh(y_pos, values, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=7)
        ax.set_xlabel('Ridge Coefficient')
        ax.set_title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig(f'taq_advanced_{symbol}.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot: taq_advanced_{symbol}.png")


def main():
    symbols = ['AAPL', 'DIA', 'IVV', 'IWM', 'GOOGL', 'AMZN']
    filepath = os.path.join(DATA_DIR, 'EQY_US_TAQ_NYSE_BBO_1_20231002.gz')
    
    all_results = []
    
    for symbol in symbols:
        print(f"\n{'='*50}")
        print(f"Processing {symbol}")
        print(f"{'='*50}")
        
        df = parse_bbo_for_symbol(filepath, symbol)
        if len(df) < 1000:
            print(f"Insufficient data for {symbol}: {len(df)} quotes")
            continue
        
        bars = build_minute_bars(df, symbol)
        bars = add_features_and_labels(bars)
        
        print(f"Quotes: {len(df)}, Minute bars: {len(bars)}")
        print(f"Avg spread: {bars['mean_spread_bps'].mean():.2f} bps")
        print(f"Avg quote intensity: {bars['quote_intensity'].mean():.2f} quotes/sec")
        
        # Volatility prediction
        vol_res, y_test, y_pred = predict_volatility(bars, symbol)
        if vol_res:
            print(f"\nVolatility prediction:")
            print(f"  R2 train: {vol_res['r2_train']:.4f}")
            print(f"  R2 test:  {vol_res['r2_test']:.4f}")
            print(f"  Top features: {sorted(vol_res['importance'].items(), key=lambda x: abs(x[1]), reverse=True)[:3]}")
            all_results.append(vol_res)
            plot_results(symbol, bars, vol_res, y_test, y_pred)
        
        # Direction prediction
        dir_res = predict_direction(bars, symbol)
        if dir_res:
            print(f"\nDirection prediction:")
            print(f"  Train accuracy: {dir_res['train_acc']:.3f}")
            print(f"  Test accuracy:  {dir_res['test_acc']:.3f}")
            print(f"  Baseline:       {dir_res['baseline']:.3f}")
            print(f"  AUC:            {dir_res['auc']:.3f}")
            all_results.append(dir_res)
    
    # Save summary
    if all_results:
        df_summary = pd.DataFrame([r for r in all_results if r is not None])
        df_summary.to_csv('taq_advanced_summary.csv', index=False)
        print("\n\n=== OVERALL SUMMARY ===")
        print(df_summary[['symbol', 'task', 'r2_test', 'test_acc', 'auc', 'n_test']].round(4))


if __name__ == '__main__':
    main()
