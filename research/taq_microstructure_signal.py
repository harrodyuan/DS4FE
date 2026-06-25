"""
NYSE TAQ Microstructure Signal Analysis
=======================================
Tests the classic microstructure hypothesis that order flow imbalance
predicts short-term price pressure. Uses only BBO depth imbalance.
"""

import gzip
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score
import os

DATA_DIR = '../data/nyse_taq_samples'


def analyze_symbol(symbol, filepath, max_quotes=2000000):
    """Analyze one symbol: depth imbalance vs. future price direction."""
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
            if len(rows) >= max_quotes:
                break
    
    df = pd.DataFrame(rows)
    valid = (df['bid_px'] > 0) & (df['ask_px'] > 0) & (df['ask_px'] > df['bid_px'])
    df = df[valid].copy()
    
    df['time'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')
    df['minute'] = df['time'].dt.floor('min')
    df['mid'] = (df['bid_px'] + df['ask_px']) / 2
    df['depth_imbalance'] = (df['bid_sz'] - df['ask_sz']) / (df['bid_sz'] + df['ask_sz'])
    
    # Build 1-minute bars
    bars = df.groupby('minute').agg(
        open_mid=('mid', 'first'),
        close_mid=('mid', 'last'),
        mean_imbalance=('depth_imbalance', 'mean'),
        last_imbalance=('depth_imbalance', 'last'),
    ).reset_index()
    
    bars['return'] = np.log(bars['close_mid'] / bars['open_mid'])
    bars['forward_return'] = bars['return'].shift(-1)
    bars['target'] = (bars['forward_return'] > 0).astype(int)
    bars = bars.iloc[3:-3].copy()
    
    df_model = bars.dropna(subset=['mean_imbalance', 'last_imbalance', 'target'])
    if len(df_model) < 30:
        return None
    
    X = df_model[['mean_imbalance', 'last_imbalance']].values
    y = df_model['target'].values
    
    split = int(0.7 * len(df_model))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_train_s, y_train)
    
    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    baseline = max(np.mean(y_test), 1 - np.mean(y_test))
    try:
        auc = roc_auc_score(y_test, y_prob)
    except:
        auc = np.nan
    
    return {
        'symbol': symbol,
        'n_bars': len(bars),
        'test_acc': acc,
        'baseline': baseline,
        'auc': auc,
        'coef_mean': model.coef_[0][0],
        'coef_last': model.coef_[0][1],
        'bars': bars,
    }


def plot_signal(bars, symbol, result):
    """Plot depth imbalance and price together."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    ax1 = axes[0]
    ax1.plot(bars['minute'], bars['close_mid'], color='black', lw=0.8)
    ax1.set_ylabel('Mid Price')
    ax1.set_title(f'{symbol}: Mid Price and Depth Imbalance')
    
    ax2 = axes[1]
    colors = ['red' if r < 0 else 'green' for r in bars['return']]
    ax2.plot(bars['minute'], bars['mean_imbalance'], color='blue', lw=0.6, alpha=0.7, label='depth imbalance')
    ax2.axhline(0, color='gray', ls='--', lw=0.8)
    ax2.set_ylabel('Depth Imbalance')
    ax2.set_xlabel('Time')
    ax2.legend()
    
    if result:
        fig.suptitle(f'Acc={result["test_acc"]:.3f}, Baseline={result["baseline"]:.3f}, AUC={result["auc"]:.3f}', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'taq_signal_{symbol}.png', dpi=150, bbox_inches='tight')
    print(f"Saved plot: taq_signal_{symbol}.png")


def main():
    symbols = ['AAPL', 'DIA', 'IVV', 'IWM', 'GOOGL', 'AMZN']
    filepath = os.path.join(DATA_DIR, 'EQY_US_TAQ_NYSE_BBO_1_20231002.gz')
    
    results = []
    for symbol in symbols:
        print(f"\nAnalyzing {symbol}...")
        res = analyze_symbol(symbol, filepath)
        if res is None:
            print(f"  Insufficient data")
            continue
        
        print(f"  Bars: {res['n_bars']}")
        print(f"  Test accuracy: {res['test_acc']:.3f} (baseline: {res['baseline']:.3f})")
        print(f"  AUC: {res['auc']:.3f}")
        print(f"  Coef (mean imbalance): {res['coef_mean']:.4f}")
        
        results.append(res)
        plot_signal(res['bars'], symbol, res)
    
    if results:
        df = pd.DataFrame([{k: v for k, v in r.items() if k != 'bars'} for r in results])
        df.to_csv('taq_microstructure_signal.csv', index=False)
        print("\n\n=== SUMMARY ===")
        print(df[['symbol', 'test_acc', 'baseline', 'auc', 'coef_mean']].round(4))


if __name__ == '__main__':
    main()
