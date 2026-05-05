"""
Generate 4g_NVDA_neighbor_sensitivity.png:
ISOMAP 2D residual vs k for NVDA October 2023.
Shows the embedding is stable across a wide range of k.
"""
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.manifold import Isomap

LOB_DIR  = 'data/lob'
FIG_DIR  = 'figures'
OBI_COLS = [f'obi_{k:02d}' for k in range(10)]
SYMBOL   = 'NVDA'

import os
FULL_PATH  = f'{LOB_DIR}/lob_mbp10_{SYMBOL}_oct2023_full.parquet'
SHORT_PATH = f'{LOB_DIR}/lob_mbp10_{SYMBOL}_calm_oct2023.parquet'
DATA_PATH  = FULL_PATH if os.path.exists(FULL_PATH) else SHORT_PATH
print(f'Loading {DATA_PATH}')

df = pd.read_parquet(DATA_PATH)
df.index = pd.DatetimeIndex(df.index).tz_convert('America/New_York')
mh = df.between_time('09:30', '16:00')

frames = {}
for k in range(10):
    b = mh[f'bid_sz_{k:02d}'].astype(np.int64)
    a = mh[f'ask_sz_{k:02d}'].astype(np.int64)
    denom = (b + a).replace(0, np.nan)
    frames[f'obi_{k:02d}'] = ((b - a) / denom).resample('1min').mean()

feat = pd.DataFrame(frames).dropna()
n_train = int(len(feat) * 0.75)
X_train = feat.iloc[:n_train][OBI_COLS].values
print(f'  Training set: {len(X_train):,} bars')

K_VALUES = [5, 10, 15, 20, 30, 50]
residuals = []

for k in K_VALUES:
    iso = Isomap(n_neighbors=k, n_components=2)
    iso.fit(X_train)
    r = iso.reconstruction_error()
    residuals.append(r)
    print(f'  k={k:>3}  residual={r:.4f}  (1-residual={1-r:.4f})')

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(K_VALUES, [1 - r for r in residuals], 'o-',
        color='#4472C4', lw=2, ms=8, zorder=3)
ax.axvline(15, color='#E74C3C', ls='--', lw=1.5, alpha=0.8, label='k=15 (per-symbol fits)')
ax.axvline(30, color='#2ECC71', ls='--', lw=1.5, alpha=0.8, label='k=30 (joint / stress fits)')
for k, r in zip(K_VALUES, residuals):
    ax.annotate(f'{1-r:.3f}', (k, 1 - r), textcoords='offset points',
                xytext=(0, 9), ha='center', fontsize=9, color='#333')
ax.set_xlabel('Number of neighbors (k)', fontsize=12)
ax.set_ylabel('1 − residual  (higher = better)', fontsize=12)
ax.set_title(f'{SYMBOL} — ISOMAP 2D Fidelity vs. k', fontsize=13, fontweight='bold')
ax.set_ylim(0.93, 1.00)
ax.set_xticks(K_VALUES)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
out = f'{FIG_DIR}/4g_NVDA_neighbor_sensitivity.png'
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {out}')
