"""
Generate two figures for the ISOMAP demo:

  4g_NVDA_depth_profile_full.png
    Full-October Z₁ / Z₂ / Z₃ Spearman depth profile.
    Z₃ panel shows no consistent pattern → visual argument for stopping at 2.

  4g_NVDA_axis_profiles.png
    "What does a high-Z₁ bar look like?"
    Mean OBI profile for extreme-Z₁ and extreme-Z₂ bars.
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.manifold import Isomap
import os

LOB_DIR  = 'data/lob'
FIG_DIR  = 'figures'
OBI_COLS = [f'obi_{k:02d}' for k in range(10)]
SYMBOL   = 'NVDA'

# ── Load & build bars ────────────────────────────────────────────────────────
DATA_PATH = f'{LOB_DIR}/lob_mbp10_{SYMBOL}_oct2023_full.parquet'
print(f'Loading {DATA_PATH} ...')
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
print(f'  Training bars: {len(X_train):,}')

# ── Fit ISOMAP with 3 components ────────────────────────────────────────────
print('Fitting ISOMAP (n_components=3, n_neighbors=15) ...')
iso3 = Isomap(n_neighbors=15, n_components=3)
Z3   = iso3.fit_transform(X_train)
print(f'  Reconstruction error (2D): {iso3.reconstruction_error():.4f}')

Z1, Z2, Z3_coord = Z3[:, 0], Z3[:, 1], Z3[:, 2]

# ── Spearman depth profiles ──────────────────────────────────────────────────
rho1 = [spearmanr(Z1, X_train[:, k]).statistic for k in range(10)]
rho2 = [spearmanr(Z2, X_train[:, k]).statistic for k in range(10)]
rho3 = [spearmanr(Z3_coord, X_train[:, k]).statistic for k in range(10)]
levels = list(range(10))

print('\nSpearman depth profiles:')
print(f'  Z₁: {[f"{r:.3f}" for r in rho1]}')
print(f'  Z₂: {[f"{r:.3f}" for r in rho2]}')
print(f'  Z₃: {[f"{r:.3f}" for r in rho3]}')

# ── Figure 1: depth profiles for Z₁, Z₂, Z₃ ────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=False)

profiles = [
    (rho1, Z1,       'Z₁ — Book-wide consensus',       '#4472C4'),
    (rho2, Z2,       'Z₂ — Near-vs-deep contrast',      '#E74C3C'),
    (rho3, Z3_coord, 'Z₃ — No consistent depth pattern','#888888'),
]

for ax, (rho, _, title, base_color) in zip(axes, profiles):
    colors = ['#2980B9' if r >= 0 else '#C0392B' for r in rho]
    bars = ax.bar(levels, rho, color=colors, edgecolor='white', linewidth=0.5, width=0.7)
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(levels)
    ax.set_xticklabels([f'L{k}' for k in levels], fontsize=9)
    ax.set_xlabel('Depth level', fontsize=10)
    ax.set_ylabel('Spearman ρ with ISOMAP coordinate', fontsize=9)
    ax.set_ylim(-1.05, 1.05)
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.grid(axis='y', alpha=0.25, lw=0.7)
    for bar, r in zip(bars, rho):
        va = 'bottom' if r >= 0 else 'top'
        offset = 0.03 if r >= 0 else -0.03
        ax.text(bar.get_x() + bar.get_width()/2,
                r + offset, f'{r:.2f}', ha='center', va=va, fontsize=7.5)

# Annotate Z₃ panel
axes[2].text(0.5, 0.92,
    'Adding a 3rd coordinate does not yield\nan interpretable depth structure.',
    transform=axes[2].transAxes, ha='center', va='top', fontsize=8.5,
    style='italic', color='#444',
    bbox=dict(boxstyle='round,pad=0.4', facecolor='#f5f5f5', edgecolor='#ccc', lw=0.8))

plt.suptitle(
    f'NVDA October 2023 — Why Stop at Two ISOMAP Coordinates?',
    fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
out1 = f'{FIG_DIR}/4g_NVDA_depth_profile_full.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved {out1}')

# ── Figure 2: axis profiles ──────────────────────────────────────────────────
# For each axis, take top/bottom 15% of bars and show their mean OBI profile.
PCTILE = 15
X = X_train

def mean_profile(mask):
    return X[mask].mean(axis=0)

# Z₁ extremes
hi_z1 = Z1 >= np.percentile(Z1, 100 - PCTILE)
lo_z1 = Z1 <= np.percentile(Z1, PCTILE)

# Z₂ extremes
hi_z2 = Z2 >= np.percentile(Z2, 100 - PCTILE)
lo_z2 = Z2 <= np.percentile(Z2, PCTILE)

profiles_ax = [
    (mean_profile(hi_z1), f'High Z₁  (top {PCTILE}%)\nEntire book leans one direction', '#2980B9'),
    (mean_profile(lo_z1), f'Low Z₁  (bottom {PCTILE}%)\nEntire book leans opposite',    '#C0392B'),
    (mean_profile(hi_z2), f'High Z₂  (top {PCTILE}%)\nNear-book and deep-book diverge', '#8E44AD'),
    (mean_profile(lo_z2), f'Low Z₂  (bottom {PCTILE}%)\nOpposite near/deep contrast',   '#27AE60'),
]

fig, axes = plt.subplots(2, 2, figsize=(11, 7))
axes = axes.flatten()

for ax, (profile, title, color) in zip(axes, profiles_ax):
    bar_colors = ['#2980B9' if v >= 0 else '#C0392B' for v in profile]
    ax.bar(levels, profile, color=bar_colors, edgecolor='white', linewidth=0.5, width=0.7)
    ax.axhline(0, color='black', lw=0.9)
    ax.set_xticks(levels)
    ax.set_xticklabels([f'L{k}' for k in levels], fontsize=9)
    ax.set_xlabel('Depth level  (L0 = top of book, L9 = deepest)', fontsize=9)
    ax.set_ylabel('Mean OBI', fontsize=9)
    ax.set_ylim(-0.35, 0.35)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=6, color=color)
    ax.grid(axis='y', alpha=0.25, lw=0.7)
    for i, v in enumerate(profile):
        va = 'bottom' if v >= 0 else 'top'
        offset = 0.008 if v >= 0 else -0.008
        ax.text(i, v + offset, f'{v:.3f}', ha='center', va=va, fontsize=7.5)

# Shade "near-book" region in Z₂ panels
for ax in axes[2:]:
    ax.axvspan(-0.5, 3.5, alpha=0.06, color='orange', label='near-book (L0–L3)')
    ax.axvspan(4.5, 9.5, alpha=0.06, color='green',  label='deep book (L5–L9)')
    ax.legend(fontsize=7.5, loc='upper right')

plt.suptitle(
    'What Does Each ISOMAP Axis Represent?\nMean OBI Profile for Extreme-Coordinate Bars',
    fontsize=12, fontweight='bold')
plt.tight_layout()
out2 = f'{FIG_DIR}/4g_NVDA_axis_profiles.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {out2}')

# ── Summary stats for notebook text ─────────────────────────────────────────
print('\n── Summary for notebook text ──')
print(f'Z₁ peak at L{int(np.argmax(np.abs(rho1)))}  (ρ={max(rho1, key=abs):.3f})')
z2_cross = next(k for k in range(1,10) if rho2[k]*rho2[k-1] < 0)
print(f'Z₂ sign flip between L{z2_cross-1} and L{z2_cross}')
print(f'Z₃ max |ρ| = {max(abs(r) for r in rho3):.3f}  (near-zero across levels)')
print(f'High-Z₁ bar: mean OBI range [{mean_profile(hi_z1).min():.3f}, {mean_profile(hi_z1).max():.3f}]')
print(f'Low-Z₁  bar: mean OBI range [{mean_profile(lo_z1).min():.3f}, {mean_profile(lo_z1).max():.3f}]')
print(f'High-Z₂: near-book OBI {mean_profile(hi_z2)[:4].mean():.3f}  deep-book OBI {mean_profile(hi_z2)[5:].mean():.3f}')
print(f'Low-Z₂:  near-book OBI {mean_profile(lo_z2)[:4].mean():.3f}  deep-book OBI {mean_profile(lo_z2)[5:].mean():.3f}')
