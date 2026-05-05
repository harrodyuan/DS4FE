"""
Generate two clean ISOMAP-specific figures for the demo:

  4g_NVDA_isomap_tod.png
    Single-panel ISOMAP embedding colored by time of day.
    Replaces the 4-method dr_tod grid.

  4g_NVDA_oos_projection.png
    Full-October OOS projection (train gray, OOS colored by week).
    Replaces the short-window 4f version.
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
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
frames['mid'] = ((mh['bid_px_00'] + mh['ask_px_00']) / 2).resample('1min').last()

feat = pd.DataFrame(frames).dropna()
feat['ret_fwd'] = feat['mid'].pct_change().shift(-1)
feat = feat.dropna()

n_train  = int(len(feat) * 0.75)
train_df = feat.iloc[:n_train]
oos_df   = feat.iloc[n_train:]
X_train  = train_df[OBI_COLS].values
X_oos    = oos_df[OBI_COLS].values

print(f'  Train: {len(X_train):,} bars  |  OOS: {len(X_oos):,} bars')
print(f'  Train: {train_df.index.min().date()} → {train_df.index.max().date()}')
print(f'  OOS:   {oos_df.index.min().date()}   → {oos_df.index.max().date()}')

# ── Fit ISOMAP ───────────────────────────────────────────────────────────────
print('Fitting ISOMAP (n_neighbors=15, n_components=2) ...')
iso = Isomap(n_neighbors=15, n_components=2)
Z_train = iso.fit_transform(X_train)
Z_oos   = iso.transform(X_oos)
print(f'  Reconstruction error: {iso.reconstruction_error():.4f}')

# ── Figure 1: single-panel ISOMAP embedding colored by time of day ──────────
tod = train_df.index.hour + train_df.index.minute / 60.0

fig, ax = plt.subplots(figsize=(7, 5.5))
sc = ax.scatter(Z_train[:, 0], Z_train[:, 1],
                c=tod, cmap='plasma', s=6, alpha=0.45,
                vmin=9.5, vmax=16.0, rasterized=True)
cbar = plt.colorbar(sc, ax=ax)
cbar.set_label('Hour (ET)', fontsize=11)
cbar.set_ticks([10, 11, 12, 13, 14, 15, 16])
cbar.set_ticklabels(['10:00', '11:00', '12:00', '13:00', '14:00', '15:00', '16:00'])

ax.set_xlabel('Z₁  (book-wide consensus)', fontsize=12)
ax.set_ylabel('Z₂  (near-book vs deep-book contrast)', fontsize=12)
ax.set_title('NVDA October 2023 — ISOMAP 2D Embedding\n(6,435 one-minute bars, colored by time of day)',
             fontsize=11, fontweight='bold')

# annotate open/close region vs mid-day
# find approximate centroid of early (09:30-10:00) and late (15:30-16:00) bars
early_mask = tod <= 10.0
late_mask  = tod >= 15.5
mid_mask   = (tod >= 11.5) & (tod <= 14.0)

for mask, label, color in [
    (early_mask | late_mask, 'Open & close\nbars', '#FFD700'),
    (mid_mask,               'Mid-day bars',       '#00BFFF'),
]:
    cx = Z_train[mask, 0].mean()
    cy = Z_train[mask, 1].mean()
    ax.annotate(label, xy=(cx, cy), fontsize=8.5, color=color, fontweight='bold',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#111', alpha=0.55,
                          edgecolor=color, lw=0.8))

ax.grid(alpha=0.2, lw=0.6)
plt.tight_layout()
out1 = f'{FIG_DIR}/4g_NVDA_isomap_tod.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved {out1}')

# ── Figure 2: OOS projection (full October) ──────────────────────────────────
# Color OOS points by calendar week so you can see the temporal structure
oos_week = oos_df.index.isocalendar().week.values

fig, ax = plt.subplots(figsize=(7, 5.5))

# Training cloud in light gray
ax.scatter(Z_train[:, 0], Z_train[:, 1],
           c='#cccccc', s=4, alpha=0.3, label='Training (Oct 2–24)', rasterized=True)

# OOS points colored by week
unique_weeks = sorted(set(oos_week))
colors_oos   = plt.cm.viridis(np.linspace(0.15, 0.9, len(unique_weeks)))
for wk, col in zip(unique_weeks, colors_oos):
    mask = oos_week == wk
    # get date range for this week
    dates = oos_df.index[mask]
    label = f'Week of {dates.min().strftime("%b %d")} (OOS)'
    ax.scatter(Z_oos[mask, 0], Z_oos[mask, 1],
               c=[col], s=12, alpha=0.7, label=label, zorder=3)

ax.set_xlabel('Z₁  (book-wide consensus)', fontsize=12)
ax.set_ylabel('Z₂  (near-book vs deep-book contrast)', fontsize=12)
ax.set_title('NVDA October 2023 — Out-of-Sample Projection\n(last 25% held out, projected via Nyström extension)',
             fontsize=11, fontweight='bold')
ax.legend(fontsize=8.5, loc='upper left', framealpha=0.85)
ax.grid(alpha=0.2, lw=0.6)
plt.tight_layout()
out2 = f'{FIG_DIR}/4g_NVDA_oos_projection.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {out2}')
