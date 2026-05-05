"""
Part 4i — Stress Period Projection: BOJ Shock Week on the Calm Manifold

Trains ISOMAP on calm October 2023 NVDA data, then projects August 5-9 2024
(BOJ shock week) onto that manifold.

Key questions:
  - Where do stress-period bars land relative to the calm training region?
  - Does the book state deviate from the calm manifold systematically?
  - Can manifold distance serve as an unsupervised regime-change signal?

Run:
    python run_4i_stress_projection.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.stats import spearmanr
from scipy.spatial import ConvexHull
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

LOB_DIR  = 'data/lob'
FIG_DIR  = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

OBI_COLS = [f'obi_{k:02d}' for k in range(10)]

# ── Helper: build 1-min OBI bars from raw mbp-10 parquet ─────────────────────
def build_obi_bars(path, label=''):
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index).tz_convert('America/New_York')
    mh = df.between_time('09:30', '16:00')

    frames = {}
    for k in range(10):
        b = mh[f'bid_sz_{k:02d}'].astype(np.int64)
        a = mh[f'ask_sz_{k:02d}'].astype(np.int64)
        d = (b + a).replace(0, np.nan)
        frames[f'obi_{k:02d}'] = ((b - a) / d).resample('1min').mean()
    frames['mid'] = ((mh['bid_px_00'] + mh['ask_px_00']) / 2).resample('1min').last()

    bars = pd.DataFrame(frames).dropna()
    bars['ret_fwd'] = bars['mid'].pct_change().shift(-1)
    bars = bars.dropna()
    if label:
        print(f'  {label}: {len(bars):,} bars')
    return bars

# ── Step 1: Build calm training bars ─────────────────────────────────────────
print('Building calm OBI bars (NVDA Oct 2023)...')
calm = build_obi_bars(f'{LOB_DIR}/lob_mbp10_NVDA_oct2023_full.parquet', 'NVDA calm')

X_calm = calm[OBI_COLS].values
n_train = int(len(X_calm) * 0.75)
X_train = X_calm[:n_train]
X_calm_oos = X_calm[n_train:]
tod_train = (pd.DatetimeIndex(calm.index[:n_train]).hour +
             pd.DatetimeIndex(calm.index[:n_train]).minute / 60)
tod_calm_oos = (pd.DatetimeIndex(calm.index[n_train:]).hour +
                pd.DatetimeIndex(calm.index[n_train:]).minute / 60)

# ── Step 2: Fit calm ISOMAP ───────────────────────────────────────────────────
print('\nFitting ISOMAP on calm training data...')
iso = Isomap(n_components=2, n_neighbors=15)
Z_train = iso.fit_transform(X_train)
Z_calm_oos = iso.transform(X_calm_oos)
print(f'  Reconstruction error: {iso.reconstruction_error():.4f} '
      f'({(1 - iso.reconstruction_error())*100:.1f}% preserved)')

# Calm manifold centroid and spread
calm_centroid = Z_train.mean(axis=0)
calm_std = Z_train.std(axis=0)

# ── Step 3: Load and build stress bars ────────────────────────────────────────
stress_path = f'{LOB_DIR}/lob_mbp10_NVDA_stress_aug2024.parquet'
if not os.path.exists(stress_path):
    raise FileNotFoundError(
        f'{stress_path} not found.\n'
        'Run:  python download_mbp10_nvda_stress.py  first.'
    )

print('\nBuilding stress OBI bars (NVDA Aug 5-9 2024)...')
stress = build_obi_bars(stress_path, 'NVDA stress')

X_stress = stress[OBI_COLS].values
tod_stress = (pd.DatetimeIndex(stress.index).hour +
              pd.DatetimeIndex(stress.index).minute / 60)
dates_stress = pd.DatetimeIndex(stress.index).normalize()
unique_days = sorted(dates_stress.unique())

print(f'\n  Stress dates: {[str(d.date()) for d in unique_days]}')

# ── Step 4: Project stress onto calm manifold ─────────────────────────────────
print('\nProjecting stress bars onto calm manifold...')
Z_stress = iso.transform(X_stress)

# Mahalanobis-style distance from calm centroid (per-axis standardised)
dist_stress = np.sqrt(((Z_stress - calm_centroid) / calm_std) ** 2).mean(axis=1)
dist_calm   = np.sqrt(((Z_train  - calm_centroid) / calm_std) ** 2).mean(axis=1)

# Also compute for calm OOS as a reference
dist_calm_oos = np.sqrt(((Z_calm_oos - calm_centroid) / calm_std) ** 2).mean(axis=1)

print(f'  Calm train  — mean dist: {dist_calm.mean():.3f}  (95th pct: {np.percentile(dist_calm, 95):.3f})')
print(f'  Calm OOS    — mean dist: {dist_calm_oos.mean():.3f}  (95th pct: {np.percentile(dist_calm_oos, 95):.3f})')
print(f'  Stress      — mean dist: {dist_stress.mean():.3f}  (95th pct: {np.percentile(dist_stress, 95):.3f})')

calm_95th = np.percentile(dist_calm, 95)
n_outside = (dist_stress > calm_95th).sum()
print(f'  Stress bars outside calm 95th pct: {n_outside}/{len(dist_stress)} '
      f'({n_outside/len(dist_stress)*100:.1f}%)')

# ── Figures ───────────────────────────────────────────────────────────────────
print('\nGenerating figures...')

DAY_COLORS = {
    unique_days[0]: '#E74C3C',   # Aug 5 — crash day
    unique_days[1]: '#E67E22',   # Aug 6
    unique_days[2]: '#F1C40F',   # Aug 7
    unique_days[3]: '#2ECC71',   # Aug 8
    unique_days[4]: '#3498DB',   # Aug 9
}
DAY_LABELS = {d: d.strftime('%b %d') for d in unique_days}

# ── Figure 1: Stress bars on calm manifold ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: calm region + stress colored by day
axes[0].scatter(Z_train[:, 0], Z_train[:, 1], c='#CCCCCC', s=4, alpha=0.3,
                rasterized=True, label='Calm train (Oct 2023)')
for day in unique_days:
    m = dates_stress == day
    axes[0].scatter(Z_stress[m, 0], Z_stress[m, 1],
                    c=DAY_COLORS[day], s=18, alpha=0.75,
                    label=DAY_LABELS[day], zorder=4, edgecolors='none')
axes[0].set_xlabel('Z₁'); axes[0].set_ylabel('Z₂')
axes[0].set_title('Stress Week on the Calm Manifold\nColored by date', fontsize=11)
axes[0].legend(fontsize=9, markerscale=2)

# Right: calm TOD + stress TOD
sc = axes[1].scatter(Z_train[:, 0], Z_train[:, 1], c=tod_train,
                     cmap='Blues', s=4, alpha=0.25, vmin=9.5, vmax=16,
                     rasterized=True)
sc2 = axes[1].scatter(Z_stress[:, 0], Z_stress[:, 1], c=tod_stress,
                      cmap='Reds', s=18, alpha=0.8, vmin=9.5, vmax=16,
                      zorder=4, edgecolors='none')
plt.colorbar(sc,  ax=axes[1], label='Calm TOD (hour ET)', shrink=0.6)
plt.colorbar(sc2, ax=axes[1], label='Stress TOD (hour ET)', shrink=0.6)
axes[1].set_xlabel('Z₁'); axes[1].set_ylabel('Z₂')
axes[1].set_title('Calm vs Stress — same time-of-day coloring\nDo stress bars shift away from their expected TOD position?',
                  fontsize=11)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_stress_embedding.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_stress_embedding.png')

# ── Figure 2: Manifold distance time series ───────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

# Top: distance for each stress bar over the week
stress_times = pd.DatetimeIndex(stress.index)
axes[0].axhline(calm_95th, color='black', lw=1.2, ls='--',
                label=f'Calm 95th percentile ({calm_95th:.2f})')
axes[0].axhspan(0, calm_95th, color='#CCEECC', alpha=0.3, label='Normal calm range')
for day in unique_days:
    m = dates_stress == day
    t = stress_times[m]
    axes[0].scatter(t, dist_stress[m], c=DAY_COLORS[day],
                    s=20, alpha=0.8, zorder=4, edgecolors='none',
                    label=DAY_LABELS[day])
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
axes[0].set_ylabel('Manifold distance from calm centroid')
axes[0].set_title('LOB Manifold Distance — BOJ Shock Week (NVDA, Aug 5–9 2024)\n'
                  'Points above the dashed line are outside the normal calm range',
                  fontsize=11)
axes[0].legend(fontsize=9, ncol=3)
axes[0].set_xlim(stress_times.min(), stress_times.max())

# Bottom: 1-min mid-price return during stress
axes[1].bar(stress_times, stress['ret_fwd'] * 10000,
            color=['#E74C3C' if r < 0 else '#2ECC71'
                   for r in stress['ret_fwd']],
            alpha=0.7, width=0.0006)
axes[1].axhline(0, color='black', lw=0.8)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %d\n%H:%M'))
axes[1].set_ylabel('1-min return (bps)')
axes[1].set_title('NVDA 1-min Returns During Stress Week', fontsize=11)
axes[1].set_xlim(stress_times.min(), stress_times.max())

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_manifold_distance.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_manifold_distance.png')

# ── Figure 3: Z1, Z2 distribution — calm vs stress ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bins = 60
for i, (ax, label, zi_calm, zi_stress) in enumerate(zip(
        axes,
        ['Z₁ (book-wide consensus)', 'Z₂ (HFT-vs-institutional contrast)'],
        [Z_train[:, 0], Z_train[:, 1]],
        [Z_stress[:, 0], Z_stress[:, 1]])):
    rng = (min(zi_calm.min(), zi_stress.min()),
           max(zi_calm.max(), zi_stress.max()))
    ax.hist(zi_calm,   bins=bins, range=rng, density=True, alpha=0.6,
            color='steelblue', label='Calm Oct 2023')
    ax.hist(zi_stress, bins=bins, range=rng, density=True, alpha=0.6,
            color='firebrick', label='Stress Aug 2024')
    ax.axvline(zi_calm.mean(),   color='steelblue', lw=1.5, ls='--')
    ax.axvline(zi_stress.mean(), color='firebrick',  lw=1.5, ls='--')
    ax.set_xlabel(label); ax.set_ylabel('Density')
    ax.set_title(f'{label}\nCalm mean={zi_calm.mean():.3f} | Stress mean={zi_stress.mean():.3f}')
    ax.legend(fontsize=9)
plt.suptitle('Z₁ / Z₂ Distributions: Calm vs Stress Period', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_z_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_z_distributions.png')

# ── Figure 4: Aug 5 minute-by-minute path on the manifold ────────────────────
aug5 = unique_days[0]
m5   = dates_stress == aug5
Z5   = Z_stress[m5]
t5   = pd.DatetimeIndex(stress.index)[m5]
tod5 = tod_stress[m5]

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(Z_train[:, 0], Z_train[:, 1], c='#DDDDDD', s=4, alpha=0.3,
           rasterized=True, zorder=1)
sc = ax.scatter(Z5[:, 0], Z5[:, 1], c=tod5, cmap='plasma',
                s=35, alpha=0.9, vmin=9.5, vmax=16, zorder=3,
                edgecolors='black', linewidths=0.3)
# Draw the minute-by-minute path
ax.plot(Z5[:, 0], Z5[:, 1], '-', color='#E74C3C', lw=0.8, alpha=0.5, zorder=2)
# Mark open and close
ax.scatter(Z5[0, 0],  Z5[0, 1],  marker='*', s=300, c='gold', zorder=5, label='09:30 open')
ax.scatter(Z5[-1, 0], Z5[-1, 1], marker='D', s=100, c='black', zorder=5, label='15:59 close')
plt.colorbar(sc, ax=ax, label='Hour ET')
ax.set_xlabel('Z₁'); ax.set_ylabel('Z₂')
ax.set_title('Aug 5 2024 — Minute-by-Minute Path on the Calm Manifold\n'
             'Grey: calm Oct 2023 | Colored path: Aug 5 in time order', fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_aug5_path.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_aug5_path.png')

# ── Figure 5: Per-day distance boxplots ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
day_data  = [dist_stress[dates_stress == d] for d in unique_days]
day_names = [DAY_LABELS[d] for d in unique_days]
bp = ax.boxplot(day_data, patch_artist=True, widths=0.5,
                medianprops=dict(color='black', lw=2))
for patch, day in zip(bp['boxes'], unique_days):
    patch.set_facecolor(DAY_COLORS[day])
    patch.set_alpha(0.8)

# Add calm reference bands
ax.axhline(dist_calm.mean(),   color='steelblue', ls='-',  lw=1.5,
           label=f'Calm mean ({dist_calm.mean():.2f})')
ax.axhline(calm_95th, color='steelblue', ls='--', lw=1.5,
           label=f'Calm 95th pct ({calm_95th:.2f})')

ax.set_xticklabels(day_names)
ax.set_ylabel('Manifold distance from calm centroid')
ax.set_title('Daily Distribution of Manifold Distance — BOJ Shock Week\n'
             'Aug 5 (crash day) should sit highest above the calm baseline',
             fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_daily_distance.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_daily_distance.png')

print('\n✓ All done. Figures saved to figures/')
print(f'\nSummary stats:')
print(f'  Calm 95th percentile distance : {calm_95th:.3f}')
print(f'  Stress mean distance           : {dist_stress.mean():.3f}')
print(f'  Stress bars outside calm range : {n_outside}/{len(dist_stress)} ({n_outside/len(dist_stress)*100:.1f}%)')
for day in unique_days:
    m = dates_stress == day
    print(f'  {DAY_LABELS[day]}: mean dist = {dist_stress[m].mean():.3f}  '
          f'({(dist_stress[m] > calm_95th).sum()}/{m.sum()} bars outside calm range)')
