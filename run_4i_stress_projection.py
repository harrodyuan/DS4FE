"""
Part 4i — Stress Period Projection: BOJ Shock Week on the Calm Manifold

Trains ISOMAP on calm October 2023 NVDA data, then projects August 5-9 2024
(BOJ shock week) onto that manifold to answer:

  1. Do stress bars fall inside or outside the calm manifold?
  2. Does the manifold cover a different region during stress?
  3. Does the Z1/Z2 signal learned in calm transfer to the stress period?
  4. What actually changes: the representation or the dynamics?

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
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

LOB_DIR  = 'data/lob'
FIG_DIR  = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

OBI_COLS = [f'obi_{k:02d}' for k in range(10)]

# ── Helper ─────────────────────────────────────────────────────────────────────
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
    frames['mid']    = ((mh['bid_px_00'] + mh['ask_px_00']) / 2).resample('1min').last()
    frames['spread'] = ((mh['ask_px_00'] - mh['bid_px_00'])
                        / ((mh['bid_px_00'] + mh['ask_px_00']) / 2) * 10000
                       ).resample('1min').mean()
    bars = pd.DataFrame(frames).dropna()
    bars['ret_fwd'] = bars['mid'].pct_change().shift(-1)
    bars = bars.dropna()
    if label:
        print(f'  {label}: {len(bars):,} bars')
    return bars

# ── Step 1: Build bars ────────────────────────────────────────────────────────
print('Building OBI bars...')
calm   = build_obi_bars(f'{LOB_DIR}/lob_mbp10_NVDA_oct2023_full.parquet',  'NVDA calm')

stress_path = f'{LOB_DIR}/lob_mbp10_NVDA_stress_aug2024.parquet'
if not os.path.exists(stress_path):
    raise FileNotFoundError(
        f'{stress_path} not found.\nRun: python download_mbp10_nvda_stress.py')
stress = build_obi_bars(stress_path, 'NVDA stress')

# ── Step 2: Fit ISOMAP on calm training set ───────────────────────────────────
n_train = int(len(calm) * 0.75)
X_train = calm[OBI_COLS].values[:n_train]
X_calm_all = calm[OBI_COLS].values

print('\nFitting ISOMAP on calm training data...')
iso = Isomap(n_components=2, n_neighbors=15)
Z_train    = iso.fit_transform(X_train)
Z_calm_all = iso.transform(X_calm_all)
Z_stress   = iso.transform(stress[OBI_COLS].values)

print(f'  Reconstruction error: {iso.reconstruction_error():.4f} '
      f'({(1 - iso.reconstruction_error())*100:.1f}% preserved)')

# TOD arrays
tod_train  = (pd.DatetimeIndex(calm.index[:n_train]).hour +
              pd.DatetimeIndex(calm.index[:n_train]).minute / 60)
tod_stress = (pd.DatetimeIndex(stress.index).hour +
              pd.DatetimeIndex(stress.index).minute / 60)
dates_stress  = pd.DatetimeIndex(stress.index).normalize()
unique_days   = sorted(dates_stress.unique())
stress_times  = pd.DatetimeIndex(stress.index)

DAY_COLORS = {
    unique_days[0]: '#C0392B',
    unique_days[1]: '#E67E22',
    unique_days[2]: '#F1C40F',
    unique_days[3]: '#27AE60',
    unique_days[4]: '#2980B9',
}
DAY_LABELS = {d: d.strftime('%b %-d') for d in unique_days}

# ── Step 3: Key statistics ────────────────────────────────────────────────────
calm_centroid = Z_train.mean(axis=0)
calm_std      = Z_train.std(axis=0)

dist_train  = np.sqrt(((Z_train   - calm_centroid) / calm_std) ** 2).mean(axis=1)
dist_stress = np.sqrt(((Z_stress  - calm_centroid) / calm_std) ** 2).mean(axis=1)
calm_95th   = np.percentile(dist_train, 95)

hull_train  = ConvexHull(Z_train)
hull_stress = ConvexHull(Z_stress)
area_ratio  = hull_stress.volume / hull_train.volume

vol_calm   = calm['ret_fwd'].std()   * np.sqrt(390 * 252) * 100
vol_stress = stress['ret_fwd'].std() * np.sqrt(390 * 252) * 100

n_outside = (dist_stress > calm_95th).sum()

print(f'\n  Calm manifold area        : {hull_train.volume:.2f}')
print(f'  Stress manifold area      : {hull_stress.volume:.2f}')
print(f'  Stress / Calm area ratio  : {area_ratio:.2f}')
print(f'  Calm return vol (ann.)    : {vol_calm:.1f}%')
print(f'  Stress return vol (ann.)  : {vol_stress:.1f}%')
print(f'  Stress bars outside 95th  : {n_outside}/{len(dist_stress)} ({n_outside/len(dist_stress)*100:.1f}%)')

# ── Figure 1: Stress bars on calm manifold, colored by day ────────────────────
print('\nGenerating figures...')
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].scatter(Z_train[:, 0], Z_train[:, 1], c='#DDDDDD', s=4, alpha=0.4,
                rasterized=True, label='Calm Oct 2023')
for day in unique_days:
    m = dates_stress == day
    axes[0].scatter(Z_stress[m, 0], Z_stress[m, 1],
                    c=DAY_COLORS[day], s=18, alpha=0.8, zorder=4,
                    edgecolors='none', label=DAY_LABELS[day])
axes[0].set_xlabel('Z₁'); axes[0].set_ylabel('Z₂')
axes[0].set_title('Stress Week on the Calm Manifold\n'
                  f'Stress covers {area_ratio*100:.0f}% of the calm training area  '
                  f'({n_outside/len(dist_stress)*100:.0f}% bars outside 95th pct)',
                  fontsize=10)
axes[0].legend(fontsize=9, markerscale=2)

# Right: calm TOD + stress TOD with shared colormap
sc1 = axes[1].scatter(Z_train[:, 0], Z_train[:, 1], c=tod_train,
                      cmap='Blues', s=4, alpha=0.25, vmin=9.5, vmax=16,
                      rasterized=True)
sc2 = axes[1].scatter(Z_stress[:, 0], Z_stress[:, 1], c=tod_stress,
                      cmap='Reds', s=18, alpha=0.85, vmin=9.5, vmax=16,
                      zorder=4, edgecolors='none')
plt.colorbar(sc1, ax=axes[1], label='Calm — hour ET', shrink=0.55, pad=0.01)
plt.colorbar(sc2, ax=axes[1], label='Stress — hour ET', shrink=0.55, pad=0.08)
axes[1].set_xlabel('Z₁'); axes[1].set_ylabel('Z₂')
axes[1].set_title('Same time-of-day coloring — calm (blue) vs stress (red)\n'
                  'Stress bars cover the same TOD structure but a compressed region',
                  fontsize=10)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_stress_embedding.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_stress_embedding.png')

# ── Figure 2: Manifold distance time series ───────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].axhline(calm_95th, color='black', lw=1.2, ls='--',
                label=f'Calm 95th pct ({calm_95th:.2f})')
axes[0].axhspan(0, calm_95th, color='#D5E8D4', alpha=0.5)
for day in unique_days:
    m = dates_stress == day
    axes[0].scatter(stress_times[m], dist_stress[m],
                    c=DAY_COLORS[day], s=16, alpha=0.75,
                    zorder=4, edgecolors='none', label=DAY_LABELS[day])
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %-d\n%H:%M'))
axes[0].set_ylabel('Manifold distance from calm centroid')
axes[0].set_title('LOB Manifold Distance During BOJ Shock Week (NVDA, Aug 5–9 2024)',
                  fontsize=11)
axes[0].legend(fontsize=9, ncol=3, loc='upper right')
axes[0].set_xlim(stress_times.min(), stress_times.max())

# Returns panel
ret_bps = stress['ret_fwd'].values * 10000
axes[1].bar(stress_times, ret_bps,
            color=['#C0392B' if r < 0 else '#27AE60' for r in ret_bps],
            alpha=0.7, width=pd.Timedelta('50s'))
axes[1].axhline(0, color='black', lw=0.8)
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%b %-d\n%H:%M'))
axes[1].set_ylabel('1-min return (bps)')
axes[1].set_title('NVDA 1-min Returns', fontsize=11)
axes[1].set_xlim(stress_times.min(), stress_times.max())

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_manifold_distance.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_manifold_distance.png')

# ── Figure 3: Aug 5 minute-by-minute path ────────────────────────────────────
aug5 = unique_days[0]
m5   = dates_stress == aug5
Z5   = Z_stress[m5]
tod5 = tod_stress[m5]

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(Z_train[:, 0], Z_train[:, 1], c='#EEEEEE', s=4, alpha=0.5,
           rasterized=True, zorder=1)
ax.plot(Z5[:, 0], Z5[:, 1], '-', color='#C0392B', lw=0.8, alpha=0.5, zorder=2)
sc = ax.scatter(Z5[:, 0], Z5[:, 1], c=tod5, cmap='plasma',
                s=40, alpha=0.9, vmin=9.5, vmax=16, zorder=3,
                edgecolors='black', linewidths=0.3)
ax.scatter(Z5[0, 0],  Z5[0, 1],  marker='*', s=350, c='gold',  zorder=5,
           edgecolors='black', lw=0.5, label='09:30 open')
ax.scatter(Z5[-1, 0], Z5[-1, 1], marker='D', s=120, c='black', zorder=5,
           label='15:59 close')
plt.colorbar(sc, ax=ax, label='Hour ET')
ax.set_xlabel('Z₁'); ax.set_ylabel('Z₂')
ax.set_title('Aug 5 2024 (Crash Day) — Minute-by-Minute Path on the Calm Manifold\n'
             'Grey: calm Oct 2023 training points', fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_aug5_path.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_aug5_path.png')

# ── Figure 4: Return vol and Z1/Z2 distributions side-by-side ────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Z1 distribution
axes[0].hist(Z_calm_all[:, 0], bins=60, density=True, alpha=0.6,
             color='steelblue', label=f'Calm  (σ={Z_calm_all[:,0].std():.2f})')
axes[0].hist(Z_stress[:, 0],   bins=60, density=True, alpha=0.6,
             color='firebrick', label=f'Stress (σ={Z_stress[:,0].std():.2f})')
axes[0].axvline(Z_calm_all[:,0].mean(), color='steelblue', lw=1.5, ls='--')
axes[0].axvline(Z_stress[:,0].mean(),   color='firebrick',  lw=1.5, ls='--')
axes[0].set_xlabel('Z₁ (book consensus)'); axes[0].set_ylabel('Density')
axes[0].set_title('Z₁ distribution'); axes[0].legend(fontsize=9)

# Z2 distribution
axes[1].hist(Z_calm_all[:, 1], bins=60, density=True, alpha=0.6,
             color='steelblue', label=f'Calm  (σ={Z_calm_all[:,1].std():.2f})')
axes[1].hist(Z_stress[:, 1],   bins=60, density=True, alpha=0.6,
             color='firebrick', label=f'Stress (σ={Z_stress[:,1].std():.2f})')
axes[1].axvline(Z_calm_all[:,1].mean(), color='steelblue', lw=1.5, ls='--')
axes[1].axvline(Z_stress[:,1].mean(),   color='firebrick',  lw=1.5, ls='--')
axes[1].set_xlabel('Z₂ (HFT-vs-institutional)'); axes[1].set_ylabel('Density')
axes[1].set_title('Z₂ distribution'); axes[1].legend(fontsize=9)

# Return vol comparison
periods  = ['Calm\nOct 2023', 'Stress\nAug 2024']
vols     = [vol_calm, vol_stress]
colors   = ['steelblue', 'firebrick']
bars = axes[2].bar(periods, vols, color=colors, alpha=0.8, edgecolor='black', lw=0.8)
for bar, v in zip(bars, vols):
    axes[2].text(bar.get_x() + bar.get_width()/2, v + 1, f'{v:.1f}%',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Annualised return vol (%)')
axes[2].set_title(f'Return volatility\n1-min NVDA returns, annualised')
axes[2].set_ylim(0, vol_stress * 1.25)

plt.suptitle('What Changes Between Calm and Stress: Not the State Space, But the Dynamics',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_calm_vs_stress.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_calm_vs_stress.png')

# ── Figure 5: Per-day distance boxplots ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
day_data  = [dist_stress[dates_stress == d] for d in unique_days]
day_names = [DAY_LABELS[d] for d in unique_days]
bp = ax.boxplot(day_data, patch_artist=True, widths=0.5,
                medianprops=dict(color='black', lw=2))
for patch, day in zip(bp['boxes'], unique_days):
    patch.set_facecolor(DAY_COLORS[day])
    patch.set_alpha(0.8)
ax.axhline(dist_train.mean(), color='steelblue', ls='-',  lw=1.5,
           label=f'Calm mean ({dist_train.mean():.2f})')
ax.axhline(calm_95th,         color='steelblue', ls='--', lw=1.5,
           label=f'Calm 95th pct ({calm_95th:.2f})')
ax.set_xticklabels(day_names)
ax.set_ylabel('Manifold distance from calm centroid')
ax.set_title('Distance from Calm Centroid by Day — Aug 5 (crash) vs later recovery days',
             fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_daily_distance.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_daily_distance.png')

print(f'\n✓ All done.')
print(f'\n── Key results ───────────────────────────────────────────────')
print(f'  Calm manifold area               : {hull_train.volume:.2f}')
print(f'  Stress manifold area             : {hull_stress.volume:.2f}  ({area_ratio*100:.0f}% of calm)')
print(f'  Calm return vol (ann.)           : {vol_calm:.1f}%')
print(f'  Stress return vol (ann.)         : {vol_stress:.1f}%')
print(f'  Stress bars outside calm 95th    : {n_outside}/{len(dist_stress)} ({n_outside/len(dist_stress)*100:.1f}%)')
for day in unique_days:
    m = dates_stress == day
    print(f'  {DAY_LABELS[day]}: mean dist = {dist_stress[m].mean():.3f}  '
          f'({(dist_stress[m] > calm_95th).sum()}/{m.sum()} above 95th pct)')
