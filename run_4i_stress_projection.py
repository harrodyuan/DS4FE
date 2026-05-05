"""
Part 4i — Stress Period Projection: BOJ Shock Week on the Calm Manifold

Trains ISOMAP on calm October 2023 NVDA data, then projects August 5-9 2024
(BOJ shock week) onto that manifold.  Two representations are compared:

  A. OBI means only (10 features) — the original Part 4f/4g representation
  B. OBI means + within-minute OBI std (20 features) — augmented representation

Key findings:
  - OBI means alone: stress lands inside the calm manifold (3.6% outside 95th pct)
  - OBI means + std:  stress clearly separates (16.8% outside, mean dist 2x calm)
  - Book depth surges ~11x during the shock week
  - Manifold path length shorter during stress (~1 SD below calm)

Run:
    python run_4i_stress_projection.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.spatial import ConvexHull
from sklearn.manifold import Isomap
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

LOB_DIR = 'data/lob'
FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

OBI_COLS = [f'obi_{k:02d}' for k in range(10)]
STD_COLS = [f'obi_std_{k:02d}' for k in range(10)]
DEP_COLS = [f'dep_{k:02d}' for k in range(10)]
AUG_COLS = OBI_COLS + STD_COLS

# ── Build bars (mean + std + depth per 1-min bar) ─────────────────────────────
def build_bars(path, label=''):
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index).tz_convert('America/New_York')
    mh = df.between_time('09:30', '16:00')
    frames = {}
    for k in range(10):
        b = mh[f'bid_sz_{k:02d}'].astype(np.int64)
        a = mh[f'ask_sz_{k:02d}'].astype(np.int64)
        d = (b + a).replace(0, np.nan)
        obi = (b - a) / d
        frames[f'obi_{k:02d}']     = obi.resample('1min').mean()
        frames[f'obi_std_{k:02d}'] = obi.resample('1min').std()
        frames[f'dep_{k:02d}']     = (b + a).resample('1min').mean()
    frames['mid'] = ((mh['bid_px_00'] + mh['ask_px_00']) / 2).resample('1min').last()
    bars = pd.DataFrame(frames).dropna()
    bars['ret_fwd']   = bars['mid'].pct_change().shift(-1)
    bars['tot_depth'] = bars[DEP_COLS].sum(axis=1)
    bars = bars.dropna()
    if label:
        print(f'  {label}: {len(bars):,} bars')
    return bars

print('Building OBI bars...')
calm = build_bars(f'{LOB_DIR}/lob_mbp10_NVDA_oct2023_full.parquet', 'NVDA calm')

stress_path = f'{LOB_DIR}/lob_mbp10_NVDA_stress_aug2024.parquet'
if not os.path.exists(stress_path):
    raise FileNotFoundError('Run: python download_mbp10_nvda_stress.py first.')
stress = build_bars(stress_path, 'NVDA stress (Aug 5-9 2024)')

n_train = int(len(calm) * 0.75)
dates_stress = pd.DatetimeIndex(stress.index).normalize()
unique_days  = sorted(dates_stress.unique())
stress_times = pd.DatetimeIndex(stress.index)

DAY_COLORS = {
    unique_days[0]: '#C0392B',
    unique_days[1]: '#E67E22',
    unique_days[2]: '#F1C40F',
    unique_days[3]: '#27AE60',
    unique_days[4]: '#2980B9',
}
DAY_LABELS = {d: d.strftime('%b %-d') for d in unique_days}

# ── Fit both ISOMAP models ────────────────────────────────────────────────────
print('\nFitting ISOMAP A (OBI means only, 10 features)...')
iso_A = Isomap(n_components=2, n_neighbors=15)
ZA_train  = iso_A.fit_transform(calm[OBI_COLS].values[:n_train])
ZA_calm   = iso_A.transform(calm[OBI_COLS].values)
ZA_stress = iso_A.transform(stress[OBI_COLS].values)
print(f'  Reconstruction error: {iso_A.reconstruction_error():.4f}  '
      f'({(1-iso_A.reconstruction_error())*100:.1f}% preserved)')

print('Fitting ISOMAP B (OBI mean + std, 20 features)...')
iso_B = Isomap(n_components=2, n_neighbors=15)
ZB_train  = iso_B.fit_transform(calm[AUG_COLS].values[:n_train])
ZB_calm   = iso_B.transform(calm[AUG_COLS].values)
ZB_stress = iso_B.transform(stress[AUG_COLS].values)
print(f'  Reconstruction error: {iso_B.reconstruction_error():.4f}  '
      f'({(1-iso_B.reconstruction_error())*100:.1f}% preserved)')

# ── Distance stats ────────────────────────────────────────────────────────────
def dist_from_centroid(Z, centroid, std):
    return np.sqrt(((Z - centroid) / std) ** 2).mean(axis=1)

ctr_A, std_A = ZA_train.mean(0), ZA_train.std(0)
ctr_B, std_B = ZB_train.mean(0), ZB_train.std(0)
dA_calm   = dist_from_centroid(ZA_train, ctr_A, std_A)
dA_stress = dist_from_centroid(ZA_stress, ctr_A, std_A)
dB_calm   = dist_from_centroid(ZB_train, ctr_B, std_B)
dB_stress = dist_from_centroid(ZB_stress, ctr_B, std_B)

p95_A, p95_B = np.percentile(dA_calm, 95), np.percentile(dB_calm, 95)
nA_out = (dA_stress > p95_A).sum()
nB_out = (dB_stress > p95_B).sum()

vol_calm   = calm['ret_fwd'].std()   * np.sqrt(390*252) * 100
vol_stress = stress['ret_fwd'].std() * np.sqrt(390*252) * 100

print(f'\n── Summary ─────────────────────────────────────────────────────────')
print(f'                          Model A (means)    Model B (means+std)')
print(f'  Calm 95th pct dist    : {p95_A:>15.3f}    {p95_B:>15.3f}')
print(f'  Stress mean dist      : {dA_stress.mean():>15.3f}    {dB_stress.mean():>15.3f}')
print(f'  Stress outside 95th   : {nA_out:>12}/{len(dA_stress)} ({nA_out/len(dA_stress)*100:.1f}%)   '
      f'{nB_out:>12}/{len(dB_stress)} ({nB_out/len(dB_stress)*100:.1f}%)')
for day in unique_days:
    m = dates_stress == day
    nA = (dA_stress[m] > p95_A).sum()
    nB = (dB_stress[m] > p95_B).sum()
    print(f'  {DAY_LABELS[day]}  outside-A={nA}/{m.sum()} ({nA/m.sum()*100:.1f}%)  '
          f'outside-B={nB}/{m.sum()} ({nB/m.sum()*100:.1f}%)')

print(f'\n  Return vol — calm: {vol_calm:.1f}%  |  stress: {vol_stress:.1f}%')
print(f'  Book depth — calm: {calm["tot_depth"].mean():,.0f}  |  '
      f'stress: {stress["tot_depth"].mean():,.0f}  '
      f'({stress["tot_depth"].mean()/calm["tot_depth"].mean():.1f}x)')

# ── Path length ───────────────────────────────────────────────────────────────
calm_dates = pd.DatetimeIndex(calm.index).normalize()
calm_lens = []
for day in sorted(calm_dates.unique()):
    m = calm_dates == day
    Z = iso_A.transform(calm[OBI_COLS].values[m])
    if len(Z) > 1:
        calm_lens.append(np.sqrt(np.diff(Z, axis=0)**2).sum(axis=1).sum())
cl_mean, cl_std = np.mean(calm_lens), np.std(calm_lens)

stress_lens = []
for day in unique_days:
    m = dates_stress == day
    Z = ZA_stress[m]
    length = np.sqrt(np.diff(Z, axis=0)**2).sum(axis=1).sum() if len(Z) > 1 else 0
    stress_lens.append(length)
    print(f'  Path {DAY_LABELS[day]}: {length:.1f}  ({(length-cl_mean)/cl_std:+.1f} SD)')

# ════════════════════════════════════════════════════════════════════════
print('\nGenerating figures...')

# ── Figure 1: Side-by-side A vs B ────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

for ax, ZA_tr, ZS, model_name, n_out, n_tot in [
    (axes[0], ZA_train, ZA_stress, 'Model A — OBI means only (10 features)',
     nA_out, len(dA_stress)),
    (axes[1], ZB_train, ZB_stress, 'Model B — OBI means + within-minute std (20 features)',
     nB_out, len(dB_stress)),
]:
    ax.scatter(ZA_tr[:, 0], ZA_tr[:, 1], c='#DDDDDD', s=4, alpha=0.4,
               rasterized=True, label='Calm Oct 2023')
    for day in unique_days:
        m = dates_stress == day
        ax.scatter(ZS[m, 0], ZS[m, 1], c=DAY_COLORS[day], s=16,
                   alpha=0.8, zorder=4, edgecolors='none', label=DAY_LABELS[day])
    ax.set_xlabel('Z₁'); ax.set_ylabel('Z₂')
    pct = n_out / n_tot * 100
    ax.set_title(f'{model_name}\n{pct:.1f}% of stress bars outside calm 95th pct', fontsize=10)
    ax.legend(fontsize=8, markerscale=2)

plt.suptitle('Adding Within-Minute OBI Variance Makes the Stress Period Detectable',
             fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_model_comparison.png')

# ── Figure 2: Stress embedding + time-of-day (model B) ───────────────────────
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

tod_train  = (pd.DatetimeIndex(calm.index[:n_train]).hour +
              pd.DatetimeIndex(calm.index[:n_train]).minute / 60)
tod_stress = (pd.DatetimeIndex(stress.index).hour +
              pd.DatetimeIndex(stress.index).minute / 60)

axes[0].scatter(ZB_train[:, 0], ZB_train[:, 1], c='#DDDDDD', s=4, alpha=0.4,
                rasterized=True, label='Calm Oct 2023')
for day in unique_days:
    m = dates_stress == day
    axes[0].scatter(ZB_stress[m, 0], ZB_stress[m, 1], c=DAY_COLORS[day],
                    s=16, alpha=0.8, zorder=4, edgecolors='none', label=DAY_LABELS[day])
axes[0].set_xlabel('Z₁'); axes[0].set_ylabel('Z₂')
axes[0].set_title('Augmented ISOMAP — stress vs calm\nColored by date', fontsize=10)
axes[0].legend(fontsize=8, markerscale=2)

sc1 = axes[1].scatter(ZB_train[:, 0], ZB_train[:, 1], c=tod_train,
                      cmap='Blues', s=4, alpha=0.25, vmin=9.5, vmax=16, rasterized=True)
sc2 = axes[1].scatter(ZB_stress[:, 0], ZB_stress[:, 1], c=tod_stress,
                      cmap='Reds', s=16, alpha=0.85, vmin=9.5, vmax=16, zorder=4)
plt.colorbar(sc1, ax=axes[1], label='Calm TOD (hour ET)', shrink=0.55, pad=0.01)
plt.colorbar(sc2, ax=axes[1], label='Stress TOD (hour ET)', shrink=0.55, pad=0.08)
axes[1].set_xlabel('Z₁'); axes[1].set_ylabel('Z₂')
axes[1].set_title('Augmented ISOMAP — time-of-day coloring\nStress open/close shift further from calm open/close',
                  fontsize=10)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_stress_embedding.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_stress_embedding.png')

# ── Figure 3: OBI std depth profile + book depth ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

x = np.arange(10)
calm_std_profile   = [calm[f'obi_std_{k:02d}'].mean() for k in range(10)]
stress_std_profile = [stress[f'obi_std_{k:02d}'].mean() for k in range(10)]
axes[0].plot(x, calm_std_profile,   'o-', color='steelblue', lw=2,
             label=f'Calm  (mean={np.mean(calm_std_profile):.3f})')
axes[0].plot(x, stress_std_profile, 's-', color='firebrick', lw=2,
             label=f'Stress (mean={np.mean(stress_std_profile):.3f})')
axes[0].fill_between(x, stress_std_profile, calm_std_profile,
                     alpha=0.15, color='firebrick')
axes[0].set_xticks(x); axes[0].set_xticklabels([f'L{k}' for k in range(10)])
axes[0].set_xlabel('OBI depth level'); axes[0].set_ylabel('Mean within-minute OBI std')
axes[0].set_title('Within-Minute OBI Volatility — Calm vs Stress\n'
                  'Stress has LOWER within-minute variance (more directional book)', fontsize=10)
axes[0].legend(fontsize=9)

# Book depth per day
calm_dep  = calm['tot_depth'].mean()
stress_day_dep = [stress['tot_depth'][dates_stress == d].mean() for d in unique_days]
colors_dep = [DAY_COLORS[d] for d in unique_days]
bars = axes[1].bar([DAY_LABELS[d] for d in unique_days], stress_day_dep,
                   color=colors_dep, alpha=0.85, edgecolor='black', lw=0.8)
axes[1].axhline(calm_dep, color='steelblue', lw=2, ls='--',
                label=f'Calm mean ({calm_dep:,.0f})')
for bar, val in zip(bars, stress_day_dep):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 200,
                 f'{val/1000:.0f}k', ha='center', va='bottom', fontsize=9)
axes[1].set_ylabel('Mean total depth (shares, all 10 levels)')
axes[1].set_title(f'Book Depth per Day — Stress vs Calm Baseline\n'
                  f'Stress depth is {stress["tot_depth"].mean()/calm_dep:.1f}x higher '
                  f'(includes 10:1 split effect)', fontsize=10)
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_obi_std_and_depth.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_obi_std_and_depth.png')

# ── Figure 4: Distance time series (model B) + returns ───────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

axes[0].axhline(p95_B, color='black', lw=1.2, ls='--',
                label=f'Calm 95th pct ({p95_B:.2f})')
axes[0].axhspan(0, p95_B, color='#D5E8D4', alpha=0.4)
for day in unique_days:
    m = dates_stress == day
    axes[0].scatter(stress_times[m], dB_stress[m],
                    c=DAY_COLORS[day], s=16, alpha=0.75, zorder=4,
                    edgecolors='none', label=DAY_LABELS[day])
axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%b %-d\n%H:%M'))
axes[0].set_ylabel('Manifold distance (augmented model)')
axes[0].set_title('Augmented ISOMAP Distance — Stress Detectable Throughout the Week',
                  fontsize=11)
axes[0].legend(fontsize=9, ncol=3, loc='upper right')
axes[0].set_xlim(stress_times.min(), stress_times.max())

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

# ── Figure 5: Aug 5 path on manifold B ───────────────────────────────────────
aug5 = unique_days[0]
m5   = dates_stress == aug5
Z5   = ZB_stress[m5]
tod5 = tod_stress[m5]

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(ZB_train[:, 0], ZB_train[:, 1], c='#EEEEEE', s=4,
           alpha=0.5, rasterized=True, zorder=1)
ax.plot(Z5[:, 0], Z5[:, 1], '-', color='#C0392B', lw=0.8, alpha=0.5, zorder=2)
sc = ax.scatter(Z5[:, 0], Z5[:, 1], c=tod5, cmap='plasma',
                s=40, alpha=0.9, vmin=9.5, vmax=16, zorder=3,
                edgecolors='black', linewidths=0.3)
ax.scatter(Z5[0, 0],  Z5[0, 1],  marker='*', s=350, c='gold', zorder=5,
           edgecolors='black', lw=0.5, label='09:30 open')
ax.scatter(Z5[-1, 0], Z5[-1, 1], marker='D', s=120, c='black', zorder=5,
           label='15:59 close')
plt.colorbar(sc, ax=ax, label='Hour ET')
ax.set_xlabel('Z₁'); ax.set_ylabel('Z₂')
ax.set_title('Aug 5 2024 (Crash Day) — Minute-by-Minute Path\n'
             'Augmented ISOMAP: grey = calm Oct 2023', fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_aug5_path.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_aug5_path.png')

# ── Figure 6: Path length + Z dist bar chart ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Path lengths
ax = axes[0]
bars = ax.bar([DAY_LABELS[d] for d in unique_days], stress_lens,
              color=[DAY_COLORS[d] for d in unique_days], alpha=0.85,
              edgecolor='black', lw=0.8)
ax.axhline(cl_mean, color='steelblue', lw=2, ls='-',
           label=f'Calm mean ({cl_mean:.0f})')
ax.axhspan(cl_mean - cl_std, cl_mean + cl_std, color='steelblue',
           alpha=0.12, label='±1 SD calm range')
for bar, val in zip(bars, stress_lens):
    sd = (val - cl_mean) / cl_std
    ax.text(bar.get_x() + bar.get_width()/2, val + 2,
            f'{sd:+.1f}σ', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_ylabel('Daily path length on ISOMAP manifold')
ax.set_title('Book State Mobility — Stress vs Calm\nNegative σ = more concentrated states',
             fontsize=10)
ax.legend(fontsize=9)

# Return vol per day
stress_day_vol = []
for day in unique_days:
    m = dates_stress == day
    v = stress['ret_fwd'][m].std() * np.sqrt(390) * 100
    stress_day_vol.append(v)
calm_daily_vol = calm['ret_fwd'].std() * np.sqrt(390) * 100

bars2 = axes[1].bar([DAY_LABELS[d] for d in unique_days], stress_day_vol,
                    color=[DAY_COLORS[d] for d in unique_days], alpha=0.85,
                    edgecolor='black', lw=0.8)
axes[1].axhline(calm_daily_vol, color='steelblue', lw=2, ls='--',
                label=f'Calm daily vol ({calm_daily_vol:.1f}%)')
for bar, val in zip(bars2, stress_day_vol):
    axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.1,
                 f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
axes[1].set_ylabel('Intraday return vol (daily, annualised)')
axes[1].set_title('Return Volatility per Day\nAug 5 has highest vol despite shortest path',
                  fontsize=10)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4i_path_and_vol.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4i_path_and_vol.png')

print(f'\n✓ All done.')
