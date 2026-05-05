"""
Generate two figures for the stress projection section:

  4i_calm_vs_stress_path.png
    Aug 5 minute-by-minute path vs a typical calm day (Oct 11)
    on the SAME augmented ISOMAP manifold. Makes "compact and directed"
    a claim against a concrete baseline.

  4i_regime_detector.png
    Simple threshold-based regime detector:
    flag each minute where Model B manifold distance > calm 95th pct.
    Shows first alert time per day and daily alert rate.
"""

import numpy as np, pandas as pd, matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.manifold import Isomap
import os, warnings
warnings.filterwarnings('ignore')

LOB_DIR  = 'data/lob'
FIG_DIR  = 'figures'
OBI_COLS = [f'obi_{k:02d}' for k in range(10)]
STD_COLS = [f'obi_std_{k:02d}' for k in range(10)]
AUG_COLS = OBI_COLS + STD_COLS

# ── Build bars (mean + within-min std) ──────────────────────────────────────
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
    bars = pd.DataFrame(frames).dropna()
    if label:
        print(f'  {label}: {len(bars):,} bars')
    return bars

print('Building bars ...')
calm   = build_bars(f'{LOB_DIR}/lob_mbp10_NVDA_oct2023_full.parquet', 'calm')
stress = build_bars(f'{LOB_DIR}/lob_mbp10_NVDA_stress_aug2024.parquet', 'stress')

n_train = int(len(calm) * 0.75)

# ── Fit augmented ISOMAP (Model B) ───────────────────────────────────────────
print('Fitting augmented ISOMAP (n_components=2, n_neighbors=15) ...')
iso_B = Isomap(n_neighbors=15, n_components=2)
ZB_train  = iso_B.fit_transform(calm[AUG_COLS].values[:n_train])
ZB_calm   = iso_B.transform(calm[AUG_COLS].values)
ZB_stress = iso_B.transform(stress[AUG_COLS].values)
print(f'  Reconstruction error: {iso_B.reconstruction_error():.4f}')

# ── Distance from training centroid ─────────────────────────────────────────
ctr   = ZB_train.mean(0)
std   = ZB_train.std(0)
dist_calm   = np.sqrt(((ZB_calm   - ctr) / std) ** 2).mean(axis=1)
dist_stress = np.sqrt(((ZB_stress - ctr) / std) ** 2).mean(axis=1)
p95  = np.percentile(dist_calm[:n_train], 95)
print(f'  Calm 95th pct distance: {p95:.3f}')

stress_times = pd.DatetimeIndex(stress.index)
dates_stress = stress_times.normalize()
unique_days  = sorted(dates_stress.unique())

DAY_COLORS = {
    unique_days[0]: '#C0392B',
    unique_days[1]: '#E67E22',
    unique_days[2]: '#F1C40F',
    unique_days[3]: '#27AE60',
    unique_days[4]: '#2980B9',
}

# ── Figure 1: calm-day path vs Aug 5 ────────────────────────────────────────
# Pick a typical calm Wednesday — Oct 11
calm_dates = pd.DatetimeIndex(calm.index).normalize()
calm_day   = pd.Timestamp('2023-10-11').date()
calm_mask  = calm_dates.date == calm_day
Z_calm_day = ZB_calm[calm_mask]
tod_calm   = (pd.DatetimeIndex(calm.index[calm_mask]).hour +
              pd.DatetimeIndex(calm.index[calm_mask]).minute / 60)

aug5     = unique_days[0]
m5       = dates_stress == aug5
Z5       = ZB_stress[m5]
tod5     = (stress_times[m5].hour + stress_times[m5].minute / 60)

print(f'\nCalm day (Oct 11): {calm_mask.sum()} bars')
print(f'Aug 5:             {m5.sum()} bars')

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, Z_path, tod, title, color, day_label in [
    (axes[0], Z_calm_day, tod_calm,
     'Oct 11 2023 — Typical Calm Day', '#2980B9', 'Calm'),
    (axes[1], Z5, tod5,
     'Aug 5 2024 — Crash Day (−16% intraday)', '#C0392B', 'Crash'),
]:
    # Training cloud
    ax.scatter(ZB_train[:, 0], ZB_train[:, 1], c='#EEEEEE', s=3,
               alpha=0.4, rasterized=True, zorder=1)
    # Path line
    ax.plot(Z_path[:, 0], Z_path[:, 1], '-', color=color,
            lw=1.0, alpha=0.6, zorder=2)
    # Points colored by time
    sc = ax.scatter(Z_path[:, 0], Z_path[:, 1], c=tod, cmap='plasma',
                    s=35, alpha=0.9, vmin=9.5, vmax=16, zorder=3,
                    edgecolors='black', linewidths=0.3)
    # Open/close markers
    ax.scatter(Z_path[0, 0],  Z_path[0, 1],  marker='*', s=300,
               c='gold', zorder=5, edgecolors='black', lw=0.5, label='09:30 open')
    ax.scatter(Z_path[-1, 0], Z_path[-1, 1], marker='D', s=100,
               c='black', zorder=5, label='15:59 close')
    plt.colorbar(sc, ax=ax, label='Hour ET', shrink=0.75)

    # Path length and % outside calm 95th
    path_len = np.sqrt(np.diff(Z_path, axis=0)**2).sum(axis=1).sum()
    dist_day = np.sqrt(((Z_path - ctr) / std)**2).mean(axis=1)
    pct_out  = (dist_day > p95).mean() * 100
    ax.set_title(f'{title}\nPath length: {path_len:.1f} | {pct_out:.0f}% outside calm 95th',
                 fontsize=10, fontweight='bold')
    ax.set_xlabel('Z₁  (book-wide consensus)', fontsize=10)
    ax.set_ylabel('Z₂  (near-vs-deep contrast)', fontsize=10)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.2, lw=0.6)

plt.suptitle('Minute-by-Minute Manifold Path: Calm vs Crash Day\n'
             'Augmented ISOMAP (OBI means + within-minute std), grey = calm training cloud',
             fontsize=11, fontweight='bold')
plt.tight_layout()
out1 = f'{FIG_DIR}/4i_calm_vs_stress_path.png'
plt.savefig(out1, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved {out1}')

# ── Figure 2: regime detector ────────────────────────────────────────────────
print('\n── Regime detector stats ──')
print(f'Calm base rate (by construction):  5.0%')
for day in unique_days:
    m     = dates_stress == day
    d_day = dist_stress[m]
    times = stress_times[m]
    flags = d_day > p95
    pct   = flags.mean() * 100
    first = times[flags][0].strftime('%H:%M') if flags.any() else 'none'
    n_consec = 0
    max_consec = 0
    for f in flags:
        n_consec = n_consec + 1 if f else 0
        max_consec = max(max_consec, n_consec)
    print(f'  {day.strftime("%b %-d")}: {pct:5.1f}% flagged | first alert {first} ET | '
          f'max consecutive: {max_consec} min')

fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=False)

# Top: alert rate per day as a bar chart
alert_rates = []
first_alerts = []
for day in unique_days:
    m     = dates_stress == day
    d_day = dist_stress[m]
    times = stress_times[m]
    flags = d_day > p95
    alert_rates.append(flags.mean() * 100)
    first = times[flags][0].strftime('%H:%M') if flags.any() else '—'
    first_alerts.append(first)

bar_colors = [DAY_COLORS[d] for d in unique_days]
day_labels  = [d.strftime('%b %-d') for d in unique_days]

bars = axes[0].bar(day_labels, alert_rates, color=bar_colors,
                   alpha=0.85, edgecolor='black', lw=0.8)
axes[0].axhline(5.0, color='steelblue', lw=2, ls='--',
                label='Calm base rate (5% by construction)')
for bar, rate, fa in zip(bars, alert_rates, first_alerts):
    axes[0].text(bar.get_x() + bar.get_width()/2, rate + 0.3,
                 f'{rate:.1f}%\nfirst: {fa}',
                 ha='center', va='bottom', fontsize=9, fontweight='bold')
axes[0].set_ylabel('% of 1-min bars flagged\n(distance > calm 95th pct)', fontsize=10)
axes[0].set_title('Simple Threshold Detector: Flag When Augmented Manifold Distance > Calm 95th Percentile',
                  fontsize=11, fontweight='bold')
axes[0].legend(fontsize=9)
axes[0].set_ylim(0, max(alert_rates) * 1.35)

# Bottom: minute-by-minute flags on Aug 5
m5_stress = dates_stress == aug5
d5        = dist_stress[m5_stress]
times5    = stress_times[m5_stress]
flags5    = d5 > p95

axes[1].fill_between(times5, 0, d5, where=~flags5,
                     color='#AAAAAA', alpha=0.5, label='Normal (< 95th pct)')
axes[1].fill_between(times5, 0, d5, where=flags5,
                     color='#C0392B', alpha=0.8, label='Flagged (≥ 95th pct)')
axes[1].axhline(p95, color='black', lw=1.5, ls='--',
                label=f'Calm 95th pct ({p95:.2f})')
axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
axes[1].set_xlabel('Time (ET), Aug 5 2024', fontsize=10)
axes[1].set_ylabel('Manifold distance', fontsize=10)
axes[1].set_title(f'Aug 5 Minute-by-Minute: '
                  f'{flags5.sum()} of {len(flags5)} bars flagged ({flags5.mean()*100:.0f}%), '
                  f'first alert {times5[flags5][0].strftime("%H:%M") if flags5.any() else "none"} ET',
                  fontsize=10, fontweight='bold')
axes[1].legend(fontsize=9, loc='upper right')
axes[1].set_xlim(times5.min(), times5.max())

plt.tight_layout()
out2 = f'{FIG_DIR}/4i_regime_detector.png'
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved {out2}')
