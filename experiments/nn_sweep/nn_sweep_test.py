"""
n_neighbors sensitivity sweep for ISOMAP on NVDA OBI.

Question: Does any choice of `n_neighbors` make ISOMAP-2 produce
   (a) different geometric scores from PCA-2,
   (b) a meaningfully different embedding shape, or
   (c) different predictive IC against forward returns
on the FULL NVDA Oct 2-19 training set (not subsampled)?

Output: figures/, results.csv, and a console summary.

Run: python3 nn_sweep_test.py
"""
from __future__ import annotations
import os, time, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ROOT     = '/Users/harold/4. RA work/Factor_Training_Eng/DS4FE'
LOB_DIR  = f'{ROOT}/data/lob'
DATA     = f'{LOB_DIR}/lob_mbp10_NVDA_oct2023_full.parquet'
OUT_DIR  = f'{ROOT}/experiments/nn_sweep'
FIG_DIR  = f'{OUT_DIR}/figures'
os.makedirs(FIG_DIR, exist_ok=True)

OBI_COLS    = [f'obi_{k:02d}' for k in range(10)]
TRAIN_CUTOFF = '2023-10-19'

# Sweep ranges
NN_GRID = [3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200]
SHOW_NN = [5, 15, 50, 150]   # which n values to plot embedding morphology for

# ---------------------------------------------------------------------------
# Data loader — same logic as notebook's build_obi_bars
# ---------------------------------------------------------------------------
def build_obi_bars(path, symbol):
    """Load mbp-10 parquet, filter to RTH, resample to 1-min OBI bars."""
    df = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index).tz_convert('America/New_York')
    mh = df.between_time('09:30', '16:00')

    frames = {}
    for k in range(10):
        b = mh[f'bid_sz_{k:02d}'].astype(np.int64)
        a = mh[f'ask_sz_{k:02d}'].astype(np.int64)
        denom = (b + a).replace(0, np.nan)
        frames[f'obi_{k:02d}'] = ((b - a) / denom).resample('1min').mean()
    frames['mid'] = ((mh['bid_px_00'] + mh['ask_px_00']) / 2).resample('1min').last()

    out = pd.DataFrame(frames).dropna()
    out['ret_fwd'] = out['mid'].pct_change().shift(-1)
    out['symbol']  = symbol
    return out.dropna()

print('Loading NVDA Oct calm data...')
df_nv = build_obi_bars(DATA, 'NVDA')
df_train = df_nv[df_nv.index.normalize() <= pd.Timestamp(TRAIN_CUTOFF, tz='America/New_York')].copy()
df_oos   = df_nv[df_nv.index.normalize() >  pd.Timestamp(TRAIN_CUTOFF, tz='America/New_York')].copy()
X_tr = df_train[OBI_COLS].values
X_oos = df_oos[OBI_COLS].values
y_tr  = df_train['ret_fwd'].values
y_oos = df_oos['ret_fwd'].values
print(f'  Training: {X_tr.shape}  OOS: {X_oos.shape}')

# ---------------------------------------------------------------------------
# Reference geodesic distances (fixed across the sweep so all ISOMAP fits are
# scored against the SAME ground-truth distance matrix). We use n=15 as the
# reference graph since that's the sklearn default; results are insensitive
# to this choice as long as n is moderate.
# ---------------------------------------------------------------------------
print('\nBuilding reference geodesic distances (n=15)...')
ref = Isomap(n_components=2, n_neighbors=15, n_jobs=-1).fit(X_tr)
D_geo_full = ref.dist_matrix_
iu = np.triu_indices(len(X_tr), k=1)
d_geo_all = D_geo_full[iu]

rng = np.random.default_rng(42)
sub_idx = rng.choice(len(d_geo_all), size=500_000, replace=False)
d_geo_sub = d_geo_all[sub_idx]

ss_tot_tr = ((X_tr  - X_tr.mean(0))  ** 2).sum()
ss_tot_oos= ((X_oos - X_oos.mean(0)) ** 2).sum()

# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------
def geodesic_rho(Z):
    return spearmanr(d_geo_sub, pdist(Z)[sub_idx])[0]

def linear_r2_train(Z):
    reg = LinearRegression().fit(Z, X_tr)
    return 1 - ((X_tr - reg.predict(Z)) ** 2).sum() / ss_tot_tr

def linear_r2_oos(Z_oos):
    reg = LinearRegression().fit(Z_oos, X_oos)
    return 1 - ((X_oos - reg.predict(Z_oos)) ** 2).sum() / ss_tot_oos

def ic_against_returns(Z, y):
    """Spearman IC of Z[:,0] and Z[:,1] separately against forward return."""
    return (spearmanr(Z[:, 0], y).statistic, spearmanr(Z[:, 1], y).statistic)

def best_axis_ic(Z, y):
    """Best linear combination axis IC (rotates Z to maximise |IC|)."""
    # Fit linear regression on (Z, y), then IC of fitted vs y.
    reg = LinearRegression().fit(Z, y)
    pred = reg.predict(Z)
    return spearmanr(pred, y).statistic

# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
print('\nSweeping n_neighbors:')
print(f'  {"n":>4} {"fit_s":>6}   {"ρ(geo)":>7} {"R²(tr)":>7} {"R²(oos)":>8}   '
      f'{"IC_Z1":>7} {"IC_Z2":>7}   {"IC_oos_Z1":>10} {"IC_oos_Z2":>10}')
print('-' * 92)

records = []
embeddings_train = {}
embeddings_oos = {}

for nn in NN_GRID:
    t0 = time.time()
    iso = Isomap(n_components=2, n_neighbors=nn, n_jobs=-1)
    Z_tr_  = iso.fit_transform(X_tr)
    Z_oos_ = iso.transform(X_oos)
    elapsed = time.time() - t0

    rho   = geodesic_rho(Z_tr_)
    r2_tr = linear_r2_train(Z_tr_)
    r2_oos= linear_r2_oos(Z_oos_)
    ic_z1, ic_z2 = ic_against_returns(Z_tr_, y_tr)
    ic_oos_z1, ic_oos_z2 = ic_against_returns(Z_oos_, y_oos)
    best_oos = best_axis_ic(Z_oos_, y_oos)   # rotate to max linear axis IC

    records.append(dict(method='ISOMAP', n_neighbors=nn, fit_seconds=elapsed,
                        geodesic_rho=rho, r2_train=r2_tr, r2_oos=r2_oos,
                        ic_train_z1=ic_z1, ic_train_z2=ic_z2,
                        ic_oos_z1=ic_oos_z1, ic_oos_z2=ic_oos_z2,
                        ic_oos_best_axis=best_oos))

    if nn in SHOW_NN:
        embeddings_train[nn] = Z_tr_
        embeddings_oos[nn]   = Z_oos_

    print(f'  {nn:>4} {elapsed:>5.1f}s  {rho:>+7.4f} {r2_tr:>+7.4f} {r2_oos:>+8.4f}   '
          f'{ic_z1:>+7.4f} {ic_z2:>+7.4f}   {ic_oos_z1:>+10.4f} {ic_oos_z2:>+10.4f}')

# PCA baselines
pca = PCA(n_components=2)
P_tr  = pca.fit_transform(X_tr)
P_oos = pca.transform(X_oos)
pca_rho   = geodesic_rho(P_tr)
pca_r2_tr = linear_r2_train(P_tr)
pca_r2_oos= linear_r2_oos(P_oos)
pca_ic_z1, pca_ic_z2 = ic_against_returns(P_tr, y_tr)
pca_ic_oos_z1, pca_ic_oos_z2 = ic_against_returns(P_oos, y_oos)
pca_best_oos = best_axis_ic(P_oos, y_oos)

records.append(dict(method='PCA', n_neighbors=None, fit_seconds=0.0,
                    geodesic_rho=pca_rho, r2_train=pca_r2_tr, r2_oos=pca_r2_oos,
                    ic_train_z1=pca_ic_z1, ic_train_z2=pca_ic_z2,
                    ic_oos_z1=pca_ic_oos_z1, ic_oos_z2=pca_ic_oos_z2,
                    ic_oos_best_axis=pca_best_oos))

print('-' * 92)
print(f'  PCA  --     {pca_rho:>+7.4f} {pca_r2_tr:>+7.4f} {pca_r2_oos:>+8.4f}   '
      f'{pca_ic_z1:>+7.4f} {pca_ic_z2:>+7.4f}   {pca_ic_oos_z1:>+10.4f} {pca_ic_oos_z2:>+10.4f}')

# Save CSV
df_results = pd.DataFrame(records)
df_results.to_csv(f'{OUT_DIR}/results.csv', index=False)
print(f'\nSaved {OUT_DIR}/results.csv  ({len(records)} rows)')

# ---------------------------------------------------------------------------
# Plot 1 — score curves vs n_neighbors with PCA baseline lines
# ---------------------------------------------------------------------------
df_iso = df_results[df_results['method'] == 'ISOMAP'].sort_values('n_neighbors')

fig, axes = plt.subplots(4, 1, figsize=(11, 16))

# (a) geodesic rho
axes[0].plot(df_iso['n_neighbors'], df_iso['geodesic_rho'],
             'o-', color='#E74C3C', lw=2, ms=8, label='ISOMAP')
axes[0].axhline(pca_rho, ls='--', color='#4472C4', lw=2, label=f'PCA = {pca_rho:.4f}')
axes[0].set_xscale('log')
axes[0].set_xlabel('n_neighbors  (log scale)')
axes[0].set_ylabel('Geodesic ρ  (preserves manifold distances)')
axes[0].set_title('Geodesic ρ vs n_neighbors  (full NVDA training)', fontsize=12)
axes[0].legend(fontsize=10)
axes[0].grid(alpha=0.3)

# (b) Euclidean R² (train) and (oos)
axes[1].plot(df_iso['n_neighbors'], df_iso['r2_train'],
             'o-', color='#E74C3C', lw=2, ms=8, label='ISOMAP — train')
axes[1].plot(df_iso['n_neighbors'], df_iso['r2_oos'],
             's-', color='#E67E22', lw=2, ms=8, label='ISOMAP — OOS')
axes[1].axhline(pca_r2_tr, ls='--', color='#4472C4', lw=2,
                label=f'PCA train = {pca_r2_tr:.4f}')
axes[1].axhline(pca_r2_oos, ls=':', color='#4472C4', lw=2,
                label=f'PCA OOS = {pca_r2_oos:.4f}')
axes[1].set_xscale('log')
axes[1].set_xlabel('n_neighbors')
axes[1].set_ylabel('Euclidean R² (best linear back-projection)')
axes[1].set_title('Euclidean R² vs n_neighbors  (PCA bound = 1.0 by construction)', fontsize=12)
axes[1].legend(fontsize=9)
axes[1].grid(alpha=0.3)

# (c) Direct IC of Z1, Z2 against forward return (OOS)
axes[2].plot(df_iso['n_neighbors'], df_iso['ic_oos_z1'],
             'o-', color='#E74C3C', lw=2, ms=8, label='ISOMAP IC(Z₁) OOS')
axes[2].plot(df_iso['n_neighbors'], df_iso['ic_oos_z2'],
             's-', color='#E67E22', lw=2, ms=8, label='ISOMAP IC(Z₂) OOS')
axes[2].plot(df_iso['n_neighbors'], df_iso['ic_oos_best_axis'],
             '^-', color='#27AE60', lw=2, ms=8, label='ISOMAP IC(best linear axis) OOS')
axes[2].axhline(pca_ic_oos_z1, ls='--', color='#4472C4', lw=1.5,
                label=f'PCA IC(PC₁) = {pca_ic_oos_z1:+.4f}')
axes[2].axhline(pca_ic_oos_z2, ls=':',  color='#4472C4', lw=1.5,
                label=f'PCA IC(PC₂) = {pca_ic_oos_z2:+.4f}')
axes[2].axhline(pca_best_oos, ls='-.', color='#2C3E50', lw=1.5,
                label=f'PCA IC(best axis) = {pca_best_oos:+.4f}')
axes[2].axhline(0, color='black', lw=0.5)
axes[2].set_xscale('log')
axes[2].set_xlabel('n_neighbors')
axes[2].set_ylabel('Spearman IC vs 1-min forward return (OOS)')
axes[2].set_title('Predictive IC vs n_neighbors  (the only chart that matters for return prediction)',
                  fontsize=12)
axes[2].legend(fontsize=8, loc='best', ncol=2)
axes[2].grid(alpha=0.3)

# (d) Fit time
axes[3].plot(df_iso['n_neighbors'], df_iso['fit_seconds'],
             'o-', color='#888', lw=2, ms=8)
axes[3].set_xscale('log')
axes[3].set_xlabel('n_neighbors')
axes[3].set_ylabel('Fit + transform seconds')
axes[3].set_title('Compute cost vs n_neighbors', fontsize=12)
axes[3].grid(alpha=0.3)

plt.suptitle(f'ISOMAP n_neighbors sweep on full NVDA training set (n={len(X_tr):,} bars)',
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/01_score_curves.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'Saved {FIG_DIR}/01_score_curves.png')

# ---------------------------------------------------------------------------
# Plot 2 — embedding morphology at selected n_neighbors (vs PCA reference)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(SHOW_NN) + 1, 1, figsize=(11, 4 * (len(SHOW_NN) + 1)))

obi_L0 = df_train[OBI_COLS[0]].values

for ax, nn in zip(axes[:-1], SHOW_NN):
    Z = embeddings_train[nn]
    extent = (np.percentile(Z[:, 0], [0.5, 99.5]).tolist()
              + np.percentile(Z[:, 1], [0.5, 99.5]).tolist())
    hb = ax.hexbin(Z[:, 0], Z[:, 1], C=obi_L0,
                   gridsize=42, cmap='RdBu', vmin=-0.4, vmax=0.4,
                   reduce_C_function=np.mean, mincnt=4,
                   extent=tuple(extent), linewidths=0.0)
    plt.colorbar(hb, ax=ax, shrink=0.85, label='mean OBI₀ per bin')
    ax.set_title(f'ISOMAP n_neighbors = {nn}', fontsize=12)
    ax.set_xlabel('Z₁'); ax.set_ylabel('Z₂')

# PCA reference
ax = axes[-1]
extent_p = (np.percentile(P_tr[:, 0], [0.5, 99.5]).tolist()
            + np.percentile(P_tr[:, 1], [0.5, 99.5]).tolist())
hb = ax.hexbin(P_tr[:, 0], P_tr[:, 1], C=obi_L0,
               gridsize=42, cmap='RdBu', vmin=-0.4, vmax=0.4,
               reduce_C_function=np.mean, mincnt=4,
               extent=tuple(extent_p), linewidths=0.0)
plt.colorbar(hb, ax=ax, shrink=0.85, label='mean OBI₀ per bin')
ax.set_title('PCA (reference)', fontsize=12, color='#4472C4', fontweight='bold')
ax.set_xlabel('PC₁'); ax.set_ylabel('PC₂')

plt.suptitle('Embedding morphology across n_neighbors (coloured by mean OBI L0)',
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_embedding_morphology.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'Saved {FIG_DIR}/02_embedding_morphology.png')

# ---------------------------------------------------------------------------
# Plot 3 — depth profile (per-level Spearman ρ) at selected n_neighbors
# Tests whether the AXIS INTERPRETATION changes with n_neighbors.
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(len(SHOW_NN) + 1, 1, figsize=(12, 3.5 * (len(SHOW_NN) + 1)))
xx = np.arange(10); ww = 0.35

for ax, nn in zip(axes[:-1], SHOW_NN):
    Z = embeddings_train[nn]
    z1c = [spearmanr(Z[:, 0], X_tr[:, k])[0] for k in range(10)]
    z2c = [spearmanr(Z[:, 1], X_tr[:, k])[0] for k in range(10)]
    ax.bar(xx - ww/2, z1c, ww, color='#E74C3C', alpha=0.85,
           edgecolor='black', lw=0.5, label='Z₁')
    ax.bar(xx + ww/2, z2c, ww, color='#4472C4', alpha=0.85,
           edgecolor='black', lw=0.5, label='Z₂')
    ax.axhline(0, color='black', lw=0.6)
    ax.set_xticks(xx); ax.set_xticklabels([f'L{k}' for k in range(10)])
    ax.set_ylim(-1, 1)
    ax.set_ylabel('Spearman ρ')
    ax.set_title(f'ISOMAP n_neighbors = {nn}', fontsize=12)
    ax.legend(fontsize=9)

# PCA reference
ax = axes[-1]
p1c = [spearmanr(P_tr[:, 0], X_tr[:, k])[0] for k in range(10)]
p2c = [spearmanr(P_tr[:, 1], X_tr[:, k])[0] for k in range(10)]
ax.bar(xx - ww/2, p1c, ww, color='#2ECC71', alpha=0.85,
       edgecolor='black', lw=0.5, label='PC₁')
ax.bar(xx + ww/2, p2c, ww, color='#9B59B6', alpha=0.85,
       edgecolor='black', lw=0.5, label='PC₂')
ax.axhline(0, color='black', lw=0.6)
ax.set_xticks(xx); ax.set_xticklabels([f'L{k}' for k in range(10)])
ax.set_ylim(-1, 1); ax.set_ylabel('Spearman ρ')
ax.set_title('PCA (reference)', fontsize=12, color='#4472C4', fontweight='bold')
ax.legend(fontsize=9)

plt.suptitle('Depth profile (per-level loadings) across n_neighbors\n'
             'Stable bar pattern => axis interpretation does not change with n_neighbors',
             fontsize=13, y=1.00)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/03_depth_profile_across_nn.png', dpi=120, bbox_inches='tight')
plt.close()
print(f'Saved {FIG_DIR}/03_depth_profile_across_nn.png')

# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
best_iso_geo = df_iso['geodesic_rho'].max()
best_iso_r2  = df_iso['r2_oos'].max()
best_iso_ic  = df_iso['ic_oos_best_axis'].abs().max()

print('\n' + '=' * 60)
print('VERDICT — does any n_neighbors make ISOMAP beat PCA?')
print('=' * 60)
print(f'Geodesic ρ:    best ISOMAP {best_iso_geo:+.4f}  vs  PCA {pca_rho:+.4f}    '
      f'Δ = {best_iso_geo - pca_rho:+.4f}')
print(f'OOS R²:        best ISOMAP {best_iso_r2:+.4f}  vs  PCA {pca_r2_oos:+.4f}    '
      f'Δ = {best_iso_r2 - pca_r2_oos:+.4f}')
print(f'OOS IC (best linear axis):')
print(f'              best ISOMAP {df_iso["ic_oos_best_axis"].abs().max():+.4f}  '
      f'vs  PCA {abs(pca_best_oos):+.4f}    '
      f'Δ = {df_iso["ic_oos_best_axis"].abs().max() - abs(pca_best_oos):+.4f}')
print()
print(f'Sweep range tested: n_neighbors ∈ {NN_GRID}')
print(f'Total fits: {len(NN_GRID)} ISOMAP + 1 PCA = {len(NN_GRID) + 1}')
print(f'Files written:')
print(f'  {OUT_DIR}/results.csv')
print(f'  {FIG_DIR}/01_score_curves.png')
print(f'  {FIG_DIR}/02_embedding_morphology.png')
print(f'  {FIG_DIR}/03_depth_profile_across_nn.png')
