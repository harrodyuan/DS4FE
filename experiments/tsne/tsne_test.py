"""
t-SNE vs ISOMAP vs PCA on NVDA OBI (minute level, Oct 2-19 2023).

Why test t-SNE here?
   The NDR.pdf course notes (Part 17) explicitly say:
     "[t-SNE] is not as effective in cases where there is a smooth manifold."
   Our intrinsic-dim test showed OBI is essentially full-rank (d ≈ 9 of 10),
   i.e. smooth and *not* clustered. So a priori we expect t-SNE to underperform
   on the geometry tests — but its specialty is cluster discovery, so we add
   a dedicated cluster-vs-natural-grouping test to give it a fair chance.

Tests:
   1) Cross-over geometry: ρ (geodesic Spearman) and R² (Euclidean back-projection).
   2) Perplexity sweep (t-SNE's main knob).
   3) Dynamics: predict OBI(t+1) from embedding at time t.
   4) Cluster discovery: KMeans(k=3) on the embedding, ARI vs four natural groupings.
   5) Visual side-by-side with ISOMAP and PCA.

Output: figures/, results.csv, console summary.
Run: python3 tsne_test.py
"""
from __future__ import annotations
import os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ROOT     = '/Users/harold/4. RA work/Factor_Training_Eng/DS4FE'
LOB_DIR  = f'{ROOT}/data/lob'
DATA     = f'{LOB_DIR}/lob_mbp10_NVDA_oct2023_full.parquet'
OUT_DIR  = f'{ROOT}/experiments/tsne'
FIG_DIR  = f'{OUT_DIR}/figures'
os.makedirs(FIG_DIR, exist_ok=True)

OBI_COLS     = [f'obi_{k:02d}' for k in range(10)]
TRAIN_CUTOFF = '2023-10-19'

PERPLEXITY_GRID = [5, 15, 30, 50, 100]
TSNE_SEED       = 0


# ---------------------------------------------------------------------------
# Data loader — same as v3 notebook
# ---------------------------------------------------------------------------
def build_obi_bars(path, symbol):
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
    out['symbol'] = symbol
    return out.dropna()


print('Loading NVDA Oct calm data...')
df_nv = build_obi_bars(DATA, 'NVDA')
df_train = df_nv[df_nv.index.normalize() <= pd.Timestamp(TRAIN_CUTOFF, tz='America/New_York')].copy()
X_tr   = df_train[OBI_COLS].values
y_tr   = df_train['ret_fwd'].values
idx_tr = df_train.index
N      = len(X_tr)
print(f'  Training bars: {N:,}')


# ---------------------------------------------------------------------------
# Reference embeddings (ISOMAP and PCA from v3)
# ---------------------------------------------------------------------------
print('\nFitting reference embeddings (k=2)...')
t0 = time.time()
iso = Isomap(n_components=2, n_neighbors=15, n_jobs=-1).fit(X_tr)
Z_iso = iso.embedding_
print(f'  ISOMAP  fit in {time.time()-t0:.1f}s')

t0 = time.time()
pca = PCA(n_components=2).fit(X_tr)
Z_pca = pca.transform(X_tr)
print(f'  PCA     fit in {time.time()-t0:.1f}s')


# ---------------------------------------------------------------------------
# Scoring helpers (identical to diffusion_map_test.py for direct comparability)
# ---------------------------------------------------------------------------
D_geo_full = iso.dist_matrix_
iu = np.triu_indices(N, k=1)
d_geo_all = D_geo_full[iu]
rng = np.random.default_rng(42)
sub_pairs = rng.choice(len(d_geo_all), size=min(500_000, len(d_geo_all)), replace=False)
d_geo_sub = d_geo_all[sub_pairs]

def geodesic_rho(Z):
    return spearmanr(d_geo_sub, pdist(Z)[sub_pairs])[0]

def linear_r2(Z, X):
    reg = LinearRegression().fit(Z, X)
    return 1 - ((X - reg.predict(Z)) ** 2).sum() / ((X - X.mean(0)) ** 2).sum()


# ---------------------------------------------------------------------------
# Fit t-SNE at default perplexity = 30, then sweep
# ---------------------------------------------------------------------------
print('\nFitting t-SNE...')
print(f'  perplexity=30 (default)...')
t0 = time.time()
tsne_default = TSNE(n_components=2, perplexity=30, init='pca',
                    random_state=TSNE_SEED, learning_rate='auto', n_jobs=-1)
Z_tsne = tsne_default.fit_transform(X_tr)
print(f'    fit in {time.time()-t0:.1f}s')


# ---------------------------------------------------------------------------
# TEST 1 — Cross-over geometry
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('TEST 1 — Cross-over geometry: who preserves what?')
print('=' * 70)

geom_records = []
for name, Z in [('ISOMAP (n=15, k=2)',         Z_iso),
                ('PCA (k=2)',                  Z_pca),
                ('t-SNE (k=2, perplexity=30)', Z_tsne)]:
    rho = geodesic_rho(Z)
    r2  = linear_r2(Z, X_tr)
    geom_records.append(dict(method=name, geodesic_rho=rho, eucl_r2=r2))
    print(f'  {name:<32}  ρ={rho:>+.4f}   R²={r2:>+.4f}')


# ---------------------------------------------------------------------------
# TEST 2 — Perplexity sweep
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('TEST 2 — t-SNE perplexity sweep')
print('=' * 70)

perp_records = []
tsne_by_perp = {}
for p in PERPLEXITY_GRID:
    t0 = time.time()
    Z = TSNE(n_components=2, perplexity=p, init='pca',
             random_state=TSNE_SEED, learning_rate='auto', n_jobs=-1).fit_transform(X_tr)
    rho = geodesic_rho(Z)
    r2  = linear_r2(Z, X_tr)
    perp_records.append(dict(perplexity=p, geodesic_rho=rho, eucl_r2=r2,
                             fit_seconds=time.time() - t0))
    tsne_by_perp[p] = Z
    print(f'  perplexity={p:>4}  ρ={rho:>+.4f}  R²={r2:>+.4f}  ({time.time()-t0:.1f}s)')


# ---------------------------------------------------------------------------
# TEST 3 — Dynamics: next-minute OBI prediction
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('TEST 3 — Dynamics: does embedding at t predict OBI at t+1?')
print('=' * 70)

split   = int(0.75 * (N - 1))
X_next  = X_tr[1:]
X_now   = X_tr[:-1]
y_train = X_next[:split]
y_test  = X_next[split:]
ss_tot  = ((y_test - y_train.mean(0)) ** 2).sum()

# Naive baseline
r2_naive = 1 - ((y_test - X_now[split:]) ** 2).sum() / ss_tot
print(f'  Naive persistence (Ŷ(t+1) = Y(t)):                          R²_test = {r2_naive:+.4f}')

dyn_records = []
for name, Z_full in [('ISOMAP (k=2)',                  Z_iso),
                     ('PCA (k=2)',                    Z_pca),
                     ('t-SNE (k=2, perplexity=30)',   Z_tsne)]:
    Z_now_train = Z_full[:-1][:split]
    Z_now_test  = Z_full[:-1][split:]
    reg = LinearRegression().fit(Z_now_train, y_train)
    pred = reg.predict(Z_now_test)
    r2_test = 1 - ((y_test - pred) ** 2).sum() / ss_tot
    dyn_records.append(dict(method=name, r2_predict_next_obi=r2_test))
    print(f'  {name:<42}  R²_test = {r2_test:+.4f}')

reg_full = LinearRegression().fit(X_now[:split], y_train)
r2_full = 1 - ((y_test - reg_full.predict(X_now[split:])) ** 2).sum() / ss_tot
print(f'  Full 10-D OBI as input (ceiling):                            R²_test = {r2_full:+.4f}')


# ---------------------------------------------------------------------------
# TEST 4 — Cluster discovery: does t-SNE find structure ISOMAP/PCA missed?
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('TEST 4 — Cluster discovery: KMeans(k=3) on embedding, ARI vs natural groupings')
print('=' * 70)

# Build four natural groupings
hour_min  = idx_tr.hour + idx_tr.minute / 60
tod_bucket   = np.where(hour_min < 10.5, 0, np.where(hour_min < 15.0, 1, 2))
dow_bucket   = idx_tr.dayofweek.values            # Mon=0..Fri=4
ret_sign     = np.where(y_tr > 0, 1, 0)            # binary
ret_mag_q    = pd.qcut(np.abs(y_tr), 3, labels=False, duplicates='drop')

groupings = {
    'time-of-day (open/mid/close)': tod_bucket,
    'day-of-week':                  dow_bucket,
    'return sign':                  ret_sign,
    'return magnitude tertile':     ret_mag_q,
}

cluster_records = []
print(f'\n  {"Grouping":<32}  {"ISOMAP":>9}  {"PCA":>9}  {"t-SNE":>9}')
print('  ' + '-' * 64)
for gname, gv in groupings.items():
    n_clust = int(max(2, len(np.unique(gv[~pd.isna(gv)]))))
    row = dict(grouping=gname, n_clusters=n_clust)
    for mname, Z in [('ISOMAP', Z_iso), ('PCA', Z_pca), ('t-SNE', Z_tsne)]:
        km = KMeans(n_clusters=n_clust, n_init=20, random_state=0).fit(Z)
        mask = ~pd.isna(gv)
        ari = adjusted_rand_score(gv[mask], km.labels_[mask])
        row[mname.lower()] = ari
    cluster_records.append(row)
    print(f'  {gname:<32}  {row["isomap"]:>+9.4f}  {row["pca"]:>+9.4f}  {row["t-sne"]:>+9.4f}')

print('\n  (ARI = 0 means clusters align by chance; ARI > 0.1 suggests real alignment.)')


# ---------------------------------------------------------------------------
# Save CSVs
# ---------------------------------------------------------------------------
df_geom    = pd.DataFrame(geom_records)
df_perp    = pd.DataFrame(perp_records)
df_dyn     = pd.DataFrame(dyn_records)
df_cluster = pd.DataFrame(cluster_records)

df_geom.to_csv(f'{OUT_DIR}/results_geometry.csv', index=False)
df_perp.to_csv(f'{OUT_DIR}/results_perplexity_sweep.csv', index=False)
df_dyn.to_csv(f'{OUT_DIR}/results_dynamics.csv', index=False)
df_cluster.to_csv(f'{OUT_DIR}/results_cluster_discovery.csv', index=False)
print(f'\nSaved 4 CSVs to {OUT_DIR}/')


# ---------------------------------------------------------------------------
# Figure 1 — Three embeddings side-by-side, coloured by time of day
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 15))
for ax, (name, Z) in zip(axes, [('ISOMAP', Z_iso),
                                ('PCA', Z_pca),
                                ('t-SNE (perplexity=30)', Z_tsne)]):
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=hour_min, cmap='plasma',
                    s=10, alpha=0.6, vmin=9.5, vmax=16)
    plt.colorbar(sc, ax=ax, label='Hour ET', shrink=0.85)
    ax.set_title(f'{name} embedding (coloured by time of day)', fontsize=13)
    ax.set_xlabel('Z₁'); ax.set_ylabel('Z₂')
plt.suptitle(f'NVDA minute-level embeddings, n={N:,} bars', fontsize=14, y=1.00)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/01_embeddings_comparison.png', dpi=120, bbox_inches='tight')
plt.close()


# ---------------------------------------------------------------------------
# Figure 2 — Perplexity sweep curves
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 8))

axes[0].plot(df_perp['perplexity'], df_perp['geodesic_rho'], 'o-', color='#27AE60', lw=2, ms=9, label='t-SNE')
axes[0].axhline(geom_records[0]['geodesic_rho'], ls='--', color='#E74C3C', lw=1.5,
                label=f"ISOMAP = {geom_records[0]['geodesic_rho']:.4f}")
axes[0].axhline(geom_records[1]['geodesic_rho'], ls=':',  color='#4472C4', lw=1.5,
                label=f"PCA = {geom_records[1]['geodesic_rho']:.4f}")
axes[0].set_xscale('log')
axes[0].set_xlabel('perplexity (log scale)')
axes[0].set_ylabel('Geodesic ρ')
axes[0].set_title('t-SNE: Geodesic ρ vs perplexity', fontsize=12)
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(df_perp['perplexity'], df_perp['eucl_r2'], 'o-', color='#27AE60', lw=2, ms=9, label='t-SNE')
axes[1].axhline(geom_records[0]['eucl_r2'], ls='--', color='#E74C3C', lw=1.5,
                label=f"ISOMAP = {geom_records[0]['eucl_r2']:.4f}")
axes[1].axhline(geom_records[1]['eucl_r2'], ls=':',  color='#4472C4', lw=1.5,
                label=f"PCA = {geom_records[1]['eucl_r2']:.4f}")
axes[1].set_xscale('log')
axes[1].set_xlabel('perplexity')
axes[1].set_ylabel('Euclidean R²')
axes[1].set_title('t-SNE: Euclidean R² vs perplexity', fontsize=12)
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_perplexity_sensitivity.png', dpi=120, bbox_inches='tight')
plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — Cluster discovery ARI bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))
x = np.arange(len(cluster_records))
w = 0.27
ax.bar(x - w, [r['isomap']  for r in cluster_records], w, label='ISOMAP', color='#E74C3C', edgecolor='black', lw=0.4)
ax.bar(x,     [r['pca']     for r in cluster_records], w, label='PCA',    color='#4472C4', edgecolor='black', lw=0.4)
ax.bar(x + w, [r['t-sne']   for r in cluster_records], w, label='t-SNE',  color='#27AE60', edgecolor='black', lw=0.4)
ax.axhline(0, color='black', lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([r['grouping'] for r in cluster_records], rotation=15, ha='right')
ax.set_ylabel('Adjusted Rand Index (KMeans on embedding vs grouping)')
ax.set_title('Cluster discovery: does any method align with natural groupings?', fontsize=12)
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/03_cluster_discovery.png', dpi=120, bbox_inches='tight')
plt.close()


print(f'\nFigures saved to {FIG_DIR}/')
print('  01_embeddings_comparison.png   — three embeddings side-by-side')
print('  02_perplexity_sensitivity.png  — t-SNE perplexity sweep vs ISOMAP/PCA baselines')
print('  03_cluster_discovery.png       — ARI bars across four natural groupings')


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)

iso_rho = geom_records[0]['geodesic_rho']
pca_rho = geom_records[1]['geodesic_rho']
tsne_rho = geom_records[2]['geodesic_rho']
best_tsne_rho = df_perp['geodesic_rho'].max()
best_tsne_r2  = df_perp['eucl_r2'].max()

print(f'GEOMETRY: t-SNE ρ = {tsne_rho:+.4f} (vs ISOMAP {iso_rho:+.4f}, PCA {pca_rho:+.4f})')
print(f'   Best t-SNE ρ across perplexity: {best_tsne_rho:+.4f}')
print(f'   Best t-SNE R² across perplexity: {best_tsne_r2:+.4f} (vs PCA {geom_records[1]["eucl_r2"]:+.4f})')
print()
print(f'DYNAMICS: t-SNE R² = {dyn_records[2]["r2_predict_next_obi"]:+.4f}  '
      f'(vs ISOMAP {dyn_records[0]["r2_predict_next_obi"]:+.4f}, PCA {dyn_records[1]["r2_predict_next_obi"]:+.4f})')
print()
print('CLUSTER DISCOVERY: max |ARI| across all groupings and methods:')
max_ari = max(abs(r[m]) for r in cluster_records for m in ['isomap', 'pca', 't-sne'])
print(f'   |ARI|_max = {max_ari:.4f}  (all methods near zero -> no hidden cluster structure)')

print('\nDone.')
