"""
Diffusion Map vs ISOMAP vs PCA on NVDA OBI (minute level, Oct 2-19 2023).

Two test families:

  1) GEOMETRY (apples-to-apples with v3 §5 cross-over):
        - Geodesic ρ: Spearman corr of pdist(embedding) vs ISOMAP geodesic distances
        - Euclidean R²: best linear back-projection from embedding to raw OBI
        Score Diffusion Map alongside ISOMAP and PCA on the SAME data.

  2) DYNAMICS (Diffusion Map's home turf):
        - Next-minute OBI prediction: train LinearRegression(Y(t)) -> OBI(t+1),
          score time-series-split R² on a held-out tail. If Diffusion Map's
          embedding genuinely captures transition structure, it should win here.
        - Transition smoothness: typical step size ||Y(t+1)-Y(t)|| relative
          to typical pair distance. Smaller = embedding lays out time smoothly.
        - Regime alignment: does kMeans on the embedding produce clusters
          that align with intraday time-of-day buckets?

Plus an ε bandwidth sensitivity sweep (DM's main hyperparameter).

Output: figures/, results.csv, console summary.
Run: python3 diffusion_map_test.py
"""
from __future__ import annotations
import os, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from scipy.spatial.distance import pdist, squareform, cdist
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths / constants
# ---------------------------------------------------------------------------
ROOT     = '/Users/harold/4. RA work/Factor_Training_Eng/DS4FE'
LOB_DIR  = f'{ROOT}/data/lob'
DATA     = f'{LOB_DIR}/lob_mbp10_NVDA_oct2023_full.parquet'
OUT_DIR  = f'{ROOT}/experiments/diffusion_map'
FIG_DIR  = f'{OUT_DIR}/figures'
os.makedirs(FIG_DIR, exist_ok=True)

OBI_COLS     = [f'obi_{k:02d}' for k in range(10)]
TRAIN_CUTOFF = '2023-10-19'


# ---------------------------------------------------------------------------
# Data loader — matches v3 notebook's build_obi_bars exactly
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
X_tr = df_train[OBI_COLS].values
N    = len(X_tr)
print(f'  Training bars: {N:,}')


# ---------------------------------------------------------------------------
# Diffusion Map — from scratch (α=1 Laplace-Beltrami normalization)
#
# Reference: Coifman & Lafon 2006. The α=1 choice ("anisotropic kernel")
# approximates the Laplace-Beltrami operator on the underlying manifold,
# independent of sampling density. This is the standard recommendation.
# ---------------------------------------------------------------------------
def fit_diffusion_map(X, epsilon, n_components, alpha=1.0):
    """Returns embedding (n, n_components), and all the pieces needed for
    OOS extension if you want it later (eigvals, eigvecs, normalisers)."""
    n = len(X)
    # Pairwise squared Euclidean distances
    D2 = squareform(pdist(X, 'sqeuclidean'))
    # Gaussian kernel
    K = np.exp(-D2 / epsilon)
    # α-normalization (anisotropic kernel)
    if alpha > 0:
        q = K.sum(axis=1)
        K = K / np.outer(q ** alpha, q ** alpha)
    # Row-normalize to Markov transition matrix P = D^{-1} K
    d = K.sum(axis=1)
    # Work with the SYMMETRIC variant S = D^{-1/2} K D^{-1/2} (eigendecomp stable)
    Dinv_sqrt = 1.0 / np.sqrt(d)
    S = K * np.outer(Dinv_sqrt, Dinv_sqrt)
    # Eigendecompose (S is symmetric)
    eigvals, eigvecs = np.linalg.eigh(S)
    # Sort descending
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Right eigenvectors of P:   ψ_i = D^{-1/2} φ_i
    psi = eigvecs * Dinv_sqrt[:, None]
    # Discard ψ_0 (constant); embedding = (λ_1 ψ_1, ..., λ_k ψ_k)
    Y = psi[:, 1:n_components + 1] * eigvals[1:n_components + 1]
    return Y, eigvals, psi


def median_epsilon(X):
    """Standard default: median of pairwise squared distances."""
    D2 = pdist(X, 'sqeuclidean')
    return float(np.median(D2))


# ---------------------------------------------------------------------------
# Fit reference embeddings (ISOMAP, PCA, Diffusion Map at default ε)
# ---------------------------------------------------------------------------
print('\nFitting reference embeddings (k=2)...')

t0 = time.time()
iso = Isomap(n_components=2, n_neighbors=15, n_jobs=-1).fit(X_tr)
Z_iso = iso.embedding_
print(f'  ISOMAP        fit in {time.time()-t0:.1f}s')

t0 = time.time()
pca = PCA(n_components=2).fit(X_tr)
Z_pca = pca.transform(X_tr)
print(f'  PCA           fit in {time.time()-t0:.1f}s')

t0 = time.time()
eps_default = median_epsilon(X_tr)
Z_dm, dm_eigvals, dm_psi = fit_diffusion_map(X_tr, eps_default, n_components=2, alpha=1.0)
print(f'  Diffusion Map fit in {time.time()-t0:.1f}s   (ε = {eps_default:.4f}, top eigvals: '
      f'{dm_eigvals[:5].round(4).tolist()})')


# ---------------------------------------------------------------------------
# Scoring helpers
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
# TEST 1 — Cross-over geometry comparison
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('TEST 1 — Cross-over geometry: who preserves what?')
print('=' * 70)

geom_records = []
for name, Z in [('ISOMAP (n=15, k=2)', Z_iso),
                ('PCA (k=2)',          Z_pca),
                ('Diffusion Map (k=2, α=1, ε=median)', Z_dm)]:
    rho = geodesic_rho(Z)
    r2  = linear_r2(Z, X_tr)
    geom_records.append(dict(method=name, geodesic_rho=rho, eucl_r2=r2))
    print(f'  {name:<42}  ρ={rho:>+.4f}   R²={r2:>+.4f}')


# ---------------------------------------------------------------------------
# TEST 2 — ε sensitivity for Diffusion Map
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('TEST 2 — Diffusion Map ε bandwidth sweep')
print('=' * 70)

eps_grid = [0.1 * eps_default, 0.3 * eps_default, eps_default,
            3 * eps_default, 10 * eps_default, 30 * eps_default]
eps_records = []
dm_embeddings_by_eps = {}
for eps in eps_grid:
    t0 = time.time()
    Z, eigvals, _ = fit_diffusion_map(X_tr, eps, n_components=2, alpha=1.0)
    rho = geodesic_rho(Z)
    r2  = linear_r2(Z, X_tr)
    eps_records.append(dict(epsilon=eps, geodesic_rho=rho, eucl_r2=r2,
                            spectral_gap=eigvals[1] - eigvals[2],
                            fit_seconds=time.time() - t0))
    dm_embeddings_by_eps[eps] = Z
    print(f'  ε={eps:>10.4f}  ρ={rho:>+.4f}  R²={r2:>+.4f}  '
          f'gap(λ₁-λ₂)={eigvals[1]-eigvals[2]:>+.4f}  ({time.time()-t0:.1f}s)')


# ---------------------------------------------------------------------------
# TEST 3 — Dynamics: next-minute OBI prediction
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('TEST 3 — Dynamics: does the embedding at t predict OBI at t+1?')
print('=' * 70)
print('Train a LinearRegression Y(t) -> OBI(t+1) on first 75% of bars,')
print('score R² on the held-out final 25%. Higher = embedding captures dynamics.\n')

split = int(0.75 * (N - 1))
# Targets: OBI at t+1
X_next = X_tr[1:]
X_now  = X_tr[:-1]
y_train, y_test = X_next[:split], X_next[split:]
x_train_idx, x_test_idx = slice(None, split), slice(split, None)

# Naive baseline: predict OBI(t+1) = OBI(t)  (persistence)
y_pred_naive = X_now[split:]
ss_res_naive = ((y_test - y_pred_naive) ** 2).sum()
ss_tot       = ((y_test - y_train.mean(0)) ** 2).sum()
r2_naive     = 1 - ss_res_naive / ss_tot
print(f'  Naive persistence baseline (Ŷ(t+1) = Y(t)):                 R²_test = {r2_naive:+.4f}')

dyn_records = []
for name, Z_full in [('ISOMAP (k=2)',                  Z_iso),
                     ('PCA (k=2)',                    Z_pca),
                     ('Diffusion Map (k=2, ε=median)', Z_dm)]:
    Z_now_train = Z_full[:-1][x_train_idx]
    Z_now_test  = Z_full[:-1][x_test_idx]
    reg = LinearRegression().fit(Z_now_train, y_train)
    pred = reg.predict(Z_now_test)
    ss_res = ((y_test - pred) ** 2).sum()
    r2_test = 1 - ss_res / ss_tot
    dyn_records.append(dict(method=name, r2_predict_next_obi=r2_test))
    print(f'  {name:<42}  R²_test = {r2_test:+.4f}')

# Direct full-OBI baseline (k=10, i.e. use the raw OBI as embedding)
reg_full = LinearRegression().fit(X_now[x_train_idx], y_train)
pred_full = reg_full.predict(X_now[x_test_idx])
r2_full = 1 - ((y_test - pred_full) ** 2).sum() / ss_tot
print(f'  Full 10-D OBI as input (ceiling):                            R²_test = {r2_full:+.4f}')


# ---------------------------------------------------------------------------
# TEST 4 — Transition smoothness
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('TEST 4 — Transition smoothness in the embedding')
print('=' * 70)
print('Compares typical t->t+1 step size to typical pair distance.')
print('Smaller ratio = embedding lays out time smoothly (dynamics-friendly).\n')

smooth_records = []
for name, Z in [('ISOMAP',         Z_iso),
                ('PCA',            Z_pca),
                ('Diffusion Map',  Z_dm)]:
    steps = np.linalg.norm(np.diff(Z, axis=0), axis=1)
    pairs = pdist(Z[np.random.default_rng(0).choice(len(Z), 500, replace=False)])
    ratio = np.median(steps) / np.median(pairs)
    smooth_records.append(dict(method=name, smoothness_ratio=ratio,
                               median_step=np.median(steps),
                               median_pair=np.median(pairs)))
    print(f'  {name:<16}  median step / median pair = {ratio:>.4f}')


# ---------------------------------------------------------------------------
# TEST 5 — Regime alignment with time-of-day
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('TEST 5 — Do embedding clusters align with intraday time-of-day buckets?')
print('=' * 70)
print('Bucket each bar into {open: 9:30-10:30, mid: 10:30-15:00, close: 15:00-16:00},')
print('then KMeans on the embedding (k=3) and score Adjusted Rand Index.\n')

idx_train = df_train.index
hour_min  = idx_train.hour + idx_train.minute / 60
tod_bucket = np.where(hour_min < 10.5, 0,
                      np.where(hour_min < 15.0, 1, 2))

regime_records = []
for name, Z in [('ISOMAP',         Z_iso),
                ('PCA',            Z_pca),
                ('Diffusion Map',  Z_dm)]:
    km = KMeans(n_clusters=3, n_init=20, random_state=0).fit(Z)
    ari = adjusted_rand_score(tod_bucket, km.labels_)
    regime_records.append(dict(method=name, ari_vs_tod=ari))
    print(f'  {name:<16}  ARI vs time-of-day buckets = {ari:>+.4f}')


# ---------------------------------------------------------------------------
# Save everything
# ---------------------------------------------------------------------------
df_geom    = pd.DataFrame(geom_records)
df_eps     = pd.DataFrame(eps_records)
df_dyn     = pd.DataFrame(dyn_records)
df_smooth  = pd.DataFrame(smooth_records)
df_regime  = pd.DataFrame(regime_records)

df_geom.to_csv(f'{OUT_DIR}/results_geometry.csv', index=False)
df_eps.to_csv(f'{OUT_DIR}/results_eps_sweep.csv', index=False)
df_dyn.to_csv(f'{OUT_DIR}/results_dynamics.csv', index=False)
df_smooth.to_csv(f'{OUT_DIR}/results_smoothness.csv', index=False)
df_regime.to_csv(f'{OUT_DIR}/results_regime.csv', index=False)
print(f'\nSaved 5 CSVs to {OUT_DIR}/')


# ---------------------------------------------------------------------------
# Figure 1 — Three embeddings side by side, coloured by time of day
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(3, 1, figsize=(11, 15))
for ax, (name, Z) in zip(axes, [('ISOMAP', Z_iso),
                                ('PCA', Z_pca),
                                ('Diffusion Map (α=1, ε=median)', Z_dm)]):
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
# Figure 2 — ε sensitivity for Diffusion Map
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 1, figsize=(11, 8))
axes[0].plot(df_eps['epsilon'], df_eps['geodesic_rho'], 'o-', color='#9B59B6', lw=2, ms=9, label='Diffusion Map')
axes[0].axhline(geom_records[0]['geodesic_rho'], ls='--', color='#E74C3C', lw=1.5, label=f"ISOMAP = {geom_records[0]['geodesic_rho']:.4f}")
axes[0].axhline(geom_records[1]['geodesic_rho'], ls=':',  color='#4472C4', lw=1.5, label=f"PCA = {geom_records[1]['geodesic_rho']:.4f}")
axes[0].set_xscale('log')
axes[0].set_xlabel('ε (Gaussian bandwidth, log scale)')
axes[0].set_ylabel('Geodesic ρ')
axes[0].set_title('Diffusion Map: Geodesic ρ vs ε', fontsize=12)
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(df_eps['epsilon'], df_eps['eucl_r2'], 'o-', color='#9B59B6', lw=2, ms=9, label='Diffusion Map')
axes[1].axhline(geom_records[0]['eucl_r2'], ls='--', color='#E74C3C', lw=1.5, label=f"ISOMAP = {geom_records[0]['eucl_r2']:.4f}")
axes[1].axhline(geom_records[1]['eucl_r2'], ls=':',  color='#4472C4', lw=1.5, label=f"PCA = {geom_records[1]['eucl_r2']:.4f}")
axes[1].set_xscale('log')
axes[1].set_xlabel('ε')
axes[1].set_ylabel('Euclidean R²')
axes[1].set_title('Diffusion Map: Euclidean R² vs ε', fontsize=12)
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/02_epsilon_sensitivity.png', dpi=120, bbox_inches='tight')
plt.close()


# ---------------------------------------------------------------------------
# Figure 3 — Dynamics test bar chart
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5))
methods_dyn = ['Naive\npersistence'] + [r['method'] for r in dyn_records] + ['Full 10-D OBI\n(ceiling)']
r2_vals     = [r2_naive] + [r['r2_predict_next_obi'] for r in dyn_records] + [r2_full]
colors_dyn  = ['#888'] + ['#E74C3C', '#4472C4', '#9B59B6'] + ['#27AE60']
bars = ax.bar(methods_dyn, r2_vals, color=colors_dyn, edgecolor='black', lw=0.6)
for b, v in zip(bars, r2_vals):
    ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005,
            f'{v:+.4f}', ha='center', fontsize=10)
ax.axhline(0, color='black', lw=0.5)
ax.set_ylabel('Test R² (held-out final 25% of training period)')
ax.set_title('Next-minute OBI prediction R² by embedding choice (k=2)', fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/03_dynamics_prediction.png', dpi=120, bbox_inches='tight')
plt.close()


print(f'\nFigures saved to {FIG_DIR}/')
print('  01_embeddings_comparison.png   — three embeddings side-by-side')
print('  02_epsilon_sensitivity.png     — DM ε sweep vs ISOMAP/PCA baselines')
print('  03_dynamics_prediction.png     — next-minute OBI prediction R²')


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)

iso_rho = geom_records[0]['geodesic_rho']
pca_rho = geom_records[1]['geodesic_rho']
dm_rho  = geom_records[2]['geodesic_rho']
dm_best_rho = df_eps['geodesic_rho'].max()
dm_best_r2  = df_eps['eucl_r2'].max()

iso_dyn = dyn_records[0]['r2_predict_next_obi']
pca_dyn = dyn_records[1]['r2_predict_next_obi']
dm_dyn  = dyn_records[2]['r2_predict_next_obi']

print(f'GEOMETRY (cross-over):')
print(f'  ρ:   DM={dm_rho:+.4f}   ISO={iso_rho:+.4f}   PCA={pca_rho:+.4f}')
print(f'  Best DM ρ across ε: {dm_best_rho:+.4f}  ({"beats" if dm_best_rho > max(iso_rho, pca_rho) else "loses to"} the best of ISO/PCA)')
print()
print(f'DYNAMICS (predict OBI(t+1)):')
print(f'  DM={dm_dyn:+.4f}   ISO={iso_dyn:+.4f}   PCA={pca_dyn:+.4f}   '
      f'(naive {r2_naive:+.4f}, ceiling {r2_full:+.4f})')

print('\nDone.')
