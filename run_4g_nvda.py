"""
Part 4g — Single-symbol DR analysis (NVDA / AAPL / MSFT / ...)
Run:  python3 run_4g_nvda.py [SYMBOL]
      python3 run_4g_nvda.py AAPL
"""

import sys
from dotenv import load_dotenv
import os, warnings, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr
from sklearn.manifold import Isomap, TSNE, trustworthiness
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import xgboost as xgb
from scipy.stats import t as t_dist

warnings.filterwarnings('ignore')
load_dotenv()

# ── Config ──────────────────────────────────────────────────────────────────
LOB_DIR  = 'data/lob'
FIG_DIR  = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)
SYMBOL   = sys.argv[1] if len(sys.argv) > 1 else 'NVDA'
OBI_COLS = [f'obi_{k:02d}' for k in range(10)]

# ── Step 1: Download if needed ───────────────────────────────────────────────
FULL_PATH  = f'{LOB_DIR}/lob_mbp10_{SYMBOL}_oct2023_full.parquet'
SHORT_PATH = f'{LOB_DIR}/lob_mbp10_{SYMBOL}_calm_oct2023.parquet'

if os.path.exists(FULL_PATH):
    DATA_PATH = FULL_PATH
    print(f'Using full-month file: {FULL_PATH}')
elif os.path.exists(SHORT_PATH):
    DATA_PATH = SHORT_PATH
    print(f'Full-month file not found — using existing 9-day file: {SHORT_PATH}')
    print('(Run with the full file for a larger dataset)')
else:
    api_key = os.environ.get('DATABENTO_API_KEY', '')
    if not api_key:
        raise ValueError('Set DATABENTO_API_KEY in .env file')
    import databento as db
    print('Downloading NVDA mbp-10 Oct 2023 full month (~$3.90)...')
    client = db.Historical(api_key)
    data   = client.timeseries.get_range(
        dataset='XNAS.ITCH', schema='mbp-10', symbols=[SYMBOL],
        start='2023-10-02', end='2023-11-01', stype_in='raw_symbol',
    )
    df_raw = data.to_df()
    df_raw.to_parquet(FULL_PATH)
    DATA_PATH = FULL_PATH
    print(f'Saved {len(df_raw):,} rows -> {FULL_PATH}')

# ── Step 2: Build 1-min OBI bars ─────────────────────────────────────────────
print('\nBuilding 1-min OBI bars...')
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

# Train = first 75%, OOS = last 25%
n_train      = int(len(feat) * 0.75)
train_df     = feat.iloc[:n_train]
oos_df       = feat.iloc[n_train:]

X_train = train_df[OBI_COLS].values
X_oos   = oos_df[OBI_COLS].values
y_train = train_df['ret_fwd'].values
y_oos   = oos_df['ret_fwd'].values

print(f'  Symbol : {SYMBOL}')
print(f'  Date   : {feat.index.min().date()} -> {feat.index.max().date()}')
print(f'  Train  : {len(X_train):,} bars  |  OOS: {len(X_oos):,} bars')

# ── Step 3: Fit all 4 DR methods ─────────────────────────────────────────────
print('\nFitting DR methods...')

pca  = PCA(n_components=2, random_state=42)
Z_pca = pca.fit_transform(X_train)
print(f'  PCA        : {pca.explained_variance_ratio_.sum()*100:.1f}% variance explained')

iso  = Isomap(n_components=2, n_neighbors=15)
Z_iso = iso.fit_transform(X_train)
iso_err = iso.reconstruction_error()
print(f'  ISOMAP     : reconstruction error = {iso_err:.4f}')

tsne = TSNE(n_components=2, perplexity=50, random_state=42, max_iter=1000)
Z_tsne = tsne.fit_transform(X_train)
print(f'  t-SNE      : KL divergence = {tsne.kl_divergence_:.4f}')

umap_model = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
Z_umap = umap_model.fit_transform(X_train)
print('  UMAP       : done')

Z_oos_iso  = iso.transform(X_oos)
Z_oos_umap = umap_model.transform(X_oos)
Z_oos_pca  = pca.transform(X_oos)

EMBEDDINGS = {'PCA': Z_pca, 'ISOMAP': Z_iso, 't-SNE': Z_tsne, 'UMAP': Z_umap}

# ── Step 4: DR quality metrics ───────────────────────────────────────────────
print('\nComputing trustworthiness & continuity...')
N_EVAL = min(2000, len(X_train))
idx    = np.random.RandomState(0).choice(len(X_train), N_EVAL, replace=False)
X_sub  = X_train[idx]

metrics = {}
for name, Z in EMBEDDINGS.items():
    Z_sub = Z[idx]
    tw = trustworthiness(X_sub, Z_sub, n_neighbors=15)
    ct = trustworthiness(Z_sub, X_sub, n_neighbors=15)
    metrics[name] = {'tw': tw, 'ct': ct}
    print(f'  {name:<8}: trustworthiness={tw:.4f}  continuity={ct:.4f}')

# ── Step 5: Scree comparison ─────────────────────────────────────────────────
pca_full = PCA().fit(X_train)
errors_iso, errors_pca = [], []
for nc in range(1, 10):
    m = Isomap(n_components=nc, n_neighbors=15).fit(X_train)
    errors_iso.append(m.reconstruction_error())
    errors_pca.append(1 - np.sum(pca_full.explained_variance_ratio_[:nc]))

# ── Step 6: K-Means regime detection ────────────────────────────────────────
print('\nRunning K-Means on UMAP embedding...')
K_RANGE    = range(2, 9)
sil_scores = []
for k in K_RANGE:
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(Z_umap)
    sil_scores.append(silhouette_score(Z_umap, labels, sample_size=3000))

best_k = list(K_RANGE)[int(np.argmax(sil_scores))]
print(f'  Best K = {best_k}  (silhouette = {max(sil_scores):.4f})')

km_final       = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_labels = km_final.fit_predict(Z_umap)
CLUSTER_COLORS = plt.cm.tab10(np.linspace(0, 1, best_k))

# ── Step 7: OOS IC comparison ────────────────────────────────────────────────
oos_regime = km_final.predict(Z_oos_umap)
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False, categories=[range(best_k)], handle_unknown='ignore')
ohe_tr  = ohe.fit_transform(cluster_labels.reshape(-1, 1))
ohe_oos = ohe.transform(oos_regime.reshape(-1, 1))

XGBp = dict(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0)
feature_sets = [
    ('Raw OBI (10D)',  X_train,                           X_oos),
    ('PCA 2D',        Z_pca,                             Z_oos_pca),
    ('ISOMAP 2D',     Z_iso,                             Z_oos_iso),
    ('UMAP 2D',       Z_umap,                            Z_oos_umap),
    ('UMAP+Regime',   np.hstack([Z_umap, ohe_tr]),       np.hstack([Z_oos_umap, ohe_oos])),
]

print(f'\nOOS IC comparison ({len(y_oos):,} bars):')
print(f'  {"Feature set":<20}  {"IC":>8}  {"t-stat":>8}  {"p":>9}')
ic_results = []
for name, Xtr, Xos in feature_sets:
    m = xgb.XGBRegressor(**XGBp)
    m.fit(Xtr, y_train)
    pred = m.predict(Xos)
    ic   = spearmanr(pred, y_oos).statistic
    n    = len(y_oos)
    ts   = ic * np.sqrt((n-2) / max(1-ic**2, 1e-9))
    pv   = 2 * (1 - t_dist.cdf(abs(ts), df=n-2))
    sig  = '**' if pv < 0.05 else ('*' if pv < 0.10 else 'n.s.')
    ic_results.append((name, ic, ts, pv))
    print(f'  {name:<20}  {ic:>+8.4f}  {ts:>+8.2f}  {pv:>9.4f}  {sig}')

# ── Step 8: Plots ─────────────────────────────────────────────────────────────
print('\nGenerating figures...')
tod = (pd.DatetimeIndex(train_df.index).hour +
       pd.DatetimeIndex(train_df.index).minute / 60)
ret = y_train
vmax_ret = np.percentile(np.abs(ret), 95)

# Figure 1: 2×2 DR embeddings colored by time-of-day
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()
for ax, (name, Z) in zip(axes, EMBEDDINGS.items()):
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=tod, cmap='plasma',
                    s=5, alpha=0.35, vmin=9.5, vmax=16)
    plt.colorbar(sc, ax=ax, label='Hour ET')
    tw = metrics[name]['tw']
    ct = metrics[name]['ct']
    ax.set_title(f'{name}  (trust={tw:.3f}, cont={ct:.3f})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
plt.suptitle(f'{SYMBOL} — Four DR Methods, 1-min OBI bars, colored by time of day', fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4g_{SYMBOL}_dr_tod.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved 4g_{SYMBOL}_dr_tod.png')

# Figure 2: 2×2 DR embeddings colored by forward return
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
axes = axes.flatten()
for ax, (name, Z) in zip(axes, EMBEDDINGS.items()):
    sc = ax.scatter(Z[:, 0], Z[:, 1], c=ret, cmap='RdBu',
                    s=5, alpha=0.35, vmin=-vmax_ret, vmax=vmax_ret)
    plt.colorbar(sc, ax=ax, label='1-min fwd return')
    ax.set_title(f'{name}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
plt.suptitle(f'{SYMBOL} — Four DR Methods, colored by forward return', fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4g_{SYMBOL}_dr_ret.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved 4g_{SYMBOL}_dr_ret.png')

# Figure 3: DR quality bar chart
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
names_m = list(metrics.keys())
tw_vals = [metrics[n]['tw'] for n in names_m]
ct_vals = [metrics[n]['ct'] for n in names_m]
colors4 = ['#4472C4', '#E74C3C', '#2ECC71', '#F39C12']
for ax, vals, label in zip(axes,
                            [tw_vals, ct_vals],
                            ['Trustworthiness (↑ better)', 'Continuity (↑ better)']):
    ax.bar(names_m, vals, color=colors4, alpha=0.8, edgecolor='black', lw=0.8)
    ax.set_ylim(0.85, 1.0)
    ax.axhline(1.0, color='grey', ls='--', lw=1)
    ax.set_ylabel(label)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.001, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
plt.suptitle(f'{SYMBOL} — DR Quality Metrics (n_neighbors=15)', fontsize=12)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4g_{SYMBOL}_dr_quality.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved 4g_{SYMBOL}_dr_quality.png')

# Figure 4: Scree comparison
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(range(1,10), errors_iso, 'o-', color='#4472C4', lw=2, ms=7, label='ISOMAP residual')
ax.plot(range(1,10), errors_pca, 's--', color='#E74C3C', lw=2, ms=7, label='PCA residual')
ax.axvline(2, color='grey', ls=':', alpha=0.7, label='2 components')
ax.set_xticks(range(1,10))
ax.set_xlabel('Number of components')
ax.set_ylabel('Residual (1 − explained)')
ax.set_title(f'{SYMBOL} OBI Manifold — ISOMAP vs PCA Scree')
ax.legend()
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4g_{SYMBOL}_scree.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved 4g_{SYMBOL}_scree.png')

# Figure 5: UMAP clusters + regime profiles
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for k in range(best_k):
    m = cluster_labels == k
    axes[0].scatter(Z_umap[m, 0], Z_umap[m, 1], c=[CLUSTER_COLORS[k]],
                    s=8, alpha=0.4, label=f'Regime {k+1}')
axes[0].set_title(f'UMAP — {best_k} Regimes (K-Means)', fontweight='bold')
axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')
axes[0].legend(fontsize=9, markerscale=2)

sc = axes[1].scatter(Z_umap[:, 0], Z_umap[:, 1], c=tod, cmap='plasma',
                     s=5, alpha=0.3, vmin=9.5, vmax=16)
plt.colorbar(sc, ax=axes[1], label='Hour ET')
for k in range(best_k):
    m = cluster_labels == k
    cx, cy = Z_umap[m, 0].mean(), Z_umap[m, 1].mean()
    axes[1].text(cx, cy, str(k+1), fontsize=13, fontweight='bold',
                 ha='center', va='center',
                 bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
axes[1].set_title('Same Embedding — Time of Day (regime numbers overlaid)')
axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4g_{SYMBOL}_umap_clusters.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved 4g_{SYMBOL}_umap_clusters.png')

# Figure 6: Regime OBI profiles
fig = plt.figure(figsize=(16, best_k * 3))
gs  = gridspec.GridSpec(best_k, 3, figure=fig, hspace=0.5, wspace=0.4)
for k in range(best_k):
    m     = cluster_labels == k
    X_k   = X_train[m]
    ret_k = y_train[m]
    tod_k = tod[m]
    ax_obi = fig.add_subplot(gs[k, 0])
    mean_obi = X_k.mean(axis=0)
    bar_cols = ['#E74C3C' if v < 0 else '#4472C4' for v in mean_obi]
    ax_obi.bar(range(10), mean_obi, yerr=X_k.std(axis=0),
               color=bar_cols, alpha=0.75, edgecolor='black', lw=0.5, capsize=3)
    ax_obi.axhline(0, color='black', lw=0.8)
    ax_obi.set_xticks(range(10))
    ax_obi.set_xticklabels([f'L{i}' for i in range(10)], fontsize=7)
    ax_obi.set_title(f'Regime {k+1} OBI Profile (n={m.sum():,})',
                     color=CLUSTER_COLORS[k], fontweight='bold')
    ax_obi.set_ylabel('Mean z-score OBI')

    ax_tod = fig.add_subplot(gs[k, 1])
    ax_tod.hist(tod_k, bins=np.arange(9.5, 16.1, 0.5),
                color=CLUSTER_COLORS[k], alpha=0.75, edgecolor='black', lw=0.4)
    ax_tod.set_xlabel('Hour ET'); ax_tod.set_ylabel('Count')
    ax_tod.set_title(f'Regime {k+1} — Time of Day')

    ax_ret = fig.add_subplot(gs[k, 2])
    vmax = np.percentile(np.abs(y_train), 97)
    ax_ret.hist(np.clip(ret_k, -vmax, vmax), bins=50,
                color=CLUSTER_COLORS[k], alpha=0.75, edgecolor='black', lw=0.4)
    ax_ret.axvline(ret_k.mean(), color='black', lw=1.5, ls='--',
                   label=f'Mean = {ret_k.mean()*1e4:.1f} bps')
    ax_ret.set_xlabel('Fwd return'); ax_ret.set_ylabel('Count')
    ax_ret.set_title(f'Regime {k+1} — Return Distribution')
    ax_ret.legend(fontsize=8)
plt.suptitle(f'{SYMBOL} Regime Profiling', fontsize=13, y=1.01)
plt.savefig(f'{FIG_DIR}/4g_{SYMBOL}_regime_profiles.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved 4g_{SYMBOL}_regime_profiles.png')

# Figure 7: OOS IC bar chart
n     = len(y_oos)
ci95  = 2 / np.sqrt(n - 3)
se    = 1 / np.sqrt(n - 3)
names_r, ics_r, ts_r, pv_r = zip(*ic_results)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(names_r, ics_r, color=['#4472C4','#9B59B6','#E74C3C','#2ECC71','#F39C12'],
       alpha=0.8, edgecolor='black', lw=0.8, yerr=[se]*len(ic_results),
       capsize=6, error_kw=dict(lw=1.5))
ax.axhspan(-ci95, ci95, color='grey', alpha=0.12, label='±2 SE noise band')
ax.axhline(0, color='black', lw=0.8)
for i, (ic, ts, pv) in enumerate(zip(ics_r, ts_r, pv_r)):
    sig = '**' if pv < 0.05 else ('*' if pv < 0.10 else '')
    ax.text(i, ic + se + 0.002, f'IC={ic:.4f}{sig}\nt={ts:.2f}',
            ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_ylabel('Spearman IC (OOS)')
ax.set_title(f'{SYMBOL} OOS IC — Raw OBI vs PCA vs ISOMAP vs UMAP vs UMAP+Regime\n'
             '** p<0.05  * p<0.10')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4g_{SYMBOL}_oos_ic.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'  Saved 4g_{SYMBOL}_oos_ic.png')

print('\n✓ All done. Figures saved to figures/')
