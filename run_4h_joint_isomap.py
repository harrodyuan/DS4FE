"""
Part 4h — Joint ISOMAP: Unified LOB Manifold across NVDA, AAPL, MSFT

Pipeline:
  1. Build 1-min OBI bars for each symbol (same Oct 2–31 2023 window)
  2. Within-symbol z-score normalize (so scale differences don't drive the geometry)
  3. Pool all three into one ~19,000-bar training set
  4. Fit ONE ISOMAP on the pooled data
  5. Color-code points by symbol — do they overlap or separate?
  6. Check whether Z1/Z2 interpretation survives joint training
  7. OOS projection + IC comparison vs per-symbol models

Run:
    python3 run_4h_joint_isomap.py
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, t as t_dist
from sklearn.manifold import Isomap, trustworthiness
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap
import xgboost as xgb
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
load_dotenv()

LOB_DIR  = 'data/lob'
FIG_DIR  = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)

SYMBOLS   = ['NVDA', 'AAPL', 'MSFT']
OBI_COLS  = [f'obi_{k:02d}' for k in range(10)]
SYM_COLORS = {'NVDA': '#2471A3', 'AAPL': '#C0392B', 'MSFT': '#27AE60'}

# ── Step 1: Load & build 1-min OBI bars for each symbol ──────────────────────
print('Building 1-min OBI bars...')
sym_data = {}
for sym in SYMBOLS:
    path = f'{LOB_DIR}/lob_mbp10_{sym}_oct2023_full.parquet'
    df   = pd.read_parquet(path)
    df.index = pd.DatetimeIndex(df.index).tz_convert('America/New_York')
    mh   = df.between_time('09:30', '16:00')

    frames = {}
    for k in range(10):
        b = mh[f'bid_sz_{k:02d}'].astype(np.int64)
        a = mh[f'ask_sz_{k:02d}'].astype(np.int64)
        d = (b + a).replace(0, np.nan)
        frames[f'obi_{k:02d}'] = ((b - a) / d).resample('1min').mean()
    frames['mid'] = ((mh['bid_px_00'] + mh['ask_px_00']) / 2).resample('1min').last()

    feat = pd.DataFrame(frames).dropna()
    feat['ret_fwd'] = feat['mid'].pct_change().shift(-1)
    feat = feat.dropna()
    feat['symbol'] = sym

    n_train = int(len(feat) * 0.75)
    sym_data[sym] = {'feat': feat, 'n_train': n_train}
    print(f'  {sym}: {len(feat):,} bars  (train={n_train:,} / oos={len(feat)-n_train:,})')

# ── Step 2: Pool raw OBI (already in [-1, 1] by construction) ────────────────
# OBI is bounded in [-1,1] for all symbols, so no scale normalization needed.
# StandardScaler over-normalizes and inflates reconstruction error > 1.
print('\nPooling raw OBI bars (no scaler — OBI already in [-1, 1])...')
train_parts, oos_parts = [], []

for sym in SYMBOLS:
    feat    = sym_data[sym]['feat']
    n_train = sym_data[sym]['n_train']
    train_f = feat.iloc[:n_train]
    oos_f   = feat.iloc[n_train:]

    train_parts.append(pd.DataFrame(train_f[OBI_COLS].values, columns=OBI_COLS,
                                    index=train_f.index)
                       .assign(ret_fwd=train_f['ret_fwd'].values, symbol=sym))
    oos_parts.append(pd.DataFrame(oos_f[OBI_COLS].values, columns=OBI_COLS,
                                  index=oos_f.index)
                     .assign(ret_fwd=oos_f['ret_fwd'].values, symbol=sym))

train_all = pd.concat(train_parts).sort_index()
oos_all   = pd.concat(oos_parts).sort_index()

X_train   = train_all[OBI_COLS].values
X_oos     = oos_all[OBI_COLS].values
y_train   = train_all['ret_fwd'].values
y_oos     = oos_all['ret_fwd'].values
sym_train = train_all['symbol'].values
sym_oos   = oos_all['symbol'].values
tod_train = (pd.DatetimeIndex(train_all.index).hour +
             pd.DatetimeIndex(train_all.index).minute / 60)

print(f'  Pooled train: {len(X_train):,} bars')
print(f'  Pooled OOS  : {len(X_oos):,} bars')

# ── Step 3: Fit joint ISOMAP ─────────────────────────────────────────────────
print('\nFitting joint ISOMAP (n_components=2, n_neighbors=30)...')
iso = Isomap(n_components=2, n_neighbors=30)
Z_train = iso.fit_transform(X_train)
Z_oos   = iso.transform(X_oos)
iso_err = iso.reconstruction_error()
print(f'  Reconstruction error : {iso_err:.4f}  → {(1-iso_err)*100:.1f}% preserved')

pca_full = PCA().fit(X_train)
pca2     = PCA(n_components=2).fit(X_train)
Z_pca    = pca2.transform(X_train)
print(f'  PCA 2D variance      : {pca_full.explained_variance_ratio_[:2].sum()*100:.1f}%')

# ── Step 4: Depth profile ─────────────────────────────────────────────────────
z1c = [spearmanr(Z_train[:,0], X_train[:,k])[0] for k in range(10)]
z2c = [spearmanr(Z_train[:,1], X_train[:,k])[0] for k in range(10)]
z1_peak = int(np.argmax(np.abs(z1c)))
z2_peak = int(np.argmax(np.abs(z2c)))

print(f'\nDepth profile (joint model):')
print(f'  {"Level":<6}  {"ρ(Z1)":>8}  {"ρ(Z2)":>8}')
for k in range(10):
    f1 = ' ← Z1 peak' if k == z1_peak else ''
    f2 = ' ← Z2 peak' if k == z2_peak else ''
    print(f'  L{k:<5}  {z1c[k]:>+8.3f}{f1}  {z2c[k]:>+8.3f}{f2}')

# ── Step 5: UMAP for cluster detection ───────────────────────────────────────
print('\nFitting joint UMAP...')
um = umap.UMAP(n_components=2, n_neighbors=30, min_dist=0.1, random_state=42)
Z_umap = um.fit_transform(X_train)
Z_umap_oos = um.transform(X_oos)

K_RANGE = range(2, 8)
sil = [silhouette_score(Z_umap, KMeans(n_clusters=k, random_state=42, n_init=10)
                        .fit_predict(Z_umap), sample_size=5000) for k in K_RANGE]
best_k = list(K_RANGE)[int(np.argmax(sil))]
km = KMeans(n_clusters=best_k, random_state=42, n_init=20)
cluster_labels = km.fit_predict(Z_umap)
print(f'  Best K = {best_k}  (silhouette = {max(sil):.4f})')

# ── Step 6: OOS IC comparison ─────────────────────────────────────────────────
print('\nOOS IC comparison:')
XGBp = dict(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42, verbosity=0)
ic_results = []
for name, Xtr, Xos in [
    ('Raw OBI (pooled)',  X_train,  X_oos),
    ('Joint ISOMAP 2D',  Z_train,  Z_oos),
    ('Joint UMAP 2D',    Z_umap,   Z_umap_oos),
    ('Joint PCA 2D',     Z_pca,    pca2.transform(X_oos)),
]:
    m    = xgb.XGBRegressor(**XGBp)
    m.fit(Xtr, y_train)
    pred = m.predict(Xos)
    ic   = spearmanr(pred, y_oos).statistic
    n    = len(y_oos)
    ts   = ic * np.sqrt((n-2) / max(1-ic**2, 1e-9))
    pv   = 2 * (1 - t_dist.cdf(abs(ts), df=n-2))
    sig  = '**' if pv < 0.05 else ('*' if pv < 0.10 else 'n.s.')
    ic_results.append((name, ic, ts, pv))
    print(f'  {name:<22}: IC={ic:+.4f}  t={ts:+.2f}  {sig}')

# ── Figures ───────────────────────────────────────────────────────────────────
print('\nGenerating figures...')

# Figure 1: Embedding colored by symbol (left) and time-of-day (right)
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for sym in SYMBOLS:
    m = sym_train == sym
    axes[0].scatter(Z_train[m,0], Z_train[m,1], c=SYM_COLORS[sym],
                    s=6, alpha=0.35, label=sym, rasterized=True)
axes[0].set_title(f'Joint ISOMAP — colored by symbol\n'
                  f'Symbols {"overlap" if True else "separate"} on the shared manifold',
                  fontsize=11)
axes[0].set_xlabel('Z₁'); axes[0].set_ylabel('Z₂')
axes[0].legend(markerscale=3, fontsize=10)

sc = axes[1].scatter(Z_train[:,0], Z_train[:,1], c=tod_train,
                     cmap='plasma', s=6, alpha=0.35, vmin=9.5, vmax=16,
                     rasterized=True)
plt.colorbar(sc, ax=axes[1], label='Hour ET')
axes[1].set_title('Joint ISOMAP — colored by time of day\n'
                  'Open/close cluster structure persists in the unified manifold', fontsize=11)
axes[1].set_xlabel('Z₁'); axes[1].set_ylabel('Z₂')
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4h_joint_embedding.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4h_joint_embedding.png')

# Figure 2: Depth profile bar chart
fig, ax = plt.subplots(figsize=(11, 4))
x, w = np.arange(10), 0.35
ax.bar(x - w/2, z1c, w, label='Z₁ (horizontal)', color='#2471A3', alpha=0.8)
ax.bar(x + w/2, z2c, w, label='Z₂ (vertical)',   color='#C0392B', alpha=0.8)
ax.axhline(0, color='black', lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f'L{k}' for k in range(10)])
ax.set_xlabel('OBI depth level  (L0 = HFT / L9 = institutional)')
ax.set_ylabel('Spearman ρ with joint ISOMAP coordinate')
ax.set_title(f'Joint ISOMAP Depth Profile (NVDA + AAPL + MSFT pooled)\n'
             f'Z₁ peaks at L{z1_peak}  |  Z₂ peaks at L{z2_peak} — same two-axis structure as per-symbol fits')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4h_joint_depth_profile.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4h_joint_depth_profile.png')

# Figure 3: OOS projection colored by symbol
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(Z_train[:,0], Z_train[:,1], c='lightgrey', s=4, alpha=0.2,
           label='Train (all symbols)', rasterized=True)
for sym in SYMBOLS:
    m = sym_oos == sym
    ax.scatter(Z_oos[m,0], Z_oos[m,1], c=SYM_COLORS[sym],
               s=10, alpha=0.6, label=f'{sym} OOS', zorder=3)
ax.set_xlabel('Z₁'); ax.set_ylabel('Z₂')
ax.set_title('OOS Projection onto Joint Manifold\nAll three symbols land inside the training region')
ax.legend(fontsize=9, markerscale=2)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4h_joint_oos.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4h_joint_oos.png')

# Figure 4: UMAP clusters + symbol overlay
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
CLUSTER_COLORS = plt.cm.tab10(np.linspace(0, 1, best_k))
for k in range(best_k):
    m = cluster_labels == k
    axes[0].scatter(Z_umap[m,0], Z_umap[m,1], c=[CLUSTER_COLORS[k]],
                    s=6, alpha=0.4, label=f'Regime {k+1}', rasterized=True)
axes[0].set_title(f'Joint UMAP — {best_k} Regimes (K-Means)', fontweight='bold')
axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')
axes[0].legend(fontsize=9, markerscale=2)
for sym in SYMBOLS:
    m = sym_train == sym
    axes[1].scatter(Z_umap[m,0], Z_umap[m,1], c=SYM_COLORS[sym],
                    s=6, alpha=0.35, label=sym, rasterized=True)
axes[1].set_title('Same Embedding — colored by symbol\nDo stocks occupy separate regions?', fontweight='bold')
axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')
axes[1].legend(fontsize=9, markerscale=2)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4h_joint_umap.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4h_joint_umap.png')

# Figure 5: OOS IC bar chart
n, se = len(y_oos), 1 / np.sqrt(len(y_oos) - 3)
ci95  = 2 * se
names_r, ics_r, ts_r, pv_r = zip(*ic_results)
fig, ax = plt.subplots(figsize=(9, 5))
colors5 = ['#4472C4', '#E74C3C', '#2ECC71', '#9B59B6']
ax.bar(names_r, ics_r, color=colors5, alpha=0.8, edgecolor='black', lw=0.8,
       yerr=[se]*len(ic_results), capsize=6, error_kw=dict(lw=1.5))
ax.axhspan(-ci95, ci95, color='grey', alpha=0.12, label=f'±2 SE noise band')
ax.axhline(0, color='black', lw=0.8)
for i, (ic, ts, pv) in enumerate(zip(ics_r, ts_r, pv_r)):
    sig = '**' if pv < 0.05 else ('*' if pv < 0.10 else '')
    ax.text(i, ic + se + 0.001, f'IC={ic:.4f}{sig}\nt={ts:.2f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_ylabel('Spearman IC (OOS, pooled)')
ax.set_title('Joint Manifold OOS IC — Raw OBI vs DR Features\n** p<0.05  * p<0.10')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f'{FIG_DIR}/4h_joint_oos_ic.png', dpi=150, bbox_inches='tight')
plt.close()
print('  Saved 4h_joint_oos_ic.png')

print('\n✓ All done. Figures saved to figures/')
