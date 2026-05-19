"""
Embedding comparison: ISOMAP vs PCA on LOB OBI features (NVDA, AAPL, MSFT).

Reuses the exact data pipeline from DS4FE_LOB_ISOMAP_v2.ipynb (Section 1)
so results are directly comparable.

Pipeline per symbol:
  1. Load mbp-10 parquet, filter to RTH (09:30-16:00 ET).
  2. Resample to 1-min mean OBI per level (10 features per bar).
  3. Train = Oct 2-19 2023 (matches v2 notebook split).
  4. Fit PCA(n_components=2) and Isomap(n_components=2, n_neighbors=15)
     on the training set.

Figures saved into ./embedding_comparison/:
  01_embeddings_per_symbol.png  - 3 rows (NVDA, AAPL, MSFT) x 2 cols (ISOMAP, PCA),
                                  each colored by OBI L0. Visual answer to:
                                  what does each embedding look like, and are the
                                  three symbols similar?
  02_cross_symbol_overlay.png   - all 3 symbols overlaid on one ISOMAP plot and
                                  one PCA plot (per-symbol z-score so axes match).
                                  Confirms whether the geometry is shared.
  03_new_information_test.png   - three quantitative tests for whether ISOMAP
                                  carries any information PCA does not:
                                    A. Cross-method linear predictability
                                    B. Per-OBI-level reconstruction R^2
                                    C. Forward-return Spearman IC

Run from repo root:
    python embedding_comparison/compare_isomap_pca.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

warnings.filterwarnings('ignore')


# ──────────────────────────────────────────────────────────────────────────────
# Configuration (mirrors DS4FE_LOB_ISOMAP_v2.ipynb)
# ──────────────────────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
LOB_DIR   = os.path.join(REPO_ROOT, 'data', 'lob')
OUT_DIR   = HERE

SYMBOLS      = ['NVDA', 'AAPL', 'MSFT']
OBI_COLS     = [f'obi_{k:02d}' for k in range(10)]
SYM_COLORS   = {'NVDA': '#E74C3C', 'AAPL': '#4472C4', 'MSFT': '#F39C12'}
DATA_PATHS   = {s: f'{LOB_DIR}/lob_mbp10_{s}_oct2023_full.parquet' for s in SYMBOLS}
TRAIN_CUTOFF = '2023-10-19'
N_NEIGHBORS  = 15

plt.rcParams.update({
    'figure.figsize'   : (12, 5),
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})


# ──────────────────────────────────────────────────────────────────────────────
# Data pipeline (identical to v2 notebook, cell `0f15179c`)
# ──────────────────────────────────────────────────────────────────────────────
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
    out['symbol']  = symbol
    return out.dropna()


def align_signs(Z, ref):
    """Flip column signs of Z so each column positively correlates with ref.
    ISOMAP coordinates are defined up to sign (and rotation); aligning with
    PCA makes visual comparison meaningful — the clouds appear in the same
    orientation."""
    Z = Z.copy()
    for k in range(Z.shape[1]):
        if np.corrcoef(Z[:, k], ref[:, k])[0, 1] < 0:
            Z[:, k] *= -1
    return Z


def standardize(arr):
    return (arr - arr.mean(axis=0)) / arr.std(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# Load + fit per symbol
# ──────────────────────────────────────────────────────────────────────────────
print('Loading data and fitting models ...')
results = {}
for sym in SYMBOLS:
    df = build_obi_bars(DATA_PATHS[sym], sym)
    is_train = df.index.date <= pd.Timestamp(TRAIN_CUTOFF).date()
    X   = df.loc[is_train, OBI_COLS].values
    y   = df.loc[is_train, 'ret_fwd'].values
    idx = df.loc[is_train].index
    tod = idx.hour + idx.minute / 60

    pca = PCA(n_components=2).fit(X)
    P   = pca.transform(X)

    iso = Isomap(n_components=2, n_neighbors=N_NEIGHBORS).fit(X)
    Z   = iso.transform(X)
    Z   = align_signs(Z, P)

    results[sym] = dict(
        X=X, y=y, tod=tod, P=P, Z=Z, iso=iso, pca=pca,
        pca_var=pca.explained_variance_ratio_.sum(),
        iso_err=iso.reconstruction_error(),
    )
    print(f'  {sym:5s}  bars={len(X):>6,}   '
          f'PCA-2 var={results[sym]["pca_var"]*100:5.1f}%   '
          f'ISOMAP-2 recon-err={results[sym]["iso_err"]:.4f}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — 3x2 embedding grid: each symbol on its own row, ISOMAP vs PCA
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(13, 14))
for r, sym in enumerate(SYMBOLS):
    P    = results[sym]['P']
    Z    = results[sym]['Z']
    obi0 = results[sym]['X'][:, 0]
    for c, (label, coords, x_lab, y_lab) in enumerate([
        ('ISOMAP', Z, 'Z₁  (consensus pressure)', 'Z₂  (book shape)'),
        ('PCA',    P, 'PC₁',                       'PC₂'),
    ]):
        ax = axes[r, c]
        sc = ax.scatter(coords[:, 0], coords[:, 1],
                        c=obi0, cmap='RdBu', s=5, alpha=0.5,
                        vmin=-0.5, vmax=0.5)
        ax.set_xlabel(x_lab, fontsize=10)
        ax.set_ylabel(y_lab, fontsize=10)
        ax.set_title(f'{sym} — {label}', fontsize=12,
                     color=SYM_COLORS[sym], fontweight='bold')
        plt.colorbar(sc, ax=ax, shrink=0.85, label='OBI L0 (top-of-book)')

plt.suptitle('ISOMAP vs PCA Embeddings — NVDA · AAPL · MSFT (Oct 2-19 2023, training)\n'
             'Each dot = 1 minute.   Color = top-of-book imbalance.   '
             'Signs aligned for visual comparison.',
             fontsize=12, y=1.0)
plt.tight_layout()
fig1_path = f'{OUT_DIR}/01_embeddings_per_symbol.png'
plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved {fig1_path}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Cross-symbol overlay: do the three books share one geometry?
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
for c, (label, key) in enumerate([('ISOMAP', 'Z'), ('PCA', 'P')]):
    ax = axes[c]
    for sym in SYMBOLS:
        coords = standardize(results[sym][key])
        ax.scatter(coords[:, 0], coords[:, 1],
                   s=4, alpha=0.30,
                   color=SYM_COLORS[sym], label=sym)
    ax.set_xlabel(f'{label} axis 1  (z-scored within symbol)')
    ax.set_ylabel(f'{label} axis 2  (z-scored within symbol)')
    ax.set_title(f'{label} — three symbols overlaid', fontsize=12, fontweight='bold')
    ax.legend(markerscale=3, loc='upper left')

plt.suptitle('Do NVDA, AAPL, MSFT Share the Same Manifold Geometry?\n'
             'Heavy overlap → yes; clean separation → no.',
             fontsize=12, y=1.02)
plt.tight_layout()
fig2_path = f'{OUT_DIR}/02_cross_symbol_overlay.png'
plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved {fig2_path}')


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 + tests — does ISOMAP carry information PCA does not?
# ══════════════════════════════════════════════════════════════════════════════
print('\n' + '─' * 72)
print('NEW-INFORMATION TESTS')
print('─' * 72)

# Test A — cross-method linear predictability.
# If ISOMAP coords are a linear function of PCA coords, R²(P→Z)→1.0.
# Lower R² = ISOMAP captures non-linear structure PCA misses.
print('\n[A] Linear cross-prediction R² (1.00 → no new info, <1 → ISOMAP captures something extra)')
test_A = {}
for sym in SYMBOLS:
    P, Z = results[sym]['P'], results[sym]['Z']
    r2_p2z = LinearRegression().fit(P, Z).score(P, Z)
    r2_z2p = LinearRegression().fit(Z, P).score(Z, P)
    test_A[sym] = (r2_p2z, r2_z2p)
    print(f'    {sym:5s}  R²(PCA → ISOMAP) = {r2_p2z:.4f}    R²(ISOMAP → PCA) = {r2_z2p:.4f}')

# Test B — per-OBI-level reconstruction.
# For each level k: how well does (P1, P2) vs (Z1, Z2) linearly predict OBI_k?
# If ISOMAP gives systematically higher R², it captures non-linear OBI structure.
print('\n[B] Per-level OBI reconstruction R² from 2 coords (linear regression)')
print('       L0     L1     L2     L3     L4     L5     L6     L7     L8     L9    mean')
test_B = {}
for sym in SYMBOLS:
    X, P, Z = results[sym]['X'], results[sym]['P'], results[sym]['Z']
    r2_pca, r2_iso = [], []
    for k in range(10):
        r2_pca.append(LinearRegression().fit(P, X[:, k]).score(P, X[:, k]))
        r2_iso.append(LinearRegression().fit(Z, X[:, k]).score(Z, X[:, k]))
    test_B[sym] = (np.array(r2_pca), np.array(r2_iso))
    pca_str = '  '.join(f'{v:.3f}' for v in r2_pca)
    iso_str = '  '.join(f'{v:.3f}' for v in r2_iso)
    print(f'    {sym:5s} PCA: {pca_str}   {np.mean(r2_pca):.3f}')
    print(f'    {sym:5s} ISO: {iso_str}   {np.mean(r2_iso):.3f}')

# Test C — forward-return Spearman IC of each axis.
# If ISOMAP coords correlate more strongly with future returns than PCA coords,
# the curved geometry is picking up tradable signal.
print('\n[C] |Spearman IC| of each axis with 1-min forward return  (training in-sample)')
test_C = {}
for sym in SYMBOLS:
    y, P, Z = results[sym]['y'], results[sym]['P'], results[sym]['Z']
    ic = (
        spearmanr(P[:, 0], y)[0],
        spearmanr(P[:, 1], y)[0],
        spearmanr(Z[:, 0], y)[0],
        spearmanr(Z[:, 1], y)[0],
    )
    test_C[sym] = ic
    print(f'    {sym:5s}  PC1={ic[0]:+.4f}  PC2={ic[1]:+.4f}  |  '
          f'Z1={ic[2]:+.4f}  Z2={ic[3]:+.4f}')


# ── Plot the three tests ─────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))

# Panel A: cross-method R²
ax = axes[0]
xs = np.arange(len(SYMBOLS)); w = 0.35
r2_p2z_vals = [test_A[s][0] for s in SYMBOLS]
r2_z2p_vals = [test_A[s][1] for s in SYMBOLS]
ax.bar(xs - w/2, r2_p2z_vals, w, label='R² (PCA → ISOMAP)',
       color='#4472C4', edgecolor='black', lw=0.6)
ax.bar(xs + w/2, r2_z2p_vals, w, label='R² (ISOMAP → PCA)',
       color='#E74C3C', edgecolor='black', lw=0.6)
ax.axhline(1.0, color='grey', ls='--', lw=0.8)
ax.set_xticks(xs); ax.set_xticklabels(SYMBOLS)
ax.set_ylabel('R² of linear cross-prediction')
ax.set_title('A. Linear equivalence of the two embeddings\n'
             'R² ≈ 1.0 ⇒ ISOMAP is just a rotated PCA (no new info)',
             fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.08)
for x, v in zip(xs - w/2, r2_p2z_vals):
    ax.text(x, v + 0.012, f'{v:.3f}', ha='center', fontsize=8.5)
for x, v in zip(xs + w/2, r2_z2p_vals):
    ax.text(x, v + 0.012, f'{v:.3f}', ha='center', fontsize=8.5)

# Panel B: per-level R² averaged across symbols
ax = axes[1]
xs = np.arange(10); w = 0.4
mean_pca = np.mean([test_B[s][0] for s in SYMBOLS], axis=0)
mean_iso = np.mean([test_B[s][1] for s in SYMBOLS], axis=0)
ax.bar(xs - w/2, mean_pca, w, label='PCA-2', color='#4472C4',
       edgecolor='black', lw=0.6)
ax.bar(xs + w/2, mean_iso, w, label='ISOMAP-2', color='#E74C3C',
       edgecolor='black', lw=0.6)
ax.set_xticks(xs); ax.set_xticklabels([f'L{k}' for k in range(10)])
ax.set_xlabel('OBI depth level')
ax.set_ylabel('R²  (linear reconstruction from 2D coords)')
ax.set_title('B. Per-level OBI reconstruction\n(averaged over NVDA, AAPL, MSFT)',
             fontsize=10)
ax.legend(fontsize=9)
ax.set_ylim(0, 1.05)

# Panel C: |IC| with forward returns
ax = axes[2]
xs  = np.arange(len(SYMBOLS)); w = 0.2
ax.bar(xs - 1.5*w, [abs(test_C[s][0]) for s in SYMBOLS], w,
       label='|IC| PC1', color='#A0BCE4', edgecolor='black', lw=0.4)
ax.bar(xs - 0.5*w, [abs(test_C[s][1]) for s in SYMBOLS], w,
       label='|IC| PC2', color='#4472C4', edgecolor='black', lw=0.4)
ax.bar(xs + 0.5*w, [abs(test_C[s][2]) for s in SYMBOLS], w,
       label='|IC| Z₁',  color='#F1A39E', edgecolor='black', lw=0.4)
ax.bar(xs + 1.5*w, [abs(test_C[s][3]) for s in SYMBOLS], w,
       label='|IC| Z₂',  color='#E74C3C', edgecolor='black', lw=0.4)
ax.set_xticks(xs); ax.set_xticklabels(SYMBOLS)
ax.set_ylabel('|Spearman IC| with 1-min forward return')
ax.set_title('C. Predictive content of each axis\n(in-sample, training set)',
             fontsize=10)
ax.legend(fontsize=8, ncol=2)

plt.suptitle('Does ISOMAP Carry Information PCA Misses?', fontsize=13, y=1.02)
plt.tight_layout()
fig3_path = f'{OUT_DIR}/03_new_information_test.png'
plt.savefig(fig3_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved {fig3_path}')


# ──────────────────────────────────────────────────────────────────────────────
# Final summary
# ──────────────────────────────────────────────────────────────────────────────
print('\n' + '═' * 72)
print('SUMMARY')
print('═' * 72)

mean_p2z = float(np.mean([test_A[s][0] for s in SYMBOLS]))
mean_z2p = float(np.mean([test_A[s][1] for s in SYMBOLS]))
mean_pca_recon = float(np.mean([test_B[s][0] for s in SYMBOLS]))
mean_iso_recon = float(np.mean([test_B[s][1] for s in SYMBOLS]))
delta_recon    = mean_iso_recon - mean_pca_recon

print(f'(A) Mean R²(PCA → ISOMAP) across symbols: {mean_p2z:.4f}')
print(f'    Mean R²(ISOMAP → PCA) across symbols: {mean_z2p:.4f}')
if mean_p2z > 0.95 and mean_z2p > 0.95:
    print('    → The two embeddings are essentially the same up to rotation.\n'
          '      ISOMAP carries NO new information at 2 components on this data.')
elif mean_p2z > 0.85:
    print('    → Most ISOMAP info is in the PCA span, but residual non-linear\n'
          '      structure is present.')
else:
    print('    → ISOMAP captures meaningful non-linear structure beyond PCA.')

print(f'\n(B) Per-level OBI reconstruction R² (mean across symbols & levels):')
print(f'    PCA-2:    {mean_pca_recon:.4f}')
print(f'    ISOMAP-2: {mean_iso_recon:.4f}   (Δ vs PCA = {delta_recon:+.4f})')

print(f'\n(C) Best |IC| achieved with returns:')
for sym in SYMBOLS:
    best_pca = max(abs(test_C[sym][0]), abs(test_C[sym][1]))
    best_iso = max(abs(test_C[sym][2]), abs(test_C[sym][3]))
    print(f'    {sym:5s}   PCA: {best_pca:.4f}    ISOMAP: {best_iso:.4f}    '
          f'Δ = {best_iso - best_pca:+.4f}')

print('\nFigures written to:', OUT_DIR)
