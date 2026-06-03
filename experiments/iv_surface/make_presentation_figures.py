"""
Presentation-grade IV-surface visuals.

Produces:
  1. iv_3d_calm_vs_stress.png  -- classic 3D vol surfaces, calm day vs 2022-selloff day
  2. iv_pca_factor_surfaces.png-- the 3 PCA factors rendered as surface deformations
                                  (level / skew / curvature) in IV units
  3. iv_factor_timeseries.png  -- PC1/PC2 scores through 2021-2023 with selloff shaded
  4. iv_surface_evolution.png  -- small-multiples of the surface at 6 dates

Run:  python experiments/iv_surface/make_presentation_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from sklearn.decomposition import PCA

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IV_DIR = os.path.join(ROOT, "data", "iv")
FIG = os.path.join(ROOT, "figures", "iv")
os.makedirs(FIG, exist_ok=True)

grid = json.load(open(os.path.join(IV_DIR, "surface_grid.json")))
MATS = np.array(grid["mats_days"])          # 7
MONEY = np.array(grid["moneyness"])         # 9
nM, nK = len(MATS), len(MONEY)

surf = pd.read_parquet(os.path.join(IV_DIR, "surfaces.parquet")).sort_index()
S = surf.values
dates = surf.index
atm30 = surf["T30_m1000"].values

MX, MY = np.meshgrid(MONEY, MATS)           # (7,9)


def reshape(row):
    return row.reshape(nM, nK) * 100.0       # to IV %


# --------------------------------------------------------------------------- #
# 1. 3D calm vs stress surfaces
# --------------------------------------------------------------------------- #
def fig_calm_vs_stress():
    i_calm = int(np.nanargmin(atm30))
    i_stress = int(np.nanargmax(atm30))
    fig = plt.figure(figsize=(15, 6.5))
    for j, (idx, tag) in enumerate([(i_calm, "Calmest day"), (i_stress, "Most stressed day")]):
        ax = fig.add_subplot(1, 2, j + 1, projection="3d")
        Z = reshape(S[idx])
        ax.plot_surface(MX, MY, Z, cmap=cm.viridis, edgecolor="k", lw=0.2,
                        antialiased=True, alpha=0.95)
        ax.set_xlabel("moneyness (K/S)")
        ax.set_ylabel("maturity (days)")
        ax.set_zlabel("implied vol (%)")
        d = pd.Timestamp(dates[idx]).date()
        ax.set_title(f"{tag}: {d}\nATM-30d IV = {atm30[idx]*100:.1f}%",
                     fontsize=12, fontweight="bold")
        ax.view_init(elev=22, azim=-60)
    fig.suptitle("SPX implied-volatility surface: calm vs stressed regime",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(FIG, "iv_3d_calm_vs_stress.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)


# --------------------------------------------------------------------------- #
# 2. PCA factors as surface deformations
# --------------------------------------------------------------------------- #
def fig_factor_surfaces():
    pca = PCA(n_components=3, svd_solver="full").fit(S)   # raw IV units
    mean_surf = pca.mean_
    evr = pca.explained_variance_ratio_
    scores = pca.transform(S)
    names = ["PC1: LEVEL", "PC2: SKEW", "PC3: CURVATURE"]

    fig = plt.figure(figsize=(16, 5.5))
    for k in range(3):
        ax = fig.add_subplot(1, 3, k + 1, projection="3d")
        amp = 2.0 * np.std(scores[:, k])       # +/- 2 sd deformation
        comp = pca.components_[k]
        Zhi = reshape(mean_surf + amp * comp)
        Zlo = reshape(mean_surf - amp * comp)
        Zm = reshape(mean_surf)
        ax.plot_surface(MX, MY, Zhi, color="#C0392B", alpha=0.55, edgecolor="none")
        ax.plot_surface(MX, MY, Zlo, color="#2980B9", alpha=0.55, edgecolor="none")
        ax.plot_wireframe(MX, MY, Zm, color="k", lw=0.4, alpha=0.5)
        ax.set_xlabel("moneyness"); ax.set_ylabel("maturity (d)"); ax.set_zlabel("IV (%)")
        ax.set_title(f"{names[k]}  ({evr[k]*100:.1f}% var)", fontsize=12, fontweight="bold")
        ax.view_init(elev=20, azim=-60)
    fig.suptitle("The 3 PCA factors as surface deformations  (red = +2sd, blue = -2sd, "
                 "black = mean)", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(FIG, "iv_pca_factor_surfaces.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)
    return scores, evr


# --------------------------------------------------------------------------- #
# 3. Factor score time series with selloff shaded
# --------------------------------------------------------------------------- #
def fig_factor_timeseries(scores):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    sel0, sel1 = pd.Timestamp("2022-01-01"), pd.Timestamp("2022-10-31")
    for ax, k, lab, col in [(axes[0], 0, "PC1 (level)", "#C0392B"),
                            (axes[1], 1, "PC2 (skew)", "#2980B9")]:
        ax.plot(dates, scores[:, k], color=col, lw=1.3)
        ax.axvspan(sel0, sel1, color="grey", alpha=0.18, label="2022 selloff")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_ylabel(lab); ax.legend(loc="upper left", fontsize=9)
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].set_title("PCA factor scores through 2021-2023 (PC1 level spikes in the 2022 selloff)",
                      fontsize=12, fontweight="bold")
    fig.tight_layout()
    p = os.path.join(FIG, "iv_factor_timeseries.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)


# --------------------------------------------------------------------------- #
# 4. Surface evolution small-multiples
# --------------------------------------------------------------------------- #
def fig_evolution():
    idxs = np.linspace(0, len(S) - 1, 6).astype(int)
    zmin, zmax = np.nanmin(S) * 100, np.nanmax(S) * 100
    fig = plt.figure(figsize=(16, 9))
    for j, idx in enumerate(idxs):
        ax = fig.add_subplot(2, 3, j + 1, projection="3d")
        ax.plot_surface(MX, MY, reshape(S[idx]), cmap=cm.plasma,
                        edgecolor="none", vmin=zmin, vmax=zmax, alpha=0.95)
        ax.set_zlim(zmin, zmax)
        ax.set_title(f"{pd.Timestamp(dates[idx]).date()}  (ATM30={atm30[idx]*100:.0f}%)",
                     fontsize=10)
        ax.set_xlabel("K/S", fontsize=8); ax.set_ylabel("mat(d)", fontsize=8)
        ax.view_init(elev=22, azim=-60)
    fig.suptitle("SPX IV surface evolution, 2021-06 to 2023-06", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = os.path.join(FIG, "iv_surface_evolution.png")
    fig.savefig(p, dpi=140); plt.close(fig); print("saved", p)


if __name__ == "__main__":
    fig_calm_vs_stress()
    scores, evr = fig_factor_surfaces()
    fig_factor_timeseries(scores)
    fig_evolution()
    print("\nPCA variance explained (PC1-3):", np.round(evr * 100, 1), "%")
    print("done -> figures/iv/")
