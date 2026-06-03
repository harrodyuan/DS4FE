"""Plot the decisive dimension-sweep result: the nonlinear kNN-vol advantage is a
2D-bottleneck artefact that vanishes once PCA gets one more dimension."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "..", "..", "figures")
A = pd.read_csv(os.path.join(OUT, "night_dim_sweep.csv"))

cur = A[A.method == "cur_vol_only"]["knn_r2_fwdvol"].mean()
raw = A[A.method == "raw_features"]["knn_r2_fwdvol"].mean()
dr = A[A.method.isin(["PCA", "Isomap", "UMAP", "Diffusion"])]
g = dr.groupby(["method", "dim"])["knn_r2_fwdvol"].agg(["mean", "std"]).reset_index()

colors = {"PCA": "#C0392B", "Isomap": "#7F8C8D", "UMAP": "#2980B9", "Diffusion": "#27AE60"}
fig, ax = plt.subplots(figsize=(9, 6))
for m in ["PCA", "Isomap", "UMAP", "Diffusion"]:
    s = g[g.method == m].sort_values("dim")
    ax.errorbar(s["dim"], s["mean"], yerr=s["std"], marker="o", capsize=3,
                lw=2.2, color=colors[m], label=m)
ax.axhline(cur, ls="--", color="black", lw=1.6, label=f"current-vol only (1 feat) = {cur:.2f}")
ax.axhline(raw, ls=":", color="#8E44AD", lw=1.6, label=f"raw 19D features = {raw:.2f}")

ax.annotate("PCA+1 dim already beats\nbest nonlinear-2D",
            xy=(3, 0.71), xytext=(3.4, 0.50), fontsize=10,
            arrowprops=dict(arrowstyle="->", color="#C0392B"))
ax.set_xticks([2, 3, 5, 10])
ax.set_xlabel("embedding dimension")
ax.set_ylabel("kNN R²  (forward 60s realized vol)")
ax.set_title("The nonlinear kNN-vol 'win' is a 2D-bottleneck artefact\n"
             "(mean ± std over 3 seeds; SPX/LOB 5-symbol calm+stress)",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
out = os.path.join(FIG, "lob_dimension_sweep.png")
fig.savefig(out, dpi=140)
print("saved", out)
