# DS4FE — Where (and why) nonlinear DR helps on financial data

**One-line answer:** Across LOB microstructure and SPX IV surfaces, **PCA is the right tool**. The one place a nonlinear method appears to beat it survives only as a *2-dimensional bottleneck artefact* — it disappears the moment PCA is given one extra dimension. This is now established by exclusion, with controls.

---

## 1. The honest "ISOMAP win" and why it does not survive

I deliberately gave ISOMAP its best shot: a **weak downstream model** (kNN, which rewards faithful neighbour geometry) on a **smooth target** (forward 60-second realized volatility), instead of a RandomForest that can carve up any layout. Under that setup, at a 2-D embedding, nonlinear methods do beat PCA:

| kNN R² (forward vol), 2-D | value |
|---|---|
| PCA-2 | 0.35 |
| Isomap-2 | 0.34 |
| UMAP-2 | 0.59 |
| Diffusion-2 | 0.61 |

But three controls dismantle the practical significance:

**(a) Dimension control — the decisive one.** Give PCA *one* more dimension and it already beats the best nonlinear 2-D embedding:

| kNN R² (forward vol) | dim 2 | dim 3 | dim 5 | dim 10 |
|---|---|---|---|---|
| **PCA** | 0.35 | **0.71** | 0.77 | 0.85 |
| Diffusion | 0.61 | 0.74 | 0.81 | — |
| Isomap | 0.34 | 0.38 | 0.76 | — |

→ The nonlinear advantage lives *only* at exactly 2-D. It is a packing efficiency, not new structure. (mean over 3 seeds; figure: `figures/lob_dimension_sweep.png`)

This generalises to the **other** forward targets (depth collapse, spread widening): nonlinear leads at 2-D, PCA closes the gap by dim 3–5 (depth: PCA overtakes at dim 5 = 0.79 AUC; spread: PCA ties Diffusion at dim 5, 0.735 vs 0.740), and raw 19-D features (~0.81 AUC) dominate every embedding. So the 2-D-artefact pattern is not vol-specific. `night_multitarget.csv`.

**(b) Persistence control.** Forward vol is dominated by trivial autocorrelation: kNN on **current vol alone (1 feature) = R² 0.88**, beating every 2-D embedding. The whole task adds little over "look at current volatility."

**(c) ISOMAP specifically never wins.** It is the weakest method at 2-D, is passed by PCA from dim 3, and wins on **0 of 5 symbols**. UMAP/Diffusion are the better nonlinear methods, not ISOMAP.

Robustness: the ranking is unchanged across kNN neighbour counts {10, 25, 50, 100} and across 3 resampling seeds (std ≤ 0.04).

---

## 2. Two datasets, one conclusion (opposite mechanisms)

| | Raw LOB (20-D sizes / OBI) | SPX IV surface (63-D grid) |
|---|---|---|
| Intrinsic dim (TwoNN) | ~9–10 | ~3.8 |
| 2-D geometry preserved (ρ_self) | PCA ties/leads | PCA 0.998 ≥ ISOMAP 0.989 |
| Forward-prediction edge for nonlinear? | only at 2-D, erased by PCA-3 | none (forward IV change ≈ unpredictable, both ≈ 0) |
| Verdict | PCA | PCA |

- **LOB:** no clean low-D manifold (too high-dimensional/noisy).
- **IV surface:** a clean low-D manifold *does* exist (~3.8 = level/skew/curvature), but it is **linear**, so PCA already captures ~99%.

**When low-dimensional structure exists in these datasets, it is linear → PCA is correct.**

---

## 3. The actual contribution (this is the result, not a failure)

A **four-control framework** for deciding whether nonlinear DR is worth it on any financial dataset, before trusting a pretty embedding:

1. **Cross-over test** — score both methods on both metrics (geodesic ρ *and* Euclidean R²).
2. **Dimension control** — does PCA at dim *k+1* match the nonlinear method at dim *k*? (Most projects skip this; it is what flips the answer here.)
3. **Persistence / trivial-baseline control** — does the embedding beat the obvious 1-feature baseline?
4. **Intrinsic-dimension estimate (TwoNN)** — is there even a low-D manifold to find?

Applied across **3 nonlinear methods (ISOMAP, UMAP, Diffusion), 2 asset classes, 2 time resolutions, 5 symbols, multiple seeds**, the conclusion is invariant.

---

## 4. Open / next (if we want to push further)

- **SVI / no-arbitrage IV fit** before DR — current surface uses linear interpolation in total variance, which could itself flatten mild curvature (the one genuine caveat).
- **Crisis-only regimes** — nonlinearity, if anywhere, is most likely in regime *transitions* rather than the full sample.

*Supporting code/data:* `experiments/lob_manifold/night_battery.py` (+ `night_dim_sweep.csv`, `night_probek.csv`, `night_persymbol.csv`), `experiments/iv_surface/iv_probe.py` (+ `iv_probe_metrics.csv`), `figures/lob_dimension_sweep.png`.
