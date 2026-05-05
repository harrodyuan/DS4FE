"""
Rebuild DS4FE_ISOMAP_Demo.ipynb with clean 8-section structure.
Run:  python build_demo_notebook.py
"""
import json, sys

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.lstrip('\n'), "outputs": []}

def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.lstrip('\n'),
    }

cells = []

# ── Cell 0: Title ─────────────────────────────────────────────────────────────
cells.append(md("""
# ISOMAP for Limit Order Books

**Goal.** The 10-level order book produces a 10-dimensional feature vector every minute. This notebook asks whether all 10 dimensions are necessary, or whether two coordinates capture the essential structure — and what those coordinates mean economically.

**Data.** Databento mbp-10 snapshots for NVDA, AAPL, MSFT — full October 2023 (22 trading days, ~50 M ticks per symbol).

**Feature.** 1-minute Order Book Imbalance at each depth level $k$:

$$\\text{OBI}_k = \\frac{B_k - A_k}{B_k + A_k} \\in [-1,\\,1]$$

$k = 0$ is the top of book; $k = 9$ is the deepest level available in the feed.
"""))

# ── Cell 1: Imports ────────────────────────────────────────────────────────────
cells.append(code("from IPython.display import Image, display"))

# ── Cell 2: Section 1 ─────────────────────────────────────────────────────────
cells.append(md("""
---
## 1. The Raw Data

Each row in the dataset is a full 10-level order book snapshot — one snapshot is generated every time any order is placed, modified, or cancelled. For NVDA in October 2023 that is roughly 50 million ticks per day.

The snapshot below is taken just after the open (09:30 ET). Bid sizes are positive, ask sizes are negative.
"""))

cells.append(code("display(Image('figures/4f_ob_snapshot.png'))"))

cells.append(md("""
The spread is tight and the book is relatively balanced at this moment. Near the open and close the profile is more extreme — wider spread, thinner inside levels, larger imbalances.

### From tick-level snapshots to 1-minute bars

Taking the mean OBI within each minute gives a stable summary of the book state for that minute. The result is a matrix of shape (minutes × 10). For October 2023 that is roughly 8,580 bars per symbol.
"""))

cells.append(code("display(Image('figures/4f_obi_structure.png'))"))

# ── Cell 6: Section 2 ─────────────────────────────────────────────────────────
cells.append(md("""
---
## 2. Why Should the Book State Be Low-Dimensional?

The 10 OBI levels are not independent. Adjacent levels tend to move together — if the bid is heavy at L2, it is usually heavy at L3 too. The correlation matrix makes this concrete:
"""))

cells.append(code("display(Image('figures/4f_obi_heatmap.png'))"))

cells.append(md("""
Nearby levels are correlated — L0 and L1 share a Spearman ρ of about 0.61, rising to 0.86 for L7–L8. The far ends of the book (L0 vs L9) are nearly uncorrelated at 0.11. The book does not fill all 10 dimensions — it lives near a lower-dimensional surface inside that space.
"""))

# ── Cell 9: Section 3 ─────────────────────────────────────────────────────────
cells.append(md("""
---
## 3. ISOMAP — A Three-Step Pipeline

ISOMAP recovers the low-dimensional manifold that the data lies near. It has three steps.

**Step 1 — k-NN graph.** Connect each 1-minute bar to its $k = 30$ nearest neighbors in the 10D OBI space. Two bars are neighbors if their full book profiles are similar at all levels.

**Step 2 — Geodesic distances.** For bars that are not direct neighbors, compute the shortest path through the graph (Dijkstra's algorithm). This is the *geodesic* distance — measured along the manifold surface rather than through empty space. If the manifold curves, the straight-line Euclidean distance cuts through regions no real bar occupies; the geodesic respects the shape.

**Step 3 — MDS embedding.** Find a 2D layout $(Z_1, Z_2)$ whose Euclidean distances match the geodesic distances as closely as possible. This is multidimensional scaling applied to the geodesic distance matrix.

In code, the full pipeline is five lines:
"""))

cells.append(md("""\
```python
from sklearn.manifold import Isomap

isomap = Isomap(n_neighbors=30, n_components=2)
Z_train = isomap.fit_transform(X_train)   # X_train: (n_bars, 10) OBI matrix
Z_oos   = isomap.transform(X_oos)         # Nyström extension for held-out bars
print(f"Reconstruction error: {isomap.reconstruction_error():.4f}")
# → 0.028  (2.8% — 97.2% of geodesic variance preserved)
```
"""))

# ── Cell 11: Section 4 ────────────────────────────────────────────────────────
cells.append(md("""
---
## 4. Does Two Dimensions Capture Enough?

The reconstruction error measures how much of the manifold's geodesic structure is lost when projecting to 2D:

$$\\text{Reconstruction error} = \\frac{\\operatorname{Var}(d_{\\text{geo}} - d_{\\text{embed}})}{\\operatorname{Var}(d_{\\text{geo}})}$$

Zero is a perfect embedding; one means the embedding is no better than predicting the mean distance. PCA minimises a similar criterion but over Euclidean distances, which assumes the manifold is flat.
"""))

cells.append(code("display(Image('figures/4g_NVDA_scree.png'))"))

cells.append(md("""
The scree plot shows ISOMAP and PCA residuals as a function of number of components. ISOMAP reaches 97.2% fidelity with 2 components; PCA needs more components to reach the same fidelity because it cannot follow the curvature of the manifold.

| | NVDA | AAPL | MSFT | Joint (pooled) |
|---|---|---|---|---|
| **ISOMAP 2D preserved** | **97.2%** | **96.7%** | **96.7%** | **97.3%** |
| PCA 2D variance | 78.6% | 82.2% | 87.0% | 81.9% |
| Gap | +18.6 pp | +14.5 pp | +9.7 pp | +15.4 pp |

The gap is not a tuning choice — it comes from curvature in the book state manifold. The ISOMAP advantage is largest for NVDA, which has the most liquid book and the most active order flow.
"""))

cells.append(code("display(Image('figures/4g_NVDA_dr_quality.png'))"))

cells.append(md("""
Trustworthiness and Continuity measure whether the embedding preserves the *local* neighborhood structure. Both exceed 0.97 for ISOMAP. The gap over PCA is consistent across all three stocks, confirming that the book state manifold has genuine curvature that a flat projection cannot follow.
"""))

# ── Cell 16: Section 5 ────────────────────────────────────────────────────────
cells.append(md("""
---
## 5. What Do Z₁ and Z₂ Capture?

The 2D ISOMAP output assigns each 1-minute bar a pair of coordinates $(Z_1, Z_2)$. ISOMAP has no knowledge of time, returns, or which stock the bar came from. The coloring below is added after the fact.
"""))

cells.append(code("display(Image('figures/4g_NVDA_dr_tod.png'))"))

cells.append(md("""
The ISOMAP panel (top-left) shows a tendency for bars from the open and close to cluster away from mid-day bars, though the separation is partial — the book visits similar states at different times of day. This structure was not given to the model; it comes from book geometry.

To label the axes economically, compute the Spearman correlation between each ISOMAP coordinate and the raw OBI at each of the 10 depth levels:
"""))

cells.append(code("display(Image('figures/4f_isomap_depth_profile.png'))"))

cells.append(md("""
**Z₁** has the same sign across all 10 levels and peaks at L5–L6 (mid-book). It measures how bullish or bearish the entire stack is — a book-wide consensus signal.

**Z₂** is positive at the shallow levels (L0–L4) and crosses zero around L4–L5, turning negative at the deep levels. It measures the contrast between the near-book and the deep book — two parts of the order book that can lean in opposite directions. The sign of this axis is arbitrary (ISOMAP sign-flips are common across fits); what matters is the crossing pattern.

The scatter grids below show Z₁ vs Z₂ for each OBI level individually. The dominant color gradient rotates as you move from shallow to deep levels, directly visualising the crossing point.
"""))

cells.append(code("display(Image('figures/4f_isomap_all_levels.png'))"))

# ── Cell 23: Section 6 ────────────────────────────────────────────────────────
cells.append(md("""
---
## 6. Out-of-Sample Generalization

The ISOMAP is trained on the first 75% of October. The remaining 25% is held out and projected using the Nyström extension (`isomap.transform()`). A well-learned manifold should accommodate OOS bars inside the training region.
"""))

cells.append(code("display(Image('figures/4f_oos_projection.png'))"))

cells.append(md("""
OOS bars land inside the region covered by the training set. The learned manifold structure is stable over the hold-out period — it is not overfitting the specific dates used for training.

Does the 2D representation help with short-horizon return prediction?
"""))

cells.append(code("display(Image('figures/4g_NVDA_oos_ic.png'))"))

cells.append(md("""
The ISOMAP 2D coordinates produce a marginally positive out-of-sample information coefficient — larger than raw OBI (which cancels across levels) and consistent in direction. The signal is small; treat it as evidence that some return-relevant structure is captured, not a trading signal on its own.
"""))

# ── Cell 28: Section 7 ────────────────────────────────────────────────────────
cells.append(md("""
---
## 7. Does the Same Structure Appear in Other Stocks?

The analysis above used NVDA. Fitting the same pipeline independently on AAPL and MSFT, and then fitting one joint model on all three pooled together, gives the following picture.

| | NVDA | AAPL | MSFT | Joint (pooled) |
|---|---|---|---|---|
| ISOMAP 2D preserved | 97.2% | 96.7% | 96.7% | 97.3% |
| Z₁ peak level | L6 | L7 | L4 | L5 |
| Z₂ sign-flip around | L4–L5 | L5–L6 | L4 | L5 |
| Best K (UMAP) | 2 | 2 | 2 | 2 |

The exact depth level where Z₁ peaks and Z₂ crosses zero shifts slightly — MSFT's book is more concentrated near the top. But the qualitative pattern (Z₁ = consensus, Z₂ = near-vs-deep contrast, K=2 regimes) replicates across all three stocks independently.

The joint embedding below fits one ISOMAP on the ~26,000-bar pooled matrix:
"""))

cells.append(code("display(Image('figures/4h_joint_embedding.png'))"))

cells.append(md("""
**Left — by symbol:** NVDA, AAPL, and MSFT points substantially overlap on the shared manifold. There is no clean region that belongs to one stock only. The book geometry is driven by *market-wide microstructure regimes* rather than stock-specific signals.

**Right — by time of day:** The familiar open/close vs. mid-day structure reappears in the pooled embedding — now spanning all three symbols. Time of day is a stronger organising dimension than stock identity.
"""))

cells.append(code("display(Image('figures/4h_joint_depth_profile.png'))"))

cells.append(md("""
The joint depth profile replicates the per-symbol pattern: Z₁ peaks at L5, Z₂ crosses zero at L4–L5. The two-axis interpretation is not a per-symbol artifact — it is a structural property of LOB geometry that survives pooling across stocks.
"""))

# ── Cell 34: Section 8 ────────────────────────────────────────────────────────
cells.append(md("""
---
## 8. Stress Period Projection: The BOJ Shock Week

On August 5 2024, the Bank of Japan raised rates unexpectedly. The yen strengthened sharply, unwinding carry trades. NVDA fell from ~$110 to ~$92 intraday (−16%). The VIX opened near 65 — highest since March 2020.

The ISOMAP trained on calm October 2023 provides a coordinate system. Projecting the August shock week onto it asks: *do stress-period book states look geometrically unusual?*

Two representations are compared:
- **Model A (OBI means only, 10 features):** the same feature set used throughout this notebook
- **Model B (OBI means + within-minute OBI std, 20 features):** adds a second moment capturing how volatile the book was within each 1-minute bar
"""))

cells.append(code("display(Image('figures/4i_model_comparison.png'))"))

cells.append(md("""
**Model A (left):** Stress bars land almost entirely inside the calm cloud. Only 3.6% fall outside the calm 95th percentile in manifold distance. The 1-minute mean OBI washes out the directional signal: during a directional crash, every update within a minute points the same way, so the mean is moderate and unremarkable.

**Model B (right):** Stress bars clearly shift away from the calm region. 16.8% are outside the calm 95th percentile, and the stress mean distance is **2× the calm mean**. The within-minute OBI standard deviation is lower during the crash (every update aligned → less within-minute variance), and this second-moment difference is what separates stress from calm on the manifold.

| | Model A (means only) | Model B (means + std) |
|---|---|---|
| Features | 10 | 20 |
| Stress bars outside calm 95th | 3.6% | **16.8%** |
| Stress mean distance / Calm mean | 0.91× | **2.04×** |

The stress separation increases through the week — 8.2% on Aug 5 (crash day, most directional) to 25.2% on Aug 9 (recovery, buyers and sellers disagreeing). The crash itself looks calmer in manifold distance than the days that follow, because price discovery during a one-way move outpaces the book's ability to explore different states.
"""))

cells.append(code("display(Image('figures/4i_aug5_path.png'))"))

cells.append(md("""
The minute-by-minute trajectory on Aug 5 starts in the upper-left (crash open, gold star) and moves through a sparse region of the manifold. The path is directed and compact — the book does not backtrack much, consistent with sustained one-way selling pressure. By the close (black diamond) the book is in a region that corresponds to calm late-afternoon configurations.
"""))

cells.append(code("display(Image('figures/4i_manifold_distance.png'))"))

cells.append(md("""
Manifold distance is persistently above the calm 95th percentile throughout the week (top panel), with the largest spikes at market open each day — when the book has the most unusual configuration relative to its calm-period counterpart.

The return volatility panel (bottom) tells the complementary story: Aug 5 has the highest daily vol but is not the most anomalous day in manifold distance. The structural unusualness of the book peaks during the recovery, when market participants are actively reassessing direction — a different kind of stress than a one-way crash.
"""))

# ── Cell 42: Conclusion ───────────────────────────────────────────────────────
cells.append(md("""
---
## Conclusion

Across NVDA, AAPL, and MSFT, the 10-level OBI matrix consistently has an intrinsic dimensionality of 2. ISOMAP recovers this structure with 97%+ fidelity versus 79–87% for PCA — the gap reflects curvature in the book state manifold that a flat projection cannot follow.

The two coordinates have a consistent interpretation across all fits, per-symbol and joint: Z₁ measures book-wide consensus (how far the entire stack tilts in one direction), and Z₂ measures the contrast between the near-book and the deep book. This pattern is not a modelling choice; it emerged from the geometry of the data in every case tested.

The stress projection result makes the distinction practical. Calm-period OBI means alone do not flag the August 2024 crash: the 1-minute mean is moderate even when every update within that minute points in the same direction. Adding the within-minute OBI standard deviation reveals the structural change: a more directional, less flickering book during the crash, detectable without labels, price data, or model retraining. 16.8% of stress bars fall outside the calm 95th percentile in manifold distance — a four-fold increase over the mean-only representation.
"""))

# ── Assemble notebook ─────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3 (ipykernel)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = "DS4FE_ISOMAP_Demo.ipynb"
with open(out, "w") as f:
    json.dump(nb, f, indent=1)

print(f"Written {len(cells)} cells → {out}")
