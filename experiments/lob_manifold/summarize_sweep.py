"""
Mine sweep_results.csv to answer: WHERE (if anywhere) does a nonlinear DR method
beat PCA on a financial task -- and specifically, where does Isomap beat PCA?

Prints:
  1. coverage (configs done, by bar/feature_set/scope)
  2. for every (bar, feature_set, n_components, fit_scope, task, probe) cell, the
     best method and its margin over PCA
  3. a focused 'Isomap beats PCA' table (margin >= 0.02), and an 'Isomap is best
     of all methods' table
  4. a global ranking: how often each method is the best, and mean margin vs PCA

Run:  python experiments/lob_manifold/summarize_sweep.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(OUT_DIR, "sweep_results.csv")
MARGIN = 0.02

# methods are stored with param suffixes; map to a base family for PCA comparison
BASES = {"PCA": "PCA", "KernelPCA_rbf": "KernelPCA", "Isomap_nn10": "Isomap",
         "Isomap_nn30": "Isomap", "LLE": "LLE", "UMAP_nn15": "UMAP",
         "UMAP_nn50": "UMAP", "Diffusion_a1.0": "Diffusion",
         "Diffusion_a0.5": "Diffusion"}
CELL = ["bar", "feature_set", "n_components", "fit_scope", "task", "probe", "metric"]


def load():
    if not os.path.exists(CSV):
        sys.exit("no sweep_results.csv yet")
    d = pd.read_csv(CSV)
    d = d.dropna(subset=["value"])
    d["family"] = d["method"].map(BASES).fillna(d["method"])
    return d


def main():
    d = load()
    pd.set_option("display.width", 240, "display.max_columns", 40,
                  "display.max_rows", 200)

    print(f"=== COVERAGE ===  rows={len(d):,}  configs={d.config_id.nunique():,}")
    print(d.groupby(["fit_scope", "bar", "feature_set"])["config_id"]
          .nunique().rename("n_configs").reset_index().to_string(index=False))

    # collapse method params: keep best variant per family within each cell
    fam = (d.groupby(CELL + ["family"])["value"].max().reset_index())

    # PCA value per cell
    pca = (fam[fam.family == "PCA"].set_index(CELL)["value"].rename("pca_value"))
    fam = fam.join(pca, on=CELL)
    fam["margin_vs_pca"] = fam["value"] - fam["pca_value"]

    # best family per cell
    idx = fam.groupby(CELL)["value"].idxmax()
    best = fam.loc[idx].copy()

    print("\n=== HOW OFTEN IS EACH METHOD THE BEST (per cell) ===")
    print(best["family"].value_counts().to_string())

    print("\n=== MEAN MARGIN vs PCA, BY FAMILY x TASK (positive => beats PCA) ===")
    piv = (fam[fam.family != "PCA"]
           .pivot_table(index="family", columns="task",
                        values="margin_vs_pca", aggfunc="mean").round(3))
    print(piv.to_string())

    # ----- ISOMAP focus ----- #
    iso = fam[fam.family == "Isomap"].copy()
    iso_win = iso[iso["margin_vs_pca"] >= MARGIN].sort_values("margin_vs_pca",
                                                              ascending=False)
    print(f"\n=== ISOMAP BEATS PCA by >= {MARGIN} (cells) : {len(iso_win)} ===")
    if len(iso_win):
        print(iso_win[CELL + ["value", "pca_value", "margin_vs_pca"]]
              .head(40).to_string(index=False))

    iso_best = best[best.family == "Isomap"]
    print(f"\n=== ISOMAP IS BEST OF ALL METHODS (cells) : {len(iso_best)} ===")
    if len(iso_best):
        print(iso_best[CELL + ["value", "pca_value", "margin_vs_pca"]]
              .sort_values("margin_vs_pca", ascending=False).head(40)
              .to_string(index=False))

    # ----- top nonlinear wins overall ----- #
    nonpca = fam[fam.family != "PCA"].sort_values("margin_vs_pca", ascending=False)
    print("\n=== TOP 25 NONLINEAR-OVER-PCA WINS (any method) ===")
    print(nonpca[CELL + ["family", "value", "pca_value", "margin_vs_pca"]]
          .head(25).to_string(index=False))

    # save digested tables
    best.to_csv(os.path.join(OUT_DIR, "sweep_best_per_cell.csv"), index=False)
    piv.to_csv(os.path.join(OUT_DIR, "sweep_margin_by_task.csv"))
    print("\nsaved sweep_best_per_cell.csv, sweep_margin_by_task.csv")


if __name__ == "__main__":
    main()
