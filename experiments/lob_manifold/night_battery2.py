"""
Extension of night_battery: does the "nonlinear edge is a 2-D artefact" conclusion
generalise beyond forward volatility to the OTHER forward microstructure targets
(depth collapse, spread widening)? Same dimension control, kNN probe.

Run:  python experiments/lob_manifold/night_battery2.py
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler

import night_battery as nb
import run_experiment as rx

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1]
DIMS = [2, 3, 5]


def run():
    t0 = time.time()
    rows = []
    for seed in SEEDS:
        print(f"\n===== SEED {seed} =====", flush=True)
        Xstd, fit_idx, eval_idx, meta = nb.prep(seed)
        Xfit, Xeval = Xstd[fit_idx], Xstd[eval_idx]
        y_depth = meta["lab_depth_collapse"].to_numpy()
        y_spread = meta["lab_spread_widen"].to_numpy()

        # raw-feature baselines (ceiling)
        rows.append(dict(seed=seed, dim=19, method="raw_features",
                         depth_auc=nb.knn_auc(Xeval, y_depth),
                         spread_auc=nb.knn_auc(Xeval, y_spread)))

        for dim in DIMS:
            methods = nb.fit_dr(Xfit, dim, seed)
            for name, model in methods.items():
                emb = rx.batched_transform(model, Xeval)
                rows.append(dict(seed=seed, dim=dim, method=name,
                                 depth_auc=nb.knn_auc(emb, y_depth),
                                 spread_auc=nb.knn_auc(emb, y_spread)))
            print(f"  dim={dim} done", flush=True)
        pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "night_multitarget.csv"), index=False)

    A = pd.DataFrame(rows)
    agg = (A.groupby(["dim", "method"])[["depth_auc", "spread_auc"]]
             .mean().round(4))
    print("\n===== MULTI-TARGET kNN AUC (mean over seeds) =====")
    print(agg.to_string())
    print(f"\nTOTAL TIME: {(time.time()-t0)/60:.1f} min  -> night_multitarget.csv")


if __name__ == "__main__":
    run()
