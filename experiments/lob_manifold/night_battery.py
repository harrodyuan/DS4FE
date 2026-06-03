"""
Overnight robustness battery for the "nonlinear DR helps kNN volatility prediction"
finding. Reuses run_experiment's data prep + DR fitting, then stress-tests the
result from every angle a sceptical referee would attack.

Experiments
-----------
A. DIMENSION SWEEP (the decisive control):
   Does PCA at higher dimension catch up to nonlinear methods at 2D? If PCA-5D
   matches Diffusion-2D on kNN-vol, the "nonlinear win" is really a dimensionality
   artefact, not curvature. Tests dims {2,3,5,10}.
B. PERSISTENCE CONTROL:
   Forward vol is autocorrelated. Baselines = kNN on current vol alone (1 feature)
   and kNN on the raw 19D features. Does the 2D embedding add info beyond these?
C. MULTI-SEED ERROR BARS:
   Re-sample landmarks/eval over seeds -> mean +/- std on every delta, so the
   headline PCA->nonlinear gap is provably not split noise.
D. PROBE-k SWEEP:
   Vary kNN neighbours {10,25,50,100} -> is the win specific to one probe setting?
E. PER-SYMBOL:
   Does the nonlinear kNN-vol win hold for all 5 symbols or one outlier?

All results stream to CSVs in this folder so partial progress survives interruption.

Run:  python experiments/lob_manifold/night_battery.py
"""
from __future__ import annotations

import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import umap

from manifold_lib import DiffusionMap
import run_experiment as rx

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
SEEDS = [0, 1, 2]
DIMS = [2, 3, 5]
N_FIT = 3000
N_EVAL = 20000          # mixed calm+stress
PROBE_K = 25


# --------------------------------------------------------------------------- #
def prep(seed):
    """Re-sample fit/eval sets under a given seed; return Xstd, meta, idx sets."""
    rx.RNG = np.random.default_rng(seed)
    df = rx.add_labels(rx.load_features())
    Xraw = rx.transform_features(df)
    scaler = StandardScaler().fit(Xraw[(df["period"] == "calm").to_numpy()])
    Xstd = np.clip(scaler.transform(Xraw), -8, 8)

    fit_idx = rx.stratified_sample(df, N_FIT // len(rx.SYMBOLS), "calm")
    eval_idx = np.concatenate([
        rx.stratified_sample(df, (N_EVAL // 2) // len(rx.SYMBOLS), "calm"),
        rx.stratified_sample(df, (N_EVAL // 2) // len(rx.SYMBOLS), "stress"),
    ])
    rx.RNG.shuffle(eval_idx)
    meta = df.iloc[eval_idx].reset_index(drop=True)
    return Xstd, fit_idx, eval_idx, meta


def fit_dr(Xfit, dim, seed):
    m = {}
    m["PCA"] = PCA(n_components=dim, random_state=0).fit(Xfit)
    m["Isomap"] = Isomap(n_neighbors=15, n_components=dim).fit(Xfit)
    m["UMAP"] = umap.UMAP(n_components=dim, n_neighbors=15, min_dist=0.1,
                          random_state=seed).fit(Xfit)
    m["Diffusion"] = DiffusionMap(n_components=dim, alpha=1.0).fit(Xfit)
    return m


def knn_r2(emb, y, k=PROBE_K):
    mask = np.isfinite(y)
    Xs = StandardScaler().fit_transform(emb[mask])
    pred = cross_val_predict(KNeighborsRegressor(n_neighbors=k), Xs, y[mask], cv=4)
    return float(r2_score(y[mask], pred))


def knn_auc(emb, y, k=PROBE_K):
    mask = np.isfinite(y)
    yy = y[mask].astype(int)
    if len(np.unique(yy)) < 2:
        return np.nan
    Xs = StandardScaler().fit_transform(emb[mask])
    proba = cross_val_predict(KNeighborsClassifier(n_neighbors=k), Xs, yy,
                              cv=4, method="predict_proba")[:, 1]
    return float(roc_auc_score(yy, proba))


def targets(meta):
    fwd_logvol = np.log(meta["fwd_max_rv_60s"].to_numpy() + 1e-9)
    cur_logvol = np.log(meta["realized_vol_60s"].to_numpy() + 1e-9)
    volspike = meta["lab_vol_spike"].to_numpy()
    return fwd_logvol, cur_logvol, volspike


# --------------------------------------------------------------------------- #
def run():
    t0 = time.time()
    rowsA, rowsE = [], []
    probek_rows = []

    for si, seed in enumerate(SEEDS):
        print(f"\n===== SEED {seed} ({si+1}/{len(SEEDS)}) =====", flush=True)
        Xstd, fit_idx, eval_idx, meta = prep(seed)
        Xfit, Xeval = Xstd[fit_idx], Xstd[eval_idx]
        fwd_logvol, cur_logvol, volspike = targets(meta)

        # --- persistence + raw-feature baselines (B) --- #
        base_cur = knn_r2(cur_logvol.reshape(-1, 1), fwd_logvol)
        base_raw = knn_r2(Xeval, fwd_logvol)
        base_raw_auc = knn_auc(Xeval, volspike)
        print(f"  baseline kNN R2 fwdvol: cur-vol-only={base_cur:.3f}  raw19D={base_raw:.3f}",
              flush=True)
        rowsA.append(dict(seed=seed, dim=0, method="cur_vol_only",
                          knn_r2_fwdvol=base_cur, knn_vol_auc=np.nan))
        rowsA.append(dict(seed=seed, dim=19, method="raw_features",
                          knn_r2_fwdvol=base_raw, knn_vol_auc=base_raw_auc))

        for dim in DIMS:
            td = time.time()
            methods = fit_dr(Xfit, dim, seed)
            for name, model in methods.items():
                emb = rx.batched_transform(model, Xeval)
                r2 = knn_r2(emb, fwd_logvol)
                auc = knn_auc(emb, volspike)
                rowsA.append(dict(seed=seed, dim=dim, method=name,
                                  knn_r2_fwdvol=r2, knn_vol_auc=auc))
                # probe-k sweep only at dim=2
                if dim == 2:
                    for k in (10, 25, 50, 100):
                        probek_rows.append(dict(
                            seed=seed, method=name, k=k,
                            knn_r2_fwdvol=knn_r2(emb, fwd_logvol, k),
                            knn_vol_auc=knn_auc(emb, volspike, k)))
                    # per-symbol (E) at dim=2
                    for sym in rx.SYMBOLS:
                        sm = (meta["symbol"] == sym).to_numpy()
                        rowsE.append(dict(
                            seed=seed, symbol=sym, method=name,
                            knn_r2_fwdvol=knn_r2(emb[sm], fwd_logvol[sm]),
                            knn_vol_auc=knn_auc(emb[sm], volspike[sm])))
            print(f"  dim={dim} done ({time.time()-td:.0f}s)", flush=True)

        # PCA at dim=10 (does linear catch up with more dims?)
        pca10 = PCA(n_components=10, random_state=0).fit(Xfit)
        emb10 = rx.batched_transform(pca10, Xeval)
        rowsA.append(dict(seed=seed, dim=10, method="PCA",
                          knn_r2_fwdvol=knn_r2(emb10, fwd_logvol),
                          knn_vol_auc=knn_auc(emb10, volspike)))

        # stream partial results
        pd.DataFrame(rowsA).to_csv(os.path.join(OUT_DIR, "night_dim_sweep.csv"), index=False)
        pd.DataFrame(probek_rows).to_csv(os.path.join(OUT_DIR, "night_probek.csv"), index=False)
        pd.DataFrame(rowsE).to_csv(os.path.join(OUT_DIR, "night_persymbol.csv"), index=False)

    # ---- aggregate + report ---- #
    A = pd.DataFrame(rowsA)
    agg = (A.groupby(["dim", "method"])[["knn_r2_fwdvol", "knn_vol_auc"]]
             .agg(["mean", "std"]).round(4))
    print("\n================ A. DIMENSION SWEEP (mean +/- std over seeds) ================")
    print(agg.to_string())

    pk = (pd.DataFrame(probek_rows).groupby(["method", "k"])[["knn_r2_fwdvol", "knn_vol_auc"]]
            .mean().round(4))
    print("\n================ D. PROBE-k SWEEP (dim=2, mean over seeds) ================")
    print(pk.to_string())

    E = (pd.DataFrame(rowsE).groupby(["symbol", "method"])[["knn_r2_fwdvol", "knn_vol_auc"]]
           .mean().round(4))
    print("\n================ E. PER-SYMBOL (dim=2, mean over seeds) ================")
    print(E.to_string())

    print(f"\nTOTAL TIME: {(time.time()-t0)/60:.1f} min")
    print("saved: night_dim_sweep.csv, night_probek.csv, night_persymbol.csv")


if __name__ == "__main__":
    run()
