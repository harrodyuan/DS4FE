"""
Give Isomap its fair shot: evaluate the SAME 2D embeddings with WEAK downstream
models (linear probe, kNN) and SMOOTH / ORDERED targets, instead of RandomForest.

Rationale
---------
RandomForest can exploit any embedding that *preserves* the information, which
neutralizes Isomap's only structural edge: the *layout* (global geodesic order).
Isomap's pitch is that it unrolls a curved manifold so a target becomes
linearly / monotonically readable. That edge only shows up with:
  - a LINEAR probe (logistic / linear regression on the 2D coords), and/or
  - a kNN probe (faithful local+global neighbours), and
  - SMOOTH, ORDERED targets (intraday liquidity cycle, forward vol level)
    rather than binary labels a forest can carve arbitrarily.

This reuses run_experiment's data prep + DR fitting, then re-scores.

Run:  python experiments/lob_manifold/linear_probe.py
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler

import run_experiment as rx

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def _auc_linear(emb, y, kind="logistic"):
    m = np.isfinite(y)
    emb, y = emb[m], y[m].astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    Xs = StandardScaler().fit_transform(emb)
    if kind == "logistic":
        clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    else:  # knn
        clf = KNeighborsClassifier(n_neighbors=25)
    proba = cross_val_predict(clf, Xs, y, cv=4, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def _r2_reg(emb, y, kind="linear"):
    m = np.isfinite(y)
    emb, y = emb[m], y[m].astype(float)
    Xs = StandardScaler().fit_transform(emb)
    if kind == "linear":
        reg = Ridge(alpha=1.0)
    else:
        reg = KNeighborsRegressor(n_neighbors=25)
    pred = cross_val_predict(reg, Xs, y, cv=4)
    return float(r2_score(y, pred))


def main():
    print("[1/3] data prep + DR fit (reusing run_experiment) ...", flush=True)
    df = rx.add_labels(rx.load_features())
    Xraw = rx.transform_features(df)
    scaler = StandardScaler().fit(Xraw[(df["period"] == "calm").to_numpy()])
    Xstd = np.clip(scaler.transform(Xraw), -8, 8)

    fit_idx = rx.stratified_sample(df, rx.N_FIT // len(rx.SYMBOLS), "calm")
    eval_idx = np.concatenate([
        rx.stratified_sample(df, rx.N_EVAL_CALM // len(rx.SYMBOLS), "calm"),
        rx.stratified_sample(df, rx.N_EVAL_STRESS // len(rx.SYMBOLS), "stress"),
    ])
    rx.RNG.shuffle(eval_idx)
    methods = rx.fit_methods(Xstd[fit_idx])
    meta = df.iloc[eval_idx].reset_index(drop=True)
    embeddings = {n: rx.batched_transform(m, Xstd[eval_idx]) for n, m in methods.items()}

    # ---- smooth / ordered targets ---- #
    t = meta["bar"].dt.tz_convert("UTC")
    minutes_since_open = (t.dt.hour * 60 + t.dt.minute - (13 * 60 + 30)).to_numpy()
    fwd_logvol = np.log(meta["fwd_max_rv_60s"].to_numpy() + 1e-9)
    cur_logvol = np.log(meta["realized_vol_60s"].to_numpy() + 1e-9)
    cur_relspread = np.log(meta["rel_spread"].to_numpy() + 1e-9)

    print("[2/3] scoring weak probes ...", flush=True)
    rows = {}
    for m in methods:
        emb = embeddings[m]
        rows[m] = dict(
            # LINEAR (logistic) probe AUC -- rewards linear layout of the target
            lin_stress_auc=_auc_linear(emb, meta["is_stress"].to_numpy()),
            lin_depth_auc=_auc_linear(emb, meta["lab_depth_collapse"].to_numpy()),
            lin_spread_auc=_auc_linear(emb, meta["lab_spread_widen"].to_numpy()),
            lin_vol_auc=_auc_linear(emb, meta["lab_vol_spike"].to_numpy()),
            # kNN probe AUC -- rewards faithful neighbour geometry
            knn_stress_auc=_auc_linear(emb, meta["is_stress"].to_numpy(), "knn"),
            knn_vol_auc=_auc_linear(emb, meta["lab_vol_spike"].to_numpy(), "knn"),
            # LINEAR regression R^2 on smooth/ordered targets
            lin_r2_intraday=_r2_reg(emb, minutes_since_open),
            lin_r2_fwdvol=_r2_reg(emb, fwd_logvol),
            lin_r2_curvol=_r2_reg(emb, cur_logvol),
            lin_r2_relspread=_r2_reg(emb, cur_relspread),
            # kNN regression R^2 (local manifold smoothness)
            knn_r2_fwdvol=_r2_reg(emb, fwd_logvol, "knn"),
        )
        print(f"  {m:10s} done", flush=True)

    summary = pd.DataFrame(rows).T.round(4)
    summary.to_csv(os.path.join(OUT_DIR, "linear_probe_metrics.csv"))

    pd.set_option("display.width", 220, "display.max_columns", 30)
    print("\n========= WEAK-PROBE METRICS (AUC for *_auc, R^2 for *_r2_*) =========")
    print(summary.to_string())

    # ---- strict Isomap verdict under weak probes ---- #
    print("\n========= ISOMAP vs PCA (weak probes) =========")
    iso, pca = summary.loc["Isomap"], summary.loc["PCA"]
    wins, best_overall = [], []
    for c in summary.columns:
        gain = iso[c] - pca[c]
        is_best = summary[c].idxmax() == "Isomap"
        flag = "  <-- ISOMAP BEST OF ALL" if is_best else ""
        print(f"  {c:18s}: Isomap {iso[c]:+.3f} vs PCA {pca[c]:+.3f}  ({gain:+.3f}){flag}")
        if gain >= 0.02:
            wins.append((c, gain))
        if is_best:
            best_overall.append(c)

    print("\nVerdict:")
    if best_overall:
        print(f"  Isomap is the BEST of all 4 methods on: {', '.join(best_overall)}")
    if wins:
        print("  Isomap beats PCA by >=0.02 on:")
        for c, g in wins:
            print(f"    - {c}: +{g:.3f}")
    if not wins and not best_overall:
        print("  Isomap still does not beat PCA even under weak probes.")


if __name__ == "__main__":
    main()
