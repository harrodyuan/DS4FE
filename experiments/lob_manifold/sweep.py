"""
Overnight DR sweep: systematically test where (if anywhere) nonlinear DR
(Isomap / LLE / KernelPCA / UMAP / diffusion maps) beats PCA on financial LOB
tasks, across many configurations.

Swept dimensions
----------------
  bar size      : 5s, 1s, 30s
  feature set   : full / state / flow / dynamics / obi
  n_components  : 2, 3, 5, 10
  DR method     : PCA, KernelPCA(rbf), Isomap(nn=10,30), LLE, UMAP(nn=15,50),
                  Diffusion(alpha=1.0, 0.5)
  fit scope     : pooled (all 5 symbols), then per-symbol (phase B)
  downstream    : logistic + kNN probes (classification),
                  ridge + kNN probes (regression)
  tasks         : stress, depth_collapse, spread_widen, vol_spike (classif.)
                  fwdvol, curvol, relspread, intraday (regression)

Design
------
  * Every (config x method x task x probe) result is appended to
    sweep_results.csv immediately (crash-safe / resumable).
  * Already-completed config_ids are skipped on restart.
  * Hard wall-clock budget (default 5h) -- stops launching new work cleanly.
  * Each config is wrapped in try/except; failures are logged, sweep continues.

Run:        python experiments/lob_manifold/sweep.py
Smoke test: SWEEP_SMOKE=1 python experiments/lob_manifold/sweep.py
Budget:     SWEEP_HOURS=5 python experiments/lob_manifold/sweep.py
"""
from __future__ import annotations

import itertools
import os
import time
import traceback
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
import umap

import build_features as bf
from manifold_lib import DiffusionMap

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(OUT_DIR, "sweep_results.csv")
ERR_LOG = os.path.join(OUT_DIR, "sweep_errors.log")

SYMBOLS = ["NVDA", "AAPL", "MSFT", "SPY", "TSLA"]
SMOKE = os.environ.get("SWEEP_SMOKE", "0") == "1"
MAX_SECONDS = float(os.environ.get("SWEEP_HOURS", "5")) * 3600
RNG = np.random.default_rng(0)

N_FIT = 3000
N_EVAL_PER = 800          # per symbol per period (pooled => ~8000 total)
CV = 3

ALL_FEATURES = bf  # placeholder; feature names defined below
FEATURE_SETS = {
    "full": ["spread", "rel_spread", "bid_sz", "ask_sz", "bid_ct", "ask_ct",
             "top_depth", "obi", "book_event_count", "signed_book_flow",
             "abs_book_flow", "trade_count", "trade_volume", "signed_trade_volume",
             "trade_imbalance", "short_return", "realized_vol_60s", "spread_change",
             "depth_change"],
    "state": ["spread", "rel_spread", "bid_sz", "ask_sz", "bid_ct", "ask_ct",
              "top_depth", "obi"],
    "flow": ["book_event_count", "signed_book_flow", "abs_book_flow", "trade_count",
             "trade_volume", "signed_trade_volume", "trade_imbalance"],
    "dynamics": ["short_return", "realized_vol_60s", "spread_change", "depth_change"],
    "obi": ["obi", "trade_imbalance"],
}
LOG_FEATURES = {"spread", "rel_spread", "bid_sz", "ask_sz", "bid_ct", "ask_ct",
                "top_depth", "book_event_count", "abs_book_flow", "trade_count",
                "trade_volume", "realized_vol_60s"}
SIGNED_LOG_FEATURES = {"signed_book_flow", "signed_trade_volume", "short_return",
                       "spread_change", "depth_change"}


# --------------------------------------------------------------------------- #
# Feature loading (cached per bar size)
# --------------------------------------------------------------------------- #
def load_bar_features(bar: str) -> pd.DataFrame:
    frames = []
    for sym in SYMBOLS:
        for pk in ("calm", "stress"):
            path = os.path.join(bf.OUT_DIR, f"feat_{sym}_{pk}_{bar}.parquet")
            if not os.path.exists(path):
                print(f"    building {os.path.basename(path)} ...", flush=True)
                d = bf.build_one(sym, pk, bar)
                d.to_parquet(path, index=False)
            frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df["bar"] = pd.to_datetime(df["bar"], utc=True)
    return add_targets(df)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    calm = df["period"] == "calm"
    rv_q90 = df.loc[calm, "realized_vol_60s"].quantile(0.90)
    df["lab_stress"] = (df["period"] == "stress").astype(float)
    df["lab_depth_collapse"] = (df["fwd_min_depth_30s"] < 0.5 * df["top_depth"]).astype(float)
    df["lab_spread_widen"] = (df["fwd_max_spread_30s"] > 2.0 * df["spread"]).astype(float)
    df["lab_vol_spike"] = (df["fwd_max_rv_60s"] >= rv_q90).astype(float)
    for c in ["fwd_min_depth_30s", "fwd_max_spread_30s", "fwd_max_rv_60s"]:
        bad = df[c].isna()
        df.loc[bad, ["lab_depth_collapse", "lab_spread_widen", "lab_vol_spike"]] = np.nan
    t = df["bar"].dt.tz_convert("UTC")
    df["reg_intraday"] = (t.dt.hour * 60 + t.dt.minute - (13 * 60 + 30)).astype(float)
    df["reg_fwdvol"] = np.log(df["fwd_max_rv_60s"] + 1e-9)
    df["reg_curvol"] = np.log(df["realized_vol_60s"] + 1e-9)
    df["reg_relspread"] = np.log(df["rel_spread"] + 1e-9)
    return df


def transform_subset(df: pd.DataFrame, cols) -> np.ndarray:
    X = df[cols].copy()
    for c in cols:
        v = X[c].astype(float)
        if c in LOG_FEATURES:
            X[c] = np.log1p(np.clip(v, 0, None))
        elif c in SIGNED_LOG_FEATURES:
            X[c] = np.sign(v) * np.log1p(np.abs(v))
        else:
            X[c] = v
    return X.to_numpy()


def sample_idx(df: pd.DataFrame, per: int, period: str, symbols) -> np.ndarray:
    out = []
    sub = df[df["period"] == period]
    for sym in symbols:
        s = sub.index[sub["symbol"] == sym].to_numpy()
        if len(s) > per:
            s = RNG.choice(s, per, replace=False)
        out.append(s)
    return np.concatenate(out) if out else np.array([], dtype=int)


# --------------------------------------------------------------------------- #
# DR method factory (all support out-of-sample .transform)
# --------------------------------------------------------------------------- #
def make_methods(n_comp: int):
    nn = max(n_comp + 1, 15)
    m = {
        "PCA": lambda: PCA(n_components=n_comp, random_state=0),
        "KernelPCA_rbf": lambda: KernelPCA(n_components=n_comp, kernel="rbf",
                                           gamma=None, random_state=0),
        "Isomap_nn10": lambda: Isomap(n_neighbors=10, n_components=n_comp),
        "Isomap_nn30": lambda: Isomap(n_neighbors=30, n_components=n_comp),
        "LLE": lambda: LocallyLinearEmbedding(n_neighbors=nn, n_components=n_comp,
                                              method="standard", random_state=0),
        "UMAP_nn15": lambda: umap.UMAP(n_components=n_comp, n_neighbors=15,
                                       min_dist=0.1, random_state=42),
        "UMAP_nn50": lambda: umap.UMAP(n_components=n_comp, n_neighbors=50,
                                       min_dist=0.1, random_state=42),
        "Diffusion_a1.0": lambda: DiffusionMap(n_components=n_comp, alpha=1.0),
        "Diffusion_a0.5": lambda: DiffusionMap(n_components=n_comp, alpha=0.5),
    }
    if SMOKE:
        return {k: m[k] for k in ["PCA", "Isomap_nn10", "Diffusion_a1.0"]}
    return m


def batched_transform(model, X, batch=4000):
    return np.vstack([model.transform(X[i:i + batch]) for i in range(0, len(X), batch)])


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
CLF_TASKS = ["lab_stress", "lab_depth_collapse", "lab_spread_widen", "lab_vol_spike"]
REG_TASKS = ["reg_fwdvol", "reg_curvol", "reg_relspread", "reg_intraday"]


def clf_auc(emb, y, probe):
    m = np.isfinite(y)
    emb, y = emb[m], y[m].astype(int)
    if len(np.unique(y)) < 2:
        return np.nan
    Xs = StandardScaler().fit_transform(emb)
    clf = (LogisticRegression(max_iter=2000, class_weight="balanced")
           if probe == "logistic" else KNeighborsClassifier(n_neighbors=25))
    proba = cross_val_predict(clf, Xs, y, cv=CV, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def reg_r2(emb, y, probe):
    m = np.isfinite(y)
    emb, y = emb[m], y[m].astype(float)
    Xs = StandardScaler().fit_transform(emb)
    reg = Ridge(alpha=1.0) if probe == "ridge" else KNeighborsRegressor(n_neighbors=25)
    pred = cross_val_predict(reg, Xs, y, cv=CV)
    return float(r2_score(y, pred))


# --------------------------------------------------------------------------- #
# Result IO
# --------------------------------------------------------------------------- #
COLUMNS = ["config_id", "bar", "feature_set", "n_features", "fit_scope",
           "method", "n_components", "task", "probe", "metric", "value",
           "fit_seconds", "n_eval", "timestamp"]


def load_done() -> set:
    if not os.path.exists(RESULTS_CSV):
        return set()
    try:
        d = pd.read_csv(RESULTS_CSV, usecols=["config_id"])
        return set(d["config_id"].unique())
    except Exception:
        return set()


def append_rows(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    header = not os.path.exists(RESULTS_CSV)
    df.to_csv(RESULTS_CSV, mode="a", header=header, index=False)


def log_err(config_id, exc):
    with open(ERR_LOG, "a") as f:
        f.write(f"\n=== {config_id} @ {time.ctime()} ===\n")
        f.write("".join(traceback.format_exception(exc)))


# --------------------------------------------------------------------------- #
# Per-config evaluation
# --------------------------------------------------------------------------- #
def eval_config(config_id, Xstd, meta, fit_idx, eval_idx, method_name, make_fn,
                n_comp, bar, fset, fit_scope, t0):
    fit_start = time.time()
    model = make_fn()
    model.fit(Xstd[fit_idx])
    emb = batched_transform(model, Xstd[eval_idx])
    fit_seconds = round(time.time() - fit_start, 2)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    rows = []
    n_eval = len(eval_idx)
    base = dict(config_id=config_id, bar=bar, feature_set=fset,
                n_features=Xstd.shape[1], fit_scope=fit_scope, method=method_name,
                n_components=n_comp, fit_seconds=fit_seconds, n_eval=n_eval,
                timestamp=ts)
    for task in CLF_TASKS:
        y = meta[task].to_numpy()
        for probe in ("logistic", "knn"):
            rows.append({**base, "task": task, "probe": probe, "metric": "auc",
                         "value": round(clf_auc(emb, y, probe), 4)})
    for task in REG_TASKS:
        y = meta[task].to_numpy()
        for probe in ("ridge", "knn"):
            rows.append({**base, "task": task, "probe": probe, "metric": "r2",
                         "value": round(reg_r2(emb, y, probe), 4)})
    return rows


# --------------------------------------------------------------------------- #
# Main sweep
# --------------------------------------------------------------------------- #
def main():
    t0 = time.time()
    done = load_done()
    print(f"[sweep] resuming: {len(done)} configs already complete", flush=True)
    print(f"[sweep] budget: {MAX_SECONDS/3600:.1f}h   smoke={SMOKE}", flush=True)

    bars = ["5s"] if SMOKE else ["5s", "1s", "30s"]
    fsets = ["full"] if SMOKE else list(FEATURE_SETS.keys())
    ncomps = [2] if SMOKE else [2, 3, 5, 10]

    n_done_now = 0

    # ---------------- Phase A: pooled fits ---------------- #
    for bar in bars:
        if time.time() - t0 > MAX_SECONDS:
            break
        print(f"\n[phase A] bar={bar}: loading features ...", flush=True)
        df = load_bar_features(bar)
        fit_idx_all = sample_idx(df, N_FIT // len(SYMBOLS), "calm", SYMBOLS)
        eval_idx = np.concatenate([
            sample_idx(df, N_EVAL_PER, "calm", SYMBOLS),
            sample_idx(df, N_EVAL_PER, "stress", SYMBOLS)])
        RNG.shuffle(eval_idx)
        meta = df.iloc[eval_idx].reset_index(drop=True)

        for fset in fsets:
            cols = FEATURE_SETS[fset]
            Xfull = transform_subset(df, cols)
            sc = StandardScaler().fit(Xfull[(df["period"] == "calm").to_numpy()])
            Xstd = np.clip(sc.transform(Xfull), -8, 8)
            for n_comp in ncomps:
                if n_comp >= len(cols):
                    continue
                for method_name, make_fn in make_methods(n_comp).items():
                    if time.time() - t0 > MAX_SECONDS:
                        print("[sweep] time budget reached -- stopping.", flush=True)
                        _final(t0, n_done_now)
                        return
                    cid = f"A|{bar}|{fset}|nc{n_comp}|{method_name}"
                    if cid in done:
                        continue
                    try:
                        rows = eval_config(cid, Xstd, meta, fit_idx_all, eval_idx,
                                           method_name, make_fn, n_comp, bar, fset,
                                           "pooled", t0)
                        append_rows(rows)
                        n_done_now += 1
                        el = time.time() - t0
                        print(f"  [{el/3600:5.2f}h] {cid:42s} ok "
                              f"({rows[0]['fit_seconds']}s fit)", flush=True)
                    except Exception as exc:
                        log_err(cid, exc)
                        print(f"  ERROR {cid}: {exc}", flush=True)

    # ---------------- Phase B: per-symbol fits (5s, full) ---------------- #
    if not SMOKE:
        bar = "5s"
        df = load_bar_features(bar)
        cols = FEATURE_SETS["full"]
        Xfull = transform_subset(df, cols)
        for sym in SYMBOLS:
            for n_comp in [2, 3]:
                fit_idx = sample_idx(df, N_FIT, "calm", [sym])
                eval_idx = np.concatenate([
                    sample_idx(df, N_EVAL_PER * 4, "calm", [sym]),
                    sample_idx(df, N_EVAL_PER * 4, "stress", [sym])])
                if len(fit_idx) < 100 or len(eval_idx) < 100:
                    continue
                RNG.shuffle(eval_idx)
                sc = StandardScaler().fit(Xfull[(df["period"] == "calm").to_numpy()])
                Xstd = np.clip(sc.transform(Xfull), -8, 8)
                meta = df.iloc[eval_idx].reset_index(drop=True)
                for method_name, make_fn in make_methods(n_comp).items():
                    if time.time() - t0 > MAX_SECONDS:
                        print("[sweep] time budget reached -- stopping.", flush=True)
                        _final(t0, n_done_now)
                        return
                    cid = f"B|{bar}|{sym}|nc{n_comp}|{method_name}"
                    if cid in done:
                        continue
                    try:
                        rows = eval_config(cid, Xstd, meta, fit_idx, eval_idx,
                                           method_name, make_fn, n_comp, bar,
                                           "full", f"sym:{sym}", t0)
                        append_rows(rows)
                        n_done_now += 1
                        el = time.time() - t0
                        print(f"  [{el/3600:5.2f}h] {cid:42s} ok", flush=True)
                    except Exception as exc:
                        log_err(cid, exc)
                        print(f"  ERROR {cid}: {exc}", flush=True)

    _final(t0, n_done_now)


def _final(t0, n_done_now):
    el = time.time() - t0
    print(f"\n[sweep] finished phase(s). new configs this run: {n_done_now}, "
          f"elapsed {el/3600:.2f}h. results -> {os.path.basename(RESULTS_CSV)}",
          flush=True)


if __name__ == "__main__":
    main()
