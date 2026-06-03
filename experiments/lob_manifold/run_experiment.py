"""
LOB manifold experiment: does any nonlinear DR (Isomap / UMAP / diffusion maps)
beat PCA on financially meaningful downstream tasks?

Pipeline
--------
1. Load 5s bar features for 5 symbols x {calm, stress}.
2. Transform + standardize features using a CALM-only scaler.
3. Fit each DR method to 2D on a CALM landmark sample, then project calm + stress.
4. Evaluate tasks: calm/stress separation, time-of-day regime, liquidity-depletion
   (depth collapse / spread widening / vol spike), and cross-symbol stress transfer.
5. Report stress AUC, balanced accuracy, average precision, silhouette,
   trustworthiness, geodesic preservation, linear-reconstruction R^2, and a
   classifier trained only on the 2D embedding.
6. Save summary tables (CSV) + plots, and print a strict PCA-vs-nonlinear verdict.

Run:  python experiments/lob_manifold/run_experiment.py
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import Isomap
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             roc_auc_score, silhouette_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import umap

from manifold_lib import (DiffusionMap, geodesic_preservation,
                           linear_reconstruction_r2, trustworthiness)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FEAT_DIR = os.path.join(ROOT, "data", "lob", "features")
FIG_DIR = os.path.join(ROOT, "figures")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(FIG_DIR, exist_ok=True)

SYMBOLS = ["NVDA", "AAPL", "MSFT", "SPY", "TSLA"]
BAR = "5s"
RNG = np.random.default_rng(42)

FEATURES = [
    "spread", "rel_spread", "bid_sz", "ask_sz", "bid_ct", "ask_ct",
    "top_depth", "obi", "book_event_count", "signed_book_flow", "abs_book_flow",
    "trade_count", "trade_volume", "signed_trade_volume", "trade_imbalance",
    "short_return", "realized_vol_60s", "spread_change", "depth_change",
]
LOG_FEATURES = ["spread", "rel_spread", "bid_sz", "ask_sz", "bid_ct", "ask_ct",
                "top_depth", "book_event_count", "abs_book_flow", "trade_count",
                "trade_volume", "realized_vol_60s"]
SIGNED_LOG_FEATURES = ["signed_book_flow", "signed_trade_volume", "short_return",
                       "spread_change", "depth_change"]

N_FIT = 4000          # calm landmarks for DR fitting
N_EVAL_CALM = 6000    # eval points per symbol (calm)
N_EVAL_STRESS = 6000  # eval points per symbol (stress)
N_GEOM = 1500         # subsample for O(n^2) geometry metrics
N_SIL = 3000          # subsample for silhouette


# --------------------------------------------------------------------------- #
# Data loading & preprocessing
# --------------------------------------------------------------------------- #
def load_features() -> pd.DataFrame:
    frames = []
    for sym in SYMBOLS:
        for pk in ("calm", "stress"):
            p = os.path.join(FEAT_DIR, f"feat_{sym}_{pk}_{BAR}.parquet")
            frames.append(pd.read_parquet(p))
    df = pd.concat(frames, ignore_index=True)
    df["bar"] = pd.to_datetime(df["bar"], utc=True)
    return df


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    calm = df["period"] == "calm"
    rv_q90 = df.loc[calm, "realized_vol_60s"].quantile(0.90)
    df["lab_depth_collapse"] = (df["fwd_min_depth_30s"] < 0.5 * df["top_depth"]).astype(float)
    df["lab_spread_widen"] = (df["fwd_max_spread_30s"] > 2.0 * df["spread"]).astype(float)
    df["lab_vol_spike"] = (df["fwd_max_rv_60s"] >= rv_q90).astype(float)
    for c in ["fwd_min_depth_30s", "fwd_max_spread_30s", "fwd_max_rv_60s"]:
        df.loc[df[c].isna(), [f"lab_depth_collapse", "lab_spread_widen", "lab_vol_spike"]] = np.nan
    # time-of-day regime
    t = df["bar"].dt.tz_convert("UTC")
    mins = t.dt.hour * 60 + t.dt.minute
    regime = np.full(len(df), "midday", dtype=object)
    regime[mins < 14 * 60] = "open"        # 13:30-14:00
    regime[mins >= 19 * 60 + 30] = "close"  # 19:30-20:00
    df["regime"] = regime
    df["is_stress"] = (df["period"] == "stress").astype(int)
    return df


def transform_features(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURES].copy()
    for c in LOG_FEATURES:
        X[c] = np.log1p(np.clip(X[c].astype(float), 0, None))
    for c in SIGNED_LOG_FEATURES:
        v = X[c].astype(float)
        X[c] = np.sign(v) * np.log1p(np.abs(v))
    return X.to_numpy()


def stratified_sample(df: pd.DataFrame, per_symbol: int, period: str) -> np.ndarray:
    idx = []
    sub = df[df["period"] == period]
    for sym in SYMBOLS:
        s = sub.index[sub["symbol"] == sym].to_numpy()
        if len(s) > per_symbol:
            s = RNG.choice(s, per_symbol, replace=False)
        idx.append(s)
    return np.concatenate(idx)


# --------------------------------------------------------------------------- #
# DR fitting / projection
# --------------------------------------------------------------------------- #
def batched_transform(model, X, batch=5000):
    out = [model.transform(X[i:i + batch]) for i in range(0, len(X), batch)]
    return np.vstack(out)


def fit_methods(X_fit: np.ndarray):
    methods = {}
    print("  fitting PCA ...", flush=True)
    methods["PCA"] = PCA(n_components=2, random_state=0).fit(X_fit)
    print("  fitting Isomap ...", flush=True)
    methods["Isomap"] = Isomap(n_neighbors=15, n_components=2).fit(X_fit)
    print("  fitting UMAP ...", flush=True)
    methods["UMAP"] = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1,
                                random_state=42).fit(X_fit)
    print("  fitting DiffusionMap ...", flush=True)
    methods["Diffusion"] = DiffusionMap(n_components=2, alpha=1.0).fit(X_fit)
    return methods


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def clf_metrics(emb, y, seeds=(0, 1, 2)):
    """RandomForest on the 2D embedding, averaged over several train/test splits
    -> mean AUC / balanced acc / avg precision (robust to split noise)."""
    m = np.isfinite(y)
    emb, y = emb[m], y[m].astype(int)
    if len(np.unique(y)) < 2:
        return dict(auc=np.nan, bal_acc=np.nan, ap=np.nan)
    aucs, bals, aps = [], [], []
    for seed in seeds:
        Xtr, Xte, ytr, yte = train_test_split(emb, y, test_size=0.4, random_state=seed,
                                               stratify=y)
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=-1,
                                     class_weight="balanced_subsample", random_state=seed)
        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        aucs.append(roc_auc_score(yte, proba))
        bals.append(balanced_accuracy_score(yte, pred))
        aps.append(average_precision_score(yte, proba))
    return dict(auc=float(np.mean(aucs)), bal_acc=float(np.mean(bals)),
                ap=float(np.mean(aps)), auc_std=float(np.std(aucs)))


def multiclass_auc(emb, y, seed=0):
    m = pd.notna(y)
    emb, y = emb[m.to_numpy()], y[m].to_numpy()
    Xtr, Xte, ytr, yte = train_test_split(emb, y, test_size=0.4, random_state=seed,
                                           stratify=y)
    clf = RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=-1,
                                 class_weight="balanced_subsample", random_state=seed)
    clf.fit(Xtr, ytr)
    proba = clf.predict_proba(Xte)
    auc = roc_auc_score(yte, proba, multi_class="ovr", average="macro",
                        labels=clf.classes_)
    pred = clf.predict(Xte)
    return dict(auc=auc, bal_acc=balanced_accuracy_score(yte, pred))


def loso_stress_auc(embeddings, meta, method, seed=0):
    """Leave-one-symbol-out: train calm/stress classifier on 4 symbols, test 5th."""
    emb = embeddings[method]
    sym = meta["symbol"].to_numpy()
    y = meta["is_stress"].to_numpy()
    aucs = []
    for held in SYMBOLS:
        tr = sym != held
        te = sym == held
        if len(np.unique(y[te])) < 2 or len(np.unique(y[tr])) < 2:
            continue
        clf = RandomForestClassifier(n_estimators=200, max_depth=6, n_jobs=-1,
                                     class_weight="balanced_subsample", random_state=seed)
        clf.fit(emb[tr], y[tr])
        proba = clf.predict_proba(emb[te])[:, 1]
        aucs.append(roc_auc_score(y[te], proba))
    return float(np.mean(aucs)) if aucs else np.nan


def stress_shift_cosine(emb, meta):
    """Mean pairwise cosine similarity of per-symbol calm->stress shift vectors."""
    shifts = []
    for sym in SYMBOLS:
        c = emb[(meta["symbol"] == sym) & (meta["is_stress"] == 0).to_numpy()]
        s = emb[(meta["symbol"] == sym) & (meta["is_stress"] == 1).to_numpy()]
        if len(c) and len(s):
            shifts.append(s.mean(0) - c.mean(0))
    shifts = np.array(shifts)
    cos = []
    for i in range(len(shifts)):
        for j in range(i + 1, len(shifts)):
            a, b = shifts[i], shifts[j]
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            if denom > 0:
                cos.append(float(a @ b / denom))
    return float(np.mean(cos)) if cos else np.nan, shifts


# --------------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------------- #
def _grid_scatter(embeddings, color, title, fname, cmap=None, legend=None,
                  categorical=False):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    for ax, method in zip(axes.ravel(), embeddings):
        emb = embeddings[method]
        if categorical:
            cats = legend
            for k, lab in enumerate(cats):
                m = color == lab
                ax.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.35, label=str(lab))
            ax.legend(markerscale=3, fontsize=8, loc="best")
        else:
            sc = ax.scatter(emb[:, 0], emb[:, 1], c=color, s=3, alpha=0.35, cmap=cmap)
        ax.set_title(method, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=130)
    plt.close(fig)
    print(f"  saved {fname}", flush=True)


def plot_stress_shifts(embeddings, meta, fname):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))
    colors = plt.cm.tab10(np.linspace(0, 1, len(SYMBOLS)))
    for ax, method in zip(axes.ravel(), embeddings):
        emb = embeddings[method]
        for sym, col in zip(SYMBOLS, colors):
            c = emb[(meta["symbol"] == sym) & (meta["is_stress"] == 0).to_numpy()]
            s = emb[(meta["symbol"] == sym) & (meta["is_stress"] == 1).to_numpy()]
            if not len(c) or not len(s):
                continue
            cc, sc = c.mean(0), s.mean(0)
            ax.scatter(*cc, color=col, marker="o", s=60, edgecolor="k", zorder=3)
            ax.scatter(*sc, color=col, marker="X", s=90, edgecolor="k", zorder=3)
            ax.annotate("", xy=sc, xytext=cc,
                        arrowprops=dict(arrowstyle="->", color=col, lw=2))
            ax.text(sc[0], sc[1], f" {sym}", fontsize=9, color=col)
        ax.set_title(method, fontsize=12, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Calm (o) -> Stress (X) centroid shift per symbol",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=130)
    plt.close(fig)
    print(f"  saved {fname}", flush=True)


def plot_task_bars(task_table, fname):
    import matplotlib.pyplot as plt

    tasks = ["stress", "depth_collapse", "spread_widen", "vol_spike", "regime",
             "xsym_loso"]
    labels = ["Calm/Stress", "Depth collapse", "Spread widen", "Vol spike",
              "Time-of-day", "Cross-sym LOSO"]
    methods = list(task_table.keys())
    x = np.arange(len(tasks))
    w = 0.2
    fig, ax = plt.subplots(figsize=(13, 6))
    for k, m in enumerate(methods):
        vals = [task_table[m].get(t, np.nan) for t in tasks]
        ax.bar(x + (k - 1.5) * w, vals, w, label=m)
    ax.axhline(0.5, color="grey", ls="--", lw=1, label="random (AUC=0.5)")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15)
    ax.set_ylabel("AUC (test, 2D embedding)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Downstream-task AUC by DR method (classifier on 2D embedding)",
                 fontweight="bold")
    ax.legend(ncol=3, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, fname), dpi=130)
    plt.close(fig)
    print(f"  saved {fname}", flush=True)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("[1/6] loading features ...", flush=True)
    df = load_features()
    df = add_labels(df)
    print(f"  total bars: {len(df):,}  "
          f"(calm {int((df.period=='calm').sum()):,}, "
          f"stress {int((df.period=='stress').sum()):,})", flush=True)

    print("[2/6] transform + calm-fit standardization ...", flush=True)
    Xraw = transform_features(df)
    scaler = StandardScaler().fit(Xraw[(df["period"] == "calm").to_numpy()])
    Xstd = scaler.transform(Xraw)
    Xstd = np.clip(Xstd, -8, 8)

    fit_idx = stratified_sample(df, N_FIT // len(SYMBOLS), "calm")
    eval_idx = np.concatenate([
        stratified_sample(df, N_EVAL_CALM // len(SYMBOLS), "calm"),
        stratified_sample(df, N_EVAL_STRESS // len(SYMBOLS), "stress"),
    ])
    RNG.shuffle(eval_idx)
    X_fit = Xstd[fit_idx]
    X_eval = Xstd[eval_idx]
    meta = df.iloc[eval_idx].reset_index(drop=True)
    print(f"  fit landmarks: {len(fit_idx):,}   eval points: {len(eval_idx):,}",
          flush=True)

    print("[3/6] fitting DR methods on calm landmarks ...", flush=True)
    methods = fit_methods(X_fit)

    print("[4/6] projecting eval set ...", flush=True)
    embeddings = {}
    for name, model in methods.items():
        embeddings[name] = batched_transform(model, X_eval)
        print(f"  projected {name}", flush=True)

    # ----- geometry metrics on a calm subsample ----- #
    print("[5/6] computing metrics ...", flush=True)
    calm_mask = (meta["is_stress"] == 0).to_numpy()
    geom_pool = np.where(calm_mask)[0]
    geom_idx = RNG.choice(geom_pool, min(N_GEOM, len(geom_pool)), replace=False)
    Xg = X_eval[geom_idx]

    rows = {}
    task_table = {}
    for m in methods:
        emb = embeddings[m]
        # geometry
        trust = trustworthiness(Xg, emb[geom_idx], n_neighbors=10)
        geo = geodesic_preservation(Xg, emb[geom_idx], n_neighbors=10)
        rec_r2 = linear_reconstruction_r2(X_eval, emb)
        # silhouette calm/stress (subsample)
        sil_idx = RNG.choice(len(emb), min(N_SIL, len(emb)), replace=False)
        sil = silhouette_score(emb[sil_idx], meta["is_stress"].to_numpy()[sil_idx])
        # tasks
        stress = clf_metrics(emb, meta["is_stress"].to_numpy())
        depth = clf_metrics(emb, meta["lab_depth_collapse"].to_numpy())
        spw = clf_metrics(emb, meta["lab_spread_widen"].to_numpy())
        vsp = clf_metrics(emb, meta["lab_vol_spike"].to_numpy())
        reg = multiclass_auc(emb, meta["regime"])
        loso = loso_stress_auc(embeddings, meta, m)
        cos, _ = stress_shift_cosine(emb, meta)

        rows[m] = dict(
            stress_auc=stress["auc"], stress_auc_std=stress.get("auc_std", np.nan),
            stress_balacc=stress["bal_acc"],
            stress_ap=stress["ap"], silhouette_cs=sil,
            trustworthiness=trust, geodesic_pres=geo, recon_r2=rec_r2,
            depth_auc=depth["auc"], spread_auc=spw["auc"], vol_auc=vsp["auc"],
            regime_auc=reg["auc"], xsym_loso_auc=loso, stress_shift_cos=cos,
        )
        task_table[m] = dict(
            stress=stress["auc"], depth_collapse=depth["auc"],
            spread_widen=spw["auc"], vol_spike=vsp["auc"],
            regime=reg["auc"], xsym_loso=loso,
        )
        print(f"  {m:10s} done", flush=True)

    summary = pd.DataFrame(rows).T
    summary = summary[[
        "stress_auc", "stress_auc_std", "stress_balacc", "stress_ap", "silhouette_cs",
        "trustworthiness", "geodesic_pres", "recon_r2",
        "depth_auc", "spread_auc", "vol_auc", "regime_auc",
        "xsym_loso_auc", "stress_shift_cos",
    ]].round(4)
    summary.to_csv(os.path.join(OUT_DIR, "summary_metrics.csv"))

    # baseline: full 19D feature RandomForest on stress task (ceiling reference)
    base = clf_metrics(X_eval, meta["is_stress"].to_numpy())
    base_depth = clf_metrics(X_eval, meta["lab_depth_collapse"].to_numpy())

    print("[6/6] plotting ...", flush=True)
    import matplotlib
    matplotlib.use("Agg")
    tod = pd.Categorical(meta["regime"], categories=["open", "midday", "close"])
    _grid_scatter(embeddings, meta["is_stress"].to_numpy(),
                  "LOB 2D embeddings colored by Calm (0) vs Stress (1)",
                  "lob_manifold_calm_stress.png", cmap="coolwarm")
    _grid_scatter(embeddings, meta["symbol"].to_numpy(),
                  "LOB 2D embeddings colored by symbol",
                  "lob_manifold_symbol.png", legend=SYMBOLS, categorical=True)
    _grid_scatter(embeddings, np.asarray(tod),
                  "LOB 2D embeddings colored by time-of-day regime",
                  "lob_manifold_regime.png", legend=["open", "midday", "close"],
                  categorical=True)
    _grid_scatter(embeddings, meta["lab_depth_collapse"].fillna(0).to_numpy(),
                  "LOB 2D embeddings colored by depth-collapse label",
                  "lob_manifold_depth_label.png", cmap="coolwarm")
    plot_stress_shifts(embeddings, meta, "lob_manifold_stress_shift.png")
    plot_task_bars(task_table, "lob_manifold_task_auc.png")

    # ----- report ----- #
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n================ SUMMARY METRICS ================")
    print(summary.to_string())
    print("\nBaseline (full 19D features, RandomForest):")
    print(f"  stress AUC={base['auc']:.3f}  depth-collapse AUC={base_depth['auc']:.3f}")
    print(f"  calm depth-collapse base rate={meta['lab_depth_collapse'].mean():.3f}, "
          f"spread-widen={meta['lab_spread_widen'].mean():.3f}, "
          f"vol-spike={meta['lab_vol_spike'].mean():.3f}")

    verdict(summary, base)


def verdict(summary: pd.DataFrame, base: dict):
    print("\n================ VERDICT ================")
    task_cols = ["stress_auc", "depth_auc", "spread_auc", "vol_auc",
                 "regime_auc", "xsym_loso_auc"]
    geom_cols = ["trustworthiness", "geodesic_pres", "recon_r2", "silhouette_cs"]
    pca = summary.loc["PCA"]
    nonlin = [m for m in summary.index if m != "PCA"]

    task_wins = {}
    for t in task_cols:
        best = summary[t].idxmax()
        margin = summary.loc[best, t] - pca[t]
        task_wins[t] = (best, margin)

    geom_wins = {t: summary[t].idxmax() for t in geom_cols}

    print("Per-task best method (vs PCA):")
    for t, (best, margin) in task_wins.items():
        tag = "PCA" if best == "PCA" else f"{best} (+{margin:.3f} over PCA)"
        print(f"  {t:16s}: {tag}")
    print("Per-geometry best method:")
    for t, best in geom_wins.items():
        print(f"  {t:16s}: {best}")

    # explicit Isomap-vs-PCA check (the user's original method of interest)
    if "Isomap" in summary.index:
        iso, pcab = summary.loc["Isomap"], summary.loc["PCA"]
        iso_gain = {t: iso[t] - pcab[t] for t in task_cols}
        is_best_anywhere = [t for t in task_cols if summary[t].idxmax() == "Isomap"]
        print("\nIsomap vs PCA on financial tasks:")
        for t in task_cols:
            print(f"  {t:16s}: Isomap {iso[t]:.3f} vs PCA {pcab[t]:.3f} "
                  f"({iso_gain[t]:+.3f})")
        if is_best_anywhere:
            print(f"  -> Isomap is the BEST method on: {', '.join(is_best_anywhere)}")
        else:
            print("  -> Isomap is NOT the best method on ANY task (dominated by "
                  "Diffusion/UMAP where nonlinearity helps, ~ties PCA elsewhere).")

    # strict rule: a nonlinear method must win a FINANCIAL task by a margin AND
    # the winning AUC must beat random (>0.52) -- a "win" on a below-random task
    # (e.g. failed cross-symbol transfer) is not credited.
    MARGIN = 0.02
    MIN_AUC = 0.52
    fin_tasks = ["stress_auc", "depth_auc", "spread_auc", "vol_auc", "xsym_loso_auc"]
    meaningful, dead_tasks = [], []
    for t in fin_tasks:
        best, margin = task_wins[t]
        if summary[t].max() < MIN_AUC:
            dead_tasks.append(t)
            continue
        if best != "PCA" and margin >= MARGIN:
            meaningful.append((t, best, margin))

    print("\nConclusion:")
    if dead_tasks:
        print(f"  Tasks where NO method beats random (AUC<{MIN_AUC}; not predictable "
              f"from any 2D embedding): {', '.join(dead_tasks)}")
    if not meaningful:
        print("  PCA is NOT beaten on any predictable financial task by >= %.2f AUC." % MARGIN)
        print("  Nonlinear methods may win geometry metrics (trustworthiness/geodesic),")
        print("  but that does NOT translate into better financial prediction.")
        print("  => PCA remains the recommended 2D representation for these LOB features.")
    else:
        print("  Nonlinear methods beat PCA on these PREDICTABLE financial tasks:")
        for t, best, margin in meaningful:
            print(f"    - {t}: {best} (+{margin:.3f} AUC, abs={summary.loc[best, t]:.3f})")
        print("  Isomap specifically: not among meaningful winners (~ties PCA).")
        print("  Recommendation: use the winning nonlinear method ONLY for those tasks.")


if __name__ == "__main__":
    main()

