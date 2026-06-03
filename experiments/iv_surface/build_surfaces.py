"""
Build the daily IV-surface matrix from cached per-day IV tables.

For each trading day we have a scattered cloud of (T, moneyness, iv) points.
We interpolate each day onto a FIXED (maturity x moneyness) grid so every day
becomes a comparable fixed-length vector. Stack -> (n_days x n_grid) matrix,
the input for PCA / ISOMAP.

Interpolation: per day, in total-implied-variance space w = iv^2 * T.
  1. smile step  : within each expiry, interp iv across log-moneyness
  2. calendar step: across expiries, interp total variance at target T
Two-step (smile then calendar) is more stable than raw 2-D scatter interp.

Output: data/iv/surfaces.parquet  (index=date, columns=grid points, values=iv)
        plus a sidecar data/iv/surface_grid.json describing the grid.
"""
import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
IV_DIR = ROOT / "data" / "iv"

# Fixed grid -----------------------------------------------------------------
MATS_DAYS = np.array([30, 60, 91, 122, 182, 273, 365])      # target tenors (days)
MONEYNESS = np.array([0.85, 0.90, 0.95, 0.975, 1.00,
                      1.025, 1.05, 1.10, 1.15])              # K / F
GRID_COLS = [f"T{d}_m{int(m*1000)}" for d in MATS_DAYS for m in MONEYNESS]


def build_one_surface(df: pd.DataFrame) -> np.ndarray | None:
    """Return a flat (len(MATS_DAYS)*len(MONEYNESS),) IV vector, or None if too sparse."""
    df = df[(df.iv > 0.01) & (df.iv < 2.0)].copy()
    if df.empty:
        return None
    df["logm"] = np.log(df.moneyness)
    df["Tdays"] = df["T"] * 365.25

    # group by expiry; need each expiry's smile sampled at target log-moneyness
    target_logm = np.log(MONEYNESS)
    expiries = sorted(df["T"].unique())
    if len(expiries) < 2:
        return None

    # smile step: for each expiry, interp iv at the target moneyness grid
    exp_T, exp_w = [], []   # T per expiry, total-variance row per expiry
    for T in expiries:
        g = df[df["T"] == T].sort_values("logm")
        if len(g) < 5:
            continue
        # average duplicate strikes (call/put) for a clean smile
        gg = g.groupby("logm", as_index=False).iv.mean().sort_values("logm")
        if gg.logm.min() > target_logm.min() or gg.logm.max() < target_logm.max():
            # allow mild extrapolation via clamping to nearest within reason
            pass
        iv_row = np.interp(target_logm, gg.logm.values, gg.iv.values)
        exp_T.append(T)
        exp_w.append((iv_row ** 2) * T)        # total implied variance

    if len(exp_T) < 2:
        return None
    exp_T = np.array(exp_T)
    exp_w = np.array(exp_w)                      # (n_exp x n_money)

    # calendar step: interp total variance at each target maturity, per moneyness col
    target_T = MATS_DAYS / 365.25
    surf = np.empty((len(target_T), len(MONEYNESS)))
    for j in range(len(MONEYNESS)):
        w_col = exp_w[:, j]
        w_interp = np.interp(target_T, exp_T, w_col)   # clamps at ends
        surf[:, j] = np.sqrt(np.maximum(w_interp, 1e-8) / target_T)
    return surf.ravel()


def main() -> int:
    iv_files = sorted(IV_DIR.glob("iv_*.parquet"))
    print(f"found {len(iv_files)} cached IV days")
    rows, dates, skipped = [], [], 0
    for f in iv_files:
        date = f.stem.replace("iv_", "")
        df = pd.read_parquet(f)
        vec = build_one_surface(df)
        if vec is None or not np.isfinite(vec).all():
            skipped += 1
            continue
        rows.append(vec)
        dates.append(date)

    if not rows:
        print("no usable surfaces yet")
        return 0

    surf_df = pd.DataFrame(rows, index=pd.to_datetime(dates), columns=GRID_COLS)
    surf_df.index.name = "date"
    surf_df = surf_df.sort_index()
    out = IV_DIR / "surfaces.parquet"
    surf_df.to_parquet(out)
    json.dump(
        {"mats_days": MATS_DAYS.tolist(), "moneyness": MONEYNESS.tolist(),
         "grid_cols": GRID_COLS},
        open(IV_DIR / "surface_grid.json", "w"), indent=2,
    )
    print(f"built {len(surf_df)} surfaces x {surf_df.shape[1]} grid points "
          f"(skipped {skipped})")
    print(f"saved -> {out.name}")
    print(f"\ndate range: {surf_df.index.min().date()} -> {surf_df.index.max().date()}")
    print("\nATM (m=1.00) term structure, last available day:")
    last = surf_df.iloc[-1]
    for d in MATS_DAYS:
        print(f"  {d:>4d}d : {last[f'T{d}_m1000']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
