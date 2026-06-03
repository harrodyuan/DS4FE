"""
ONE-DAY validation pull for the IV-surface pivot (BILLABLE, ~$0.017).

Steps:
  1. Pull SPX option `definition` for the day  -> instrument_id -> (strike, expiry, C/P)
  2. Pull `cbbo-1m` for the 15:45 ET minute     -> bid/ask per instrument
  3. Join, compute mid prices
  4. Per expiration: forward F + discount via put-call parity regression
  5. Invert Black-76 to implied vol per contract
  6. Save raw + computed to parquet; print sanity diagnostics

Raw data is cached to data/iv/ so we never re-download (and re-pay) the same day.
"""
import os
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


# ----------------------------------------------------------------------------- config
ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "iv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "OPRA.PILLAR"
PARENT = "SPX.OPT"
DAY = "2023-06-01"          # validation day
SNAP_UTC_START = f"{DAY}T19:45:00"   # 15:45 ET (EDT = UTC-4)
SNAP_UTC_END = f"{DAY}T19:46:00"


def load_env(env_path: pathlib.Path) -> None:
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


# ----------------------------------------------------------------------------- pull
def pull_raw():
    import databento as db

    load_env(ROOT / ".env")
    client = db.Historical(os.environ["DATABENTO_API_KEY"])

    def_path = OUT_DIR / f"def_{DAY}.parquet"
    snap_path = OUT_DIR / f"cbbo_{DAY}_1545.parquet"

    if def_path.exists():
        print(f"  using cached {def_path.name}")
        defs = pd.read_parquet(def_path)
    else:
        print("  downloading definition (1 day) ...")
        defs = client.timeseries.get_range(
            dataset=DATASET, symbols=PARENT, stype_in="parent",
            schema="definition", start=f"{DAY}T00:00:00", end=f"{DAY}T23:59:00",
        ).to_df()
        defs.to_parquet(def_path)
        print(f"  saved {def_path.name}  ({len(defs):,} rows)")

    if snap_path.exists():
        print(f"  using cached {snap_path.name}")
        snap = pd.read_parquet(snap_path)
    else:
        print("  downloading cbbo-1m 15:45 window ...")
        snap = client.timeseries.get_range(
            dataset=DATASET, symbols=PARENT, stype_in="parent",
            schema="cbbo-1m", start=SNAP_UTC_START, end=SNAP_UTC_END,
        ).to_df()
        snap.to_parquet(snap_path)
        print(f"  saved {snap_path.name}  ({len(snap):,} rows)")

    return defs, snap


# ----------------------------------------------------------------------------- IV math
def black76_price(F, K, T, sigma, r, is_call):
    """Undiscounted-forward Black-76 then discounted by exp(-rT)."""
    if sigma <= 0 or T <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return np.exp(-r * T) * intrinsic
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        px = F * norm.cdf(d1) - K * norm.cdf(d2)
    else:
        px = K * norm.cdf(-d2) - F * norm.cdf(-d1)
    return np.exp(-r * T) * px


def implied_vol(price, F, K, T, r, is_call):
    intrinsic = np.exp(-r * T) * (max(F - K, 0.0) if is_call else max(K - F, 0.0))
    if price <= intrinsic + 1e-8 or T <= 0:
        return np.nan
    try:
        return brentq(
            lambda s: black76_price(F, K, T, s, r, is_call) - price,
            1e-4, 5.0, maxiter=100, xtol=1e-6,
        )
    except (ValueError, RuntimeError):
        return np.nan


# ----------------------------------------------------------------------------- main
def main() -> int:
    defs, snap = pull_raw()

    print("\n--- definition columns ---")
    print(list(defs.columns))
    print("\n--- cbbo columns ---")
    print(list(snap.columns))

    # Build instrument map from definition
    dmap = defs[["instrument_id", "raw_symbol", "strike_price",
                 "expiration", "instrument_class"]].drop_duplicates("instrument_id").copy()
    dmap["strike"] = dmap["strike_price"]            # .to_df() already returns dollars
    dmap["is_call"] = dmap["instrument_class"].astype(str).str.upper().str[0] == "C"
    dmap["expiry"] = pd.to_datetime(dmap["expiration"]).dt.tz_localize(None).dt.normalize()

    # Keep latest quote per instrument in the snapshot minute
    snap = snap.sort_index()
    px = snap.groupby("instrument_id").last().reset_index()
    bid_col = "bid_px_00" if "bid_px_00" in px.columns else "bid_px"
    ask_col = "ask_px_00" if "ask_px_00" in px.columns else "ask_px"
    px["bid"] = px[bid_col]                           # .to_df() already returns dollars
    px["ask"] = px[ask_col]
    px = px[["instrument_id", "bid", "ask"]]

    df = px.merge(dmap, on="instrument_id", how="inner")
    df = df[(df.bid > 0) & (df.ask > 0) & (df.ask >= df.bid)].copy()
    df["mid"] = 0.5 * (df.bid + df.ask)
    asof = pd.Timestamp(DAY)
    df["T"] = (df.expiry - asof).dt.days / 365.25
    df = df[df["T"] > 0]

    print(f"\nclean quotes: {len(df):,}  expiries: {df.expiry.nunique()}  "
          f"calls: {df.is_call.sum():,}  puts: {(~df.is_call).sum():,}")

    # Forward + discount per expiry via put-call parity: C - P = exp(-rT)(F - K)
    rows = []
    for exp, g in df.groupby("expiry"):
        piv = g.pivot_table(index="strike", columns="is_call", values="mid")
        if True not in piv.columns or False not in piv.columns:
            continue
        pair = piv.dropna()
        if len(pair) < 4:
            continue
        K = pair.index.values.astype(float)
        diff = (pair[True] - pair[False]).values   # C - P
        # linear fit diff = a + b*K  -> b = -exp(-rT), a = exp(-rT)*F
        b, a = np.polyfit(K, diff, 1)
        if b >= 0:
            continue
        disc = -b
        F = a / disc
        T = g["T"].iloc[0]
        r = -np.log(disc) / T if disc > 0 else 0.0
        for _, row in g.iterrows():
            iv = implied_vol(row.mid, F, row.strike, T, r, row.is_call)
            rows.append({
                "expiry": exp, "T": T, "strike": row.strike, "F": F, "r": r,
                "moneyness": row.strike / F, "is_call": row.is_call,
                "mid": row.mid, "iv": iv,
            })

    out = pd.DataFrame(rows)
    out = out[out.iv.notna() & (out.iv > 0.01) & (out.iv < 2.0)]
    out_path = OUT_DIR / f"iv_{DAY}.parquet"
    out.to_parquet(out_path)

    print(f"\nIV computed for {len(out):,} contracts across {out.expiry.nunique()} expiries")
    print(f"saved -> {out_path.name}")
    print("\n--- forward / rate per expiry ---")
    print(out.groupby("expiry").agg(T=("T", "first"), F=("F", "first"),
                                    r=("r", "first"), n=("iv", "size")).round(4).to_string())
    print("\n--- ATM-ish IV (0.95<moneyness<1.05) by expiry ---")
    atm = out[(out.moneyness > 0.95) & (out.moneyness < 1.05)]
    print(atm.groupby("expiry").iv.mean().round(4).to_string())
    print("\n--- skew check: IV vs moneyness, nearest expiry ---")
    near = out[out.expiry == out.expiry.min()].sort_values("moneyness")
    print(near[["moneyness", "iv", "is_call"]].head(20).round(4).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
