"""
Reusable IV-surface helpers shared by the one-day and multi-day pull scripts.

Validated against Databento OPRA.PILLAR SPX options (2023-06-01):
  - .to_df() returns prices already in dollars (no /1e9 scaling)
  - forward + discount per expiry via put-call parity: C - P = exp(-rT)(F - K)
  - SPX options are European -> clean Black-76 inversion
"""
import os
import pathlib

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm


def load_env(env_path: pathlib.Path) -> None:
    """Minimal .env loader (KEY=VALUE per line) into os.environ."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def black76_price(F, K, T, sigma, r, is_call):
    if sigma <= 0 or T <= 0:
        intrinsic = max(F - K, 0.0) if is_call else max(K - F, 0.0)
        return np.exp(-r * T) * intrinsic
    sq = sigma * np.sqrt(T)
    d1 = (np.log(F / K) + 0.5 * sigma**2 * T) / sq
    d2 = d1 - sq
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


def build_iv_table(defs: pd.DataFrame, snap: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Join definition + snapshot, compute forward/rate per expiry, invert to IV."""
    dmap = defs[["instrument_id", "raw_symbol", "strike_price",
                 "expiration", "instrument_class"]].drop_duplicates("instrument_id").copy()
    dmap["strike"] = dmap["strike_price"]                # dollars (to_df already scaled)
    dmap["is_call"] = dmap["instrument_class"].astype(str).str.upper().str[0] == "C"
    dmap["expiry"] = pd.to_datetime(dmap["expiration"]).dt.tz_localize(None).dt.normalize()

    snap = snap.sort_index()
    px = snap.groupby("instrument_id").last().reset_index()
    bid_col = "bid_px_00" if "bid_px_00" in px.columns else "bid_px"
    ask_col = "ask_px_00" if "ask_px_00" in px.columns else "ask_px"
    px = px[[ "instrument_id", bid_col, ask_col]].rename(columns={bid_col: "bid", ask_col: "ask"})

    df = px.merge(dmap, on="instrument_id", how="inner")
    df = df[(df.bid > 0) & (df.ask > 0) & (df.ask >= df.bid)].copy()
    df["mid"] = 0.5 * (df.bid + df.ask)
    df["T"] = (df.expiry - asof).dt.days / 365.25
    df = df[df["T"] > 0]

    rows = []
    for exp, g in df.groupby("expiry"):
        piv = g.pivot_table(index="strike", columns="is_call", values="mid")
        if True not in piv.columns or False not in piv.columns:
            continue
        pair = piv.dropna()
        if len(pair) < 4:
            continue
        K = pair.index.values.astype(float)
        diff = (pair[True] - pair[False]).values
        b, a = np.polyfit(K, diff, 1)
        if b >= 0:
            continue
        disc = -b
        if disc <= 0:
            continue
        F = a / disc
        T = float(g["T"].iloc[0])
        r = -np.log(disc) / T
        for _, row in g.iterrows():
            iv = implied_vol(row.mid, F, row.strike, T, r, row.is_call)
            rows.append({
                "expiry": exp, "T": T, "strike": float(row.strike), "F": F, "r": r,
                "moneyness": row.strike / F, "is_call": bool(row.is_call),
                "mid": float(row.mid), "iv": iv,
            })

    out = pd.DataFrame(rows)
    if len(out):
        out = out[out.iv.notna() & (out.iv > 0.01) & (out.iv < 2.0)].reset_index(drop=True)
    return out
