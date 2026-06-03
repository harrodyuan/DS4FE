"""
Build fixed-interval (default 5s) LOB state features from MBP-1 + trades.

For each (symbol, period) we read the event-level MBP-1 parquet and the trades
parquet in memory-safe pyarrow batches (SPY files are ~1 GB), aggregate to bars
restricted to US regular trading hours (13:30-20:00 UTC; Oct-2023 and Aug-2024
are both EDT), then compute the contemporaneous state features and the
forward-looking liquidity-depletion targets.

Output: data/lob/features/feat_{symbol}_{period}_{bar}.parquet (one row per bar).

Run:  python experiments/lob_manifold/build_features.py
"""
from __future__ import annotations

import os
from datetime import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOB_DIR = os.path.join(ROOT, "data", "lob")
OUT_DIR = os.path.join(LOB_DIR, "features")
os.makedirs(OUT_DIR, exist_ok=True)

SYMBOLS = ["NVDA", "AAPL", "MSFT", "SPY", "TSLA"]
PERIODS = {"calm": "calm_oct2023", "stress": "stress_aug2024"}
BAR = "5s"
RTH_START = time(13, 30)   # 09:30 ET in UTC (EDT)
RTH_END = time(20, 0)      # 16:00 ET in UTC (EDT)
BATCH = 1_000_000

BOOK_COLS = [
    "ts_event", "action", "side", "size",
    "bid_px_00", "ask_px_00", "bid_sz_00", "ask_sz_00", "bid_ct_00", "ask_ct_00",
]
TRADE_COLS = ["ts_event", "side", "size"]


def _rth(df: pd.DataFrame) -> pd.DataFrame:
    t = df["ts_event"].dt.tz_convert("UTC").dt.time
    return df[(t >= RTH_START) & (t < RTH_END)]


def build_book_bars(path: str, bar: str = BAR) -> pd.DataFrame:
    """Aggregate event-level MBP-1 to per-bar book state + book-flow features."""
    pf = pq.ParquetFile(path)
    state_parts, flow_parts = [], []
    for batch in pf.iter_batches(batch_size=BATCH, columns=BOOK_COLS):
        df = batch.to_pandas()
        df = _rth(df)
        if df.empty:
            continue
        df["bar"] = df["ts_event"].dt.floor(bar)

        # --- book-update flow (Add / Cancel only; data has no Modify) ---
        bk = df[df["action"].isin(["A", "C"])].copy()
        if not bk.empty:
            side_sgn = np.where(bk["side"] == "B", 1.0,
                                np.where(bk["side"] == "A", -1.0, 0.0))
            act_sgn = np.where(bk["action"] == "A", 1.0, -1.0)
            sz = bk["size"].astype(float).to_numpy()
            bk["signed_book_flow"] = side_sgn * act_sgn * sz
            bk["abs_book_flow"] = sz
            flow = bk.groupby("bar").agg(
                book_event_count=("size", "size"),
                signed_book_flow=("signed_book_flow", "sum"),
                abs_book_flow=("abs_book_flow", "sum"),
            )
            flow_parts.append(flow)

        # --- end-of-bar book snapshot (each row carries post-event top-of-book) ---
        st = df.groupby("bar").agg(
            bid_px=("bid_px_00", "last"), ask_px=("ask_px_00", "last"),
            bid_sz=("bid_sz_00", "last"), ask_sz=("ask_sz_00", "last"),
            bid_ct=("bid_ct_00", "last"), ask_ct=("ask_ct_00", "last"),
            _lastts=("ts_event", "last"),
        )
        state_parts.append(st)

    if not state_parts:
        return pd.DataFrame()

    flow = (pd.concat(flow_parts).groupby("bar").sum()
            if flow_parts else pd.DataFrame())
    state = (pd.concat(state_parts).sort_values("_lastts")
             .groupby("bar").last().drop(columns="_lastts"))
    out = state.join(flow, how="left")
    for c in ["book_event_count", "signed_book_flow", "abs_book_flow"]:
        if c not in out:
            out[c] = 0.0
    out[["book_event_count", "signed_book_flow", "abs_book_flow"]] = \
        out[["book_event_count", "signed_book_flow", "abs_book_flow"]].fillna(0.0)
    return out


def build_trade_bars(path: str, bar: str = BAR) -> pd.DataFrame:
    """Aggregate trades to per-bar trade-flow features."""
    pf = pq.ParquetFile(path)
    parts = []
    for batch in pf.iter_batches(batch_size=BATCH, columns=TRADE_COLS):
        df = batch.to_pandas()
        df = _rth(df)
        if df.empty:
            continue
        df["bar"] = df["ts_event"].dt.floor(bar)
        sgn = np.where(df["side"] == "B", 1.0,
                       np.where(df["side"] == "A", -1.0, 0.0))
        df["signed_trade_volume"] = sgn * df["size"].astype(float)
        g = df.groupby("bar").agg(
            trade_count=("size", "size"),
            trade_volume=("size", "sum"),
            signed_trade_volume=("signed_trade_volume", "sum"),
        )
        parts.append(g)
    if not parts:
        return pd.DataFrame(columns=["trade_count", "trade_volume", "signed_trade_volume"])
    return pd.concat(parts).groupby("bar").sum()


def _fwd(series: pd.Series, horizon: int, how: str) -> pd.Series:
    """Aggregate over the *next* `horizon` bars (t+1 .. t+horizon)."""
    nxt = series.shift(-1)
    rev = nxt[::-1].rolling(horizon, min_periods=1)
    agg = rev.min() if how == "min" else rev.max()
    return agg[::-1]


def derive_features(book: pd.DataFrame, trade: pd.DataFrame, bar: str = BAR) -> pd.DataFrame:
    df = book.join(trade, how="left")
    for c in ["trade_count", "trade_volume", "signed_trade_volume"]:
        if c not in df:
            df[c] = 0.0
    df[["trade_count", "trade_volume", "signed_trade_volume"]] = \
        df[["trade_count", "trade_volume", "signed_trade_volume"]].fillna(0.0)

    df = df.reset_index().rename(columns={"index": "bar"})
    # cast unsigned-int snapshot columns to float to avoid wraparound in diffs/ratios
    for c in ["bid_sz", "ask_sz", "bid_ct", "ask_ct"]:
        df[c] = df[c].astype("float64")
    df = df.dropna(subset=["bid_px", "ask_px"])
    df = df[(df["bid_sz"] > 0) & (df["ask_sz"] > 0)]
    df = df[df["ask_px"] >= df["bid_px"]].reset_index(drop=True)

    df["mid"] = 0.5 * (df["bid_px"] + df["ask_px"])
    df["spread"] = df["ask_px"] - df["bid_px"]
    df["rel_spread"] = df["spread"] / df["mid"]
    df["top_depth"] = df["bid_sz"] + df["ask_sz"]
    df["obi"] = (df["bid_sz"] - df["ask_sz"]) / df["top_depth"]
    df["trade_imbalance"] = np.where(
        df["trade_volume"] > 0, df["signed_trade_volume"] / df["trade_volume"], 0.0)

    df["date"] = df["bar"].dt.date

    bars_per_min = int(round(60 / pd.Timedelta(bar).total_seconds()))
    h30 = max(1, bars_per_min // 2)   # 30s
    h60 = bars_per_min                # 60s

    def per_day(g: pd.DataFrame) -> pd.DataFrame:
        g = g.sort_values("bar").copy()
        logmid = np.log(g["mid"])
        g["short_return"] = logmid.diff().fillna(0.0)
        g["realized_vol_60s"] = (g["short_return"].rolling(h60, min_periods=2)
                                 .std().bfill().fillna(0.0))
        g["realized_vol_30s"] = (g["short_return"].rolling(h30, min_periods=2)
                                 .std().bfill().fillna(0.0))
        g["spread_change"] = g["spread"].diff().fillna(0.0)
        g["depth_change"] = g["top_depth"].diff().fillna(0.0)
        # forward-looking raw quantities for labels
        g["fwd_min_depth_30s"] = _fwd(g["top_depth"], h30, "min")
        g["fwd_max_spread_30s"] = _fwd(g["spread"], h30, "max")
        g["fwd_max_rv_60s"] = _fwd(g["realized_vol_60s"], h60, "max")
        return g

    df = df.groupby("date", group_keys=False).apply(per_day)
    return df.reset_index(drop=True)


def build_one(symbol: str, period_key: str, bar: str = BAR) -> pd.DataFrame:
    suffix = PERIODS[period_key]
    book_path = os.path.join(LOB_DIR, f"lob_mbp1_{symbol}_{suffix}.parquet")
    trade_path = os.path.join(LOB_DIR, f"trades_{symbol}_{suffix}.parquet")
    print(f"  reading book  {os.path.basename(book_path)}", flush=True)
    book = build_book_bars(book_path, bar)
    print(f"  reading trades {os.path.basename(trade_path)}", flush=True)
    trade = build_trade_bars(trade_path, bar)
    df = derive_features(book, trade, bar)
    df["symbol"] = symbol
    df["period"] = period_key
    return df


def main():
    for sym in SYMBOLS:
        for pk in PERIODS:
            out_path = os.path.join(OUT_DIR, f"feat_{sym}_{pk}_{BAR}.parquet")
            if os.path.exists(out_path):
                print(f"[skip] {os.path.basename(out_path)} exists", flush=True)
                continue
            print(f"[build] {sym} {pk}", flush=True)
            df = build_one(sym, pk, BAR)
            df.to_parquet(out_path, index=False)
            print(f"  -> {len(df):,} bars saved to {os.path.basename(out_path)}",
                  flush=True)


if __name__ == "__main__":
    main()
