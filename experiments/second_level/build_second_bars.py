"""
Build second-level OBI bars for NVDA, AAPL, MSFT (Oct 2-31 2023).

Aggregation rules (matching the minute-level notebook, scaled to seconds):
  - Within each second:           OBI_k = mean of all per-tick (bid_sz_k - ask_sz_k)/(bid_sz_k + ask_sz_k)
                                  for ticks landing in that second.
  - Seconds with NO orderbook update   forward-fill the last seen OBI vector
                                  (book state is unchanged until the next
                                   message, so the snapshot is the previous one).
  - Mid price aggregated as last-tick of the second, then forward-filled the
    same way.
  - Forward returns computed at three horizons: 1s, 10s, 60s.

Output: experiments/second_level/data/{sym}_second.parquet
"""
from __future__ import annotations
import os, time
import numpy as np
import pandas as pd

ROOT     = '/Users/harold/4. RA work/Factor_Training_Eng/DS4FE'
LOB_DIR  = f'{ROOT}/data/lob'
OUT_DIR  = f'{ROOT}/experiments/second_level/data'
os.makedirs(OUT_DIR, exist_ok=True)

DATA_PATHS = {
    'NVDA': f'{LOB_DIR}/lob_mbp10_NVDA_oct2023_full.parquet',
    'AAPL': f'{LOB_DIR}/lob_mbp10_AAPL_oct2023_full.parquet',
    'MSFT': f'{LOB_DIR}/lob_mbp10_MSFT_oct2023_full.parquet',
}

OBI_COLS = [f'obi_{k:02d}' for k in range(10)]


def build_second_bars(path: str, symbol: str) -> pd.DataFrame:
    """Read mbp-10 parquet, filter RTH, resample to 1-second OBI bars,
    forward-fill stale seconds, attach forward returns at 1s/10s/60s."""
    print(f'\n=== {symbol} ===')
    t0 = time.time()

    cols_needed = (
        [f'bid_sz_{k:02d}' for k in range(10)] +
        [f'ask_sz_{k:02d}' for k in range(10)] +
        ['bid_px_00', 'ask_px_00']
    )
    df = pd.read_parquet(path, columns=cols_needed)
    print(f'  loaded {len(df):>11,} ticks  ({time.time()-t0:.1f}s)')

    # NY time, RTH only
    df.index = pd.DatetimeIndex(df.index).tz_convert('America/New_York')
    df = df.between_time('09:30', '15:59:59')
    print(f'  RTH    {len(df):>11,} ticks')

    # Compute per-tick OBI for each level, then resample to 1s mean
    t1 = time.time()
    frames = {}
    for k in range(10):
        b = df[f'bid_sz_{k:02d}'].astype(np.int64)
        a = df[f'ask_sz_{k:02d}'].astype(np.int64)
        denom = (b + a).replace(0, np.nan)
        obi_tick = (b - a) / denom
        frames[f'obi_{k:02d}'] = obi_tick.resample('1s').mean()
    frames['mid'] = ((df['bid_px_00'] + df['ask_px_00']) / 2).resample('1s').last()
    out = pd.DataFrame(frames)
    print(f'  resampled to 1s  ({time.time()-t1:.1f}s)  -> {len(out):,} second-bars')

    # Restrict to RTH seconds (resample may extend outside)
    out = out.between_time('09:30', '15:59:59')

    # Sparsity stats BEFORE forward-fill
    n_total   = len(out)
    is_empty  = out[OBI_COLS].isna().all(axis=1)
    n_empty   = int(is_empty.sum())
    pct_empty = 100.0 * n_empty / max(n_total, 1)
    print(f'  empty seconds (no update): {n_empty:>9,} / {n_total:,}  ({pct_empty:.2f}%)')

    # Forward-fill: stale seconds inherit the last seen book state
    out[OBI_COLS] = out[OBI_COLS].ffill()
    out['mid']    = out['mid'].ffill()

    # Drop any leading NaNs that have no prior data to fill from
    out = out.dropna(subset=OBI_COLS + ['mid'])

    # Forward returns at 1s, 10s, 60s horizons (in fraction)
    out['ret_fwd_1s']  = out['mid'].pct_change(1).shift(-1)
    out['ret_fwd_10s'] = out['mid'].pct_change(10).shift(-10)
    out['ret_fwd_60s'] = out['mid'].pct_change(60).shift(-60)
    out['symbol'] = symbol
    out['was_empty'] = is_empty.reindex(out.index).fillna(False).values

    # Drop rows whose forward returns can't be computed (last 60 seconds of each day)
    out = out.dropna(subset=['ret_fwd_1s', 'ret_fwd_10s', 'ret_fwd_60s'])

    print(f'  final  {len(out):,} bars  ({time.time()-t0:.1f}s total)')
    return out


def main():
    for sym, path in DATA_PATHS.items():
        bars = build_second_bars(path, sym)
        out_path = f'{OUT_DIR}/{sym}_second.parquet'
        bars.to_parquet(out_path)
        size_mb = os.path.getsize(out_path) / 1e6
        print(f'  -> wrote {out_path}  ({size_mb:.1f} MB)')

    print('\nDone.')


if __name__ == '__main__':
    main()
