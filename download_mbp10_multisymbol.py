"""
Download mbp-10 data for multi-symbol LOB manifold analysis (Part 4g).

Symbols : AAPL, MSFT, SPY, TSLA  (NVDA calm already exists)
Period  : 2023-10-02 → 2023-10-13  (same calm window as NVDA Part 4f)
Schema  : mbp-10 (10-level order book snapshots)
Dataset : XNAS.ITCH

Output files saved to data/lob/:
  lob_mbp10_AAPL_calm_oct2023.parquet
  lob_mbp10_MSFT_calm_oct2023.parquet
  lob_mbp10_SPY_calm_oct2023.parquet
  lob_mbp10_TSLA_calm_oct2023.parquet

Run:
    python download_mbp10_multisymbol.py
"""

import databento as db
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("DATABENTO_API_KEY", "")
if not API_KEY:
    raise ValueError("Set DATABENTO_API_KEY in .env before running.")

SYMBOLS  = ["AAPL", "MSFT", "SPY", "TSLA"]   # NVDA already downloaded
DATASET  = "XNAS.ITCH"
SCHEMA   = "mbp-10"
START    = "2023-10-02"
END      = "2023-10-13"   # exclusive — covers Oct 2–12 (9 trading days)
OUT_DIR  = "data/lob"

os.makedirs(OUT_DIR, exist_ok=True)
client = db.Historical(API_KEY)

for sym in SYMBOLS:
    out_path = f"{OUT_DIR}/lob_mbp10_{sym}_calm_oct2023.parquet"

    if os.path.exists(out_path):
        df_existing = pd.read_parquet(out_path)
        print(f"[skip] {out_path} already exists ({len(df_existing):,} rows)")
        continue

    print(f"\nDownloading mbp-10 | {sym} | {START} → {END} ...")
    try:
        data = client.timeseries.get_range(
            dataset  = DATASET,
            schema   = SCHEMA,
            symbols  = [sym],
            start    = START,
            end      = END,
            stype_in = "raw_symbol",
        )
        df = data.to_df()
        df.to_parquet(out_path)
        print(f"  Saved {len(df):,} rows → {out_path}")
    except Exception as e:
        print(f"  ERROR for {sym}: {e}")

print("\nAll downloads complete.")
for sym in SYMBOLS:
    path = f"{OUT_DIR}/lob_mbp10_{sym}_calm_oct2023.parquet"
    if os.path.exists(path):
        df = pd.read_parquet(path)
        idx = pd.DatetimeIndex(df.index)
        print(f"  {sym}: {len(df):,} rows | {idx.min().date()} → {idx.max().date()}")
