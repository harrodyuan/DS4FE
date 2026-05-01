"""
Download mbp-10 data for NVDA — extended calm period (full October 2023).

Current file: lob_mbp10_NVDA_calm_oct2023.parquet  (Oct 2–12, 9 trading days)
New file:     lob_mbp10_NVDA_oct2023_full.parquet   (Oct 2–31, ~21 trading days)

Run:
    python download_mbp10_nvda_extended.py
"""

import databento as db
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("DATABENTO_API_KEY", "")
if not API_KEY:
    raise ValueError("Set DATABENTO_API_KEY in .env before running.")

SYMBOL   = "NVDA"
DATASET  = "XNAS.ITCH"
SCHEMA   = "mbp-10"
START    = "2023-10-02"
END      = "2023-11-01"   # exclusive — covers all of October (21 trading days)
OUT_PATH = "data/lob/lob_mbp10_NVDA_oct2023_full.parquet"

os.makedirs("data/lob", exist_ok=True)
client = db.Historical(API_KEY)

# Cost check before downloading
print("Checking cost ...")
cost = client.metadata.get_cost(
    dataset=DATASET, schema=SCHEMA,
    symbols=[SYMBOL], start=START, end=END,
    stype_in="raw_symbol"
)
print(f"Estimated cost: ${cost:.2f}")
confirm = input("Proceed? [y/N] ").strip().lower()
if confirm != "y":
    print("Aborted.")
    exit()

if os.path.exists(OUT_PATH):
    df_existing = pd.read_parquet(OUT_PATH)
    print(f"[skip] {OUT_PATH} already exists ({len(df_existing):,} rows)")
else:
    print(f"\nDownloading mbp-10 | {SYMBOL} | {START} → {END} ...")
    data = client.timeseries.get_range(
        dataset  = DATASET,
        schema   = SCHEMA,
        symbols  = [SYMBOL],
        start    = START,
        end      = END,
        stype_in = "raw_symbol",
    )
    df = data.to_df()
    df.to_parquet(OUT_PATH)
    idx = pd.DatetimeIndex(df.index)
    print(f"Saved {len(df):,} rows → {OUT_PATH}")
    print(f"Date range: {idx.min().date()} → {idx.max().date()}")
    print(f"Trading days: {df.index.normalize().nunique()}")
