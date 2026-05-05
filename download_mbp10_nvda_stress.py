"""
Download mbp-10 data for NVDA — BOJ shock stress week (Aug 5–9, 2024).

Aug 5 2024: VIX hit ~65 intraday, NVDA fell ~$15 on the day.
Aug 5–9 covers the full shock and partial recovery — 5 trading days.

Run:
    python download_mbp10_nvda_stress.py
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
START    = "2024-08-05"
END      = "2024-08-10"   # exclusive — covers Aug 5–9 (5 trading days)
OUT_PATH = "data/lob/lob_mbp10_NVDA_stress_aug2024.parquet"

os.makedirs("data/lob", exist_ok=True)

if os.path.exists(OUT_PATH):
    df = pd.read_parquet(OUT_PATH)
    idx = pd.DatetimeIndex(df.index)
    print(f"[skip] {OUT_PATH} already exists")
    print(f"  {len(df):,} rows | {idx.min().date()} → {idx.max().date()}")
else:
    client = db.Historical(API_KEY)
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
