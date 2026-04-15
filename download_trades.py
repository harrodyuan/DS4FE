"""
DS4FE Trade Tape Download Script
=================================
Downloads tick-level trade data (Databento `trades` schema) for the same
5 symbols and date windows used for the LOB (mbp-1) data.

Trade data enables building internship features that require aggressor-side
information and VWAP, e.g.:
  - mid_vwap_bias / vwap_mid_diff
  - signed_volume_ratio / trade imbalance
  - directional_volume_value
  - net_volume_delta / net_volume_accel
  - liquidity_consumption_ratio

Date windows (same as LOB data):
  - Calm period:  2023-10-02  →  2023-10-13  (9 trading days)
  - Stress period: 2024-08-05  →  2024-08-09  (4 trading days, incl. BOJ shock)

Output files (data/lob/):
  trades_AAPL_calm_oct2023.parquet
  trades_AAPL_stress_aug2024.parquet
  ... (one file per symbol per period)

Run once:
    python download_trades.py
"""

import databento as db
import pandas as pd
import os

# ── Config ─────────────────────────────────────────────────────────────────────
API_KEY = os.environ.get("DATABENTO_API_KEY", "")
if not API_KEY:
    raise ValueError("Set DATABENTO_API_KEY environment variable before running.")

SYMBOLS  = ["NVDA", "AAPL", "TSLA", "MSFT", "SPY"]
DATASET  = "XNAS.ITCH"
SCHEMA   = "trades"
STY      = "continuous"          # stype_in — use symbol names directly

# Date windows must extend 1 day past the last day (Databento end is exclusive)
PERIODS = {
    "calm_oct2023"  : ("2023-10-02", "2023-10-13"),
    "stress_aug2024": ("2024-08-05", "2024-08-09"),
}

OUT_DIR = "data/lob"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Download ───────────────────────────────────────────────────────────────────
client = db.Historical(API_KEY)

for period_name, (start, end) in PERIODS.items():
    for sym in SYMBOLS:
        out_path = f"{OUT_DIR}/trades_{sym}_{period_name}.parquet"

        if os.path.exists(out_path):
            df_existing = pd.read_parquet(out_path)
            print(f"  [skip] {out_path} already exists ({len(df_existing):,} rows)")
            continue

        print(f"\nDownloading trades | {sym} | {period_name} | {start} → {end}")

        try:
            data = client.timeseries.get_range(
                dataset   = DATASET,
                schema    = SCHEMA,
                symbols   = [sym],
                start     = start,
                end       = end,
                stype_in  = "raw_symbol",
            )
            df = data.to_df()

            df.to_parquet(out_path)
            print(f"  Saved {len(df):,} rows → {out_path}")

        except Exception as e:
            print(f"  ERROR for {sym} {period_name}: {e}")

print("\nAll downloads complete.")
print("Files in data/lob/:")
for f in sorted(os.listdir(OUT_DIR)):
    if f.startswith("trades_"):
        path = os.path.join(OUT_DIR, f)
        df = pd.read_parquet(path)
        print(f"  {f}: {len(df):,} rows  |  {df.index[0]} to {df.index[-1]}")
