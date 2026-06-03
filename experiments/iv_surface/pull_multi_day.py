"""
MULTI-DAY SPX IV-surface pull (BILLABLE).

Loops over NYSE trading days in [START, END], pulls the 15:45-ET definition +
cbbo-1m snapshot for each day, computes the IV table, and caches everything to
data/iv/. Designed to be safely resumable: any day already cached is skipped, so
re-running never re-downloads (or re-pays for) the same day.

Cost guidance: ~$0.017 per trading day (~$8-10 for two years).

Usage:
    python3 pull_multi_day.py --start 2021-06-01 --end 2023-06-30
    python3 pull_multi_day.py --start 2021-06-01 --end 2023-06-30 --dry-run
"""
import argparse
import pathlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import pandas as pd

from iv_lib import build_iv_table, load_env

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "data" / "iv"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASET = "OPRA.PILLAR"
PARENT = "SPX.OPT"
ET = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")
SNAP_HOUR, SNAP_MIN = 15, 45        # 15:45 ET


def snap_window_utc(day: pd.Timestamp):
    """Return (start_utc_iso, end_utc_iso) for the 15:45-15:46 ET minute, DST-aware."""
    start_et = pd.Timestamp(day.year, day.month, day.day, SNAP_HOUR, SNAP_MIN, tz=ET)
    end_et = start_et + pd.Timedelta(minutes=1)
    fmt = "%Y-%m-%dT%H:%M:%S"
    return (start_et.astimezone(UTC).strftime(fmt), end_et.astimezone(UTC).strftime(fmt))


def trading_days(start, end):
    """NYSE sessions if pandas_market_calendars is available, else business days."""
    try:
        import pandas_market_calendars as mcal
        sched = mcal.get_calendar("XNYS").schedule(start_date=start, end_date=end)
        return [pd.Timestamp(d).normalize() for d in sched.index]
    except Exception:
        return list(pd.bdate_range(start, end))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--dry-run", action="store_true",
                    help="List days + estimate cost; download nothing.")
    ap.add_argument("--workers", type=int, default=6,
                    help="Parallel download workers (cost is unchanged by speed).")
    args = ap.parse_args()

    load_env(ROOT / ".env")
    days = trading_days(args.start, args.end)
    todo = [d for d in days if not (OUT_DIR / f"iv_{d.date()}.parquet").exists()]
    print(f"sessions in range : {len(days)}")
    print(f"already cached    : {len(days) - len(todo)}")
    print(f"to download       : {len(todo)}")
    print(f"est. cost         : ~${0.017 * len(todo):.2f}  (@ ~$0.017/day)")

    if args.dry_run or not todo:
        print("\n(dry-run / nothing to do — no data downloaded)")
        return 0

    import databento as db
    import os
    client = db.Historical(os.environ["DATABENTO_API_KEY"])

    counter = {"ok": 0, "fail": 0, "empty": 0, "done": 0}
    lock = threading.Lock()
    t0 = time.time()
    n = len(todo)

    def process(day):
        ds = str(day.date())
        def_path = OUT_DIR / f"def_{ds}.parquet"
        snap_path = OUT_DIR / f"cbbo_{ds}_1545.parquet"
        iv_path = OUT_DIR / f"iv_{ds}.parquet"
        s_utc, e_utc = snap_window_utc(day)
        try:
            if def_path.exists():
                defs = pd.read_parquet(def_path)
            else:
                defs = client.timeseries.get_range(
                    dataset=DATASET, symbols=PARENT, stype_in="parent",
                    schema="definition", start=f"{ds}T00:00:00", end=f"{ds}T23:59:00",
                ).to_df()
                defs.to_parquet(def_path)

            if snap_path.exists():
                snap = pd.read_parquet(snap_path)
            else:
                snap = client.timeseries.get_range(
                    dataset=DATASET, symbols=PARENT, stype_in="parent",
                    schema="cbbo-1m", start=s_utc, end=e_utc,
                ).to_df()
                snap.to_parquet(snap_path)

            if len(snap) == 0 or len(defs) == 0:
                return ds, "empty", 0, 0

            out = build_iv_table(defs, snap, pd.Timestamp(day))
            out.to_parquet(iv_path)
            return ds, "ok", len(out), out.expiry.nunique()
        except Exception as ex:  # noqa: BLE001
            return ds, f"error: {ex}", 0, 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, d): d for d in todo}
        for fut in as_completed(futs):
            ds, status, ncon, nexp = fut.result()
            with lock:
                counter["done"] += 1
                i = counter["done"]
                if status == "ok":
                    counter["ok"] += 1
                elif status == "empty":
                    counter["empty"] += 1
                else:
                    counter["fail"] += 1
                rate = (time.time() - t0) / i
                eta = rate * (n - i) / 60
                if status == "ok":
                    print(f"  [{i}/{n}] {ds}  IV={ncon:,} exp={nexp}  (ETA {eta:.1f} min)")
                elif status == "empty":
                    print(f"  [{i}/{n}] {ds}  EMPTY (holiday) — skipped")
                else:
                    print(f"  [{i}/{n}] {ds}  {status}")

    print(f"\ndone: {counter['ok']} ok, {counter['empty']} empty, "
          f"{counter['fail']} failed, total {(time.time()-t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
