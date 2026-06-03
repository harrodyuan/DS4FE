"""
Databento connectivity + FREE cost-estimate checks for the IV-surface pivot.

This script makes ONLY free metadata calls:
  - list available date range for OPRA.PILLAR
  - estimate (get_cost) the data size/price for a 1-day SPX pull

It does NOT download any billable data. Run before any real pull.
"""
import os
import pathlib


def load_env(env_path: pathlib.Path) -> None:
    """Minimal .env loader (KEY=VALUE per line) into os.environ."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def main() -> int:
    root = pathlib.Path(__file__).resolve().parents[2]
    load_env(root / ".env")

    key = os.environ.get("DATABENTO_API_KEY")
    if not key:
        print("ERROR: DATABENTO_API_KEY not found in environment or .env")
        return 1
    print(f"key loaded: yes (len={len(key)}, prefix={key[:4]}...)")

    import databento as db

    client = db.Historical(key)

    DATASET = "OPRA.PILLAR"

    # --- free: available date range for the dataset ---
    rng = client.metadata.get_dataset_range(dataset=DATASET)
    print(f"\n{DATASET} available range: {rng}")

    # --- free: cost estimate for ONE day of SPX options ---
    # Note: OPRA timestamps are UTC. 15:45 ET on 2023-06-01 (EDT, UTC-4) = 19:45 UTC.
    queries = [
        # (label, schema, start, end)
        ("definition (full day)", "definition",
         "2023-06-01T00:00:00", "2023-06-02T00:00:00"),
        ("cbbo-1m (full day)", "cbbo-1m",
         "2023-06-01T00:00:00", "2023-06-02T00:00:00"),
        ("cbbo-1m (15:45 window)", "cbbo-1m",
         "2023-06-01T19:45:00", "2023-06-01T19:46:00"),
    ]

    for label, schema, start, end in queries:
        try:
            cost = client.metadata.get_cost(
                dataset=DATASET, symbols="SPX.OPT", stype_in="parent",
                schema=schema, start=start, end=end,
            )
            size = client.metadata.get_record_count(
                dataset=DATASET, symbols="SPX.OPT", stype_in="parent",
                schema=schema, start=start, end=end,
            )
            print(f"\n{label:26s}  est_cost=${cost:.4f}  est_records={size:,}")
        except Exception as e:  # noqa: BLE001
            print(f"\n{label:26s}  ERROR: {e}")

    print("\nNOTE: these are FREE estimates. No data was downloaded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
