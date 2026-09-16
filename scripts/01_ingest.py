"""Stage 1 - build the compact activity store from the raw day files.

Usage
-----
    python scripts/01_ingest.py                # process every missing day
    python scripts/01_ingest.py --limit 3      # smoke test on three days
    python scripts/01_ingest.py --force        # re-process everything

The job is resumable: interrupt it at any point and re-run to continue.
"""

import argparse

import _bootstrap  # noqa: F401  (side effect: sys.path)

from src import config as C
from src.ingest import ingest
from src.profiling import system_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None, help="process at most N days")
    parser.add_argument("--force", action="store_true", help="ignore the manifest")
    parser.add_argument(
        "--days",
        default=None,
        help="comma-separated YYYY-MM-DD dates to (re)process, e.g. 2013-11-03,2013-11-04",
    )
    args = parser.parse_args()
    only = [d.strip() for d in args.days.split(",")] if args.days else None

    print("[ingest] host:", system_summary())
    print(f"[ingest] target store: {C.INTERNET_STORE}")
    print(f"[ingest] shape: ({C.N_SLOTS}, {C.N_SQUARES}) float32 "
          f"= {C.N_SLOTS * C.N_SQUARES * 4 / (1 << 20):.0f} MB")

    frame = ingest(force=args.force, limit=args.limit, only=only)

    if frame.empty:
        print("[ingest] nothing to do.")
        return

    print("\n[ingest] summary of this run")
    print(f"  days processed      : {len(frame)}")
    print(f"  raw bytes streamed  : {frame['raw_mb'].sum() / 1024:.2f} GB")
    print(f"  rows parsed         : {frame['rows'].sum():,}")
    print(f"  rows out of range   : {frame['rows_out_of_range'].sum():,}")
    print(f"  mean parse time     : {frame['parse_seconds'].mean():.1f} s/day")
    print(f"  max peak RSS        : {frame['peak_rss_mb'].max():.0f} MB")
    print(f"  log written to      : {C.INGEST_LOG}")


if __name__ == "__main__":
    main()
