"""Stage 2 - quantify the memory cost of four loading strategies.

Produces the "before and after optimisation" evidence for the report:
``results/memory_benchmark.csv`` and ``results/memory_footprint_summary.csv``.

Requires one raw ``.txt`` day file to be present. If the ingestion stage has
already deleted them all, fetch one back with::

    python scripts/01_ingest.py --limit 1 --force
"""

import _bootstrap  # noqa: F401

from src import config as C
from src.memory_benchmark import projection_summary, run
from src.profiling import system_summary


def main() -> None:
    print("[memory] host:", system_summary(), "\n")
    table = run()

    print("\n[memory] per-strategy results")
    print(table.to_string(index=False))

    print("\n[memory] whole-dataset footprint arithmetic")
    print(projection_summary().to_string(index=False))
    print(f"\n[memory] tables written to {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
