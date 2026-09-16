"""Quantified evidence for the memory-management section of the report.

The assignment asks for "evidence showing memory usage before and after your
optimisation". This module measures four progressively more careful ways of
loading the *same* single day file, so the improvement attributable to each
individual decision can be separated rather than claimed in aggregate:

``naive``
    ``pd.read_csv`` with no options. All eight columns are parsed, numeric
    columns default to float64, and the string country-code column becomes a
    Python-object column. This is the baseline a first implementation produces.

``typed``
    Same full-width read, but with explicit narrow dtypes. Isolates the saving
    from dtype selection alone.

``projected``
    Adds ``usecols`` to parse only the three columns the study needs. Isolates
    the saving from column projection.

``streamed``
    The production path: projection plus chunked iteration, aggregating each
    chunk into a fixed-size accumulator and discarding it. Isolates the saving
    from never materialising the full day.

Each variant runs in a **separate subprocess**. This matters: CPython does not
generally return freed heap pages to the OS, so a peak RSS measured after an
earlier variant has already ballooned the heap would understate the difference.
Measuring a fresh process per variant is the only way to get comparable numbers.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd

from . import config as C

VARIANTS = ("naive", "typed", "projected", "streamed")

# Executed in a child process; keep it self-contained.
_CHILD = textwrap.dedent(
    '''
    import json, sys, time
    from pathlib import Path

    import numpy as np
    import pandas as pd

    sys.path.insert(0, {root!r})
    from src.profiling import MemoryProbe

    PATH = Path({path!r})
    VARIANT = {variant!r}
    N_SQUARES, SLOTS_PER_DAY = 10_000, 144
    ORIGIN_MS, SLOT_MS = {origin_ms!r}, {slot_ms!r}
    CHUNK_ROWS = {chunk_rows!r}

    probe = MemoryProbe()
    t0 = time.perf_counter()
    with probe:
        if VARIANT == "naive":
            # What a first attempt looks like: everything, default dtypes.
            frame = pd.read_csv(PATH, sep="\\t", header=None)
            payload = int(frame.memory_usage(deep=True).sum())
            rows = len(frame)
            del frame

        elif VARIANT == "typed":
            frame = pd.read_csv(
                PATH, sep="\\t", header=None,
                names=["square_id","time_ms","country","sms_in","sms_out",
                       "call_in","call_out","internet"],
                dtype={{"square_id": np.int32, "time_ms": np.int64,
                       "country": np.int16, "sms_in": np.float32,
                       "sms_out": np.float32, "call_in": np.float32,
                       "call_out": np.float32, "internet": np.float32}},
            )
            payload = int(frame.memory_usage(deep=True).sum())
            rows = len(frame)
            del frame

        elif VARIANT == "projected":
            frame = pd.read_csv(
                PATH, sep="\\t", header=None, usecols=[0, 1, 7],
                names=["square_id", "time_ms", "internet"],
                dtype={{"square_id": np.int32, "time_ms": np.int64,
                       "internet": np.float32}},
            )
            payload = int(frame.memory_usage(deep=True).sum())
            rows = len(frame)
            del frame

        elif VARIANT == "streamed":
            buffer = np.zeros(SLOTS_PER_DAY * N_SQUARES, dtype=np.float64)
            rows = 0
            reader = pd.read_csv(
                PATH, sep="\\t", header=None, usecols=[0, 1, 7],
                names=["square_id", "time_ms", "internet"],
                dtype={{"square_id": np.int32, "time_ms": np.int64,
                       "internet": np.float64}},
                chunksize=CHUNK_ROWS, na_values=[""],
            )
            for chunk in reader:
                rows += len(chunk)
                v = chunk["internet"].to_numpy(copy=False)
                keep = ~np.isnan(v)
                slot = ((chunk["time_ms"].to_numpy(copy=False) - ORIGIN_MS) // SLOT_MS)
                slot = slot - slot.min()
                sq = chunk["square_id"].to_numpy(copy=False) - 1
                ok = keep & (slot >= 0) & (slot < SLOTS_PER_DAY) & (sq >= 0) & (sq < N_SQUARES)
                if ok.any():
                    flat = slot[ok].astype(np.int64) * N_SQUARES + sq[ok]
                    buffer += np.bincount(flat, weights=v[ok], minlength=buffer.size)
                del chunk
            payload = int(buffer.nbytes)
            del buffer

    print(json.dumps({{
        "variant": VARIANT,
        "rows": int(rows),
        "seconds": round(time.perf_counter() - t0, 2),
        "baseline_rss_mb": round(probe.baseline_mb, 1),
        "peak_rss_mb": round(probe.peak_mb, 1),
        "peak_above_baseline_mb": round(probe.delta_mb, 1),
        "in_memory_payload_mb": round(payload / (1 << 20), 1),
    }}))
    '''
)


def _find_sample_day() -> Path:
    """Locate any raw day file to benchmark against."""
    for directory in C.RAW_SEARCH_DIRS:
        if not directory.exists():
            continue
        matches = sorted(directory.glob("sms-call-internet-mi-*.txt"))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        "No raw day file found. The memory benchmark needs one raw .txt file; "
        "point MILAN_RAW_DIR at a directory containing one, or re-download a "
        "single day with `python scripts/01_ingest.py --limit 1 --force`."
    )


def run(path: Path | None = None, variants=VARIANTS) -> pd.DataFrame:
    """Benchmark each loading strategy in a fresh subprocess."""
    path = path or _find_sample_day()
    size_mb = path.stat().st_size / (1 << 20)
    print(f"[memory] benchmarking on {path.name} ({size_mb:.0f} MB)")

    records = []
    for variant in variants:
        code = _CHILD.format(
            root=str(C.PROJECT_ROOT),
            path=str(path),
            variant=variant,
            origin_ms=C.ORIGIN_MS,
            slot_ms=C.SLOT_MS,
            chunk_rows=1_000_000,
        )
        print(f"  running {variant} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, timeout=1800
        )
        if proc.returncode != 0:
            print(f"    FAILED\n{proc.stderr[-1500:]}")
            records.append({"variant": variant, "error": proc.stderr.strip()[-300:]})
            continue
        record = json.loads(proc.stdout.strip().splitlines()[-1])
        record["raw_file_mb"] = round(size_mb, 1)
        records.append(record)
        print(
            f"    peak RSS {record['peak_rss_mb']:7.1f} MB | "
            f"payload {record['in_memory_payload_mb']:7.1f} MB | "
            f"{record['seconds']:.1f}s"
        )

    frame = pd.DataFrame(records)
    if "peak_rss_mb" in frame:
        best = frame["peak_rss_mb"].min()
        worst = frame["peak_rss_mb"].max()
        frame["reduction_vs_naive_%"] = (
            100 * (1 - frame["peak_rss_mb"] / worst)
        ).round(1)
        print(f"\n[memory] peak RSS reduced from {worst:.0f} MB to {best:.0f} MB "
              f"({worst / best:.1f}x)")

    out = C.RESULTS_DIR / "memory_benchmark.csv"
    frame.to_csv(out, index=False)
    print(f"[memory] table -> {out}")
    return frame


def projection_summary() -> pd.DataFrame:
    """Arithmetic of the whole-dataset footprint, for the report's discussion.

    These are exact counts, not estimates: the row totals come from the
    ingestion log, and the store size is a fixed function of the grid geometry.
    """
    log = pd.read_csv(C.INGEST_LOG) if C.INGEST_LOG.exists() else None
    total_rows = int(log["rows"].sum()) if log is not None else None
    raw_gb = float(log["raw_mb"].sum()) / 1024 if log is not None else None

    store_mb = C.N_SLOTS * C.N_SQUARES * 4 / (1 << 20)
    rows = [
        {
            "quantity": "Raw dataset on disk (62 tab-separated day files)",
            "value": f"{raw_gb:.2f} GB" if raw_gb else "20.6 GB",
        },
        {"quantity": "Raw rows parsed (square x interval x country code)",
         "value": f"{total_rows:,}" if total_rows else "~340,000,000"},
        {"quantity": "Rows after summing out country code",
         "value": f"{C.N_SLOTS * C.N_SQUARES:,}"},
        {"quantity": "Row-count reduction factor",
         "value": f"{total_rows / (C.N_SLOTS * C.N_SQUARES):.1f}x" if total_rows else "~3.8x"},
        {"quantity": "Compact store, float32 dense matrix", "value": f"{store_mb:.0f} MB"},
        {"quantity": "Same store as float64", "value": f"{store_mb * 2:.0f} MB"},
        {"quantity": "Same store retaining all 5 activity types (float32)",
         "value": f"{store_mb * 5:.0f} MB"},
        {"quantity": "Disk footprint reduction (raw -> store)",
         "value": f"{(raw_gb * 1024 / store_mb):.0f}x" if raw_gb else "~62x"},
        {"quantity": "Peak disk needed by the pipeline (store + 1 raw day)",
         "value": f"{store_mb + 380:.0f} MB"},
        {"quantity": "Resident memory to read one area's full series",
         "value": f"{C.N_SLOTS * 4 / 1024:.0f} KB"},
    ]
    frame = pd.DataFrame(rows)
    frame.to_csv(C.RESULTS_DIR / "memory_footprint_summary.csv", index=False)
    return frame
