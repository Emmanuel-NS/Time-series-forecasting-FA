"""Memory-bounded ingestion of the Milan telecommunications dataset.

Problem
-------
The full dataset is 62 tab-separated text files (one per day) totalling
approximately 20.6 GB. Each file holds one row per
``(square_id, 10-minute interval, country_code)`` combination, i.e. roughly
5-6 million rows per day and ~340 million rows overall. Loading even a single
day with ``pandas.read_csv`` default settings materialises eight float64
columns plus a row index; loading all of them at once is impossible on a
commodity laptop.

Strategy
--------
The pipeline never holds more than one chunk of one day in memory:

1. **Aggregate away the dimension we do not need.** The forecasting task is
   defined per geographical area, not per country code, so country code is
   summed out during ingestion. This alone collapses ~340 million rows to
   ``62 x 144 x 10000 = 89.28`` million cells.
2. **Project to the target column.** Only ``square_id``, ``time_interval`` and
   ``internet`` are parsed (``usecols``); the four SMS/call columns are never
   allocated.
3. **Chunked streaming parse.** Each day is read in fixed-size row chunks, so
   peak resident memory is a function of the chunk size, not the file size.
4. **Dense, compact, memory-mapped output.** Because the grid is complete and
   regular, the aggregate is naturally a dense ``(n_slots, n_squares)`` matrix.
   Stored as ``float32`` this is 357 MB on disk and is accessed later through
   ``numpy.memmap``, so downstream analysis pages in only the columns it uses.
5. **Download-process-discard.** Raw files are fetched one at a time and
   deleted immediately after aggregation, bounding peak disk use at
   ``store size + one raw file`` (~750 MB) rather than 20.6 GB.

Accumulation is performed in float64 and cast to float32 only when the day is
complete, so that summing many small activity values does not accumulate
float32 rounding error.

The job is idempotent and resumable: completed days are recorded in a manifest
and skipped on subsequent runs.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent import futures
from dataclasses import dataclass, asdict
from datetime import timedelta
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from . import config as C
from .profiling import MemoryProbe

# Rows may carry a timestamp slightly outside the nominal day (the published
# files are cut on server-side boundaries). A four-hour margin on each side is
# generous enough to absorb this while keeping the per-day buffer small.
SLOT_MARGIN = 24

#: Rows per chunk. 1e6 rows x 3 columns x 8 bytes ~= 24 MB of payload, which
#: keeps peak RSS comfortably inside a 1 GB budget while still amortising the
#: per-chunk overhead of the C parser.
CHUNK_ROWS = 1_000_000

#: Concurrent HTTP range requests per file. Harvard Dataverse throttles each
#: individual connection to roughly 0.17 MB/s for this dataset while the host
#: link sustains >5 MB/s, so throughput is limited by concurrency rather than
#: bandwidth. Eight is a deliberate compromise: it recovers most of the
#: available bandwidth without behaving abusively towards a public archive.
DOWNLOAD_WORKERS = int(os.environ.get("MILAN_DOWNLOAD_WORKERS", "8"))

#: Bytes per range request. 16 MB keeps ``workers x segment_bytes`` (128 MB)
#: well inside the memory budget while amortising per-request latency.
SEGMENT_BYTES = 16 << 20


# ---------------------------------------------------------------------------
# Manifest / bookkeeping
# ---------------------------------------------------------------------------


@dataclass
class DayStat:
    """Per-day ingestion record, written to ``results/ingest_log.csv``."""

    date: str
    raw_mb: float
    rows: int
    rows_out_of_range: int
    nonzero_cells: int
    parse_seconds: float
    peak_rss_mb: float
    downloaded: bool


def _day_dates() -> list[str]:
    return [
        (C.FIRST_DAY + timedelta(days=i)).strftime("%Y-%m-%d")
        for i in range(C.N_DAYS)
    ]


def _load_meta() -> dict:
    if C.STORE_META.exists():
        return json.loads(C.STORE_META.read_text())
    return {
        "n_slots": C.N_SLOTS,
        "n_squares": C.N_SQUARES,
        "dtype": "float32",
        "origin_ms": C.ORIGIN_MS,
        "slot_minutes": C.SLOT_MINUTES,
        "completed_days": [],
    }


def _save_meta(meta: dict) -> None:
    C.STORE_META.write_text(json.dumps(meta, indent=2))


def open_store(mode: str = "r") -> np.memmap:
    """Open the compact activity store as a ``(n_slots, n_squares)`` memmap."""
    if mode in ("r", "r+") and not C.INTERNET_STORE.exists():
        raise FileNotFoundError(
            f"{C.INTERNET_STORE} not found - run `python -m scripts.01_ingest` first."
        )
    return np.memmap(
        C.INTERNET_STORE,
        dtype=np.float32,
        mode=mode,
        shape=(C.N_SLOTS, C.N_SQUARES),
    )


def _ensure_store() -> np.memmap:
    """Create the zero-filled store on first use, then return it read-write."""
    if not C.INTERNET_STORE.exists():
        store = np.memmap(
            C.INTERNET_STORE,
            dtype=np.float32,
            mode="w+",
            shape=(C.N_SLOTS, C.N_SQUARES),
        )
        store.flush()
        del store
    return open_store("r+")


# ---------------------------------------------------------------------------
# Raw file acquisition
# ---------------------------------------------------------------------------


#: Harvard Dataverse rejects requests carrying the default ``Python-urllib``
#: agent with HTTP 403, so an explicit agent string is required.
HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) milan-traffic-forecasting/1.0 "
        "(academic research; contact via repository)"
    ),
    "Accept": "*/*",
}


def _request(url: str, extra_headers: dict[str, str] | None = None) -> urllib.request.Request:
    """Build an authenticated Dataverse request.

    The dataset carries a download guestbook, so an unauthenticated request is
    rejected with HTTP 400 ("You may not download this file without the required
    Guestbook response"). Supplying ``X-Dataverse-key`` identifies the caller,
    and Dataverse then records the guestbook entry on their behalf.
    """
    headers = dict(HTTP_HEADERS)
    if C.DATAVERSE_API_TOKEN:
        headers["X-Dataverse-key"] = C.DATAVERSE_API_TOKEN
    headers.update(extra_headers or {})

    request = urllib.request.Request(url)
    for key, value in headers.items():
        request.add_header(key, value)
    return request


def _dataverse_catalogue() -> dict[str, dict]:
    """Map ``filename -> {"id", "size"}`` for the dataset, cached on disk."""
    if C.FILE_ID_CACHE.exists():
        return json.loads(C.FILE_ID_CACHE.read_text())

    with urllib.request.urlopen(_request(C.DATAVERSE_FILE_LIST), timeout=120) as resp:
        payload = json.load(resp)

    catalogue = {
        f["dataFile"]["filename"]: {
            "id": int(f["dataFile"]["id"]),
            "size": int(f["dataFile"]["filesize"]),
        }
        for f in payload["data"]["latestVersion"]["files"]
    }
    C.FILE_ID_CACHE.write_text(json.dumps(catalogue, indent=2))
    return catalogue


def _find_local(filename: str) -> Path | None:
    for directory in C.RAW_SEARCH_DIRS:
        candidate = directory / filename
        if candidate.exists() and candidate.stat().st_size > 0:
            return candidate
    return None


class _SignedUrlPool:
    """Issues and refreshes guestbook-signed download URLs for one file.

    Harvard Dataverse protects this dataset with a download guestbook
    ("Privacy risk assessment", guestbook id 96). Since Dataverse 6.10 the plain
    ``GET /api/access/datafile/{id}`` endpoint returns HTTP 400 unless a
    guestbook response is supplied, *even for an authenticated user*. The
    supported programmatic route is to POST the response to the same endpoint
    with ``?signed=true``, which records the guestbook entry and returns a
    time-limited signed URL.

    The signed URL must be used within roughly one minute of issue, so the pool
    re-issues it on demand rather than caching it for the whole transfer.
    """

    TTL_SECONDS = 35.0

    def __init__(self, file_id: int) -> None:
        self.file_id = file_id
        self._lock = threading.Lock()
        self._url: str | None = None
        self._issued_at = 0.0

    def _issue(self) -> str:
        body = json.dumps({"guestbookResponse": _guestbook_response()}).encode("utf-8")
        request = _request(
            f"{C.DATAVERSE_ACCESS}/{self.file_id}?signed=true",
            {"Content-Type": "application/json"},
        )
        request.method = "POST"
        request.data = body
        with urllib.request.urlopen(request, timeout=120) as resp:
            payload = json.load(resp)
        return payload["data"]["signedUrl"]

    def get(self, force: bool = False) -> str:
        with self._lock:
            stale = time.monotonic() - self._issued_at > self.TTL_SECONDS
            if force or stale or self._url is None:
                self._url = self._issue()
                self._issued_at = time.monotonic()
            return self._url


def _guestbook_response() -> dict:
    """The guestbook payload. Guestbook 96 requires an email address only."""
    return {
        "name": os.environ.get("DATAVERSE_NAME", "Milan traffic forecasting study"),
        "email": os.environ.get("DATAVERSE_EMAIL", ""),
        "institution": os.environ.get("DATAVERSE_INSTITUTION", "African Leadership University"),
        "position": os.environ.get("DATAVERSE_POSITION", "Student"),
        "answers": [],
    }


def _fetch_segment(pool: _SignedUrlPool, lo: int, hi: int, retries: int = 5) -> bytes:
    """Download byte range ``[lo, hi]`` inclusive, refreshing the URL on failure."""
    for attempt in range(1, retries + 1):
        try:
            url = pool.get(force=attempt > 1)
            request = _request(url, {"Range": f"bytes={lo}-{hi}"})
            with urllib.request.urlopen(request, timeout=300) as resp:
                buf = bytearray()
                while True:
                    block = resp.read(1 << 20)
                    if not block:
                        break
                    buf.extend(block)
            if len(buf) == hi - lo + 1:
                return bytes(buf)
            raise OSError(f"short read: {len(buf)} of {hi - lo + 1} bytes")
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as exc:
            if attempt == retries:
                raise RuntimeError(f"segment {lo}-{hi} failed: {exc}") from exc
            time.sleep(2.0 * attempt)
    raise RuntimeError("unreachable")


def _download(
    filename: str,
    dest: Path,
    workers: int = DOWNLOAD_WORKERS,
    segment_bytes: int = SEGMENT_BYTES,
) -> Path:
    """Fetch one day file using parallel HTTP range requests.

    Why parallel: Harvard Dataverse throttles an individual connection to
    roughly 0.17 MB/s for this dataset, while the host link sustains over
    5 MB/s. Measured aggregate throughput scales with the number of concurrent
    range requests, so the 20.6 GB dataset becomes a ~2 hour job instead of a
    ~34 hour one. Segments are written directly into a pre-allocated sparse file
    at their correct offset, so peak memory is ``workers x segment_bytes``
    (128 MB by default), not the file size.
    """
    if not C.DATAVERSE_API_TOKEN:
        raise RuntimeError(
            "DATAVERSE_API_TOKEN is not set. The dataset has a download guestbook, "
            "so anonymous downloads are refused. See .env.example for instructions, "
            "or place the raw .txt files in one of: "
            + ", ".join(str(p) for p in C.RAW_SEARCH_DIRS)
        )

    entry = _dataverse_catalogue()[filename]
    size = entry["size"]
    pool = _SignedUrlPool(entry["id"])
    tmp = dest.with_suffix(dest.suffix + ".part")

    bounds = [
        (lo, min(lo + segment_bytes, size) - 1) for lo in range(0, size, segment_bytes)
    ]

    with open(tmp, "wb") as fh:
        fh.truncate(size)

    write_lock = threading.Lock()
    done = {"bytes": 0}
    t0 = time.monotonic()

    def task(index: int) -> None:
        lo, hi = bounds[index]
        payload = _fetch_segment(pool, lo, hi)
        with write_lock, open(tmp, "r+b") as fh:
            fh.seek(lo)
            fh.write(payload)
            done["bytes"] += len(payload)

    with futures.ThreadPoolExecutor(max_workers=workers) as pool_exec:
        pending = [pool_exec.submit(task, i) for i in range(len(bounds))]
        for future in futures.as_completed(pending):
            future.result()  # re-raise the first failure

    if tmp.stat().st_size != size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{filename}: expected {size} bytes, got {tmp.stat().st_size}")

    elapsed = time.monotonic() - t0
    print(
        f"           downloaded {size / (1 << 20):.0f} MB in {elapsed:.0f}s "
        f"({size / (1 << 20) / max(elapsed, 1e-9):.2f} MB/s, {workers} streams)",
        flush=True,
    )
    tmp.replace(dest)
    return dest


# ---------------------------------------------------------------------------
# Streaming aggregation
# ---------------------------------------------------------------------------


def _chunks(path: Path) -> Iterator[pd.DataFrame]:
    """Yield row chunks holding only the three columns the study needs.

    ``dtype`` is pinned so the C parser writes straight into pre-typed buffers
    instead of inferring types and then copying.
    """
    return pd.read_csv(
        path,
        sep="\t",
        header=None,
        usecols=[C.COL_SQUARE, C.COL_TIME, C.COL_INTERNET],
        names=["square_id", "time_ms", "internet"],
        dtype={"square_id": np.int32, "time_ms": np.int64, "internet": np.float64},
        chunksize=CHUNK_ROWS,
        engine="c",
        na_values=[""],
        low_memory=True,
    )


def aggregate_day(path: Path, day_index: int, store: np.memmap) -> tuple[int, int, int]:
    """Aggregate one raw day file into ``store``.

    Country codes are summed out and the result is written to the rows of
    ``store`` covering ``day_index``.

    Returns
    -------
    (rows_read, rows_out_of_range, nonzero_cells)
    """
    day_start = day_index * C.SLOTS_PER_DAY
    lo = day_start - SLOT_MARGIN
    n_local = C.SLOTS_PER_DAY + 2 * SLOT_MARGIN

    # float64 accumulator: exact summation over country codes, 15 MB.
    buffer = np.zeros(n_local * C.N_SQUARES, dtype=np.float64)

    rows_read = 0
    rows_dropped = 0

    for chunk in _chunks(path):
        rows_read += len(chunk)

        internet = chunk["internet"].to_numpy(copy=False)
        keep = ~np.isnan(internet)
        if not keep.any():
            continue

        slot = ((chunk["time_ms"].to_numpy(copy=False) - C.ORIGIN_MS) // C.SLOT_MS) - lo
        square = chunk["square_id"].to_numpy(copy=False) - 1

        valid = keep & (slot >= 0) & (slot < n_local) & (square >= 0) & (square < C.N_SQUARES)
        rows_dropped += int((keep & ~valid).sum())
        if not valid.any():
            continue

        flat = slot[valid].astype(np.int64) * C.N_SQUARES + square[valid]
        buffer += np.bincount(flat, weights=internet[valid], minlength=buffer.size)

        del chunk, internet, slot, square, valid, flat

    day = buffer.reshape(n_local, C.N_SQUARES)[SLOT_MARGIN : SLOT_MARGIN + C.SLOTS_PER_DAY]
    store[day_start : day_start + C.SLOTS_PER_DAY, :] = day.astype(np.float32)
    nonzero = int(np.count_nonzero(day))

    del buffer, day
    return rows_read, rows_dropped, nonzero


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _append_log(stat: DayStat) -> None:
    """Append one day's record immediately.

    Written per day rather than per run so that an interrupted or failed
    ingestion still leaves complete evidence for the days it did process.
    """
    frame = pd.DataFrame([asdict(stat)])
    frame.to_csv(C.INGEST_LOG, mode="a", header=not C.INGEST_LOG.exists(), index=False)


def ingest(
    force: bool = False, limit: int | None = None, only: list[str] | None = None
) -> pd.DataFrame:
    """Run the ingestion, skipping days already present in the manifest.

    Parameters
    ----------
    force:
        Re-process days even if the manifest lists them as complete.
    limit:
        Stop after this many days (useful for smoke tests).
    only:
        Restrict processing to these ``YYYY-MM-DD`` dates.
    """
    meta = _load_meta()
    completed = set() if force else set(meta.get("completed_days", []))
    store = _ensure_store()
    stats: list[DayStat] = []

    dates = _day_dates()
    todo = [(i, d) for i, d in enumerate(dates) if d not in completed]
    if only:
        wanted = set(only)
        todo = [(i, d) for i, d in enumerate(dates) if d in wanted]
    if limit is not None:
        todo = todo[:limit]

    print(f"[ingest] {len(completed)}/{len(dates)} days already done; processing {len(todo)}")

    for day_index, date in todo:
        filename = C.FILENAME_TEMPLATE.format(date=date)
        local = _find_local(filename)
        downloaded = local is None

        if downloaded:
            local = C.RAW_DIR / filename
            print(f"[ingest] {date}: downloading ...", flush=True)
            _download(filename, local)

        raw_mb = local.stat().st_size / (1 << 20)
        probe = MemoryProbe()
        t0 = time.perf_counter()
        with probe:
            rows, dropped, nonzero = aggregate_day(local, day_index, store)
        elapsed = time.perf_counter() - t0

        store.flush()

        # Discard the raw file only if this run fetched it; never delete data
        # the user placed on disk themselves.
        if downloaded:
            local.unlink(missing_ok=True)

        stat = DayStat(
            date=date,
            raw_mb=round(raw_mb, 1),
            rows=rows,
            rows_out_of_range=dropped,
            nonzero_cells=nonzero,
            parse_seconds=round(elapsed, 2),
            peak_rss_mb=round(probe.peak_mb, 1),
            downloaded=downloaded,
        )
        stats.append(stat)
        _append_log(stat)
        meta.setdefault("completed_days", []).append(date)
        _save_meta(meta)

        print(
            f"[ingest] {date}: {rows:,} rows, {raw_mb:.0f} MB raw -> "
            f"{nonzero:,} non-zero cells in {elapsed:.1f}s "
            f"(peak RSS {probe.peak_mb:.0f} MB, dropped {dropped})",
            flush=True,
        )

    return pd.DataFrame([asdict(s) for s in stats])
