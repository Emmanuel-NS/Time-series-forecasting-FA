"""Access layer over the compact activity store.

The store is a ``(8928, 10000)`` float32 matrix on disk. Everything downstream
reads it through :func:`open_store`, which returns a ``numpy.memmap`` so that
the operating system pages in only the bytes actually touched. Extracting the
time series for one geographical area therefore costs 8928 float32 values
(35 KB) of resident memory, not 341 MB.

This module also owns the **input representation** used by the supervised
models, so that every model is trained and evaluated on exactly the same
windows and the same scaling.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd

from . import config as C
from .ingest import open_store

SQUARE_TOTALS_CACHE = C.PROCESSED_DIR / "square_totals.npy"
DAILY_PROFILE_CACHE = C.PROCESSED_DIR / "square_daily_profile.npy"
CACHE_FINGERPRINT = C.PROCESSED_DIR / "aggregate_fingerprint.json"


# ---------------------------------------------------------------------------
# Global index helpers
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def time_index() -> pd.DatetimeIndex:
    """Timezone-aware index of the 8928 ten-minute slots in the study window."""
    return pd.date_range(
        start=C.FIRST_DAY, periods=C.N_SLOTS, freq=f"{C.SLOT_MINUTES}min", tz=C.LOCAL_TZ
    )


def days_ingested() -> int:
    if not C.STORE_META.exists():
        return 0
    meta = json.loads(C.STORE_META.read_text())
    return len(set(meta.get("completed_days", [])))


def store_is_complete() -> bool:
    return days_ingested() == C.N_DAYS


def _cache_is_current() -> bool:
    """True if the cached spatial aggregates were built from the current store.

    Without this check, running the exploratory analysis while ingestion is still
    in progress would write a `square_totals.npy` covering only part of the
    period, and every later run would silently reuse it. Since the identity of
    the top-3 areas -- and therefore every downstream experiment -- depends on
    those totals, a stale cache would be a serious and invisible error.
    """
    if not CACHE_FINGERPRINT.exists():
        return False
    return json.loads(CACHE_FINGERPRINT.read_text()).get("days") == days_ingested()


def _stamp_cache() -> None:
    CACHE_FINGERPRINT.write_text(json.dumps({"days": days_ingested()}))


# ---------------------------------------------------------------------------
# Spatial aggregates (computed once, cached to disk)
# ---------------------------------------------------------------------------


def square_totals(recompute: bool = False) -> np.ndarray:
    """Total Internet activity per geographical area over the whole period.

    Computed by streaming the store in row blocks so that peak memory stays at
    one block rather than the full matrix.
    """
    if SQUARE_TOTALS_CACHE.exists() and _cache_is_current() and not recompute:
        return np.load(SQUARE_TOTALS_CACHE)

    store = open_store("r")
    totals = np.zeros(C.N_SQUARES, dtype=np.float64)
    block = 512  # rows; 512 x 10000 x 4 B = 20 MB per read
    for start in range(0, C.N_SLOTS, block):
        totals += store[start : start + block].sum(axis=0, dtype=np.float64)
    del store

    np.save(SQUARE_TOTALS_CACHE, totals)
    _stamp_cache()
    return totals


def mean_daily_profile(recompute: bool = False) -> np.ndarray:
    """Mean activity per ``(slot-of-day, square)``, shape ``(144, 10000)``.

    Used by the exploratory analysis and by the seasonal-naive style baselines.
    """
    if DAILY_PROFILE_CACHE.exists() and _cache_is_current() and not recompute:
        return np.load(DAILY_PROFILE_CACHE)

    store = open_store("r")
    acc = np.zeros((C.SLOTS_PER_DAY, C.N_SQUARES), dtype=np.float64)
    for day in range(C.N_DAYS):
        s = day * C.SLOTS_PER_DAY
        acc += store[s : s + C.SLOTS_PER_DAY]
    del store
    acc /= C.N_DAYS

    profile = acc.astype(np.float32)
    np.save(DAILY_PROFILE_CACHE, profile)
    _stamp_cache()
    return profile


def top_squares(k: int = 3) -> list[int]:
    """The ``k`` square IDs with the highest total Internet traffic (1-indexed)."""
    totals = square_totals()
    order = np.argsort(totals)[::-1][:k]
    return [int(i) + 1 for i in order]


def study_squares() -> list[int]:
    """Task 2 study set: top-3 areas followed by the two reference squares."""
    return top_squares(3) + list(C.REFERENCE_SQUARES)


# ---------------------------------------------------------------------------
# Series extraction
# ---------------------------------------------------------------------------


def series(square_id: int) -> pd.Series:
    """Internet activity time series for one area, indexed by local time."""
    if not 1 <= square_id <= C.N_SQUARES:
        raise ValueError(f"square_id must be in [1, {C.N_SQUARES}], got {square_id}")
    store = open_store("r")
    values = np.array(store[:, square_id - 1], dtype=np.float64)
    del store
    return pd.Series(values, index=time_index(), name=f"square_{square_id}")


def frame(square_ids) -> pd.DataFrame:
    """Wide frame of several areas' series, one column per area."""
    return pd.DataFrame({f"square_{s}": series(s) for s in square_ids})


# ---------------------------------------------------------------------------
# Chronological splits
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Split:
    """Slot boundaries of the three chronological partitions.

    The split is strictly chronological -- no shuffling -- because shuffled
    splits leak future information into training and are invalid for
    forecasting evaluation.
    """

    train: tuple[int, int]
    val: tuple[int, int]
    test: tuple[int, int]

    def describe(self) -> pd.DataFrame:
        idx = time_index()
        rows = []
        for name in ("train", "val", "test"):
            lo, hi = getattr(self, name)
            rows.append(
                {
                    "partition": name,
                    "start": idx[lo].strftime("%Y-%m-%d %H:%M"),
                    "end": idx[hi - 1].strftime("%Y-%m-%d %H:%M"),
                    "slots": hi - lo,
                    "days": round((hi - lo) / C.SLOTS_PER_DAY, 2),
                }
            )
        return pd.DataFrame(rows)


def default_split() -> Split:
    return Split(train=(0, C.TRAIN_END_SLOT), val=C.VAL_SLICE, test=C.TEST_SLICE)


def require_complete_store(allow_partial: bool = False) -> None:
    """Abort unless the full 62-day store is present.

    Every downstream result depends on the whole period: the top-3 areas come
    from two-month totals, and the test week is the last week of December. Running
    on a partially ingested store produces plausible-looking numbers that are
    wrong, so this is a hard stop rather than a warning.
    """
    done = days_ingested()
    if done == C.N_DAYS:
        return
    message = (
        f"The activity store holds {done} of {C.N_DAYS} days. Results computed now "
        "would be invalid: the highest-traffic areas are defined over the full "
        "two-month period and the test week is 16-22 December. "
        "Run `python scripts/01_ingest.py` to completion first."
    )
    if allow_partial:
        print(f"WARNING: {message}")
        return
    raise SystemExit(f"ERROR: {message}")


# ---------------------------------------------------------------------------
# Input representation for the supervised models
# ---------------------------------------------------------------------------


class LogStandardScaler:
    """``log1p`` compression followed by standardisation.

    Rationale
    ---------
    Cell-level Internet activity is strictly non-negative, strongly
    right-skewed, and spans two to three orders of magnitude between the
    nightly trough and the daily peak. Training a neural network directly on
    such a series makes the loss dominated by the few largest peaks.
    ``log1p`` converts the multiplicative spread into an additive one; the
    subsequent standardisation puts the input in the range where ``tanh`` and
    ``ReLU`` units have useful gradients.

    Statistics are estimated on the training partition only, so no information
    from the validation or test weeks reaches the model.
    """

    def __init__(self) -> None:
        self.mu: float = 0.0
        self.sigma: float = 1.0

    def fit(self, values: np.ndarray) -> "LogStandardScaler":
        z = np.log1p(np.clip(np.asarray(values, dtype=np.float64), 0.0, None))
        self.mu = float(z.mean())
        self.sigma = float(z.std())
        if self.sigma < 1e-12:
            self.sigma = 1.0
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        z = np.log1p(np.clip(np.asarray(values, dtype=np.float64), 0.0, None))
        return (z - self.mu) / self.sigma

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        """Map scaled predictions back to activity units.

        Two clips are applied, both for correctness rather than cosmetics. The
        upper clip guards ``expm1`` against overflow if an untrained network
        emits a large value. The lower clip at zero enforces the physical
        constraint that traffic volume is non-negative: because ``log1p`` maps
        ``[0, inf)`` onto ``[0, inf)``, any scaled prediction below
        ``-mu/sigma`` would otherwise invert to a negative activity.
        """
        z = np.asarray(values, dtype=np.float64) * self.sigma + self.mu
        return np.maximum(np.expm1(np.clip(z, None, 50.0)), 0.0)

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return f"LogStandardScaler(mu={self.mu:.4f}, sigma={self.sigma:.4f})"


def make_windows(
    values: np.ndarray, lookback: int, horizon: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window over a 1-D series to build a supervised dataset.

    Returns ``(X, y)`` with ``X.shape == (n, lookback)`` and
    ``y.shape == (n,)``, where ``y[i]`` is the value ``horizon`` steps after the
    end of ``X[i]``. Implemented with a strided view so no data is copied until
    the caller materialises a batch.
    """
    values = np.ascontiguousarray(values, dtype=np.float32)
    n = len(values) - lookback - horizon + 1
    if n <= 0:
        raise ValueError(
            f"series of length {len(values)} is too short for lookback={lookback}"
        )
    windows = np.lib.stride_tricks.sliding_window_view(values, lookback)[:n]
    targets = values[lookback + horizon - 1 :][:n]
    return windows, targets


@dataclass
class SupervisedData:
    """Everything a supervised model needs for one area, in one object."""

    square_id: int
    lookback: int
    scaler: LogStandardScaler
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    y_val_raw: np.ndarray
    y_test_raw: np.ndarray
    test_index: pd.DatetimeIndex

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"SupervisedData(square={self.square_id}, lookback={self.lookback}, "
            f"train={self.x_train.shape}, val={self.x_val.shape}, test={self.x_test.shape})"
        )


def supervised(
    square_id: int, lookback: int = 144, horizon: int = 1, split: Split | None = None
) -> SupervisedData:
    """Build scaled, windowed train/val/test tensors for one area.

    Windows are cut so that a window belongs to the partition of its *target*.
    Validation and test windows are allowed to read history from the preceding
    partition -- that is legitimate, because at forecast time step ``t+1`` the
    observations up to ``t`` are genuinely available. What must never happen,
    and does not happen here, is a *target* from a later partition informing
    training or the scaler.
    """
    split = split or default_split()
    raw = series(square_id).to_numpy()

    scaler = LogStandardScaler().fit(raw[split.train[0] : split.train[1]])
    scaled = scaler.transform(raw)

    def cut(lo: int, hi: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Target slots are [lo, hi); each needs `lookback` prior observations.
        start = lo - lookback - horizon + 1
        if start < 0:
            raise ValueError(
                f"partition starting at slot {lo} has fewer than {lookback} "
                "preceding observations"
            )
        x, y = make_windows(scaled[start:hi], lookback, horizon)
        return x, y, raw[lo:hi]

    x_tr, y_tr, _ = cut(split.train[0] + lookback + horizon - 1, split.train[1])
    x_va, y_va, raw_va = cut(*split.val)
    x_te, y_te, raw_te = cut(*split.test)

    return SupervisedData(
        square_id=square_id,
        lookback=lookback,
        scaler=scaler,
        x_train=x_tr,
        y_train=y_tr,
        x_val=x_va,
        y_val=y_va,
        x_test=x_te,
        y_test=y_te,
        y_val_raw=raw_va,
        y_test_raw=raw_te,
        test_index=time_index()[split.test[0] : split.test[1]],
    )
