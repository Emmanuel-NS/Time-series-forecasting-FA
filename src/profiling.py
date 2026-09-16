"""Lightweight instrumentation used to produce the memory-management evidence.

``MemoryProbe`` samples the resident set size (RSS) of the current process from
a background thread. Sampling rather than relying on a single before/after
reading matters here: the quantity of interest is the *peak* allocation during
a parse, which is transient and would be invisible to an end-of-block
measurement.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import psutil


class MemoryProbe:
    """Context manager that records peak process RSS while active.

    Parameters
    ----------
    interval:
        Sampling period in seconds. 20 ms is fast enough to catch the short
        spikes produced by a ``pandas`` chunk allocation without measurably
        perturbing the workload.
    """

    def __init__(self, interval: float = 0.02) -> None:
        self.interval = interval
        self._process = psutil.Process()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.baseline_mb = 0.0
        self.peak_mb = 0.0

    # -- context manager -------------------------------------------------
    def __enter__(self) -> "MemoryProbe":
        self.baseline_mb = self._rss_mb()
        self.peak_mb = self.baseline_mb
        self._stop.clear()
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.peak_mb = max(self.peak_mb, self._rss_mb())

    # -- internals -------------------------------------------------------
    def _rss_mb(self) -> float:
        return self._process.memory_info().rss / (1 << 20)

    def _sample(self) -> None:
        while not self._stop.is_set():
            self.peak_mb = max(self.peak_mb, self._rss_mb())
            self._stop.wait(self.interval)

    # -- reporting -------------------------------------------------------
    @property
    def delta_mb(self) -> float:
        """Peak RSS attributable to the profiled block."""
        return self.peak_mb - self.baseline_mb


@dataclass
class Timer:
    """Accumulating wall-clock timer with a context-manager interface."""

    label: str = ""
    elapsed: float = 0.0
    _t0: float = field(default=0.0, repr=False)

    def __enter__(self) -> "Timer":
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed += time.perf_counter() - self._t0


def system_summary() -> dict:
    """Hardware/OS facts recorded alongside every timing measurement."""
    import platform

    vm = psutil.virtual_memory()
    freq = psutil.cpu_freq()
    return {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "cpu_max_mhz": round(freq.max, 0) if freq else None,
        "total_ram_gb": round(vm.total / (1 << 30), 2),
    }
