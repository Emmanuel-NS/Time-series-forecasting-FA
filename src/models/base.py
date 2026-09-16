"""Common interface shared by every forecaster in this study.

Having one interface matters for the comparison: it guarantees that all four
models are handed identical inputs, produce predictions on identical target
slots, and have their training and inference time measured the same way. Any
difference in the reported metrics is then attributable to the model, not to an
inconsistency in the harness.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FitReport:
    """Timing and convergence information captured during ``fit``."""

    train_seconds: float = 0.0
    n_parameters: int = 0
    epochs_run: int = 0
    best_epoch: int | None = None
    history: list[dict] = field(default_factory=list)
    notes: str = ""


class Forecaster(ABC):
    """One-step-ahead forecaster operating on a single geographical area.

    Subclasses receive data already split chronologically and, where relevant,
    already scaled. Predictions must always be returned in the *original*
    activity units so that metrics are directly comparable across models.
    """

    #: Human-readable name used in tables and figure legends.
    name: str = "forecaster"

    #: Whether the model consumes the windowed/scaled representation
    #: (``SupervisedData``) or the raw series.
    needs_windows: bool = True

    def __init__(self) -> None:
        self.report = FitReport()

    @abstractmethod
    def fit(self, data) -> "Forecaster":
        """Train on the training partition, using validation only for selection."""

    @abstractmethod
    def predict_test(self, data) -> np.ndarray:
        """Return one-step-ahead predictions for every slot of the test week."""

    def predict_val(self, data) -> np.ndarray:
        """Predictions over the validation week, used for hyperparameter search."""
        raise NotImplementedError

    # -- helpers ---------------------------------------------------------
    def _timed_fit(self, fn) -> None:
        t0 = time.perf_counter()
        fn()
        self.report.train_seconds = time.perf_counter() - t0

    def timed_predict(self, fn) -> tuple[np.ndarray, float]:
        """Run a prediction callable and return ``(predictions, seconds)``."""
        t0 = time.perf_counter()
        out = fn()
        return out, time.perf_counter() - t0

    def hyperparameters(self) -> dict:
        """Parameters that define this configuration, for the results tables."""
        return {}

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        params = ", ".join(f"{k}={v}" for k, v in self.hyperparameters().items())
        return f"{self.__class__.__name__}({params})"
