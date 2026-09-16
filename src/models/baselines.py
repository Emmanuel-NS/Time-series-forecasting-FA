"""Reference baselines.

These are not among the three models under comparison. They exist because a
forecast error is meaningless in isolation: a MAPE of 12% sounds good until you
discover that repeating yesterday's value achieves 11%. Every learned model in
this study is therefore reported against the seasonal-naive baseline via MASE.

Two baselines are included:

``PersistenceForecaster``
    :math:`\\hat{x}(t+1) = x(t)`. The hardest baseline to beat at a 10-minute
    resolution, because consecutive observations are extremely similar. It is a
    lower bound on "is the model learning anything at all beyond continuity".

``SeasonalNaiveForecaster``
    :math:`\\hat{x}(t+1) = x(t+1-144)`, i.e. the same time of day, one day
    earlier. This is the reference that MASE is scaled by, and it directly
    exploits the daily seasonality identified in the exploratory analysis.
"""

from __future__ import annotations

import numpy as np

from .. import config as C
from .. import data as D
from .base import Forecaster


class PersistenceForecaster(Forecaster):
    """Last-value carried forward (random-walk forecast)."""

    name = "Persistence"
    needs_windows = False

    def fit(self, data) -> "PersistenceForecaster":
        self._timed_fit(lambda: None)  # closed form: nothing to estimate
        self.report.notes = "closed form, no parameters estimated"
        return self

    def _predict(self, square_id: int, lo: int, hi: int) -> np.ndarray:
        raw = D.series(square_id).to_numpy()
        return raw[lo - 1 : hi - 1]

    def predict_test(self, data) -> np.ndarray:
        lo, hi = C.TEST_SLICE
        return self._predict(data.square_id, lo, hi)

    def predict_val(self, data) -> np.ndarray:
        lo, hi = C.VAL_SLICE
        return self._predict(data.square_id, lo, hi)


class SeasonalNaiveForecaster(Forecaster):
    """Same slot, previous day (or previous week with ``period=1008``)."""

    name = "Seasonal naive"
    needs_windows = False

    def __init__(self, period: int = C.SLOTS_PER_DAY) -> None:
        super().__init__()
        self.period = period

    def hyperparameters(self) -> dict:
        return {"period": self.period}

    def fit(self, data) -> "SeasonalNaiveForecaster":
        self._timed_fit(lambda: None)
        self.report.notes = f"closed form, seasonal period {self.period} steps"
        return self

    def _predict(self, square_id: int, lo: int, hi: int) -> np.ndarray:
        raw = D.series(square_id).to_numpy()
        return raw[lo - self.period : hi - self.period]

    def predict_test(self, data) -> np.ndarray:
        lo, hi = C.TEST_SLICE
        return self._predict(data.square_id, lo, hi)

    def predict_val(self, data) -> np.ndarray:
        lo, hi = C.VAL_SLICE
        return self._predict(data.square_id, lo, hi)
