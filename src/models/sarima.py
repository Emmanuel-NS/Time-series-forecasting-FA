"""Model 1 - Seasonal ARIMA.

Why a seasonal ARIMA at all
---------------------------
It is the standard linear-statistical reference for this literature (Box and
Jenkins; Hyndman and Athanasopoulos), and it is the model whose assumptions the
exploratory analysis can directly test. If a linear model with a daily seasonal
term already captures most of the structure, then the extra capacity of a neural
network is not justified, and reporting that is a genuine finding rather than a
failure.

Implementation choice: explicit seasonal differencing
-----------------------------------------------------
The daily seasonal period at a 10-minute resolution is :math:`s = 144`. Fitting
a full ``SARIMAX(p,d,q)(P,D,Q)_144`` through ``statsmodels`` is not practical
here: with ``P`` or ``Q`` non-zero the state-space representation carries on the
order of 144 states, so each Kalman filter pass over ~8,000 observations
becomes minutes of computation on a 2-core CPU, and it must be repeated for
every optimiser iteration and every grid point.

This implementation therefore realises ``SARIMA(p,d,q)(0,1,0)_144``, but applies
the seasonal difference explicitly instead of delegating it to the state space:

.. math::

    y_t = \\log(1 + x_t), \\qquad z_t = y_t - y_{t-144}

An ``ARIMA(p,d,q)`` is fitted to :math:`z`, and predictions are mapped back with

.. math::

    \\hat{x}_{t+1} = \\exp(\\hat{z}_{t+1} + y_{t+1-144}) - 1

This is mathematically the same model, but the state dimension now depends only
on ``max(p, q+1)``, which makes the grid search feasible. It is also easier to
inspect and to defend, because the seasonal step is visible in the code rather
than hidden in a matrix.

The ``log1p`` transform is applied for the same reason as for the neural models:
the raw series has strongly heteroskedastic noise (variance grows with the
level), which violates the constant-variance assumption of ARIMA. Working on the
log scale stabilises it.

One-step-ahead evaluation
-------------------------
Parameters are estimated **once**, on the training partition only. The fitted
parameters are then applied to the longer series with ``ARIMAResults.apply`` and
one-step-ahead predictions are read off with ``dynamic=False``, so each forecast
for slot ``t+1`` is conditioned on the true observations up to ``t`` but never on
test-period parameter re-estimation. This mirrors an operational deployment in
which the model is refitted infrequently but observes traffic continuously.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from .. import config as C
from .. import data as D
from .base import Forecaster


class SarimaForecaster(Forecaster):
    """``SARIMA(p,d,q)(0,1,0)_144`` on the ``log1p`` scale."""

    name = "SARIMA"
    needs_windows = False

    def __init__(
        self,
        order: tuple[int, int, int] = (2, 0, 1),
        seasonal_period: int = C.SLOTS_PER_DAY,
        trend: str = "n",
    ) -> None:
        """
        Parameters
        ----------
        trend:
            ``'n'`` for no deterministic term (the default) or ``'c'`` to include
            a constant. Note that ``statsmodels`` treats ``trend=None`` as *its*
            default, which for ``d = 0`` silently includes a constant; ``'n'`` is
            the only way to actually exclude one. A constant on an already
            seasonally differenced series implies a deterministic linear trend in
            the original series, which is not something the exploratory analysis
            supports, so it is excluded by default and tested explicitly during
            order selection instead.
        """
        super().__init__()
        self.order = order
        self.seasonal_period = seasonal_period
        self.trend = trend
        self._params = None
        self._log_series: np.ndarray | None = None
        self._diff: np.ndarray | None = None
        self.aic: float | None = None

    def hyperparameters(self) -> dict:
        p, d, q = self.order
        return {
            "p": p,
            "d": d,
            "q": q,
            "seasonal_period": self.seasonal_period,
            "trend": self.trend,
        }

    @property
    def spec(self) -> str:
        p, d, q = self.order
        suffix = " + const" if self.trend == "c" else ""
        return f"SARIMA({p},{d},{q})(0,1,0)[{self.seasonal_period}]{suffix}"

    # -- fitting ---------------------------------------------------------
    def fit(self, data) -> "SarimaForecaster":
        raw = D.series(data.square_id).to_numpy()
        self._log_series = np.log1p(np.clip(raw, 0.0, None))
        s = self.seasonal_period
        self._diff = self._log_series[s:] - self._log_series[:-s]

        # Training targets end where the validation week begins. Index shift of
        # `s` accounts for the seasonal difference consuming the first s points.
        train_end = C.TRAIN_END_SLOT - s

        def _run() -> None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = ARIMA(
                    self._diff[:train_end],
                    order=self.order,
                    trend=self.trend,
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                )
                result = model.fit(method="statespace", method_kwargs={"maxiter": 200})
            self._params = result.params
            self.aic = float(result.aic)

        self._timed_fit(_run)
        self.report.n_parameters = int(len(self._params))
        self.report.notes = f"{self.spec}, AIC={self.aic:.1f}, log1p scale"
        return self

    # -- prediction ------------------------------------------------------
    def _one_step(self, lo: int, hi: int) -> np.ndarray:
        """One-step-ahead predictions for global slots ``[lo, hi)``."""
        if self._params is None:
            raise RuntimeError("call fit() before predicting")
        s = self.seasonal_period

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            applied = ARIMA(
                self._diff[: hi - s],
                order=self.order,
                trend=self.trend,
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).filter(self._params)
            pred = applied.get_prediction(start=lo - s, end=hi - s - 1, dynamic=False)

        z_hat = np.asarray(pred.predicted_mean, dtype=np.float64)
        # Undo the seasonal difference using the observed value one day earlier.
        y_hat = z_hat + self._log_series[lo - s : hi - s]
        return np.expm1(np.clip(y_hat, None, 50.0))

    def predict_test(self, data) -> np.ndarray:
        return self._one_step(*C.TEST_SLICE)

    def predict_val(self, data) -> np.ndarray:
        return self._one_step(*C.VAL_SLICE)


def order_search_table(
    square_id: int, candidates: list[tuple[int, int, int]]
) -> pd.DataFrame:
    """Fit each candidate order and report AIC plus validation MAE.

    AIC alone is not used to pick the final order: it measures in-sample fit of
    the differenced series, whereas the quantity we care about is out-of-sample
    error in the original units. Both are reported so the choice is auditable.
    """
    from ..data import supervised
    from ..metrics import mae, rmse

    data = supervised(square_id, lookback=1)
    rows = []
    for order in candidates:
        model = SarimaForecaster(order=order)
        try:
            model.fit(data)
            pred = model.predict_val(data)
            rows.append(
                {
                    "order": f"({order[0]},{order[1]},{order[2]})",
                    "n_params": model.report.n_parameters,
                    "AIC": round(model.aic, 1),
                    "val_MAE": round(mae(data.y_val_raw, pred), 2),
                    "val_RMSE": round(rmse(data.y_val_raw, pred), 2),
                    "fit_seconds": round(model.report.train_seconds, 1),
                }
            )
            print(f"  {model.spec:28s} AIC={model.aic:9.1f}  val MAE={rows[-1]['val_MAE']:8.2f}"
                  f"  ({model.report.train_seconds:.1f}s)")
        except Exception as exc:  # pragma: no cover - diagnostics
            print(f"  order {order} failed: {exc}")
    return pd.DataFrame(rows)
