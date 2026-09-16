"""Forecast accuracy metrics, all evaluated in the original activity units.

Every model in this study trains on a ``log1p``-standardised target, but all
reported errors are computed after inverting that transform. Comparing errors
in transformed space would flatter the neural models, because an error made at
a night-time trough would count as heavily as one made at a daily peak.

Metric choice
-------------
* **MAE** and **RMSE** are required by the assignment. They are reported
  together deliberately: RMSE/MAE > 1 quantifies how far the error
  distribution is dominated by a few large misses, which is exactly the
  failure mode expected at sharp traffic peaks.
* **MAPE** is required but is unreliable on this data, because night-time
  activity approaches zero and the ratio explodes. It is reported alongside
  **sMAPE**, which is bounded, and **MASE**, which is scale-free and stated
  relative to a seasonal-naive forecast. This lets the comparison be read
  without being misled by MAPE artefacts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C


def _clean(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    return y_true, y_pred


def mae(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    """Mean absolute percentage error (%). Guarded against division by zero."""
    y_true, y_pred = _clean(y_true, y_pred)
    denom = np.maximum(np.abs(y_true), C.MAPE_EPS)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred) -> float:
    """Symmetric MAPE (%), bounded in [0, 200] and stable near zero."""
    y_true, y_pred = _clean(y_true, y_pred)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.maximum(denom, C.MAPE_EPS)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def mase(y_true, y_pred, seasonal_naive_mae: float) -> float:
    """Mean absolute scaled error against a seasonal-naive reference.

    A value below 1 means the model beats "same time yesterday"; above 1 means
    it does not, and the extra complexity is unjustified.
    """
    if seasonal_naive_mae <= 0:
        return float("nan")
    return mae(y_true, y_pred) / seasonal_naive_mae


def r2(y_true, y_pred) -> float:
    y_true, y_pred = _clean(y_true, y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def evaluate(y_true, y_pred, seasonal_naive_mae: float | None = None) -> dict[str, float]:
    """Full metric bundle for one model on one area."""
    out = {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE_%": mape(y_true, y_pred),
        "sMAPE_%": smape(y_true, y_pred),
        "R2": r2(y_true, y_pred),
    }
    out["RMSE/MAE"] = out["RMSE"] / out["MAE"] if out["MAE"] > 0 else float("nan")
    if seasonal_naive_mae is not None:
        out["MASE"] = mase(y_true, y_pred, seasonal_naive_mae)
    return out


def error_by_hour(y_true, y_pred, index: pd.DatetimeIndex) -> pd.DataFrame:
    """Absolute error aggregated by hour of day - used for failure analysis."""
    y_true, y_pred = _clean(y_true, y_pred)
    frame = pd.DataFrame(
        {"abs_err": np.abs(y_true - y_pred), "actual": y_true, "hour": index.hour}
    )
    grouped = frame.groupby("hour").agg(
        mae=("abs_err", "mean"),
        mean_actual=("actual", "mean"),
    )
    grouped["relative_%"] = 100 * grouped["mae"] / grouped["mean_actual"]
    return grouped
