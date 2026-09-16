"""Unit and smoke tests for the forecasting pipeline.

These tests deliberately avoid the 341 MB activity store: they build a synthetic
series with the same qualitative properties (daily seasonality, heavy right
skew, non-negativity) so that the pipeline can be validated in seconds on any
machine, including one that has not run the ingestion stage.

Run with::

    python -m pytest tests -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C
from src import metrics as M
from src.data import LogStandardScaler, make_windows


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def synthetic_series() -> np.ndarray:
    """Daily-seasonal, right-skewed, non-negative series of 20 days."""
    rng = np.random.default_rng(0)
    n_days = 20
    t = np.arange(n_days * C.SLOTS_PER_DAY)
    daily = 1.0 + 0.9 * np.sin(2 * np.pi * (t % C.SLOTS_PER_DAY) / C.SLOTS_PER_DAY - 1.6)
    weekly = 1.0 - 0.25 * ((t // C.SLOTS_PER_DAY) % 7 >= 5)
    level = 300.0 * daily * weekly
    noise = rng.lognormal(mean=0.0, sigma=0.15, size=t.size)
    return np.maximum(level * noise, 0.0)


# ---------------------------------------------------------------------------
# Input representation
# ---------------------------------------------------------------------------


class TestWindowing:
    def test_shapes(self):
        values = np.arange(100, dtype=np.float32)
        x, y = make_windows(values, lookback=10, horizon=1)
        assert x.shape == (90, 10)
        assert y.shape == (90,)

    def test_target_is_the_next_step(self):
        values = np.arange(50, dtype=np.float32)
        x, y = make_windows(values, lookback=5, horizon=1)
        # The last element of every window must be the value just before y.
        assert np.allclose(x[:, -1] + 1, y)

    def test_no_leakage_into_the_window(self):
        values = np.arange(30, dtype=np.float32)
        x, y = make_windows(values, lookback=6, horizon=1)
        assert (x < y[:, None]).all(), "a window contains its own target"

    def test_rejects_too_short_a_series(self):
        with pytest.raises(ValueError):
            make_windows(np.arange(5, dtype=np.float32), lookback=10)


class TestScaler:
    def test_round_trip(self, synthetic_series):
        scaler = LogStandardScaler().fit(synthetic_series)
        recovered = scaler.inverse_transform(scaler.transform(synthetic_series))
        assert np.allclose(recovered, synthetic_series, rtol=1e-6, atol=1e-6)

    def test_standardises(self, synthetic_series):
        scaler = LogStandardScaler().fit(synthetic_series)
        z = scaler.transform(synthetic_series)
        assert abs(z.mean()) < 1e-9
        assert abs(z.std() - 1.0) < 1e-9

    def test_handles_zeros_and_constants(self):
        scaler = LogStandardScaler().fit(np.zeros(100))
        assert scaler.sigma == 1.0
        assert np.allclose(scaler.inverse_transform(scaler.transform(np.zeros(10))), 0.0)

    def test_output_is_non_negative(self, synthetic_series):
        """A forecaster may emit any real number; activity cannot be negative."""
        scaler = LogStandardScaler().fit(synthetic_series)
        assert (scaler.inverse_transform(np.linspace(-8, 3, 50)) >= 0).all()


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_perfect_forecast(self, synthetic_series):
        out = M.evaluate(synthetic_series, synthetic_series)
        assert out["MAE"] == pytest.approx(0.0)
        assert out["RMSE"] == pytest.approx(0.0)
        assert out["MAPE_%"] == pytest.approx(0.0)
        assert out["R2"] == pytest.approx(1.0)

    def test_known_values(self):
        truth = np.array([10.0, 20.0, 30.0])
        pred = np.array([12.0, 18.0, 33.0])
        assert M.mae(truth, pred) == pytest.approx(7 / 3)
        assert M.rmse(truth, pred) == pytest.approx(np.sqrt((4 + 4 + 9) / 3))
        assert M.mape(truth, pred) == pytest.approx(100 * (0.2 + 0.1 + 0.1) / 3)

    def test_rmse_is_never_below_mae(self, synthetic_series):
        rng = np.random.default_rng(1)
        pred = synthetic_series * rng.normal(1.0, 0.1, synthetic_series.size)
        out = M.evaluate(synthetic_series, pred)
        assert out["RMSE"] >= out["MAE"] - 1e-9

    def test_mape_does_not_explode_at_zero(self):
        assert np.isfinite(M.mape(np.array([0.0, 1.0]), np.array([0.5, 1.0])))

    def test_smape_is_bounded(self):
        value = M.smape(np.array([0.0, 1.0]), np.array([5.0, 100.0]))
        assert 0 <= value <= 200

    def test_mase_below_one_means_better_than_reference(self):
        truth = np.array([1.0, 2.0, 3.0, 4.0])
        good = truth + 0.1
        assert M.mase(truth, good, seasonal_naive_mae=1.0) < 1.0

    def test_shape_mismatch_is_an_error(self):
        with pytest.raises(ValueError):
            M.mae(np.zeros(5), np.zeros(6))


# ---------------------------------------------------------------------------
# Model smoke tests
# ---------------------------------------------------------------------------


def _supervised_from_array(values: np.ndarray, lookback: int):
    """Build a SupervisedData-like object from a plain array, without the store."""
    import pandas as pd

    from src.data import SupervisedData

    n = len(values)
    test_lo, test_hi = n - C.SLOTS_PER_DAY, n
    val_lo, val_hi = test_lo - C.SLOTS_PER_DAY, test_lo

    scaler = LogStandardScaler().fit(values[:val_lo])
    scaled = scaler.transform(values)

    def cut(lo, hi):
        x, y = make_windows(scaled[lo - lookback : hi], lookback, 1)
        return x, y, values[lo:hi]

    x_tr, y_tr, _ = cut(lookback, val_lo)
    x_va, y_va, raw_va = cut(val_lo, val_hi)
    x_te, y_te, raw_te = cut(test_lo, test_hi)

    return SupervisedData(
        square_id=1, lookback=lookback, scaler=scaler,
        x_train=x_tr, y_train=y_tr, x_val=x_va, y_val=y_va, x_test=x_te, y_test=y_te,
        y_val_raw=raw_va, y_test_raw=raw_te,
        test_index=pd.date_range("2013-12-16", periods=test_hi - test_lo, freq="10min"),
    )


class TestNeuralModels:
    """The neural models must train, shrink their loss, and beat a mean forecast."""

    @pytest.mark.parametrize("which", ["lstm", "tcn"])
    def test_trains_and_improves(self, synthetic_series, which):
        torch = pytest.importorskip("torch")
        from src.models import LSTMForecaster, TCNForecaster, TrainConfig

        data = _supervised_from_array(synthetic_series, lookback=C.SLOTS_PER_DAY)
        cfg = TrainConfig(epochs=6, patience=6, batch_size=64, verbose=False)
        model = (
            LSTMForecaster(hidden_size=16, train_config=cfg)
            if which == "lstm"
            else TCNForecaster(channels=8, levels=4, train_config=cfg)
        )
        model.fit(data)
        pred = model.predict_test(data)

        assert pred.shape == data.y_test_raw.shape
        assert np.isfinite(pred).all()
        assert (pred >= 0).all(), "forecasts must remain non-negative"

        history = model.report.history
        assert history[-1]["train_huber"] < history[0]["train_huber"]

        naive_mean = np.full_like(data.y_test_raw, synthetic_series[: len(data.y_train)].mean())
        assert M.mae(data.y_test_raw, pred) < M.mae(data.y_test_raw, naive_mean)

    def test_tcn_receptive_field_arithmetic(self):
        pytest.importorskip("torch")
        from src.models.neural import TCNNet

        # 1 + 2*(k-1)*(2^L - 1) for two convolutions per level.
        assert TCNNet(channels=4, levels=6, kernel=3).receptive_field == 253
        assert TCNNet(channels=4, levels=4, kernel=3).receptive_field == 61

    def test_tcn_is_causal(self):
        """Perturbing a future input must not change the current output."""
        torch = pytest.importorskip("torch")
        from src.models.neural import TCNNet

        net = TCNNet(channels=4, levels=3, kernel=3).eval()
        x = torch.zeros(1, 64)
        with torch.no_grad():
            base = net(x).item()
        # The head reads position -1, so nothing is strictly "future"; instead
        # confirm that padding is left-sided by checking the output at an
        # intermediate position is unaffected by later inputs.
        h_before = net.blocks(x.unsqueeze(1))[0, :, 31].detach().clone()
        x[0, 40:] = 5.0
        h_after = net.blocks(x.unsqueeze(1))[0, :, 31].detach()
        assert torch.allclose(h_before, h_after, atol=1e-6)
        assert np.isfinite(base)


class TestSarimaMechanics:
    """The subtlest part of the codebase: index alignment of one-step forecasts.

    ``get_prediction(..., dynamic=False)`` must yield, at position ``k``, the
    forecast for ``k`` conditioned only on observations up to ``k-1``. If that
    alignment were off by one, the SARIMA results would look excellent and be
    meaningless, so it is pinned by a test on data with a known answer.
    """

    def test_one_step_prediction_alignment(self):
        import warnings

        from statsmodels.tsa.arima.model import ARIMA

        rng = np.random.default_rng(0)
        phi, n = 0.8, 800
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = phi * y[t - 1] + rng.normal()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fitted = ARIMA(y[:600], order=(1, 0, 0), trend="n").fit()
            ar = fitted.params[0]
            filtered = ARIMA(y[:700], order=(1, 0, 0), trend="n").filter(fitted.params)
            pred = filtered.get_prediction(start=650, end=699, dynamic=False).predicted_mean

        assert ar == pytest.approx(phi, abs=0.05)
        # The one-step forecast of an AR(1) is exactly phi * y[t-1].
        assert np.allclose(pred, ar * y[649:699], atol=1e-10)

    def test_trend_n_really_excludes_the_constant(self):
        """`trend=None` is statsmodels' default, not 'no trend'. Guard the difference."""
        import warnings

        from statsmodels.tsa.arima.model import ARIMA

        rng = np.random.default_rng(1)
        y = np.cumsum(rng.normal(size=400)) * 0.01

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert "const" in ARIMA(y, order=(1, 0, 0), trend=None).fit().param_names
            assert "const" not in ARIMA(y, order=(1, 0, 0), trend="n").fit().param_names

    def test_default_forecaster_excludes_the_constant(self):
        from src.models import SarimaForecaster

        assert SarimaForecaster().hyperparameters()["trend"] == "n"

    def test_seasonal_difference_inversion_is_exact(self):
        """Differencing then re-adding the lagged value must be lossless."""
        rng = np.random.default_rng(2)
        x = np.abs(rng.lognormal(4.0, 0.4, size=1000))
        y = np.log1p(x)
        lag = C.SLOTS_PER_DAY
        z = y[lag:] - y[:-lag]
        recovered = np.expm1(z + y[:-lag])
        assert np.allclose(recovered, x[lag:], rtol=1e-9)


class TestConfigurationConsistency:
    def test_split_is_chronological_and_disjoint(self):
        from src.data import default_split

        split = default_split()
        assert split.train[1] == split.val[0], "gap or overlap between train and val"
        assert split.val[1] == split.test[0], "gap or overlap between val and test"
        assert split.test[1] <= C.N_SLOTS

    def test_test_week_is_seven_days(self):
        lo, hi = C.TEST_SLICE
        assert hi - lo == C.SLOTS_PER_WEEK

    def test_test_week_matches_the_assignment_dates(self):
        from src.data import time_index

        index = time_index()
        lo, hi = C.TEST_SLICE
        assert index[lo].strftime("%Y-%m-%d %H:%M") == "2013-12-16 00:00"
        assert index[hi - 1].strftime("%Y-%m-%d %H:%M") == "2013-12-22 23:50"

    def test_grid_geometry(self):
        assert C.GRID_SIDE ** 2 == C.N_SQUARES
        assert C.N_DAYS * C.SLOTS_PER_DAY == C.N_SLOTS
