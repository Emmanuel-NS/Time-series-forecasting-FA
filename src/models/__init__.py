"""Forecasting models compared in this study.

The three models under comparison are deliberately drawn from three different
families so that the comparison is informative rather than an ablation:

===================  ==========================  ==================================
Model                Family                      Aggregates history by
===================  ==========================  ==================================
``SARIMA``           linear statistical          explicit differencing + ARMA terms
``LSTM``             recurrent neural            gated sequential state
``TCN``              convolutional neural        dilated causal convolutions
===================  ==========================  ==================================

``Persistence`` and ``Seasonal naive`` are reference baselines, not competitors;
they exist to make the learned models' errors interpretable.
"""

from .base import Forecaster, FitReport
from .baselines import PersistenceForecaster, SeasonalNaiveForecaster
from .sarima import SarimaForecaster, order_search_table

__all__ = [
    "Forecaster",
    "FitReport",
    "PersistenceForecaster",
    "SeasonalNaiveForecaster",
    "SarimaForecaster",
    "order_search_table",
    "LSTMForecaster",
    "TCNForecaster",
    "TrainConfig",
]


def __getattr__(name: str):
    """Import the torch-dependent models lazily.

    This keeps the exploratory-analysis scripts runnable in an environment
    without PyTorch installed.
    """
    if name in {"LSTMForecaster", "TCNForecaster", "TrainConfig"}:
        from . import neural

        return getattr(neural, name)
    raise AttributeError(name)
