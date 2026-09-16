"""Experiment harness: tuning studies, final evaluation, figures and tables.

Design goals
------------
1. **One code path for every model.** The same function fits, times and scores
   all four forecasters, so reported differences cannot come from harness
   inconsistencies.
2. **Auditable tuning.** Every configuration tried is appended to a CSV with its
   validation score, so the report can show the search rather than assert its
   conclusion. Selection uses validation MAE only; the test week is touched once,
   at the end.
3. **Honest timing.** Training time is wall-clock for the full fit including
   early stopping. Inference time is the median of repeated runs over the whole
   test week, because a single measurement on a 2-core laptop is dominated by
   scheduling noise.
"""

from __future__ import annotations

import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass, field

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Match the figure style used in ``eda`` so every figure in the report is
# consistent. ``savefig.bbox="tight"`` matters rather than being cosmetic: these
# figures carry two stacked axes with rotated date labels, and without it the
# bottom axis label is cropped out of the saved PNG.
plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

from . import config as C
from . import data as D
from . import metrics as M
from .models import (
    PersistenceForecaster,
    SarimaForecaster,
    SeasonalNaiveForecaster,
)
from .profiling import system_summary

INFERENCE_REPEATS = 5

# ---------------------------------------------------------------------------
# Compute budget
# ---------------------------------------------------------------------------
# Measured on the host (Intel i5-6300U, 2 cores): one epoch over the ~5,300
# training windows costs roughly 6 s for the LSTM and 15-30 s for the TCN,
# because every dilation level of the TCN convolves the whole padded 144-step
# sequence. A 12-configuration search at 60 epochs each would therefore take
# most of a day. During tuning the epoch count is capped and a wall-clock budget
# is applied so that no single configuration can starve the rest of the search;
# the finally selected configurations are then retrained with a larger budget.
TUNE_EPOCHS = 30
TUNE_PATIENCE = 5
TUNE_BUDGET_S = 420.0

# The final runs use a budget generous enough that the selected configurations
# stop on their own. This matters for credibility: a headline result produced by
# a truncated run would carry the same confound that invalidated the first TCN
# search (see ``tcn_receptive_field_study``). The selected TCN reached its best
# epoch at 20 and the selected LSTM at 18, so 40 epochs with patience 8 leaves
# clear headroom, and the 1,800 s cap exists only to bound a pathological run
# rather than to bind a normal one. It is set well above the slowest per-epoch
# cost observed under CPU contention (~90 s) times the expected epoch count.
FINAL_EPOCHS = 40
FINAL_PATIENCE = 8
FINAL_BUDGET_S = 1800.0


# ===========================================================================
# Core evaluation
# ===========================================================================


@dataclass
class RunResult:
    """Outcome of fitting and evaluating one model on one area."""

    model: str
    square_id: int
    partition: str
    metrics: dict = field(default_factory=dict)
    train_seconds: float = 0.0
    inference_seconds: float = 0.0
    inference_ms_per_step: float = 0.0
    n_parameters: int = 0
    best_epoch: int | None = None
    hyperparameters: dict = field(default_factory=dict)
    notes: str = ""
    predictions: np.ndarray | None = None

    def row(self) -> dict:
        base = {
            "model": self.model,
            "square_id": self.square_id,
            "partition": self.partition,
            **{k: round(v, 4) for k, v in self.metrics.items()},
            "train_seconds": round(self.train_seconds, 3),
            "inference_seconds": round(self.inference_seconds, 4),
            "inference_ms_per_step": round(self.inference_ms_per_step, 4),
            "n_parameters": self.n_parameters,
            "best_epoch": self.best_epoch,
            "hyperparameters": json.dumps(self.hyperparameters),
            "notes": self.notes,
        }
        return base


def baseline_mae(square_id: int, partition: str = "test") -> float:
    """Seasonal-naive MAE for one area, used as the MASE denominator."""
    model = SeasonalNaiveForecaster()
    data = D.supervised(square_id, lookback=1)
    model.fit(data)
    pred = model.predict_test(data) if partition == "test" else model.predict_val(data)
    truth = data.y_test_raw if partition == "test" else data.y_val_raw
    return M.mae(truth, pred)


def run_model(
    model,
    data,
    partition: str = "test",
    scale_reference: float | None = None,
    repeats: int = INFERENCE_REPEATS,
) -> RunResult:
    """Fit ``model``, predict the requested partition, and time both phases."""
    model.fit(data)

    predict = model.predict_test if partition == "test" else model.predict_val
    truth = data.y_test_raw if partition == "test" else data.y_val_raw

    timings = []
    predictions = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        predictions = predict(data)
        timings.append(time.perf_counter() - t0)
    inference = statistics.median(timings)

    predictions = np.asarray(predictions, dtype=np.float64)
    if len(predictions) != len(truth):
        raise ValueError(
            f"{model.name}: produced {len(predictions)} predictions for "
            f"{len(truth)} target slots"
        )

    return RunResult(
        model=model.name,
        square_id=data.square_id,
        partition=partition,
        metrics=M.evaluate(truth, predictions, scale_reference),
        train_seconds=model.report.train_seconds,
        inference_seconds=inference,
        inference_ms_per_step=1000 * inference / len(predictions),
        n_parameters=model.report.n_parameters,
        best_epoch=model.report.best_epoch,
        hyperparameters=model.hyperparameters(),
        notes=model.report.notes,
        predictions=predictions,
    )


# ===========================================================================
# Tuning studies
# ===========================================================================


class ExperimentLog:
    """Append-only record of every configuration evaluated during tuning."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.path = C.RESULTS_DIR / f"tuning_{name}.csv"
        self.rows: list[dict] = []

    def add(self, experiment: int, rationale: str, result: RunResult) -> None:
        row = {
            "experiment": experiment,
            "rationale": rationale,
            **result.row(),
        }
        self.rows.append(row)
        pd.DataFrame(self.rows).to_csv(self.path, index=False)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)

    def best(self, metric: str = "MAE") -> dict:
        frame = self.frame()
        return frame.loc[frame[metric].idxmin()].to_dict()


def sarima_study(square_id: int) -> pd.DataFrame:
    """Sequential order selection for SARIMA, guided by the PACF and by AIC.

    Experiment order follows a Box-Jenkins style argument rather than a blind
    grid: start from the simplest model that the seasonal differencing leaves
    plausible, then add AR and MA terms only while validation error improves.
    """
    log = ExperimentLog(f"sarima_sq{square_id}")
    data = D.supervised(square_id, lookback=1)
    reference = baseline_mae(square_id, "val")

    plan = [
        ((1, 0, 0), "n", "Baseline: a single AR term on the seasonally differenced series."),
        ((0, 0, 1), "n", "A pure MA term at the same complexity, to see which the data prefers."),
        ((1, 0, 1), "n", "ARMA(1,1): the standard first upgrade when both ACF and PACF decay."),
        ((2, 0, 1), "n", "The PACF shows significant mass at lag 2; add a second AR term."),
        ((3, 0, 1), "n", "Probe whether AR order keeps helping or starts to overfit."),
        ((2, 0, 2), "n", "Trade an AR term for an extra MA term at equal parameter count."),
        ((2, 1, 1), "n", "Add a non-seasonal difference to test for residual drift."),
        ((2, 0, 1), "c", "Re-test the best order with a constant, to check whether the "
                         "seasonally differenced series carries any residual mean offset. "
                         "Worth testing explicitly because statsmodels includes a constant "
                         "by default and it is easy to fit one unintentionally."),
    ]

    for i, (order, trend, rationale) in enumerate(plan, start=1):
        print(f"\n[SARIMA {i}/{len(plan)}] order={order} trend={trend!r}\n  reason: {rationale}")
        try:
            result = run_model(
                SarimaForecaster(order=order, trend=trend), data, "val", reference, repeats=1
            )
        except Exception as exc:
            print(f"  failed: {exc}")
            continue
        log.add(i, rationale, result)
        print(f"  val MAE={result.metrics['MAE']:.2f}  RMSE={result.metrics['RMSE']:.2f}  "
              f"MASE={result.metrics['MASE']:.3f}  fit={result.train_seconds:.1f}s")

    return log.frame()


def lstm_study(square_id: int) -> pd.DataFrame:
    """Iterative LSTM tuning.

    The sequence of experiments is deliberately sequential rather than a grid:
    each step changes one thing and is motivated by what the previous step
    showed. On a 2-core CPU an exhaustive grid over five hyperparameters is not
    affordable, so the search has to be guided.
    """
    from .models import LSTMForecaster, TrainConfig

    log = ExperimentLog(f"lstm_sq{square_id}")
    reference = baseline_mae(square_id, "val")
    cache: dict[int, D.SupervisedData] = {}

    def data_for(lookback: int):
        if lookback not in cache:
            cache[lookback] = D.supervised(square_id, lookback=lookback)
        return cache[lookback]

    plan = [
        dict(
            lookback=36, hidden_size=32, num_layers=1, lr=1e-3, batch=128,
            rationale="Start small: 6 hours of history, 32 units. Establishes whether "
                      "short-range continuity alone already beats the baseline.",
        ),
        dict(
            lookback=144, hidden_size=32, num_layers=1, lr=1e-3, batch=128,
            rationale="Extend the window to a full day (144 steps). The ACF shows a "
                      "strong peak at lag 144, so the model should be able to use "
                      "yesterday's same-time value.",
        ),
        dict(
            lookback=144, hidden_size=64, num_layers=1, lr=1e-3, batch=128,
            rationale="Double the hidden width at the better lookback to test whether "
                      "capacity, not context, is the binding constraint.",
        ),
        dict(
            lookback=144, hidden_size=64, num_layers=2, dropout=0.1, lr=1e-3, batch=128,
            rationale="Add depth with light dropout: a second layer can compose "
                      "short-range and daily features hierarchically.",
        ),
        dict(
            lookback=144, hidden_size=64, num_layers=1, lr=3e-3, batch=256,
            rationale="Larger batches with a proportionally higher learning rate. "
                      "Tests whether the earlier runs were optimisation-limited "
                      "rather than capacity-limited, at roughly half the epoch cost.",
        ),
        dict(
            lookback=288, hidden_size=64, num_layers=1, lr=1e-3, batch=256,
            rationale="Two days of history, to check whether the weekly structure "
                      "visible in the EDA is reachable through a longer window.",
        ),
    ]

    for i, spec in enumerate(plan, start=1):
        rationale = spec.pop("rationale")
        lookback = spec.pop("lookback")
        lr, batch = spec.pop("lr"), spec.pop("batch")
        print(f"\n[LSTM {i}/{len(plan)}] lookback={lookback} {spec} lr={lr} batch={batch}")
        print(f"  reason: {rationale}")
        model = LSTMForecaster(
            **spec,
            train_config=TrainConfig(
                learning_rate=lr, batch_size=batch, epochs=TUNE_EPOCHS,
                patience=TUNE_PATIENCE, time_budget_seconds=TUNE_BUDGET_S, verbose=False,
            ),
        )
        result = run_model(model, data_for(lookback), "val", reference, repeats=2)
        result.hyperparameters["lookback"] = lookback
        log.add(i, rationale, result)
        print(f"  val MAE={result.metrics['MAE']:.2f}  MASE={result.metrics['MASE']:.3f}  "
              f"params={result.n_parameters:,}  train={result.train_seconds:.0f}s")

    return log.frame()


def tcn_study(square_id: int) -> pd.DataFrame:
    """Iterative TCN tuning, driven by receptive field before capacity.

    For a TCN the receptive field is a hard architectural limit, so it is tuned
    first: a network that physically cannot see 144 steps back cannot exploit the
    daily seasonality no matter how wide it is.
    """
    from .models import TCNForecaster, TrainConfig

    log = ExperimentLog(f"tcn_sq{square_id}")
    reference = baseline_mae(square_id, "val")
    cache: dict[int, D.SupervisedData] = {}

    def data_for(lookback: int):
        if lookback not in cache:
            cache[lookback] = D.supervised(square_id, lookback=lookback)
        return cache[lookback]

    plan = [
        dict(
            lookback=144, channels=16, levels=4, kernel=3, dropout=0.1,
            rationale="Receptive field 61 steps (~10 h): deliberately shorter than one "
                      "day, to demonstrate that the daily lag is out of reach.",
        ),
        dict(
            lookback=144, channels=16, levels=6, kernel=3, dropout=0.1,
            rationale="Two more dilation levels push the receptive field to 253 steps, "
                      "which covers the lag-144 daily peak identified by the ACF.",
        ),
        dict(
            lookback=144, channels=32, levels=6, kernel=3, dropout=0.1,
            rationale="With the receptive field sufficient, widen the channels to test "
                      "whether representational capacity is now the limit.",
        ),
        dict(
            lookback=144, channels=32, levels=6, kernel=3, dropout=0.0,
            rationale="Remove dropout: with only ~5k training windows the noise it "
                      "injects may cost more than the regularisation it buys.",
        ),
        dict(
            lookback=144, channels=24, levels=5, kernel=4, dropout=0.0,
            rationale="Receptive field 187 with one fewer dilation level, trading depth "
                      "for kernel width. The deepest level dominates cost because it "
                      "convolves the most heavily padded sequence, so this is markedly "
                      "cheaper at a comparable receptive field.",
        ),
        dict(
            lookback=288, channels=24, levels=7, kernel=3, dropout=0.0,
            rationale="Receptive field 509 over a two-day window, matching the "
                      "longest-context LSTM configuration for a fair comparison.",
        ),
    ]

    for i, spec in enumerate(plan, start=1):
        rationale = spec.pop("rationale")
        lookback = spec.pop("lookback")
        rf = 1 + 2 * (spec["kernel"] - 1) * (2 ** spec["levels"] - 1)
        print(f"\n[TCN {i}/{len(plan)}] lookback={lookback} {spec} -> receptive field {rf}")
        print(f"  reason: {rationale}")
        model = TCNForecaster(
            **spec,
            train_config=TrainConfig(
                epochs=TUNE_EPOCHS, patience=TUNE_PATIENCE, batch_size=256,
                learning_rate=2e-3, time_budget_seconds=TUNE_BUDGET_S, verbose=False,
            ),
        )
        result = run_model(model, data_for(lookback), "val", reference, repeats=2)
        result.hyperparameters["lookback"] = lookback
        log.add(i, rationale, result)
        print(f"  val MAE={result.metrics['MAE']:.2f}  MASE={result.metrics['MASE']:.3f}  "
              f"params={result.n_parameters:,}  train={result.train_seconds:.0f}s")

    return log.frame()


def tcn_receptive_field_study(square_id: int, epochs: int = 24) -> pd.DataFrame:
    """Controlled re-test of receptive field at matched capacity and epochs.

    Why this exists. In :func:`tcn_study` the configuration with the *shortest*
    receptive field (61 steps, about 10 hours) achieved the best validation error,
    which contradicts the expectation set by the lag-144 autocorrelation peak.
    That result cannot be trusted as stated, because it is confounded by compute:
    the small model was the only one to train freely to convergence (20 epochs),
    while every deeper configuration hit the wall-clock budget after 4 to 15
    epochs. Depth increases cost per epoch, so a fixed wall-clock budget
    systematically under-trains exactly the models being tested.

    This study removes the confound. Channel width is held at 16, kernel at 3,
    and the lookback at 144, so the *only* variable is the number of dilation
    levels and therefore the receptive field. Every configuration is trained for
    the same number of epochs with no wall-clock cap and with early stopping
    disabled, so all models receive equal optimisation effort. Cost is reported
    but no longer restricts training.
    """
    from .models import TCNForecaster, TrainConfig

    log = ExperimentLog(f"tcn_receptive_field_sq{square_id}")
    reference = baseline_mae(square_id, "val")
    data = D.supervised(square_id, lookback=C.SLOTS_PER_DAY)

    # Levels 4-6 span the informative range. A seventh level would give a
    # receptive field of 509 steps, but the input window is only 144 steps long,
    # so anything past 253 already sees the entire window: further depth adds
    # capacity, not context, and would not test the question this study asks.
    plan = [
        (4, "Receptive field 61 steps (~10 h): cannot reach the lag-144 daily peak."),
        (5, "Receptive field 125 steps (~21 h): just short of one full day."),
        (6, "Receptive field 253 steps (~42 h): covers the whole 144-step window."),
    ]

    for i, (levels, rationale) in enumerate(plan, start=1):
        rf = 1 + 2 * (3 - 1) * (2 ** levels - 1)
        print(f"\n[TCN-RF {i}/{len(plan)}] levels={levels} -> receptive field {rf}, "
              f"{epochs} epochs, no time cap\n  reason: {rationale}")
        model = TCNForecaster(
            channels=16, levels=levels, kernel=3, dropout=0.1,
            train_config=TrainConfig(
                epochs=epochs,
                patience=epochs + 1,   # disable early stopping: equalise effort
                batch_size=256,
                learning_rate=2e-3,
                time_budget_seconds=None,
                verbose=False,
            ),
        )
        result = run_model(model, data, "val", reference, repeats=2)
        result.hyperparameters["lookback"] = C.SLOTS_PER_DAY
        log.add(i, rationale, result)
        print(f"  val MAE={result.metrics['MAE']:.2f}  MASE={result.metrics['MASE']:.3f}  "
              f"params={result.n_parameters:,}  train={result.train_seconds:.0f}s  "
              f"({result.train_seconds / epochs:.1f}s/epoch)")

    return log.frame()


# ===========================================================================
# Final evaluation
# ===========================================================================


def build_final_models(config: dict):
    """Instantiate the tuned models. ``config`` comes from the tuning stage."""
    from .models import LSTMForecaster, TCNForecaster, TrainConfig

    lstm_cfg = config["lstm"]
    tcn_cfg = config["tcn"]

    return [
        (PersistenceForecaster(), 1),
        (SeasonalNaiveForecaster(), 1),
        (
            SarimaForecaster(
                order=tuple(config["sarima"]["order"]),
                trend=config["sarima"].get("trend", "n"),
            ),
            1,
        ),
        (
            LSTMForecaster(
                hidden_size=lstm_cfg["hidden_size"],
                num_layers=lstm_cfg["num_layers"],
                dropout=lstm_cfg.get("dropout", 0.0),
                train_config=TrainConfig(
                    learning_rate=lstm_cfg["lr"],
                    batch_size=lstm_cfg["batch_size"],
                    epochs=FINAL_EPOCHS,
                    patience=FINAL_PATIENCE,
                    time_budget_seconds=FINAL_BUDGET_S,
                    verbose=False,
                ),
            ),
            lstm_cfg["lookback"],
        ),
        (
            TCNForecaster(
                channels=tcn_cfg["channels"],
                levels=tcn_cfg["levels"],
                kernel=tcn_cfg["kernel"],
                dropout=tcn_cfg.get("dropout", 0.0),
                train_config=TrainConfig(
                    epochs=FINAL_EPOCHS,
                    patience=FINAL_PATIENCE,
                    batch_size=tcn_cfg.get("batch_size", 256),
                    learning_rate=2e-3,
                    time_budget_seconds=FINAL_BUDGET_S,
                    verbose=False,
                ),
            ),
            tcn_cfg["lookback"],
        ),
    ]


def final_evaluation(square_ids: list[int], config: dict) -> tuple[pd.DataFrame, dict]:
    """Train and evaluate every model on the test week for each area.

    Returns the metric table and a nested dict of predictions for plotting.
    """
    rows, predictions = [], {}

    for square_id in square_ids:
        print(f"\n{'=' * 72}\nArea {square_id}\n{'=' * 72}")
        reference = baseline_mae(square_id, "test")
        predictions[square_id] = {}

        for model, lookback in build_final_models(config):
            data = D.supervised(square_id, lookback=lookback)
            print(f"  fitting {model.name} (lookback={lookback}) ...", flush=True)
            result = run_model(model, data, "test", reference)
            result.hyperparameters["lookback"] = lookback
            rows.append(result.row())
            predictions[square_id][model.name] = result.predictions
            print(
                f"    MAE={result.metrics['MAE']:8.2f}  RMSE={result.metrics['RMSE']:8.2f}  "
                f"MAPE={result.metrics['MAPE_%']:6.2f}%  MASE={result.metrics['MASE']:.3f}  "
                f"train={result.train_seconds:6.1f}s  infer={result.inference_ms_per_step:.3f} ms/step"
            )
        predictions[square_id]["actual"] = data.y_test_raw
        predictions[square_id]["index"] = data.test_index

    frame = pd.DataFrame(rows)
    frame.to_csv(C.RESULTS_DIR / "final_results.csv", index=False)

    # Persist the predictions themselves, not just their summary statistics.
    # Figures can then be redrawn without retraining, which matters here for a
    # specific reason: the training loop is bounded by wall-clock time as well as
    # by epochs, so a run on a differently loaded machine may stop at a different
    # epoch and produce different numbers. The seed alone does not make the
    # pipeline reproducible; these files are what pin the reported results.
    for square_id, bundle in predictions.items():
        out = pd.DataFrame({"actual": bundle["actual"]}, index=bundle["index"])
        for name, values in bundle.items():
            if name not in ("actual", "index"):
                out[name] = values
        out.index.name = "timestamp"
        out.to_csv(C.RESULTS_DIR / f"predictions_square_{square_id}.csv")

    return frame, predictions


# ===========================================================================
# Reporting artefacts
# ===========================================================================

#: The three models under comparison, in the order used throughout the report.
COMPARED = ("SARIMA", "LSTM", "TCN")


def per_area_tables(frame: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """One MAE/MAPE/RMSE table per area, as required by the assignment."""
    tables = {}
    for square_id, group in frame.groupby("square_id"):
        table = (
            group.set_index("model")[
                ["MAE", "MAPE_%", "RMSE", "sMAPE_%", "MASE", "R2", "RMSE/MAE"]
            ]
            .round(3)
            .reindex(["Persistence", "Seasonal naive", *COMPARED])
        )
        tables[int(square_id)] = table
        table.to_csv(C.RESULTS_DIR / f"metrics_square_{square_id}.csv")
    return tables


def timing_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Training and inference cost, averaged across the three areas."""
    grouped = (
        frame.groupby("model")
        .agg(
            train_seconds_mean=("train_seconds", "mean"),
            train_seconds_min=("train_seconds", "min"),
            train_seconds_max=("train_seconds", "max"),
            inference_seconds_week=("inference_seconds", "mean"),
            inference_ms_per_step=("inference_ms_per_step", "mean"),
            n_parameters=("n_parameters", "max"),
            best_epoch=("best_epoch", "mean"),
        )
        .round(4)
        .reindex(["Persistence", "Seasonal naive", *COMPARED])
    )
    grouped.to_csv(C.RESULTS_DIR / "timing.csv")

    (C.RESULTS_DIR / "hardware.json").write_text(
        json.dumps(
            {
                **system_summary(),
                "torch_threads": _torch_threads(),
                "measurement_protocol": (
                    "Training time is wall-clock for a complete fit including early "
                    f"stopping, measured with time.perf_counter. Inference time is the "
                    f"median of {INFERENCE_REPEATS} repeated one-step-ahead passes over "
                    "all 1008 slots of the test week. Values are averaged over the three "
                    "study areas. PyTorch is pinned to 2 threads to match the 2 physical "
                    "cores and keep repeats comparable."
                ),
            },
            indent=2,
        )
    )
    return grouped


def _torch_threads() -> int | None:
    try:
        import torch

        return torch.get_num_threads()
    except Exception:
        return None


def prediction_figures(predictions: dict, models=COMPARED) -> list[str]:
    """The nine required actual-vs-predicted plots, plus per-area overlays."""
    paths = []
    for square_id, bundle in predictions.items():
        index = bundle["index"]
        actual = bundle["actual"]

        for model in models:
            if model not in bundle:
                continue
            pred = bundle[model]
            fig, (ax, axr) = plt.subplots(
                2, 1, figsize=(11.5, 5.0), sharex=True,
                gridspec_kw={"height_ratios": [3, 1]},
            )
            ax.plot(index, actual, lw=1.0, color="#1a202c", label="Observed")
            ax.plot(index, pred, lw=1.0, color="#c53030", alpha=0.85, label=f"{model} forecast")
            ax.set_ylabel("Internet activity")
            ax.set_title(
                f"{model} one-step-ahead forecast - square {square_id}, "
                f"16-22 December 2013   "
                f"(MAE {M.mae(actual, pred):.1f}, RMSE {M.rmse(actual, pred):.1f})",
                loc="left",
            )
            ax.legend(loc="upper right", fontsize=8)

            axr.plot(index, actual - pred, lw=0.7, color="#2b6cb0")
            axr.axhline(0, color="k", lw=0.6)
            axr.set_ylabel("residual")
            axr.set_xlabel("Local time (CET)")
            axr.xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%H:%M"))
            axr.xaxis.set_major_locator(mdates.DayLocator())

            path = C.FIGURES_DIR / f"forecast_sq{square_id}_{model.replace(' ', '_')}.png"
            fig.savefig(path)
            plt.close(fig)
            paths.append(str(path))

        # Combined overlay: makes the model-to-model differences directly visible.
        fig, ax = plt.subplots(figsize=(11.5, 4.2))
        ax.plot(index, actual, lw=1.3, color="#1a202c", label="Observed", zorder=5)
        for model, colour in zip(models, ("#c53030", "#2b6cb0", "#2f855a")):
            if model in bundle:
                ax.plot(index, bundle[model], lw=0.85, alpha=0.8, color=colour, label=model)
        ax.set(
            title=f"All models, square {square_id}, test week",
            ylabel="Internet activity", xlabel="Local time (CET)",
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.legend(fontsize=8, ncol=4)
        path = C.FIGURES_DIR / f"forecast_sq{square_id}_all_models.png"
        fig.savefig(path)
        plt.close(fig)
        paths.append(str(path))

    print(f"[figures] wrote {len(paths)} forecast figures")
    return paths


def tuning_figure(studies: dict[str, pd.DataFrame]) -> str:
    """Validation MAE across the experiment sequence for each model family."""
    fig, axes = plt.subplots(1, len(studies), figsize=(4.2 * len(studies), 3.4))
    for ax, (name, frame) in zip(np.atleast_1d(axes), studies.items()):
        if frame.empty:
            continue
        ax.plot(frame["experiment"], frame["MAE"], "o-", color="#2b6cb0")
        best = frame["MAE"].idxmin()
        ax.plot(frame.loc[best, "experiment"], frame.loc[best, "MAE"], "o",
                ms=11, mfc="none", mec="#c53030", mew=2)
        ax.set(title=f"{name} tuning sequence", xlabel="experiment #", ylabel="validation MAE")
        ax.set_xticks(frame["experiment"])
    fig.suptitle("Validation error over the documented experiment sequence "
                 "(circled = selected)", y=1.04)
    path = C.FIGURES_DIR / "tuning_sequences.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def failure_analysis(predictions: dict, square_id: int, models=COMPARED) -> pd.DataFrame:
    """Locate and visualise the worst-forecast period of the test week.

    The window is chosen automatically as the 6-hour span with the highest mean
    absolute error, averaged over the three compared models, so the example is
    selected by evidence rather than picked to flatter or embarrass a model.
    """
    bundle = predictions[square_id]
    index, actual = bundle["index"], bundle["actual"]

    stacked = np.vstack([np.abs(actual - bundle[m]) for m in models if m in bundle])
    mean_err = stacked.mean(axis=0)

    win = 36  # 6 hours
    rolling = pd.Series(mean_err).rolling(win).mean()
    end = int(rolling.idxmax())
    start = max(0, end - win)

    fig, (ax, axe) = plt.subplots(
        2, 1, figsize=(11.0, 5.4), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    sl = slice(max(0, start - 72), min(len(index), end + 72))
    ax.plot(index[sl], actual[sl], lw=1.6, color="#1a202c", label="Observed", zorder=5)
    for model, colour in zip(models, ("#c53030", "#2b6cb0", "#2f855a")):
        if model in bundle:
            ax.plot(index[sl], bundle[model][sl], lw=1.1, color=colour, label=model)
    ax.axvspan(index[start], index[end - 1], color="orange", alpha=0.15, lw=0,
               label="worst 6-hour window")
    ax.set(title=f"Worst-forecast window, square {square_id}", ylabel="Internet activity")
    ax.legend(fontsize=8, ncol=5)

    for model, colour in zip(models, ("#c53030", "#2b6cb0", "#2f855a")):
        if model in bundle:
            axe.plot(index[sl], (actual - bundle[model])[sl], lw=0.9, color=colour)
    axe.axhline(0, color="k", lw=0.6)
    axe.set(ylabel="residual", xlabel="Local time (CET)")
    axe.xaxis.set_major_formatter(mdates.DateFormatter("%a %d\n%H:%M"))
    fig.savefig(C.FIGURES_DIR / f"failure_window_sq{square_id}.png")
    plt.close(fig)

    rows = []
    for model in models:
        if model not in bundle:
            continue
        pred = bundle[model]
        rows.append(
            {
                "model": model,
                "window_start": index[start].strftime("%a %d %b %H:%M"),
                "window_end": index[end - 1].strftime("%a %d %b %H:%M"),
                "window_MAE": M.mae(actual[start:end], pred[start:end]),
                "week_MAE": M.mae(actual, pred),
                "ratio": M.mae(actual[start:end], pred[start:end]) / M.mae(actual, pred),
                "window_bias": float(np.mean(pred[start:end] - actual[start:end])),
            }
        )
    table = pd.DataFrame(rows).round(3)
    table.to_csv(C.RESULTS_DIR / f"failure_analysis_sq{square_id}.csv", index=False)

    # Error by hour of day, which explains *when* the models struggle.
    hourly = pd.concat(
        {
            model: M.error_by_hour(actual, bundle[model], index)
            for model in models
            if model in bundle
        },
        axis=1,
    )
    hourly.round(3).to_csv(C.RESULTS_DIR / f"error_by_hour_sq{square_id}.csv")

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    for model, colour in zip(models, ("#c53030", "#2b6cb0", "#2f855a")):
        if model in bundle:
            ax.plot(hourly.index, hourly[(model, "mae")], "o-", color=colour, label=model, ms=3)
    ax2 = ax.twinx()
    ax2.fill_between(hourly.index, hourly[(models[0], "mean_actual")],
                     color="grey", alpha=0.15, lw=0)
    ax2.set_ylabel("mean observed activity (shaded)")
    ax2.grid(False)
    ax.set(title=f"Forecast error by hour of day, square {square_id}",
           xlabel="hour of day (CET)", ylabel="MAE")
    ax.legend(fontsize=8)
    fig.savefig(C.FIGURES_DIR / f"error_by_hour_sq{square_id}.png")
    plt.close(fig)

    return table
