"""Stage 4 - iterative hyperparameter experimentation.

Tuning is performed on the highest-traffic area only, and selection uses the
validation week (9-15 December) exclusively. The test week (16-22 December) is
never read at this stage. The chosen configuration is then applied unchanged to
all three areas in stage 5, so the comparison across areas measures
generalisation rather than per-area overfitting.

Every configuration tried, with the reason it was tried, is appended to
``results/tuning_<model>_sq<id>.csv``. The selected configuration is written to
``results/tuned_config.json``.
"""

import argparse
import json

import _bootstrap  # noqa: F401

import pandas as pd

from src import config as C
from src import data as D
from src import experiments as E
from src.profiling import system_summary


def _select_lstm(frame: pd.DataFrame) -> dict:
    best = frame.loc[frame["MAE"].idxmin()]
    hp = json.loads(best["hyperparameters"])
    return {
        "lookback": int(hp["lookback"]),
        "hidden_size": int(hp["hidden_size"]),
        "num_layers": int(hp["num_layers"]),
        "dropout": float(hp["dropout"]),
        "lr": float(hp["lr"]),
        "batch_size": int(hp["batch_size"]),
        "val_MAE": float(best["MAE"]),
        "experiment": int(best["experiment"]),
    }


def _select_tcn(frame: pd.DataFrame) -> dict:
    best = frame.loc[frame["MAE"].idxmin()]
    hp = json.loads(best["hyperparameters"])
    return {
        "lookback": int(hp["lookback"]),
        "channels": int(hp["channels"]),
        "levels": int(hp["levels"]),
        "kernel": int(hp["kernel"]),
        "dropout": float(hp["dropout"]),
        "batch_size": int(hp["batch_size"]),
        "receptive_field": int(hp["receptive_field"]),
        "val_MAE": float(best["MAE"]),
        "experiment": int(best["experiment"]),
    }


def _select_tcn_receptive_field(frame: pd.DataFrame) -> dict:
    """Pick a TCN depth from the controlled receptive-field study.

    Selection is deliberately *not* ``argmin(MAE)``. The study varies only depth,
    so if receptive field mattered the errors would vary monotonically with it.
    They do not: the ordering is non-monotonic, and the size of the non-monotonic
    dip is a lower bound on run-to-run variation, because with one seed per
    configuration nothing else can explain a middle configuration scoring worse
    than both of its neighbours.

    That bound is used as the decision threshold. Any configuration within it of
    the best is treated as indistinguishable, and the **shallowest** such
    configuration is chosen: fewest parameters, cheapest per epoch, and no
    reliance on a difference the experiment cannot resolve. This is the same
    parsimony rule applied to the SARIMA order search.
    """
    runs = []
    for _, row in frame.iterrows():
        hp = json.loads(row["hyperparameters"])
        runs.append((int(hp["levels"]), int(hp["receptive_field"]), float(row["MAE"]), row))
    runs.sort()

    best_mae = min(mae for _, _, mae, _ in runs)

    # Lower bound on run-to-run noise: the largest amount by which any interior
    # configuration exceeds *both* of its depth-neighbours. With one seed per
    # configuration and depth the only variable, such a dip cannot be explained
    # by receptive field, so it must be variation. Zero if the ordering is
    # monotonic, in which case only exact ties are pooled.
    noise = 0.0
    for i in range(1, len(runs) - 1):
        dip = min(runs[i][2] - runs[i - 1][2], runs[i][2] - runs[i + 1][2])
        noise = max(noise, dip)

    # Shallowest configuration indistinguishable from the best.
    chosen = next(row for _, _, mae, row in runs if mae - best_mae <= noise)
    hp = json.loads(chosen["hyperparameters"])
    scores = ", ".join(f"RF {rf}: {mae:.2f}" for _, rf, mae, _ in runs)

    return {
        "lookback": int(hp["lookback"]),
        "channels": int(hp["channels"]),
        "levels": int(hp["levels"]),
        "kernel": int(hp["kernel"]),
        "dropout": float(hp["dropout"]),
        "batch_size": int(hp["batch_size"]),
        "receptive_field": int(hp["receptive_field"]),
        "val_MAE": float(chosen["MAE"]),
        "experiment": f"receptive-field study #{int(chosen['experiment'])}",
        "selection_note": (
            f"Chosen for parsimony, not for best validation MAE. Controlled study "
            f"(equal epochs, early stopping disabled) gave {scores}. The ordering is "
            f"non-monotonic, bounding run-to-run variation at >= {noise:.2f} MAE with a "
            f"single seed per configuration; this configuration is within that bound of "
            f"the best ({best_mae:.2f}) and is the shallowest such, so the difference is "
            f"not evidence of a real effect. Substantive finding: receptive field has no "
            f"systematic effect on one-step-ahead accuracy over this range."
        ),
    }


def _select_sarima(frame: pd.DataFrame) -> dict:
    best = frame.loc[frame["MAE"].idxmin()]
    hp = json.loads(best["hyperparameters"])
    return {
        "order": [int(hp["p"]), int(hp["d"]), int(hp["q"])],
        "seasonal_period": int(hp["seasonal_period"]),
        "trend": str(hp["trend"]),
        "val_MAE": float(best["MAE"]),
        "experiment": int(best["experiment"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--square", type=int, default=None,
                        help="area to tune on (default: highest total traffic)")
    parser.add_argument("--models", default="sarima,lstm,tcn",
                        help="comma-separated subset to tune: sarima, lstm, tcn, tcn-rf")
    args = parser.parse_args()

    pd.set_option("display.width", 220)
    D.require_complete_store()
    square = args.square or D.top_squares(1)[0]
    wanted = {m.strip().lower() for m in args.models.split(",")}

    print("[tune] host:", system_summary())
    print(f"[tune] tuning area: square {square}")
    print(D.default_split().describe().to_string(index=False))

    config_path = C.RESULTS_DIR / "tuned_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    config["tuning_square"] = square
    studies = {}

    if "sarima" in wanted:
        print("\n" + "#" * 78)
        print("# SARIMA order selection")
        print("#" * 78)
        frame = E.sarima_study(square)
        studies["SARIMA"] = frame
        config["sarima"] = _select_sarima(frame)
        print(f"\n  selected: {config['sarima']}")

    if "lstm" in wanted:
        print("\n" + "#" * 78)
        print("# LSTM tuning")
        print("#" * 78)
        frame = E.lstm_study(square)
        studies["LSTM"] = frame
        config["lstm"] = _select_lstm(frame)
        print(f"\n  selected: {config['lstm']}")

    if "tcn" in wanted:
        print("\n" + "#" * 78)
        print("# TCN tuning")
        print("#" * 78)
        frame = E.tcn_study(square)
        studies["TCN"] = frame
        config["tcn"] = _select_tcn(frame)
        print(f"\n  selected: {config['tcn']}")

    if "tcn-rf" in wanted:
        print("\n" + "#" * 78)
        print("# TCN receptive-field study (controlled: matched capacity and epochs)")
        print("#" * 78)
        frame = E.tcn_receptive_field_study(square)
        studies["TCN receptive field"] = frame
        config["tcn"] = _select_tcn_receptive_field(frame)
        print(f"\n  selected: {config['tcn']}")

    config_path.write_text(json.dumps(config, indent=2))
    print(f"\n[tune] selected configuration -> {config_path}")

    if studies:
        print("[tune] figure ->", E.tuning_figure(studies))


if __name__ == "__main__":
    main()
