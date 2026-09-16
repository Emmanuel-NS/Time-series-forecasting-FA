"""Stage 5 - final forecasting experiments on the 16-22 December test week.

Trains the tuned SARIMA, LSTM and TCN (plus the two reference baselines) on each
of the three highest-traffic areas and produces every artefact the report needs:

  * ``figures/forecast_sq<id>_<model>.png``  - the nine required actual-vs-forecast plots
  * ``results/metrics_square_<id>.csv``      - the three required metric tables
  * ``results/timing.csv`` + ``hardware.json`` - training/inference cost and the host
  * ``figures/failure_window_sq<id>.png``    - worst-forecast period
  * ``results/error_by_hour_sq<id>.csv``     - when each model struggles
"""

import argparse
import json

import _bootstrap  # noqa: F401

import pandas as pd

from src import config as C
from src import data as D
from src import experiments as E


DEFAULT_CONFIG = {
    "sarima": {"order": [2, 0, 1], "seasonal_period": C.SLOTS_PER_DAY},
    "lstm": {"lookback": 144, "hidden_size": 64, "num_layers": 1, "dropout": 0.0,
             "lr": 1e-3, "batch_size": 128},
    "tcn": {"lookback": 144, "channels": 32, "levels": 6, "kernel": 3,
            "dropout": 0.0, "batch_size": 128},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--squares", default=None,
                        help="comma-separated area ids (default: top 3 by total traffic)")
    args = parser.parse_args()

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    D.require_complete_store()

    config_path = C.RESULTS_DIR / "tuned_config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
        print(f"[final] using tuned configuration from {config_path.name}")
    else:
        config = DEFAULT_CONFIG
        print("[final] tuned_config.json not found; falling back to documented defaults")
    for key in ("sarima", "lstm", "tcn"):
        config.setdefault(key, DEFAULT_CONFIG[key])

    squares = (
        [int(s) for s in args.squares.split(",")] if args.squares else D.top_squares(3)
    )
    print(f"[final] evaluation areas: {squares}")
    print(D.default_split().describe().to_string(index=False))
    print("\n[final] model configuration:")
    print(json.dumps({k: config[k] for k in ("sarima", "lstm", "tcn")}, indent=2))

    frame, predictions = E.final_evaluation(squares, config)

    print("\n" + "=" * 78)
    print("Per-area performance tables (test week 16-22 December 2013)")
    print("=" * 78)
    for square_id, table in E.per_area_tables(frame).items():
        print(f"\nSquare {square_id}")
        print(table.to_string())

    print("\n" + "=" * 78)
    print("Training and inference cost (mean over the three areas)")
    print("=" * 78)
    print(E.timing_table(frame).to_string())

    E.prediction_figures(predictions)

    print("\n" + "=" * 78)
    print("Failure analysis")
    print("=" * 78)
    for square_id in squares:
        print(f"\nSquare {square_id}")
        print(E.failure_analysis(predictions, square_id).to_string(index=False))

    print(f"\n[final] figures -> {C.FIGURES_DIR}")
    print(f"[final] tables  -> {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
