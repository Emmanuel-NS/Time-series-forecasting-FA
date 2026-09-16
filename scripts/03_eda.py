"""Stage 3 - exploratory analysis and time-series characterisation.

Answers Task 2 of the assignment:
  * distribution of total Internet traffic across the 10,000 areas
  * the three highest-traffic areas, plus squares 4159 and 4556, over two weeks
  * two further analyses of the highest-traffic area (autocorrelation and
    periodicity; stationarity and STL decomposition)

Writes figures to ``figures/`` and the numbers behind them to ``results/``.
"""

import argparse

import _bootstrap  # noqa: F401

import pandas as pd

from src import config as C
from src import data as D
from src import eda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="run against an incomplete store (smoke testing only; results invalid)",
    )
    args = parser.parse_args()

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    D.require_complete_store(allow_partial=args.allow_partial)

    print("=" * 78)
    print("2.1  Distribution of total Internet traffic across geographical areas")
    print("=" * 78)
    summary = eda.spatial_distribution()
    print(summary.T.to_string(header=["value"]))

    top3 = D.top_squares(3)
    squares = D.study_squares()
    print(f"\nTop-3 areas by total traffic: {top3}")
    print(f"Study set (top-3 + reference squares {C.REFERENCE_SQUARES}): {squares}")

    print("\n" + "=" * 78)
    print("2.2  Traffic time series, first two weeks, five areas")
    print("=" * 78)
    area_stats = eda.two_week_series(squares)
    print(area_stats.to_string(index=False))
    eda.weekly_profiles(squares)

    print("\nDiurnal signatures and land-use interpretation:")
    print(eda.diurnal_signatures(squares).to_string(index=False))

    print("\n" + "=" * 78)
    print("2.3  Additional analysis A - autocorrelation and periodicity")
    print("=" * 78)
    acf_result = eda.autocorrelation_analysis()
    print("\nAutocorrelation at candidate seasonal lags:")
    print(acf_result.seasonal_peaks.to_string(index=False))
    print("\nStrongest spectral peaks:")
    print(acf_result.summary.to_string(index=False))

    print("\n" + "=" * 78)
    print("2.4  Additional analysis B - stationarity and decomposition")
    print("=" * 78)
    tests = eda.stationarity_and_decomposition()
    print(tests.to_string(index=False))

    print(f"\nFigures -> {C.FIGURES_DIR}")
    print(f"Tables  -> {C.RESULTS_DIR}")


if __name__ == "__main__":
    main()
