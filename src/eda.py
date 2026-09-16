"""Exploratory analysis and time-series characterisation.

Each function produces one piece of evidence required by Task 2 of the
assignment and returns the numbers behind the figure, so that the report can
quote statistics rather than describe pictures.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")  # headless: figures are written to disk, never displayed

import matplotlib.dates as mdates
import matplotlib.patheffects as patheffects
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller, kpss, pacf

from . import config as C
from . import data as D
from . import grid

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

FIG = C.FIGURES_DIR
TAB = C.RESULTS_DIR


def _save(fig: plt.Figure, name: str) -> str:
    path = FIG / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  figure -> {path.name}")
    return str(path)


# ===========================================================================
# 2.1  Spatial distribution of total Internet traffic
# ===========================================================================


def spatial_distribution() -> pd.DataFrame:
    """Distribution of two-month total Internet traffic across the 10,000 areas.

    Produces a three-panel figure: the raw histogram (to show the skew), the
    same data on a log axis (to show the underlying shape), and the traffic
    rendered on the 100x100 Milan grid (to show that the skew is spatially
    structured rather than random).
    """
    totals = D.square_totals()
    positive = totals[totals > 0]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))

    axes[0].hist(totals, bins=120, color="#2b6cb0")
    axes[0].set(
        title="(a) Total Internet activity per area",
        xlabel="Total activity (arbitrary units)",
        ylabel="Number of areas",
    )
    axes[0].set_yscale("log")

    axes[1].hist(np.log10(positive), bins=120, color="#2f855a")
    axes[1].set(
        title="(b) Same distribution, $\\log_{10}$ scale",
        xlabel="$\\log_{10}$ total activity",
        ylabel="Number of areas",
    )

    # Panel (c): a geographically correct map. The grid convention
    # (row = north, col = east) is verified in src/grid.py against the official
    # cell centroids, so `origin="lower"` really is north-up.
    grid_values = totals.reshape(C.GRID_SIDE, C.GRID_SIDE)
    extent = None
    if grid.available():
        coords = grid.centroids()
        extent = (
            coords[:, 0].min(), coords[:, 0].max(),
            coords[:, 1].min(), coords[:, 1].max(),
        )
    # At 45.5 deg N one degree of longitude spans ~78 km against ~111 km for one
    # degree of latitude, so an equal-units aspect would stretch the city
    # east-west. Scaling by 1/cos(latitude) renders true proportions.
    aspect = 1.0 / np.cos(np.radians(45.46)) if extent else "equal"
    im = axes[2].imshow(
        np.log10(np.maximum(grid_values, 1.0)),
        cmap="magma", origin="lower", aspect=aspect, extent=extent,
    )
    if extent:
        # Dark outline behind every annotation: the map is bright in the centre,
        # which is exactly where the labels sit.
        stroke = [patheffects.withStroke(linewidth=1.7, foreground="#0b0b12")]

        # A small, well-separated set of landmarks, with label offsets chosen to
        # push text away from the crowded city centre.
        placements = {
            "Duomo (historic centre)": (6, -11),
            "Milano Centrale station": (5, 4),
            "San Siro stadium": (-6, 5),
            "Linate airport": (-10, 5),
        }
        for name, offset in placements.items():
            lat, lon = grid.LANDMARKS[name]
            axes[2].plot(lon, lat, "o", ms=3.0, mfc="none", mec="#67e8f9", mew=1.0)
            axes[2].annotate(
                name.split(" (")[0], (lon, lat), fontsize=5.8, color="#cffafe",
                xytext=offset, textcoords="offset points", path_effects=stroke,
            )
        for rank, cell in enumerate(np.argsort(totals)[::-1][:3], 1):
            lon, lat = coords[cell]
            axes[2].plot(lon, lat, "s", ms=7, mfc="none", mec="#4ade80", mew=1.5)
        axes[2].annotate(
            "top 3 areas", coords[int(np.argmax(totals))], fontsize=6.2,
            color="#4ade80", fontweight="bold", xytext=(9, 7),
            textcoords="offset points", path_effects=stroke,
        )
        for sq, offset in zip(C.REFERENCE_SQUARES, ((-24, -9), (-24, 5))):
            lon, lat = coords[sq - 1]
            axes[2].plot(lon, lat, "^", ms=6, mfc="none", mec="#fb7185", mew=1.4)
            axes[2].annotate(
                str(sq), (lon, lat), fontsize=6.2, color="#fecdd3",
                xytext=offset, textcoords="offset points", path_effects=stroke,
            )
        axes[2].set(title="(c) Geographic distribution", xlabel="longitude ($\\degree$E)",
                    ylabel="latitude ($\\degree$N)")
    else:
        axes[2].set(title="(c) Spatial layout of the Milan grid",
                    xlabel="grid column (east)", ylabel="grid row (north)")
    axes[2].grid(False)
    fig.colorbar(im, ax=axes[2], label="$\\log_{10}$ total activity", fraction=0.046)

    fig.suptitle(
        "Distribution of total Internet traffic across 10,000 Milan areas "
        "(1 Nov 2013 - 1 Jan 2014)",
        y=1.04,
    )
    _save(fig, "eda_01_spatial_distribution")

    shares = np.sort(totals)[::-1] / totals.sum()
    summary = pd.DataFrame(
        [
            {
                "n_areas": int(totals.size),
                "n_zero_areas": int((totals == 0).sum()),
                "mean": totals.mean(),
                "median": np.median(totals),
                "std": totals.std(),
                "min": totals.min(),
                "max": totals.max(),
                "max_over_median": totals.max() / np.median(totals),
                "skewness": float(stats.skew(totals)),
                "kurtosis_excess": float(stats.kurtosis(totals)),
                "share_top_1pct_%": 100 * shares[:100].sum(),
                "share_top_10pct_%": 100 * shares[:1000].sum(),
                "share_bottom_50pct_%": 100 * shares[5000:].sum(),
                "gini": _gini(totals),
            }
        ]
    )
    summary.T.to_csv(TAB / "eda_spatial_summary.csv", header=["value"])

    # Concentration curve: how much of the city's traffic the busiest areas carry.
    cumulative = np.cumsum(shares)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    ax.plot(100 * np.arange(1, len(shares) + 1) / len(shares), 100 * cumulative,
            color="#2b6cb0", lw=1.4)
    ax.plot([0, 100], [0, 100], "--", color="grey", lw=0.9, label="perfect uniformity")
    for pct in (1, 10, 25):
        n = int(pct / 100 * len(shares))
        ax.plot(pct, 100 * cumulative[n - 1], "o", color="#c53030", ms=4)
        ax.annotate(f"top {pct}% of areas\ncarry {100 * cumulative[n - 1]:.0f}%",
                    (pct, 100 * cumulative[n - 1]), fontsize=6.8,
                    xytext=(7, -9), textcoords="offset points")
    ax.set(title="Spatial concentration of Internet traffic",
           xlabel="areas, ranked by total traffic (%)",
           ylabel="cumulative share of city traffic (%)")
    ax.legend(fontsize=7.5, loc="lower right")
    _save(fig, "eda_01b_concentration_curve")

    if grid.available():
        locations = pd.DataFrame(grid.describe_many(np.argsort(totals)[::-1][:5] + 1))
        locations["total_activity"] = np.sort(totals)[::-1][:5]
        locations["share_of_city_%"] = 100 * locations["total_activity"] / totals.sum()
        locations.round(4).to_csv(TAB / "eda_top_area_locations.csv", index=False)
        print("\n  Location of the highest-traffic areas:")
        print(locations.to_string(index=False))

    return summary


def _gini(values: np.ndarray) -> float:
    """Gini coefficient: 0 = every area identical, 1 = all traffic in one area."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    n = x.size
    cum = np.cumsum(x)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


# ===========================================================================
# 2.2  Time series of the five required areas over the first two weeks
# ===========================================================================


def two_week_series(squares: list[int] | None = None) -> pd.DataFrame:
    """Plot the first two weeks for the top-3 areas plus squares 4159 and 4556.

    A shared y-axis would render the two reference squares as flat lines next to
    the top-3 areas, so each area gets its own panel with its own scale, and a
    second normalised figure makes the *shape* comparison possible.
    """
    squares = squares or D.study_squares()
    end = 14 * C.SLOTS_PER_DAY
    wide = D.frame(squares).iloc[:end]
    labels = _labels(squares)

    fig, axes = plt.subplots(len(squares), 1, figsize=(11.5, 2.05 * len(squares)), sharex=True)
    for ax, sq in zip(axes, squares):
        s = wide[f"square_{sq}"]
        ax.plot(s.index, s.values, lw=0.7, color="#2b6cb0")
        ax.set_ylabel("activity")
        ax.set_title(labels[sq], loc="left", fontsize=8.2)
        # Shade weekends: the assignment asks about temporal dynamics, and the
        # weekly cycle is the single most visible feature.
        for day in pd.date_range(s.index[0].normalize(), s.index[-1], freq="D"):
            if day.weekday() >= 5:
                ax.axvspan(day, day + pd.Timedelta(days=1), color="grey", alpha=0.12, lw=0)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axes[-1].set_xlabel("Local date (CET), 1-14 November 2013.  Shaded = weekend")
    fig.suptitle("Internet traffic in five Milan areas, first two weeks", y=1.0)
    _save(fig, "eda_02_two_week_series")

    # Normalised overlay: divides each series by its own mean, isolating shape.
    fig, ax = plt.subplots(figsize=(11.5, 4.0))
    for sq in squares:
        s = wide[f"square_{sq}"]
        ax.plot(s.index, s.values / s.mean(), lw=0.8, alpha=0.85, label=labels[sq])
    ax.set(
        title="Mean-normalised traffic: comparison of temporal shape, not magnitude",
        ylabel="activity / area mean",
        xlabel="Local date (CET)",
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(fontsize=7.5, ncol=2)
    _save(fig, "eda_03_two_week_normalised")

    stats_rows = []
    full = D.frame(squares)
    for sq in squares:
        s = full[f"square_{sq}"]
        by_dow = s.groupby(s.index.dayofweek).mean()
        weekday, weekend = by_dow.loc[:4].mean(), by_dow.loc[5:].mean()
        night = s[(s.index.hour >= 3) & (s.index.hour < 5)].mean()
        stats_rows.append(
            {
                "square": sq,
                "role": labels[sq],
                "total": s.sum(),
                "mean": s.mean(),
                "std": s.std(),
                "cv": s.std() / s.mean(),
                "min": s.min(),
                "max": s.max(),
                "peak_over_trough": s.max() / max(night, 1e-9),
                "weekday_mean": weekday,
                "weekend_mean": weekend,
                "weekend_ratio": weekend / weekday,
                "peak_hour": int(s.groupby(s.index.hour).mean().idxmax()),
            }
        )
    table = pd.DataFrame(stats_rows)
    table.to_csv(TAB / "eda_area_statistics.csv", index=False)
    return table


def diurnal_signatures(squares: list[int] | None = None) -> pd.DataFrame:
    """Weekday vs weekend hourly profile for each study area.

    This is the analysis that connects the statistics to the city. The two
    reference squares the assignment specifies are not arbitrary: square 4159
    lies ~0.4 km from the Bocconi University campus and square 4556 ~0.4 km from
    the Navigli nightlife district, whereas the highest-traffic areas sit beside
    the Duomo. If land use drives demand, then these areas should show
    *qualitatively different* daily shapes rather than merely different volumes,
    and the weekday/weekend contrast should separate a campus from a nightlife
    district. Testing that prediction is what makes the observed patterns
    explanatory rather than decorative.
    """
    squares = squares or D.study_squares()
    labels = _labels(squares)

    fig, axes = plt.subplots(
        1, len(squares), figsize=(2.65 * len(squares), 3.2), sharey=True
    )
    rows = []
    for ax, sq in zip(np.atleast_1d(axes), squares):
        s = D.series(sq)
        is_weekend = s.index.dayofweek >= 5
        weekday = s[~is_weekend].groupby(s[~is_weekend].index.hour).mean()
        weekend = s[is_weekend].groupby(s[is_weekend].index.hour).mean()
        overall = s.mean()

        ax.plot(weekday.index, weekday / overall, lw=1.5, color="#2b6cb0", label="Mon-Fri")
        ax.plot(weekend.index, weekend / overall, lw=1.5, color="#c53030", label="Sat-Sun")
        ax.axhline(1.0, color="grey", lw=0.6, ls=":")
        ax.set(xlabel="hour (CET)", xticks=[0, 6, 12, 18, 23])
        ax.set_title(_short_label(sq), fontsize=7.8)
        if ax is np.atleast_1d(axes)[0]:
            ax.set_ylabel("activity / area mean")
            ax.legend(fontsize=6.8)

        night = s[(s.index.hour >= 22) | (s.index.hour < 2)].mean()
        office = s[(s.index.hour >= 9) & (s.index.hour < 18)].mean()
        location = grid.describe(sq) if grid.available() else {}
        rows.append(
            {
                "square": sq,
                "nearest_landmark": location.get("nearest_landmark", "n/a"),
                "distance_km": location.get("distance_km", np.nan),
                "mean": overall,
                "late_night_ratio": night / overall,
                "office_hours_ratio": office / overall,
                "night_over_office": night / office,
                "weekend_over_weekday": weekend.mean() / weekday.mean(),
                "weekday_peak_hour": int(weekday.idxmax()),
                "weekend_peak_hour": int(weekend.idxmax()),
            }
        )

    fig.suptitle(
        "Weekday and weekend diurnal signatures, normalised by each area's own mean",
        y=1.03,
    )
    _save(fig, "eda_04b_diurnal_signatures")

    table = pd.DataFrame(rows).round(4)
    table.to_csv(TAB / "eda_diurnal_signatures.csv", index=False)
    return table


#: Short, human-readable descriptions of the areas the study focuses on, used
#: where a full label would not fit. Derived from the verified cell centroids.
PLACE_ABBREVIATIONS = {
    "Duomo (historic centre)": "Duomo",
    "Milano Centrale station": "Centrale stn",
    "Bocconi University": "Bocconi Univ.",
    "Politecnico (Leonardo)": "Politecnico",
    "San Siro stadium": "San Siro",
    "Linate airport": "Linate",
    "Navigli nightlife district": "Navigli",
    "Fiera Milano City / Portello": "Portello",
}


def _short_label(square_id: int) -> str:
    """Two-line label: identity on the first line, location on the second."""
    top = D.top_squares(3)
    rank = f"#{top.index(square_id) + 1} by traffic" if square_id in top else "reference"
    if grid.available():
        info = grid.describe(square_id)
        place = PLACE_ABBREVIATIONS.get(info["nearest_landmark"], info["nearest_landmark"])
        return f"Square {square_id}  ({rank})\n{info['distance_km']} km from {place}"
    return f"Square {square_id}  ({rank})"


def _labels(squares: list[int]) -> dict[int, str]:
    """Label each area by its rank and, where known, its real-world location."""
    top = D.top_squares(3)
    out = {}
    for sq in squares:
        role = (
            f"rank #{top.index(sq) + 1} by total traffic"
            if sq in top
            else "reference area"
        )
        place = ""
        if grid.available():
            info = grid.describe(sq)
            place = f", {info['distance_km']} km from {info['nearest_landmark'].split(' (')[0]}"
        out[sq] = f"Square {sq} - {role}{place}"
    return out


def weekly_profiles(squares: list[int] | None = None) -> pd.DataFrame:
    """Average day-of-week x hour profile for each study area (heatmaps)."""
    squares = squares or D.study_squares()
    labels = _labels(squares)
    fig, axes = plt.subplots(1, len(squares), figsize=(3.0 * len(squares), 3.1), sharey=True)

    panels = []
    for ax, sq in zip(np.atleast_1d(axes), squares):
        s = D.series(sq)
        pivot = (
            s.groupby([s.index.dayofweek, s.index.hour])
            .mean()
            .unstack()
            .reindex(range(7))
        )
        panels.append((ax, sq, pivot.div(pivot.to_numpy().mean()).to_numpy()))

    # One shared colour scale so the panels are directly comparable, with the
    # upper limit taken from the data rather than guessed. A fixed vmax of 2.0
    # saturated the central-Milan panels into flat white and hid their structure.
    vmax = float(np.ceil(max(values.max() for _, _, values in panels) * 10) / 10)

    rows = []
    for ax, sq, values in panels:
        im = ax.imshow(values, aspect="auto", cmap="magma", origin="lower", vmin=0, vmax=vmax)
        ax.set(xlabel="hour of day", yticks=range(7), yticklabels=list("MTWTFSS"))
        ax.set_title(_short_label(sq), fontsize=7.6)
        ax.grid(False)
        rows.append(
            {
                "square": sq,
                "role": labels[sq],
                "weekly_profile_min": round(float(values.min()), 4),
                "weekly_profile_max": round(float(values.max()), 4),
                "weekly_profile_range": round(float(values.max() - values.min()), 4),
            }
        )
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), label="activity / area mean", fraction=0.03)
    fig.suptitle("Mean weekly activity profile (day of week x hour), full two months", y=1.05)
    _save(fig, "eda_04_weekly_profiles")

    table = pd.DataFrame(rows)
    table.to_csv(TAB / "eda_weekly_profile_range.csv", index=False)
    return table


# ===========================================================================
# 2.3  Additional analysis A - autocorrelation and periodicity
# ===========================================================================


@dataclass
class AutocorrelationResult:
    top_square: int
    acf_values: np.ndarray
    pacf_values: np.ndarray
    seasonal_peaks: pd.DataFrame
    summary: pd.DataFrame


def autocorrelation_analysis(square_id: int | None = None, nlags: int = 3 * C.SLOTS_PER_DAY):
    """ACF/PACF plus a periodogram for the highest-traffic area.

    This is the analysis that most directly informs the forecasting design: the
    ACF tells us how far back a useful history extends (hence the lookback
    length), and the periodogram confirms which seasonal periods a model must
    represent.
    """
    square_id = square_id or D.top_squares(1)[0]
    s = D.series(square_id)
    x = s.to_numpy()

    a = acf(x, nlags=nlags, fft=True)
    p = pacf(x, nlags=48, method="ywm")

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))

    lags = np.arange(len(a))
    axes[0].plot(lags / C.SLOTS_PER_DAY, a, lw=0.9, color="#2b6cb0")
    axes[0].axhline(0, color="k", lw=0.6)
    conf = 1.96 / np.sqrt(len(x))
    axes[0].axhspan(-conf, conf, color="grey", alpha=0.25, lw=0)
    for d in (1, 2, 3):
        axes[0].axvline(d, color="crimson", ls="--", lw=0.8)
    axes[0].set(
        title=f"(a) ACF, square {square_id} (up to 3 days)",
        xlabel="lag (days)", ylabel="autocorrelation",
    )

    axes[1].stem(np.arange(len(p)), p, basefmt=" ", markerfmt="o", linefmt="-")
    axes[1].axhspan(-conf, conf, color="grey", alpha=0.25, lw=0)
    axes[1].set(
        title="(b) PACF, first 48 lags (8 hours)",
        xlabel="lag (10-min steps)", ylabel="partial autocorrelation",
    )

    # Periodogram of the mean-removed series, plotted against period in hours.
    detrended = x - x.mean()
    freqs = np.fft.rfftfreq(len(detrended), d=C.SLOT_MINUTES / 60.0)  # cycles per hour
    power = np.abs(np.fft.rfft(detrended)) ** 2
    keep = freqs > 0
    periods = 1.0 / freqs[keep]
    normalised = power[keep] / power[keep].max()
    # Both axes are logarithmic. On a linear power axis the 24-hour peak is so
    # dominant that the 12-hour harmonic and the weekly component are
    # indistinguishable from zero, which would hide evidence the report relies on.
    axes[2].loglog(periods, np.maximum(normalised, 1e-8), lw=0.7, color="#2f855a")
    for label, hours, colour in (
        ("8 h", 8, "#a0aec0"), ("12 h", 12, "crimson"),
        ("24 h", 24, "crimson"), ("1 week", 168, "#2b6cb0"),
    ):
        axes[2].axvline(hours, color=colour, ls="--", lw=0.8)
        axes[2].text(hours, 2.0, label, ha="center", fontsize=6.6, color=colour)
    axes[2].set(
        title="(c) Periodogram: dominant cycles",
        xlabel="period (hours, log scale)", ylabel="normalised power (log scale)",
        ylim=(1e-8, 6), xlim=(1, 2000),
    )

    fig.suptitle(f"Temporal dependence structure, square {square_id}", y=1.05)
    _save(fig, "eda_05_autocorrelation")

    # Which seasonal lags carry the most correlation?
    peaks = []
    for name, lag in (
        ("10 min", 1),
        ("1 hour", 6),
        ("6 hours", 36),
        ("12 hours", 72),
        ("1 day", C.SLOTS_PER_DAY),
        ("2 days", 2 * C.SLOTS_PER_DAY),
        ("3 days", 3 * C.SLOTS_PER_DAY),
    ):
        if lag < len(a):
            peaks.append({"lag_label": name, "lag_steps": lag, "acf": float(a[lag])})
    seasonal_peaks = pd.DataFrame(peaks)
    seasonal_peaks.to_csv(TAB / "eda_acf_at_seasonal_lags.csv", index=False)

    # Report only cycles shorter than 10 days. Longer "periods" are dominated by
    # the finite length of the record itself (the 62-day window appears as a
    # spurious 1488-hour component) and carry no information about periodicity.
    interpretable = periods <= 240
    ranked = np.argsort(power[keep][interpretable])[::-1][:8]
    summary = pd.DataFrame(
        {
            "period_hours": np.round(periods[interpretable][ranked], 3),
            "normalised_power": np.round(
                power[keep][interpretable][ranked] / power[keep][interpretable].max(), 4
            ),
        }
    )
    summary.to_csv(TAB / "eda_periodogram_peaks.csv", index=False)

    return AutocorrelationResult(square_id, a, p, seasonal_peaks, summary)


# ===========================================================================
# 2.4  Additional analysis B - stationarity and decomposition
# ===========================================================================


def stationarity_and_decomposition(square_id: int | None = None) -> pd.DataFrame:
    """ADF/KPSS tests plus an STL decomposition of the highest-traffic area.

    Two tests are used because they have opposite null hypotheses; agreeing on
    "stationary" is much stronger evidence than either test alone. The tests are
    run on the raw series, on the ``log1p`` series, and on the seasonally
    differenced series, because those are precisely the three representations
    the forecasting models will use.
    """
    square_id = square_id or D.top_squares(1)[0]
    s = D.series(square_id)
    x = s.to_numpy()

    log_x = np.log1p(x)
    lag = C.SLOTS_PER_DAY
    variants = {
        "raw": x,
        "log1p": log_x,
        "first difference": np.diff(x),
        "seasonal difference (lag 144)": x[lag:] - x[:-lag],
        "log1p + seasonal difference": log_x[lag:] - log_x[:-lag],
    }

    rows = []
    for name, v in variants.items():
        adf_stat, adf_p, *_ = adfuller(v, autolag="AIC")
        kpss_stat, kpss_p, *_ = kpss(v, regression="c", nlags="auto")
        rows.append(
            {
                "series": name,
                "ADF_stat": adf_stat,
                "ADF_p": adf_p,
                "ADF_verdict": "stationary" if adf_p < 0.05 else "non-stationary",
                "KPSS_stat": kpss_stat,
                "KPSS_p": kpss_p,
                "KPSS_verdict": "stationary" if kpss_p > 0.05 else "non-stationary",
                "std": float(np.std(v)),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(TAB / "eda_stationarity_tests.csv", index=False)

    # STL with a daily period. Robust fitting limits the influence of the
    # anomalous spikes seen in the raw series.
    stl = STL(
        pd.Series(np.log1p(x), index=s.index),
        period=C.SLOTS_PER_DAY,
        seasonal=145,
        robust=True,
    ).fit()

    fig, axes = plt.subplots(4, 1, figsize=(11.5, 8.0), sharex=True)
    for ax, (series_, label, colour) in zip(
        axes,
        [
            (stl.observed, "observed  $\\log(1+x)$", "#1a202c"),
            (stl.trend, "trend", "#2b6cb0"),
            (stl.seasonal, "daily seasonal", "#2f855a"),
            (stl.resid, "remainder", "#c53030"),
        ],
    ):
        ax.plot(series_.index, series_.values, lw=0.55, color=colour)
        ax.set_ylabel(label, fontsize=8)
    axes[-1].axhline(0, color="k", lw=0.6)
    axes[-1].set_xlabel("Local date (CET)")
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    fig.suptitle(f"STL decomposition (daily period = 144 steps), square {square_id}", y=0.995)
    _save(fig, "eda_06_stl_decomposition")

    # Component strengths, following Wang, Smith and Hyndman (2006). Reporting
    # Var(component)/Var(observed) would be misleading, because the STL
    # components are correlated and those ratios need not sum to 100%. The
    # strength measures below are bounded in [0, 1] and are the standard summary:
    # each asks how much of the variation in "component + remainder" the
    # component itself accounts for.
    var_resid = float(np.var(stl.resid))
    strength_trend = max(0.0, 1 - var_resid / float(np.var(stl.trend + stl.resid)))
    strength_seasonal = max(0.0, 1 - var_resid / float(np.var(stl.seasonal + stl.resid)))

    shares = pd.DataFrame(
        [
            {"quantity": "Trend strength F_T (0-1)", "value": round(strength_trend, 4)},
            {"quantity": "Daily seasonal strength F_S (0-1)", "value": round(strength_seasonal, 4)},
            {"quantity": "Var(observed), log scale", "value": round(float(np.var(stl.observed)), 4)},
            {"quantity": "Var(trend)", "value": round(float(np.var(stl.trend)), 4)},
            {"quantity": "Var(seasonal)", "value": round(float(np.var(stl.seasonal)), 4)},
            {"quantity": "Var(remainder)", "value": round(var_resid, 4)},
            {
                "quantity": "Remainder std as % of observed std",
                "value": round(100 * np.sqrt(var_resid) / np.sqrt(float(np.var(stl.observed))), 2),
            },
        ]
    )
    shares.to_csv(TAB / "eda_stl_variance_shares.csv", index=False)

    # Residual outliers = candidate anomalies, useful for the failure analysis.
    resid = stl.resid
    z = (resid - resid.mean()) / resid.std()
    anomalies = z[np.abs(z) > 4].sort_values()
    anomalies.to_csv(TAB / "eda_stl_anomalies.csv", header=["z_score"])

    print(f"  STL variance shares:\n{shares.to_string(index=False)}")
    print(f"  |z|>4 remainder outliers: {len(anomalies)}")
    return table
