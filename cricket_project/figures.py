"""Report figures, rendered with matplotlib to PNG for the PDF report.

Styling follows one set of rules for every figure: categorical colours in a
fixed order (slot 1 blue, slot 2 orange), thin bars with a gap between them,
a light horizontal grid behind the data, one y-axis per panel, text in
neutral ink rather than series colours, and a legend whenever there are two
or more series.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed; also makes the output deterministic in CI
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # validated categorical slots 1-3, in order
# Secondary encodings so every figure still reads in greyscale print:
# a 45-degree hatch on the second bar series, and a distinct marker and
# line style per line series.
HATCHES = (None, "////", "\\\\\\\\")
MARKERS = ("o", "s", "^")
LINESTYLES = ("-", "--", ":")
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
DPI = 200


def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "font.family": "DejaVu Sans",
        "hatch.color": SURFACE,
        "hatch.linewidth": 1.2,
    })


def _save(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def first_innings_totals(totals: pd.DataFrame, path: Path) -> Path:
    """Histogram of first-innings totals, one panel per limited-overs format."""
    _style()
    first = totals.loc[(totals["innings"] == 1) & totals["match_type"].isin(["ODI", "T20"])]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, match_type in zip(axes, ["ODI", "T20"]):
        runs = first.loc[first["match_type"] == match_type, "runs"]
        ax.hist(runs, bins=25, color=SERIES[0], edgecolor=SURFACE, linewidth=1)
        median = runs.median()
        ax.axvline(median, color=INK, linewidth=1, linestyle="--")
        ax.annotate(f"median {median:.0f}", (median, ax.get_ylim()[1] * 0.95),
                    xytext=(5, 0), textcoords="offset points", fontsize=8, color=INK,
                    bbox={"boxstyle": "square,pad=0.2", "facecolor": SURFACE, "edgecolor": "none"})
        ax.set_title(f"{match_type} (n = {len(runs)} innings)", loc="left")
        ax.set_xlabel("First-innings total (runs)")
    axes[0].set_ylabel("Innings")
    return _save(fig, path)


def phase_run_rates(rates: pd.DataFrame, path: Path, team: str = "Sri Lanka") -> Path:
    """Grouped bars: run rate by phase, focus team vs opposition, one panel per format."""
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2), sharey=True)
    sides = [team, "Opposition"]
    width = 0.36
    for ax, match_type in zip(axes, ["ODI", "T20"]):
        subset = rates.loc[rates["match_type"] == match_type]
        phases = [str(p) for p in subset["phase"].cat.categories]
        x = np.arange(len(phases))
        for offset, (side, colour, hatch) in enumerate(zip(sides, SERIES, HATCHES)):
            values = subset.loc[subset["side"] == side].set_index("phase")["run_rate"].reindex(phases)
            bars = ax.bar(x + (offset - 0.5) * (width + 0.03), values, width, color=colour,
                          hatch=hatch, label=side)
            ax.bar_label(bars, fmt="%.1f", fontsize=7.5, color=INK_SECONDARY, padding=2)
        ax.set_xticks(x, phases)
        ax.set_title(match_type, loc="left")
    axes[0].set_ylabel("Runs per 6 legal balls")
    axes[1].legend(loc="upper left")
    return _save(fig, path)


def factor_win_rates(tests: pd.DataFrame, path: Path) -> Path:
    """Win % with vs without each factor, per format, with the chi-square p-value."""
    _style()
    labels = {"home": "Playing at home", "won_toss": "Won the toss", "batted_first": "Batted first"}
    factors = [f for f in labels if f in set(tests["factor"])]
    fig, axes = plt.subplots(1, len(factors), figsize=(9, 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    width = 0.36
    for ax, factor in zip(axes, factors):
        subset = tests.loc[tests["factor"] == factor].set_index("match_type")
        formats = list(subset.index)
        x = np.arange(len(formats))
        yes = ax.bar(x - (width + 0.03) / 2, subset["win_pct_if_true"], width, color=SERIES[0], label="Yes")
        no = ax.bar(x + (width + 0.03) / 2, subset["win_pct_if_false"], width, color=SERIES[1],
                    hatch=HATCHES[1], label="No")
        for bars in (yes, no):
            ax.bar_label(bars, fmt="%.0f", fontsize=7.5, color=INK_SECONDARY, padding=2)
        for xi, p in zip(x, subset["p_value"]):
            ax.annotate(f"p = {p:.3f}", (xi, 82), ha="center", fontsize=7.5, color=INK)
        ax.set_xticks(x, formats)
        ax.set_ylim(0, 90)
        ax.set_title(labels[factor], loc="left")
    axes[0].set_ylabel("Sri Lanka win % (decided matches)")
    axes[-1].legend(loc="upper right", bbox_to_anchor=(1.0, 0.92))
    return _save(fig, path)


def storage_results(results: pd.DataFrame, path: Path) -> Path:
    """Size on disk and median timings (min-max whiskers), one panel per measure.

    Log scales: the formats differ by up to three orders of magnitude, and on a
    linear axis every fast format collapses to an invisible sliver.
    """
    _style()
    measures = [("size", "Size on disk (MB)"), ("write", "Write (s)"),
                ("read", "Read all (s)"), ("read_column", "Read one column (s)")]
    order = list(dict.fromkeys(results["format"]))
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True)
    y = np.arange(len(order))
    for ax, (measure, title) in zip(axes, measures):
        ax.set_xscale("log")
        if measure == "size":
            values = results.drop_duplicates("format").set_index("format")["size_mb"].reindex(order)
            ax.barh(y, values, 0.6, color=SERIES[0])
            ax.bar_label(ax.containers[0], fmt="%.3g", fontsize=7, color=INK_SECONDARY, padding=2)
        else:
            rows = results.loc[results["operation"] == measure].set_index("format").reindex(order)
            error = [rows["median_s"] - rows["min_s"], rows["max_s"] - rows["median_s"]]
            ax.barh(y, rows["median_s"], 0.6, color=SERIES[0], xerr=error,
                    error_kw={"ecolor": INK_SECONDARY, "elinewidth": 0.8, "capsize": 2})
            ax.bar_label(ax.containers[1], fmt="%.3g", fontsize=7, color=INK_SECONDARY, padding=4)
            ax.set_xlim(right=ax.get_xlim()[1] * 8)  # room for the labels
        ax.set_title(title + ", log scale", loc="left", fontsize=9.5)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlim(right=axes[0].get_xlim()[1] * 8)
    axes[0].set_yticks(y, order)
    axes[0].invert_yaxis()
    return _save(fig, path)


def spark_scaling(results: pd.DataFrame, path: Path) -> Path:
    """Median time vs data size for each engine, min-max shaded."""
    _style()
    labels = {"pandas": "pandas", "spark": "Spark, 200 shuffle partitions",
              "spark_tuned": "Spark, 1 shuffle partition per core"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    engines = list(dict.fromkeys(results["engine"]))
    for engine, colour, marker, style in zip(engines, SERIES, MARKERS, LINESTYLES):
        rows = results.loc[results["engine"] == engine].sort_values("rows")
        ax.plot(rows["rows"] / 1e6, rows["median_s"], color=colour, linewidth=2, marker=marker,
                linestyle=style, markersize=6, label=labels.get(engine, engine))
        ax.fill_between(rows["rows"] / 1e6, rows["min_s"], rows["max_s"], color=colour, alpha=0.15, linewidth=0)
        last = rows.iloc[-1]
        short = {"pandas": "pandas", "spark": "Spark (200)", "spark_tuned": "Spark (per core)"}
        ax.annotate(short.get(engine, engine), (last["rows"] / 1e6, last["median_s"]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=INK)
    ax.set_xlabel("Rows (millions)")
    ax.set_ylabel("Median time (s)")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper left")
    return _save(fig, path)
