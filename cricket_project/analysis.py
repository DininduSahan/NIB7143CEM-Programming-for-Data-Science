"""Explore and Model stages (Task 3).

Question: for Sri Lanka men's international cricket, does batting first or
winning the toss go with winning, does that differ at home and away, and how
does scoring tempo differ by phase of the innings?

All functions take the *scrubbed* deliveries table and return new tables;
none of them mutates its input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

FOCUS_TEAM = "Sri Lanka"
# Cities of every Sri Lankan international ground in the data (after scrubbing).
HOME_CITIES = frozenset({"Colombo", "Kandy", "Galle", "Dambulla", "Hambantota"})
LIMITED_OVERS = ("ODI", "T20")

# (first over, last over, label) with Cricsheet's 0-based over numbers.
PHASES: dict[str, tuple[tuple[int, int, str], ...]] = {
    "ODI": ((0, 9, "Powerplay"), (10, 39, "Middle"), (40, 49, "Death")),
    "T20": ((0, 5, "Powerplay"), (6, 14, "Middle"), (15, 19, "Death")),
}
PHASE_ORDER = ("Powerplay", "Middle", "Death")

MATCH_COLUMNS = ["match_type", "match_date", "season", "venue", "city",
                 "toss_winner", "toss_decision", "winner", "result"]


def _require(frame: pd.DataFrame, columns: set[str]) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"expected a DataFrame, got {type(frame).__name__}")
    missing = columns - set(frame.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")


def match_summary(frame: pd.DataFrame, team: str = FOCUS_TEAM) -> pd.DataFrame:
    """One row per match, from ``team``'s point of view.

    Adds ``batted_first``, ``won_toss``, ``home``, ``year`` and ``outcome``
    (``won`` / ``lost`` / ``draw`` / ``tie`` / ``no result``).

    Raises:
        ValueError: ``frame`` is missing a required column.
    """
    _require(frame, {"match_id", "innings", "batting_team", *MATCH_COLUMNS})
    matches = frame.groupby("match_id", observed=True)[MATCH_COLUMNS].first()
    first_batting = frame.loc[frame["innings"] == 1].groupby("match_id", observed=True)["batting_team"].first()
    matches["batted_first"] = first_batting.reindex(matches.index).astype(str).eq(team)
    matches["won_toss"] = matches["toss_winner"].astype(str).eq(team)
    matches["home"] = matches["city"].astype(str).isin(HOME_CITIES)
    matches["year"] = pd.to_datetime(matches["match_date"]).dt.year
    winner = matches["winner"].astype("object")
    result = matches["result"].astype("object")
    matches["outcome"] = np.select(
        [winner.eq(team), winner.notna(), result.eq("draw"), result.eq("tie")],
        ["won", "lost", "draw", "tie"],
        default="no result",
    )
    return matches.reset_index()


def win_rates(summary: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Played / won / lost / win % grouped by ``by``. No-result matches are excluded.

    Raises:
        ValueError: ``by`` is empty or names a column not in ``summary``.
    """
    if not by:
        raise ValueError("by must name at least one column")
    _require(summary, {"outcome", *by})
    played = summary.loc[summary["outcome"] != "no result"]
    table = played.groupby(by, observed=True)["outcome"].value_counts().unstack(fill_value=0)
    for outcome in ("won", "lost", "draw", "tie"):
        if outcome not in table.columns:
            table[outcome] = 0
    table["played"] = table[["won", "lost", "draw", "tie"]].sum(axis=1)
    table["win_pct"] = (100 * table["won"] / table["played"]).round(1)
    return table[["played", "won", "lost", "draw", "tie", "win_pct"]].reset_index()


def independence_test(summary: pd.DataFrame, factor: str, match_type: str) -> dict[str, float]:
    """Chi-square test of independence between a boolean ``factor`` and winning.

    Uses only matches of ``match_type`` that ended in a win or a loss (2x2 table).

    Returns:
        ``{"match_type", "factor", "n", "chi2", "p_value", "win_pct_if_true",
        "win_pct_if_false", "valid"}``. ``valid`` is False when an expected cell
        count is below 5, the usual condition for the chi-square approximation.

    Raises:
        ValueError: missing columns, or no decided matches of ``match_type``.
    """
    _require(summary, {factor, "outcome", "match_type"})
    decided = summary.loc[(summary["match_type"] == match_type) & summary["outcome"].isin(["won", "lost"])]
    if decided.empty:
        raise ValueError(f"no decided {match_type} matches to test")
    table = pd.crosstab(decided[factor], decided["outcome"]).reindex(
        index=[True, False], columns=["won", "lost"], fill_value=0
    )
    chi2, p_value, _, expected = chi2_contingency(table.to_numpy())
    rates = 100 * table["won"] / table.sum(axis=1)
    return {
        "match_type": match_type,
        "factor": factor,
        "n": int(table.to_numpy().sum()),
        "chi2": round(float(chi2), 3),
        "p_value": round(float(p_value), 4),
        "win_pct_if_true": round(float(rates[True]), 1),
        "win_pct_if_false": round(float(rates[False]), 1),
        "valid": bool((expected >= 5).all()),
    }


def innings_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Runs, dismissals and legal balls per innings."""
    _require(frame, {"match_id", "innings", "match_type", "batting_team",
                     "runs_total", "is_dismissal", "is_legal_ball"})
    return (
        frame.groupby(["match_id", "innings"], observed=True)
        .agg(
            match_type=("match_type", "first"),
            batting_team=("batting_team", "first"),
            runs=("runs_total", "sum"),
            wickets=("is_dismissal", "sum"),
            legal_balls=("is_legal_ball", "sum"),
        )
        .reset_index()
    )


def assign_phase(frame: pd.DataFrame) -> pd.Series:
    """Phase label (Powerplay / Middle / Death) for each limited-overs delivery, else NaN."""
    _require(frame, {"match_type", "over"})
    phase = pd.Series(pd.NA, index=frame.index, dtype="object")
    for match_type, bands in PHASES.items():
        is_type = frame["match_type"] == match_type
        for first, last, label in bands:
            phase[is_type & frame["over"].between(first, last)] = label
    return phase


def phase_run_rates(frame: pd.DataFrame, team: str = FOCUS_TEAM) -> pd.DataFrame:
    """Runs per six legal balls by format, phase and side (``team`` vs opposition)."""
    _require(frame, {"match_type", "over", "batting_team", "runs_total", "is_legal_ball"})
    limited = frame.loc[frame["match_type"].isin(LIMITED_OVERS)]
    working = pd.DataFrame({
        "match_type": limited["match_type"].astype(str),
        "phase": assign_phase(limited),
        "side": np.where(limited["batting_team"].astype(str) == team, team, "Opposition"),
        "runs": limited["runs_total"],
        "legal": limited["is_legal_ball"].astype(int),
    })
    grouped = working.groupby(["match_type", "phase", "side"]).agg(runs=("runs", "sum"), balls=("legal", "sum"))
    grouped["run_rate"] = (6 * grouped["runs"] / grouped["balls"]).round(2)
    result = grouped.reset_index()
    result["phase"] = pd.Categorical(result["phase"], categories=PHASE_ORDER, ordered=True)
    return result.sort_values(["match_type", "phase", "side"]).reset_index(drop=True)
