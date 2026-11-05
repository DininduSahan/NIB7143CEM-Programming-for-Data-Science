"""Tests for the cleaning rules and the transactional Scrubber."""

from __future__ import annotations

import pandas as pd
import pytest

from cricket_project.cleaning import (
    CleaningRule,
    FlagMiscountedOverRule,
    FlagNonDismissalRule,
    ImputeCityRule,
    NormaliseSeasonRule,
    RemoveSuperOverRule,
    Scrubber,
    StandardiseVenueRule,
    default_rules,
)
from cricket_project.data_loader import MatchFile, MatchLoader
from cricket_project.exceptions import ScrubError


def _frame(**columns) -> pd.DataFrame:
    rows = len(next(iter(columns.values())))
    columns.setdefault("match_id", [f"m{i}" for i in range(rows)])
    return pd.DataFrame(columns)


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("R Premadasa Stadium, Colombo", "R Premadasa Stadium"),
        ("R.Premadasa Stadium, Khettarama", "R Premadasa Stadium"),
        ("Galle International Stadium", "Galle International Stadium"),
        ("Feroz Shah Kotla", "Arun Jaitley Stadium"),
        ("Nehru Stadium, Fatorda", "Nehru Stadium, Fatorda"),  # pinned: must not merge with Pune's Nehru Stadium
    ],
)
def test_venue_canonical_names(raw, canonical):
    assert StandardiseVenueRule.canonical(raw) == canonical


def test_venue_rule_counts_rows_and_matches():
    frame = _frame(
        match_id=["a", "a", "b", "c"],
        venue=["R Premadasa Stadium, Colombo"] * 2 + ["R.Premadasa Stadium", "R Premadasa Stadium"],
    )
    cleaned, report = StandardiseVenueRule().apply(frame)

    assert set(cleaned["venue"]) == {"R Premadasa Stadium"}
    assert report.rows_affected == 3 and report.matches_affected == 2


def test_impute_city_uses_same_venue():
    frame = _frame(
        match_id=["a", "b", "c"],
        venue=["Galle International Stadium", "Galle International Stadium", "Unknown Ground"],
        city=["Galle", None, None],
    )
    cleaned, report = ImputeCityRule().apply(frame)

    assert cleaned["city"].tolist()[:2] == ["Galle", "Galle"]
    assert pd.isna(cleaned["city"].iloc[2])  # no evidence for this ground: left missing, not guessed
    assert report.rows_affected == 1


def test_season_rule_converts_ints_to_str():
    frame = _frame(season=pd.Series([2017, "2016/17"], dtype=object))
    cleaned, report = NormaliseSeasonRule().apply(frame)
    assert cleaned["season"].tolist() == ["2017", "2016/17"]
    assert report.rows_affected == 1


def test_super_over_rows_removed_and_counted():
    frame = _frame(match_id=["a", "a", "a"], is_super_over=[False, True, True])
    cleaned, report = RemoveSuperOverRule().apply(frame)
    assert len(cleaned) == 1 and report.rows_affected == 2


def test_miscounted_over_flagged_not_dropped():
    frame = _frame(
        match_id=["a"] * 8,
        innings=[1] * 8,
        over=[0] * 8,
        extras_type=[None] * 6 + ["wides", None],  # 7 legal balls + 1 wide
    )
    cleaned, report = FlagMiscountedOverRule().apply(frame)
    assert len(cleaned) == 8
    assert cleaned["over_miscounted"].all() and cleaned["is_legal_ball"].sum() == 7
    assert report.rows_affected == 8


def test_retirement_is_not_a_dismissal():
    frame = _frame(wicket_count=[1, 1, 0], wicket_kind=["caught", "retired hurt", None])
    cleaned, _ = FlagNonDismissalRule().apply(frame)
    assert cleaned["is_dismissal"].tolist() == [True, False, False]


def test_rule_rejects_frame_missing_columns():
    with pytest.raises(ValueError, match="missing required columns"):
        StandardiseVenueRule().apply(pd.DataFrame({"match_id": ["a"]}))


class _ExplodingRule(CleaningRule):
    name = "exploding"
    required_columns = frozenset({"match_id"})

    def _detect(self, frame):
        return pd.Series(True, index=frame.index)

    def _fix(self, frame, mask):
        frame["venue"] = "CHANGED"
        raise KeyError("simulated failure half-way through a rule")


def test_failed_scrub_leaves_input_and_last_report_unchanged():
    frame = _frame(match_id=["a"], venue=["R Premadasa Stadium, Colombo"])
    before = frame.copy()
    scrubber = Scrubber([StandardiseVenueRule(), _ExplodingRule()])

    with pytest.raises(ScrubError, match="exploding") as caught:
        scrubber.run(frame)

    pd.testing.assert_frame_equal(frame, before)  # no half-changed state
    assert scrubber.last_reports == ()
    assert isinstance(caught.value.__cause__, KeyError)


def test_scrubber_rejects_non_rules_and_duplicate_names():
    with pytest.raises(ValueError, match="not a CleaningRule"):
        Scrubber([StandardiseVenueRule(), "not a rule"])
    with pytest.raises(ValueError, match="unique"):
        Scrubber([StandardiseVenueRule(), StandardiseVenueRule()])


def test_default_rules_run_end_to_end(tmp_path, write_match, match_payload):
    write_match("m1", match_payload)
    raw = MatchLoader(tmp_path).load_frame()

    clean, reports = Scrubber(default_rules()).run(raw)

    assert len(reports) == len(default_rules())
    assert clean["venue"].unique().tolist() == ["R Premadasa Stadium"]
    assert {"is_legal_ball", "over_miscounted", "is_dismissal"} <= set(clean.columns)
    assert raw["venue"].iat[0] == "R Premadasa Stadium, Colombo"  # raw untouched
    assert MatchFile("m1", match_payload).fingerprint[0] == "T20"
