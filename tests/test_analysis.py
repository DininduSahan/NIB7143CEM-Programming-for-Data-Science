"""Tests for the Explore/Model functions."""

from __future__ import annotations

import pandas as pd
import pytest

from cricket_project import analysis
from cricket_project.cleaning import Scrubber, default_rules
from cricket_project.data_loader import MatchLoader


@pytest.fixture
def clean(tmp_path, write_match, match_payload) -> pd.DataFrame:
    write_match("m1", match_payload)
    return Scrubber(default_rules()).run(MatchLoader(tmp_path).load_frame())[0]


def test_match_summary_from_sri_lanka_view(clean):
    row = analysis.match_summary(clean).iloc[0]
    assert row["batted_first"] and not row["won_toss"] and row["home"]
    assert row["outcome"] == "won" and row["year"] == 2024


def test_assign_phase_boundaries():
    frame = pd.DataFrame({"match_type": ["T20", "T20", "T20", "ODI", "ODI", "Test"],
                          "over": [5, 6, 15, 9, 40, 3]})
    assert analysis.assign_phase(frame).tolist() == ["Powerplay", "Middle", "Death", "Powerplay", "Death", pd.NA]


def test_phase_run_rates_ignore_wides_in_ball_count(clean):
    rates = analysis.phase_run_rates(clean).set_index("side")
    # Sri Lanka: 4 + 1 (wide) + 0 = 5 runs from 2 legal balls -> 15 per over.
    assert rates.loc["Sri Lanka", "runs"] == 5 and rates.loc["Sri Lanka", "balls"] == 2
    assert rates.loc["Sri Lanka", "run_rate"] == 15.0


def test_win_rates_requires_grouping_column(clean):
    summary = analysis.match_summary(clean)
    with pytest.raises(ValueError, match="at least one column"):
        analysis.win_rates(summary, [])
