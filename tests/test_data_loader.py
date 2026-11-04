"""Tests for MatchFile and MatchLoader."""

from __future__ import annotations

import pytest

from cricket_project.data_loader import DELIVERY_COLUMNS, MatchFile, MatchLoader
from cricket_project.exceptions import DuplicateMatchError, InvalidMatchError


def test_iter_deliveries_flattens_every_ball(match_payload):
    rows = [dict(zip(DELIVERY_COLUMNS, row)) for row in MatchFile("m1", match_payload).iter_deliveries()]

    assert len(rows) == 4
    wide, wicket, chase = rows[1], rows[2], rows[3]
    assert wide["extras_type"] == "wides" and wide["runs_total"] == 1
    assert wicket["wicket_count"] == 1 and wicket["wicket_kind"] == "bowled"
    assert wicket["batter_id"] == "id-a"
    assert chase["innings"] == 2 and chase["batting_team"] == "India" and chase["bowling_team"] == "Sri Lanka"


def test_from_path_rejects_invalid_json_with_value_error(write_match):
    path = write_match("broken", "{ this is not json")
    with pytest.raises(ValueError, match="not valid UTF-8 JSON"):
        MatchFile.from_path(path)


def test_non_dict_payload_is_value_error():
    with pytest.raises(ValueError, match="payload must be a dict"):
        MatchFile("m1", ["not", "a", "match"])


def test_missing_info_key_is_invalid_match_error(match_payload):
    del match_payload["info"]["venue"]
    with pytest.raises(InvalidMatchError, match="venue"):
        MatchFile("m1", match_payload)


def test_unsupported_version_is_invalid_match_error(match_payload):
    match_payload["meta"]["data_version"] = "9.9.9"
    with pytest.raises(InvalidMatchError, match="unsupported data_version"):
        MatchFile("m1", match_payload)


def test_innings_by_unknown_team_is_invalid_match_error(match_payload):
    match_payload["innings"][1]["team"] = "Atlantis"
    with pytest.raises(InvalidMatchError, match="Atlantis"):
        MatchFile("m1", match_payload)


def test_invalid_match_error_is_not_a_value_error(match_payload):
    """Callers must be able to tell malformed input from a domain-rule failure."""
    match_payload["meta"]["data_version"] = "9.9.9"
    with pytest.raises(InvalidMatchError) as caught:
        MatchFile("m1", match_payload)
    assert not isinstance(caught.value, ValueError)


def test_lenient_loader_skips_and_records_bad_files(tmp_path, write_match, match_payload):
    write_match("good", match_payload)
    write_match("bad", "not json")

    loader = MatchLoader(tmp_path)
    frame = loader.load_frame()

    assert loader.loaded_count == 1
    assert [name for name, _ in loader.failures] == ["bad.json"]
    assert set(frame["match_id"]) == {"good"}
    assert str(frame["match_date"].dtype).startswith("datetime64")


def test_strict_loader_raises_on_duplicate_match(tmp_path, write_match, match_payload):
    write_match("100", match_payload)
    write_match("200", match_payload)  # same match, different id

    with pytest.raises(DuplicateMatchError, match="'200' duplicates already-loaded match '100'"):
        MatchLoader(tmp_path, strict=True).load_frame()


def test_loader_state_resets_between_loads(tmp_path, write_match, match_payload):
    write_match("100", match_payload)
    loader = MatchLoader(tmp_path)
    loader.load_frame()
    loader.load_frame()  # second load must not report the match as its own duplicate
    assert loader.loaded_count == 1 and loader.failures == ()


def test_loader_rejects_missing_directory(tmp_path):
    with pytest.raises(ValueError, match="not a directory"):
        MatchLoader(tmp_path / "nope")


def test_empty_directory_is_invalid_match_error(tmp_path):
    with pytest.raises(InvalidMatchError, match="no valid match files"):
        MatchLoader(tmp_path).load_frame()
