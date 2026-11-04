"""Shared fixtures: a tiny hand-built Cricsheet match, so tests never touch the real data."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

SAMPLE_MATCH = {
    "meta": {"data_version": "1.2.0", "created": "2024-01-01", "revision": 1},
    "info": {
        "dates": ["2024-01-10"],
        "match_type": "T20",
        "season": 2024,
        "teams": ["Sri Lanka", "India"],
        "venue": "R Premadasa Stadium, Colombo",
        "city": "Colombo",
        "outcome": {"winner": "Sri Lanka", "by": {"runs": 5}},
        "toss": {"winner": "India", "decision": "field"},
        "event": {"name": "Test Series"},
        "registry": {"people": {"A Bat": "id-a", "B Bat": "id-b", "C Bowl": "id-c", "D Bowl": "id-d"}},
    },
    "innings": [
        {
            "team": "Sri Lanka",
            "overs": [
                {
                    "over": 0,
                    "deliveries": [
                        {"batter": "A Bat", "bowler": "D Bowl", "non_striker": "B Bat",
                         "runs": {"batter": 4, "extras": 0, "total": 4}},
                        {"batter": "A Bat", "bowler": "D Bowl", "non_striker": "B Bat",
                         "extras": {"wides": 1}, "runs": {"batter": 0, "extras": 1, "total": 1}},
                        {"batter": "A Bat", "bowler": "D Bowl", "non_striker": "B Bat",
                         "runs": {"batter": 0, "extras": 0, "total": 0},
                         "wickets": [{"player_out": "A Bat", "kind": "bowled"}]},
                    ],
                }
            ],
        },
        {
            "team": "India",
            "overs": [
                {
                    "over": 0,
                    "deliveries": [
                        {"batter": "C Bowl", "bowler": "B Bat", "non_striker": "D Bowl",
                         "runs": {"batter": 1, "extras": 0, "total": 1}},
                    ],
                }
            ],
        },
    ],
}


@pytest.fixture
def match_payload() -> dict:
    """A fresh deep copy each time, so a test can break it without affecting others."""
    return copy.deepcopy(SAMPLE_MATCH)


@pytest.fixture
def write_match(tmp_path: Path):
    """Write a payload (or raw text) as ``<match_id>.json`` in a temp folder and return the path."""

    def _write(match_id: str, payload: dict | str, folder: Path = tmp_path) -> Path:
        path = folder / f"{match_id}.json"
        text = payload if isinstance(payload, str) else json.dumps(payload)
        path.write_text(text, encoding="utf-8")
        return path

    return _write
