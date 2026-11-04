"""Loading and validating Cricsheet JSON match files.

Two classes:

* :class:`MatchFile` wraps one validated match. Its raw payload is private;
  callers read it through properties and the :meth:`MatchFile.iter_deliveries`
  generator, so a match cannot be modified after it has been validated.
* :class:`MatchLoader` walks a directory, builds a :class:`MatchFile` for every
  ``*.json`` file, tracks which match ids it has already seen and which files
  failed, and flattens everything into one ball-by-ball ``DataFrame``.

Cricsheet format reference: https://cricsheet.org/format/json/
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd

from .exceptions import DuplicateMatchError, InvalidMatchError

SUPPORTED_VERSIONS = frozenset({"1.0.0", "1.1.0", "1.2.0", "1.3.0"})
REQUIRED_INFO_KEYS = frozenset({"dates", "match_type", "teams", "venue", "outcome", "toss", "season"})

# One row per delivery. A tuple, not a list: the column order is part of the
# contract between iter_deliveries() and load_frame() and must not change.
DELIVERY_COLUMNS: tuple[str, ...] = (
    "match_id", "data_version", "match_type", "match_date", "season", "event_name",
    "venue", "city", "batting_team", "bowling_team", "toss_winner", "toss_decision",
    "winner", "result", "win_method", "innings", "is_super_over", "over", "ball",
    "batter", "bowler", "non_striker", "batter_id", "bowler_id",
    "runs_batter", "runs_extras", "runs_total", "extras_type",
    "wicket_count", "wicket_kind", "player_out",
)


class MatchFile:
    """One Cricsheet match, validated on construction and read-only afterwards.

    Contract:
        * ``match_id`` must be a non-empty ``str``; ``payload`` must be a ``dict``.
          Otherwise ``ValueError`` (malformed input).
        * ``payload`` must have ``meta``, ``info`` and ``innings``; ``info`` must
          contain every key in :data:`REQUIRED_INFO_KEYS`; ``data_version`` must be
          in :data:`SUPPORTED_VERSIONS`; there must be exactly two teams and every
          innings must be batted by one of them. Otherwise :class:`InvalidMatchError`.

    Args:
        match_id: Identifier of the match (Cricsheet uses the file stem).
        payload: The decoded JSON document.
        source: Path the payload was read from, used only in error messages.
    """

    def __init__(self, match_id: str, payload: dict[str, Any], source: Path | None = None) -> None:
        if not isinstance(match_id, str) or not match_id.strip():
            raise ValueError("match_id must be a non-empty string")
        if not isinstance(payload, dict):
            raise ValueError(f"payload must be a dict decoded from JSON, got {type(payload).__name__}")
        self._match_id = match_id
        self._source = source
        self._validate(payload)
        self._version: str = payload["meta"]["data_version"]
        self._info: dict[str, Any] = payload["info"]
        self._innings: list[dict[str, Any]] = payload["innings"]

    @classmethod
    def from_path(cls, path: Path | str) -> MatchFile:
        """Read and validate one match file.

        Args:
            path: Path to a ``.json`` file. The file stem becomes the match id.

        Raises:
            FileNotFoundError: The file does not exist.
            ValueError: The file is not valid UTF-8 JSON, or is not a JSON object.
            InvalidMatchError: The JSON is not a valid Cricsheet match.
        """
        path = Path(path)
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError(f"{path.name} is not valid UTF-8 JSON: {exc}") from exc
        return cls(path.stem, payload, source=path)

    def _validate(self, payload: dict[str, Any]) -> None:
        missing_top = {"meta", "info", "innings"} - payload.keys()
        if missing_top:
            raise InvalidMatchError(f"missing top-level keys {sorted(missing_top)}", self._source)
        version = payload["meta"].get("data_version")
        if version not in SUPPORTED_VERSIONS:
            raise InvalidMatchError(f"unsupported data_version {version!r}", self._source)
        info = payload["info"]
        # Set difference is O(len(REQUIRED_INFO_KEYS)) and reports every missing key at once.
        missing_info = REQUIRED_INFO_KEYS - info.keys()
        if missing_info:
            raise InvalidMatchError(f"info is missing {sorted(missing_info)}", self._source)
        teams = info["teams"]
        if not isinstance(teams, list) or len(teams) != 2:
            raise InvalidMatchError(f"expected exactly two teams, got {teams!r}", self._source)
        if not isinstance(payload["innings"], list):
            raise InvalidMatchError("innings must be a list", self._source)
        for number, innings in enumerate(payload["innings"], start=1):
            if innings.get("team") not in teams:
                raise InvalidMatchError(
                    f"innings {number} batted by {innings.get('team')!r}, not one of {teams}", self._source
                )

    @property
    def match_id(self) -> str:
        """The match identifier."""
        return self._match_id

    @property
    def version(self) -> str:
        """Cricsheet ``data_version`` of the source file."""
        return self._version

    @property
    def match_type(self) -> str:
        """Format as recorded by Cricsheet (``Test``, ``ODI``, ``T20``, ``IT20``...)."""
        return self._info["match_type"]

    @property
    def teams(self) -> tuple[str, str]:
        """The two teams, in the order Cricsheet lists them."""
        return tuple(self._info["teams"])

    @property
    def fingerprint(self) -> tuple[str, tuple[str, ...], tuple[str, ...], str]:
        """``(match_type, dates, sorted teams, venue)``: identifies a match independently of its id.

        A tuple (not a list) because it must be hashable to be used as a dict key.
        """
        info = self._info
        return (info["match_type"], tuple(info["dates"]), tuple(sorted(info["teams"])), info["venue"])

    @property
    def innings_count(self) -> int:
        """Number of innings in the match, including super overs."""
        return len(self._innings)

    def iter_deliveries(self) -> Iterator[tuple[Any, ...]]:
        """Yield one tuple per delivery, in the order of :data:`DELIVERY_COLUMNS`.

        A generator, so a 1,500-ball Test match never needs a second in-memory copy.
        Values are taken as recorded; no cleaning happens here (that is the
        Scrub stage's job, see :mod:`cricket_project.cleaning`).
        """
        info = self._info
        outcome = info["outcome"]
        people = info.get("registry", {}).get("people", {})
        teams = info["teams"]
        match_level = (
            self._match_id,
            self._version,
            info["match_type"],
            info["dates"][0],
            info["season"],
            info.get("event", {}).get("name"),
            info["venue"],
            info.get("city"),
        )
        toss = (info["toss"].get("winner"), info["toss"].get("decision"))
        # A tie decided by a super over records the winner as "eliminator".
        result = (outcome.get("winner", outcome.get("eliminator")), outcome.get("result"), outcome.get("method"))

        for innings_number, innings in enumerate(self._innings, start=1):
            batting = innings["team"]
            bowling = teams[1] if batting == teams[0] else teams[0]
            is_super_over = bool(innings.get("super_over", False))
            for over in innings.get("overs", []):
                for ball, delivery in enumerate(over["deliveries"], start=1):
                    runs = delivery["runs"]
                    extras = delivery.get("extras", {})
                    wickets = delivery.get("wickets", [])
                    first_wicket = wickets[0] if wickets else {}
                    yield (
                        *match_level,
                        batting,
                        bowling,
                        *toss,
                        *result,
                        innings_number,
                        is_super_over,
                        over["over"],
                        ball,
                        delivery["batter"],
                        delivery["bowler"],
                        delivery["non_striker"],
                        people.get(delivery["batter"]),
                        people.get(delivery["bowler"]),
                        runs["batter"],
                        runs["extras"],
                        runs["total"],
                        "+".join(sorted(extras)) if extras else None,
                        len(wickets),
                        first_wicket.get("kind"),
                        first_wicket.get("player_out"),
                    )


class MatchLoader:
    """Load every Cricsheet match in a directory into one ball-by-ball table.

    Contract:
        * ``directory`` must be an existing directory, otherwise ``ValueError``.
        * In lenient mode (``strict=False``, the default) a bad file is recorded
          in :attr:`failures` and skipped. In strict mode the first
          ``ValueError`` / :class:`InvalidMatchError` / :class:`DuplicateMatchError`
          is re-raised and loading stops.

    Args:
        directory: Folder containing ``<match_id>.json`` files.
        strict: Raise on the first bad file instead of skipping it.
    """

    def __init__(self, directory: Path | str, strict: bool = False) -> None:
        directory = Path(directory)
        if not directory.is_dir():
            raise ValueError(f"{directory} is not a directory")
        self._directory = directory
        self._strict = strict
        # dict keyed by a tuple fingerprint: duplicate lookup is O(1) per file.
        # Scanning a list of earlier matches instead would be O(n) per file and
        # O(n^2) over the directory.
        self._seen: dict[tuple, str] = {}
        self._failures: list[tuple[str, str]] = []

    @property
    def failures(self) -> tuple[tuple[str, str], ...]:
        """``(file name, reason)`` for every file skipped by the last load."""
        return tuple(self._failures)

    @property
    def loaded_count(self) -> int:
        """Number of distinct matches loaded by the last load."""
        return len(self._seen)

    def iter_matches(self) -> Iterator[MatchFile]:
        """Yield each valid, non-duplicate match in file-name order, resetting state first.

        A match is a duplicate if an earlier file has the same
        :attr:`MatchFile.fingerprint` (same format, dates, teams and venue)
        under a different id.
        """
        self._seen = {}
        self._failures = []
        for path in sorted(self._directory.glob("*.json")):
            try:
                match = MatchFile.from_path(path)
                if match.fingerprint in self._seen:
                    raise DuplicateMatchError(match.match_id, self._seen[match.fingerprint])
            except (ValueError, InvalidMatchError, DuplicateMatchError) as exc:
                if self._strict:
                    raise
                self._failures.append((path.name, str(exc)))
            else:
                self._seen[match.fingerprint] = match.match_id
                yield match

    def iter_deliveries(self) -> Iterator[tuple[Any, ...]]:
        """Yield every delivery of every valid match (see :meth:`MatchFile.iter_deliveries`)."""
        for match in self.iter_matches():
            yield from match.iter_deliveries()

    def load_frame(self) -> pd.DataFrame:
        """Return all deliveries as a ``DataFrame`` with columns :data:`DELIVERY_COLUMNS`.

        Only ``match_date`` is converted (to ``datetime64``); every other value is
        left exactly as recorded so the Scrub stage can see the raw problems.

        Raises:
            InvalidMatchError: The directory contains no valid match files.
        """
        frame = pd.DataFrame.from_records(self.iter_deliveries(), columns=list(DELIVERY_COLUMNS))
        if frame.empty:
            raise InvalidMatchError(f"no valid match files found in {self._directory}")
        frame["match_date"] = pd.to_datetime(frame["match_date"], format="%Y-%m-%d")
        return frame
