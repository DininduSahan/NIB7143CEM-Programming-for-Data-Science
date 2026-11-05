"""The Scrub stage: data-quality rules and a transactional runner.

Every rule is a subclass of :class:`CleaningRule` and overrides two hooks:

* ``_detect(frame)`` returns a boolean mask of the rows that have the problem;
* ``_fix(frame, mask)`` returns a frame with the problem dealt with.

The base class's :meth:`CleaningRule.apply` (a template method) validates the
input, calls both hooks and builds a :class:`RuleReport` with the counts, so
every rule reports in the same shape and none can forget to. The
:class:`Scrubber` runs rules in order and either returns the fully cleaned
frame or raises :class:`~cricket_project.exceptions.ScrubError` with the
caller's frame untouched.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import pandas as pd

from .exceptions import ScrubError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuleReport:
    """What one rule found and did. Frozen so a report cannot be edited after the fact."""

    rule: str
    problem: str
    rows_affected: int
    matches_affected: int
    action: str
    reason: str


class CleaningRule(ABC):
    """Base class for a single data-quality rule.

    Contract:
        * :meth:`apply` accepts a ``pandas.DataFrame`` containing every column in
          ``required_columns``; anything else raises ``ValueError``.
        * :meth:`apply` never mutates its argument. It relies on pandas
          Copy-on-Write (default from pandas 3.0), so ``frame.copy(deep=False)``
          is cheap yet any change a rule makes stays in the copy.

    Subclasses set ``name``, ``problem``, ``action``, ``reason`` and
    ``required_columns`` and override ``_detect`` and ``_fix``.
    """

    name: str = "rule"
    problem: str = ""
    action: str = ""
    reason: str = ""
    required_columns: frozenset[str] = frozenset()

    def apply(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, RuleReport]:
        """Detect and fix this rule's problem.

        Args:
            frame: The deliveries table.

        Returns:
            ``(cleaned_frame, report)``.

        Raises:
            ValueError: ``frame`` is not a DataFrame or lacks a required column.
        """
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"{self.name}: expected a DataFrame, got {type(frame).__name__}")
        missing = self.required_columns - set(frame.columns)
        if missing:
            raise ValueError(f"{self.name}: missing required columns {sorted(missing)}")
        mask = self._detect(frame)
        cleaned = self._fix(frame.copy(deep=False), mask)
        report = RuleReport(
            rule=self.name,
            problem=self.problem,
            rows_affected=int(mask.sum()),
            matches_affected=int(frame.loc[mask, "match_id"].nunique()),
            action=self.action,
            reason=self.reason,
        )
        return cleaned, report

    @abstractmethod
    def _detect(self, frame: pd.DataFrame) -> pd.Series:
        """Return a boolean Series, True where the row has the problem."""

    @abstractmethod
    def _fix(self, frame: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        """Return ``frame`` with the problem dealt with. ``frame`` is already a copy."""


class NormaliseSeasonRule(CleaningRule):
    """Cricsheet stores single-year seasons as ints and split seasons as ``"2016/17"``."""

    name = "normalise_season"
    problem = "season stored as int for some matches and str for others"
    action = "converted every season to str"
    reason = "a mixed-type column cannot be written to Parquet and sorts inconsistently"
    required_columns = frozenset({"match_id", "season"})

    def _detect(self, frame):
        return frame["season"].map(lambda value: not isinstance(value, str)).astype(bool)

    def _fix(self, frame, mask):
        frame["season"] = frame["season"].astype(str)
        return frame


class NormaliseMatchTypeRule(CleaningRule):
    """Fold ``IT20`` (non-ICC-status international T20) into ``T20``."""

    name = "normalise_match_type"
    problem = "two labels (T20, IT20) for the same 20-over international format"
    action = "relabelled IT20 as T20"
    reason = "playing conditions are identical; analysis is by format, not by ICC status"
    required_columns = frozenset({"match_id", "match_type"})
    ALIASES = {"IT20": "T20"}

    def _detect(self, frame):
        return frame["match_type"].isin(self.ALIASES.keys())

    def _fix(self, frame, mask):
        frame["match_type"] = frame["match_type"].replace(self.ALIASES)
        return frame


class StandardiseVenueRule(CleaningRule):
    """Give each ground one name.

    Two passes, both dictionary lookups (O(1) per distinct name):

    1. An explicit alias table for renamed grounds and spelling variants.
    2. Otherwise drop a trailing ``", <place>"`` suffix, so
       ``"R Premadasa Stadium, Colombo"`` becomes ``"R Premadasa Stadium"``.

    Names whose suffix is needed to tell grounds apart are pinned in the alias
    table so the suffix rule never merges them.
    """

    name = "standardise_venue"
    problem = "same ground recorded under several names (suffixes, renamings, spellings)"
    action = "mapped every variant to one canonical ground name"
    reason = "grouping by venue would otherwise split one ground's matches across up to four labels"
    required_columns = frozenset({"match_id", "venue"})

    ALIASES: dict[str, str] = {
        "R.Premadasa Stadium": "R Premadasa Stadium",
        "R.Premadasa Stadium, Khettarama": "R Premadasa Stadium",
        "Sinhalese Sports Club": "Sinhalese Sports Club Ground",
        "P Saravanamuttu Stadium": "P Sara Oval",
        "Sher-e-Bangla National Cricket Stadium": "Shere Bangla National Stadium",
        "Chittagong Divisional Stadium": "Zahur Ahmed Chowdhury Stadium",
        "Zayed Cricket Stadium, Abu Dhabi": "Sheikh Zayed Stadium",
        "Feroz Shah Kotla": "Arun Jaitley Stadium",
        "Jade Stadium": "AMI Stadium",
        "Sharjah Cricket Association Stadium": "Sharjah Cricket Stadium",
        "Simonds Stadium, South Geelong": "GMHBA Stadium",
        "Boland Bank Park, Paarl": "Boland Park",
        "Goodyear Park, Bloemfontein": "Mangaung Oval",
        "Chevrolet Park": "Mangaung Oval",
        "Sardar Patel (Gujarat) Stadium, Motera": "Narendra Modi Stadium",
        "Sardar Patel Stadium": "Narendra Modi Stadium",
        "Sardar Patel Stadium, Motera": "Narendra Modi Stadium",
        "Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium":
            "Andhra Cricket Association-Visakhapatnam District Cricket Association Stadium",
        "Beausejour Stadium, Gros Islet": "Daren Sammy National Cricket Stadium",
        "The Oval": "Kennington Oval",
        "The Wanderers Stadium": "New Wanderers Stadium",
        "Punjab Cricket Association Stadium, Mohali": "Punjab Cricket Association IS Bindra Stadium",
        # Pinned: the suffix is what distinguishes these grounds.
        "Nehru Stadium, Fatorda": "Nehru Stadium, Fatorda",
        "National Cricket Stadium, St George's": "National Cricket Stadium, St George's",
    }

    @classmethod
    def canonical(cls, venue: str) -> str:
        """Return the canonical name for one raw venue string."""
        if venue in cls.ALIASES:
            return cls.ALIASES[venue]
        base = venue.split(",", 1)[0].strip()
        return cls.ALIASES.get(base, base)

    def _mapping(self, frame: pd.DataFrame) -> dict[str, str]:
        # Resolve each *distinct* name once (about 200) rather than 660k times.
        return {venue: self.canonical(venue) for venue in frame["venue"].unique()}

    def _detect(self, frame):
        mapping = self._mapping(frame)
        return frame["venue"] != frame["venue"].map(mapping)

    def _fix(self, frame, mask):
        frame["venue"] = frame["venue"].map(self._mapping(frame))
        return frame


class StandardiseCityRule(CleaningRule):
    """Fix city names that are renamed, or that record an island or region instead of a city."""

    name = "standardise_city"
    problem = "city recorded under old and new names, or as an island/region instead of a city"
    action = "mapped to the current city name"
    reason = "Bangalore/Bengaluru etc. would otherwise count as two cities"
    required_columns = frozenset({"match_id", "city"})
    ALIASES: dict[str, str] = {
        "Bangalore": "Bengaluru",
        "Chittagong": "Chattogram",
        "Port Elizabeth": "Gqeberha",
        "Pallekele": "Kandy",
        "Mirpur": "Dhaka",
        "Barbados": "Bridgetown",
        "Jamaica": "Kingston",
        "Trinidad": "Port of Spain",
        "Guyana": "Providence",
        "St Lucia": "Gros Islet",
        "Antigua": "North Sound",
        "Grenada": "St George's",
        "Victoria": "Geelong",
    }

    def _detect(self, frame):
        return frame["city"].isin(self.ALIASES.keys())

    def _fix(self, frame, mask):
        frame["city"] = frame["city"].replace(self.ALIASES)
        return frame


class ImputeCityRule(CleaningRule):
    """Fill a missing city from the same ground's city in other matches.

    Must run after :class:`StandardiseVenueRule` and :class:`StandardiseCityRule`,
    otherwise the lookup would miss grounds recorded under a different name.
    Rows whose ground never has a city recorded stay missing.
    """

    name = "impute_city"
    problem = "city missing although the ground's city is known from other matches"
    action = "filled from the most common city recorded for the same ground"
    reason = "a ground does not move; leaving these blank would drop them from any home/away split"
    required_columns = frozenset({"match_id", "venue", "city"})

    def _venue_city(self, frame: pd.DataFrame) -> dict[str, str]:
        known = frame.dropna(subset=["city"]).drop_duplicates(["match_id"])
        return known.groupby("venue")["city"].agg(lambda cities: cities.mode().iat[0]).to_dict()

    def _detect(self, frame):
        return frame["city"].isna() & frame["venue"].isin(self._venue_city(frame).keys())

    def _fix(self, frame, mask):
        frame.loc[mask, "city"] = frame.loc[mask, "venue"].map(self._venue_city(frame))
        return frame


class FillMissingEventRule(CleaningRule):
    """Label matches with no event name explicitly instead of leaving a null."""

    name = "fill_missing_event"
    problem = "event (series/tournament) name not recorded"
    action = "set to the literal '(not recorded)'"
    reason = "keeps the rows; an explicit label stops groupby() silently dropping them"
    required_columns = frozenset({"match_id", "event_name"})
    PLACEHOLDER = "(not recorded)"

    def _detect(self, frame):
        return frame["event_name"].isna()

    def _fix(self, frame, mask):
        frame["event_name"] = frame["event_name"].fillna(self.PLACEHOLDER)
        return frame


class RemoveSuperOverRule(CleaningRule):
    """Remove super-over deliveries from the main table.

    The rows are dropped, but counted in the report, and the match result
    (``winner``) still records who won the super over.
    """

    name = "remove_super_overs"
    problem = "super-over tie-breaker deliveries mixed in with regulation innings"
    action = "removed from the deliveries table (match winner kept)"
    reason = "a one-over tie-breaker is not part of an innings; it would distort per-innings totals and run rates"
    required_columns = frozenset({"match_id", "is_super_over"})

    def _detect(self, frame):
        return frame["is_super_over"].astype(bool)

    def _fix(self, frame, mask):
        return frame.loc[~mask].reset_index(drop=True)


class FlagMiscountedOverRule(CleaningRule):
    """Flag overs that contain more than six legal deliveries.

    Under the Laws an over has six legal balls, but umpires occasionally
    miscount and the extra ball stands. The data is a true record of what
    happened, so the rows are kept and flagged rather than "corrected".
    Adds ``is_legal_ball`` and ``over_miscounted`` columns.
    """

    name = "flag_miscounted_overs"
    problem = "overs with more than six legal deliveries (umpire miscount)"
    action = "kept, flagged with over_miscounted=True"
    reason = "the extra ball was really bowled and its runs count; deleting it would make totals wrong"
    required_columns = frozenset({"match_id", "innings", "over", "extras_type"})
    # A tuple of the extras that make a delivery illegal (it does not count towards the over).
    ILLEGAL_EXTRAS: tuple[str, ...] = ("wides", "noballs")

    def _legal(self, frame: pd.DataFrame) -> pd.Series:
        extras = frame["extras_type"].fillna("")
        return ~extras.str.contains("|".join(self.ILLEGAL_EXTRAS))

    def _detect(self, frame):
        legal_per_over = self._legal(frame).groupby(
            [frame["match_id"], frame["innings"], frame["over"]]
        ).transform("sum")
        return legal_per_over > 6

    def _fix(self, frame, mask):
        frame["is_legal_ball"] = self._legal(frame)
        frame["over_miscounted"] = mask
        return frame


class FlagNonDismissalRule(CleaningRule):
    """Retirements are recorded as wickets but are not dismissals.

    Adds ``is_dismissal``, so a bowler is never credited, and a batter never
    charged, with a wicket for a retirement.
    """

    name = "flag_non_dismissals"
    problem = "retired hurt / retired not out recorded in the wickets field"
    action = "kept, is_dismissal=False for those rows"
    reason = "counting them as dismissals inflates wicket counts and lowers batting averages"
    required_columns = frozenset({"match_id", "wicket_count", "wicket_kind"})
    NOT_DISMISSALS = frozenset({"retired hurt", "retired not out"})

    def _detect(self, frame):
        return frame["wicket_kind"].isin(self.NOT_DISMISSALS)

    def _fix(self, frame, mask):
        frame["is_dismissal"] = (frame["wicket_count"] > 0) & ~mask
        return frame


class RunsConsistencyRule(CleaningRule):
    """Check that ``runs_total == runs_batter + runs_extras`` on every delivery."""

    name = "check_runs_consistency"
    problem = "runs_total differs from runs_batter + runs_extras"
    action = "recompute runs_total from its parts"
    reason = "the parts are recorded per event and are the more reliable source"
    required_columns = frozenset({"match_id", "runs_batter", "runs_extras", "runs_total"})

    def _detect(self, frame):
        return frame["runs_total"] != frame["runs_batter"] + frame["runs_extras"]

    def _fix(self, frame, mask):
        frame.loc[mask, "runs_total"] = frame.loc[mask, "runs_batter"] + frame.loc[mask, "runs_extras"]
        return frame


def default_rules() -> list[CleaningRule]:
    """The rules used by the project pipeline, in the order they must run."""
    return [
        NormaliseSeasonRule(),
        NormaliseMatchTypeRule(),
        StandardiseVenueRule(),
        StandardiseCityRule(),
        ImputeCityRule(),
        FillMissingEventRule(),
        RemoveSuperOverRule(),
        FlagMiscountedOverRule(),
        FlagNonDismissalRule(),
        RunsConsistencyRule(),
    ]


class Scrubber:
    """Run a sequence of :class:`CleaningRule` objects as one all-or-nothing step.

    Contract:
        * ``rules`` must be a non-empty sequence of :class:`CleaningRule`
          instances with unique names, otherwise ``ValueError``.
        * :meth:`run` either returns the frame with every rule applied, or raises
          :class:`ScrubError`. On failure the caller's frame is unchanged and
          :attr:`last_reports` still holds the previous successful run.

    Args:
        rules: Rules to apply, in order.
    """

    def __init__(self, rules: Sequence[CleaningRule]) -> None:
        if not rules:
            raise ValueError("Scrubber needs at least one rule")
        for rule in rules:
            if not isinstance(rule, CleaningRule):
                raise ValueError(f"{rule!r} is not a CleaningRule")
        names = [rule.name for rule in rules]
        if len(names) != len(set(names)):
            raise ValueError(f"rule names must be unique, got {names}")
        self._rules: tuple[CleaningRule, ...] = tuple(rules)
        self._last_reports: tuple[RuleReport, ...] = ()

    @property
    def rules(self) -> tuple[CleaningRule, ...]:
        """The rules in the order they run."""
        return self._rules

    @property
    def last_reports(self) -> tuple[RuleReport, ...]:
        """Reports from the most recent *successful* :meth:`run`."""
        return self._last_reports

    def run(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, tuple[RuleReport, ...]]:
        """Apply every rule in order.

        Args:
            frame: Raw deliveries table from :class:`~cricket_project.data_loader.MatchLoader`.

        Returns:
            ``(cleaned_frame, reports)``, one report per rule.

        Raises:
            ValueError: ``frame`` is not a DataFrame.
            ScrubError: A rule failed; nothing is committed.
        """
        if not isinstance(frame, pd.DataFrame):
            raise ValueError(f"expected a DataFrame, got {type(frame).__name__}")
        working = frame
        reports: list[RuleReport] = []
        current = self._rules[0]
        started = time.perf_counter()
        try:
            for current in self._rules:
                working, report = current.apply(working)
                reports.append(report)
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            # Nothing has been committed: `frame` was never mutated and
            # `_last_reports` is only replaced in the else branch below.
            raise ScrubError(current.name, str(exc)) from exc
        else:
            self._last_reports = tuple(reports)
            return working, self._last_reports
        finally:
            # Runs on success and on failure, so every attempt is logged.
            logger.info(
                "scrub: %d/%d rules applied in %.2fs",
                len(reports), len(self._rules), time.perf_counter() - started,
            )

    @staticmethod
    def report_frame(reports: Sequence[RuleReport]) -> pd.DataFrame:
        """Turn reports into a table for the README/report."""
        return pd.DataFrame([asdict(report) for report in reports])
