"""Domain-specific exceptions for the cricket_project package.

The package separates two kinds of failure:

* ``ValueError`` (built-in) - the *input is malformed*: wrong type, not JSON,
  an empty argument. The caller passed something that could never work.
* ``CricketDataError`` and its subclasses - the input is well-formed but
  breaks a *domain rule*: a match file that is valid JSON but is not a valid
  Cricsheet match, a duplicate match, a raw file whose checksum changed, a
  cleaning rule that could not be applied.

Keeping them apart lets a caller decide, for example, to skip a corrupt
match file but stop the whole run if the raw data has been tampered with.
"""

from __future__ import annotations

from pathlib import Path


class CricketDataError(Exception):
    """Base class for every domain-specific failure raised by this package."""


class InvalidMatchError(CricketDataError):
    """A match payload is valid JSON but violates the Cricsheet contract.

    Args:
        message: Human-readable explanation of which rule was broken.
        source: The file the payload came from, if known.
    """

    def __init__(self, message: str, source: Path | str | None = None) -> None:
        self.source = Path(source) if source is not None else None
        prefix = f"{self.source.name}: " if self.source is not None else ""
        super().__init__(prefix + message)


class DuplicateMatchError(CricketDataError):
    """The same match appears twice in one directory under different ids."""

    def __init__(self, match_id: str, duplicate_of: str) -> None:
        self.match_id = match_id
        self.duplicate_of = duplicate_of
        super().__init__(f"match {match_id!r} duplicates already-loaded match {duplicate_of!r}")


class RawDataIntegrityError(CricketDataError):
    """The immutable raw extract no longer matches its recorded checksums."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = list(problems)
        shown = "; ".join(self.problems[:5])
        more = f" (+{len(self.problems) - 5} more)" if len(self.problems) > 5 else ""
        super().__init__(f"raw data integrity check failed: {shown}{more}")


class ScrubError(CricketDataError):
    """A cleaning rule failed. The dataset passed in is left unchanged.

    Args:
        rule_name: Name of the rule that failed.
        message: What went wrong.
    """

    def __init__(self, rule_name: str, message: str) -> None:
        self.rule_name = rule_name
        super().__init__(f"cleaning rule {rule_name!r} failed, no changes applied: {message}")


class StorageFormatError(CricketDataError):
    """A storage format was requested that this package does not support."""
