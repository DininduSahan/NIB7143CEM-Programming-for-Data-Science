"""Data minimisation before anything is written to disk (Task 5, GDPR Art. 5(1)(c)).

Cricsheet data names real people (players and officials). The analysis in this
project works at team, format and venue level, so names are not needed. The
processed dataset keeps Cricsheet's own registry identifiers for batter and
bowler (pseudonymous: they can only be linked back to a name with the
Cricsheet people register) and drops every name column.
"""

from __future__ import annotations

import pandas as pd

# Columns that hold a person's name directly.
DIRECT_IDENTIFIERS: tuple[str, ...] = ("batter", "bowler", "non_striker", "player_out")


def pseudonymise(frame: pd.DataFrame, drop: tuple[str, ...] = DIRECT_IDENTIFIERS) -> pd.DataFrame:
    """Return a copy of ``frame`` without the name columns in ``drop``.

    Raises:
        ValueError: ``frame`` is not a DataFrame, or a column in ``drop`` is absent
            (so a renamed column cannot silently slip through un-dropped).
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"expected a DataFrame, got {type(frame).__name__}")
    absent = [column for column in drop if column not in frame.columns]
    if absent:
        raise ValueError(f"cannot pseudonymise: columns {absent} not found")
    return frame.drop(columns=list(drop))
