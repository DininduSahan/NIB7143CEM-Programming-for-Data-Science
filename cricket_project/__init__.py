"""cricket_project: load, clean, store and analyse Cricsheet ball-by-ball data.

Public interface::

    from cricket_project import MatchLoader, Scrubber, default_rules, get_format

    raw = MatchLoader("Data/Raw").load_frame()
    clean, reports = Scrubber(default_rules()).run(raw)
    get_format("parquet", "zstd").write(clean, "deliveries.parquet")
"""

from .cleaning import CleaningRule, RuleReport, Scrubber, default_rules
from .data_loader import DELIVERY_COLUMNS, MatchFile, MatchLoader
from .exceptions import (
    CricketDataError,
    DuplicateMatchError,
    InvalidMatchError,
    RawDataIntegrityError,
    ScrubError,
    StorageFormatError,
)
from .privacy import pseudonymise
from .storage import (
    CsvFormat,
    JsonFormat,
    ParquetFormat,
    PickleFormat,
    StorageFormat,
    compact_dtypes,
    get_format,
)

__version__ = "1.0.0"

__all__ = [
    "DELIVERY_COLUMNS",
    "CleaningRule",
    "CricketDataError",
    "CsvFormat",
    "DuplicateMatchError",
    "InvalidMatchError",
    "JsonFormat",
    "MatchFile",
    "MatchLoader",
    "ParquetFormat",
    "PickleFormat",
    "RawDataIntegrityError",
    "RuleReport",
    "ScrubError",
    "Scrubber",
    "StorageFormat",
    "StorageFormatError",
    "compact_dtypes",
    "default_rules",
    "get_format",
    "pseudonymise",
]
