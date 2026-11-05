"""Storage formats behind one shared interface (Task 2).

:class:`StorageFormat` defines ``write``, ``read`` and ``read_column``. Each
subclass overrides the private hooks for its own library call. The benchmark
code treats them all the same way and never needs to know which format it is
timing (polymorphism through a shared interface).

``read_column`` has a default implementation that reads the whole file and
then selects the column, because that is all a row-oriented format can do.
CSV and Parquet override it with a real column-pruned read.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from .exceptions import StorageFormatError


class StorageFormat(ABC):
    """A file format that can store and reload the deliveries table.

    Contract:
        * ``compression`` must be one of ``supported_compressions``, otherwise ``ValueError``.
        * :meth:`write` accepts a non-empty ``DataFrame`` only (``ValueError`` otherwise).
        * :meth:`read` and :meth:`read_column` raise ``FileNotFoundError`` for a missing file.

    Args:
        compression: Codec name, or ``None`` for uncompressed.
    """

    name: str = "format"
    extension: str = ""
    supported_compressions: tuple[str | None, ...] = (None,)

    def __init__(self, compression: str | None = None) -> None:
        if compression not in self.supported_compressions:
            raise ValueError(
                f"{self.name} does not support compression {compression!r}; "
                f"choose from {self.supported_compressions}"
            )
        self._compression = compression

    @property
    def compression(self) -> str | None:
        """The codec in use, or ``None``."""
        return self._compression

    @property
    def label(self) -> str:
        """Short name used in result tables, e.g. ``csv+gzip``."""
        return f"{self.name}+{self._compression}" if self._compression else self.name

    def path_in(self, directory: Path | str) -> Path:
        """The file this format writes to inside ``directory``."""
        suffix = ".gz" if self._compression == "gzip" and self.name != "parquet" else ""
        return Path(directory) / f"deliveries_{self.label.replace('+', '_')}{self.extension}{suffix}"

    def write(self, frame: pd.DataFrame, path: Path | str) -> Path:
        """Write ``frame`` to ``path`` (overwriting) and return the path.

        Raises:
            ValueError: ``frame`` is not a non-empty DataFrame.
        """
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("write() needs a non-empty DataFrame")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write(frame, path)
        return path

    def read(self, path: Path | str) -> pd.DataFrame:
        """Read the full table back.

        Raises:
            FileNotFoundError: ``path`` does not exist.
        """
        return self._read(self._existing(path))

    def read_column(self, path: Path | str, column: str) -> pd.Series:
        """Read a single column.

        The default implementation parses the whole file, as any row-oriented
        format must. Columnar subclasses override it.

        Raises:
            ValueError: ``column`` is not a non-empty string.
            FileNotFoundError: ``path`` does not exist.
            KeyError: the column is not in the file.
        """
        if not isinstance(column, str) or not column:
            raise ValueError("column must be a non-empty string")
        return self._read(self._existing(path))[column]

    @staticmethod
    def _existing(path: Path | str) -> Path:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path

    @abstractmethod
    def _write(self, frame: pd.DataFrame, path: Path) -> None:
        """Library-specific write."""

    @abstractmethod
    def _read(self, path: Path) -> pd.DataFrame:
        """Library-specific full read."""


class CsvFormat(StorageFormat):
    """Comma-separated text. Loses dtypes (dates come back as strings)."""

    name = "csv"
    extension = ".csv"
    supported_compressions = (None, "gzip")

    def _write(self, frame, path):
        frame.to_csv(path, index=False, compression=self._compression)

    def _read(self, path):
        return pd.read_csv(path, compression=self._compression, low_memory=False)

    def read_column(self, path, column):
        """Parse only one column (``usecols``). The whole file is still scanned."""
        if not isinstance(column, str) or not column:
            raise ValueError("column must be a non-empty string")
        return pd.read_csv(self._existing(path), usecols=[column], compression=self._compression)[column]


class JsonFormat(StorageFormat):
    """JSON Lines: one record per line, every key repeated on every row."""

    name = "json"
    extension = ".jsonl"
    supported_compressions = (None, "gzip")

    def _write(self, frame, path):
        frame.to_json(path, orient="records", lines=True, date_format="iso", compression=self._compression)

    def _read(self, path):
        return pd.read_json(path, orient="records", lines=True, compression=self._compression)


class PickleFormat(StorageFormat):
    """Python pickle. Fast and exact, but Python-only and unsafe to load from untrusted sources."""

    name = "pickle"
    extension = ".pkl"
    supported_compressions = (None, "gzip")

    def _write(self, frame, path):
        frame.to_pickle(path, compression=self._compression)

    def _read(self, path):
        return pd.read_pickle(path, compression=self._compression)


class ParquetFormat(StorageFormat):
    """Apache Parquet via pyarrow: columnar, typed, compressed per column chunk."""

    name = "parquet"
    extension = ".parquet"
    supported_compressions = (None, "snappy", "gzip", "zstd")

    def _write(self, frame, path):
        # pyarrow treats compression=None as "uncompressed".
        frame.to_parquet(path, engine="pyarrow", index=False, compression=self._compression)

    def _read(self, path):
        return pd.read_parquet(path, engine="pyarrow")

    def read_column(self, path, column):
        """Read only the requested column chunk from disk."""
        if not isinstance(column, str) or not column:
            raise ValueError("column must be a non-empty string")
        return pd.read_parquet(self._existing(path), engine="pyarrow", columns=[column])[column]


def compact_dtypes(frame: pd.DataFrame, max_categories: int = 5_000) -> pd.DataFrame:
    """Return a copy using the smallest sensible dtypes.

    * string columns with at most ``max_categories`` distinct values become
      ``category`` (each value stored once, rows hold a small integer code);
    * integer columns are downcast to the smallest integer type that fits.

    Raises:
        ValueError: ``frame`` is not a DataFrame or ``max_categories`` < 1.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError(f"expected a DataFrame, got {type(frame).__name__}")
    if max_categories < 1:
        raise ValueError("max_categories must be at least 1")
    compact = frame.copy(deep=False)
    for column in compact.columns:
        series = compact[column]
        if pd.api.types.is_string_dtype(series) and series.nunique() <= max_categories:
            compact[column] = series.astype("category")
        elif pd.api.types.is_integer_dtype(series) and not pd.api.types.is_bool_dtype(series):
            compact[column] = pd.to_numeric(series, downcast="integer")
    return compact


FORMATS: dict[str, type[StorageFormat]] = {
    cls.name: cls for cls in (CsvFormat, JsonFormat, PickleFormat, ParquetFormat)
}


def get_format(name: str, compression: str | None = None) -> StorageFormat:
    """Build a format by name, e.g. ``get_format("parquet", "zstd")``.

    Raises:
        StorageFormatError: ``name`` is not a known format.
        ValueError: the format does not support ``compression``.
    """
    try:
        cls = FORMATS[name]
    except KeyError:
        raise StorageFormatError(f"unknown format {name!r}; known formats: {sorted(FORMATS)}") from None
    return cls(compression)
