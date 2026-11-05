"""Measuring storage formats (Task 2): size, write time, read time, one-column read time.

Method, as required by the brief: every timing is repeated ``repeats`` times,
the first ``warmup`` runs are discarded, and the median is reported with the
minimum and maximum of the remaining runs.
"""

from __future__ import annotations

import gc
import os
import platform
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .storage import StorageFormat


@dataclass(frozen=True)
class TimingResult:
    """Kept timings for one operation; the warm-up runs are already removed."""

    samples: tuple[float, ...]

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def minimum(self) -> float:
        return min(self.samples)

    @property
    def maximum(self) -> float:
        return max(self.samples)


def time_callable(func: Callable[[], object], repeats: int = 6, warmup: int = 1) -> TimingResult:
    """Call ``func`` ``repeats`` times and keep all but the first ``warmup`` timings.

    Raises:
        ValueError: ``warmup`` is negative or ``repeats`` is not greater than ``warmup``.
    """
    if warmup < 0 or repeats <= warmup:
        raise ValueError(f"need repeats > warmup >= 0, got repeats={repeats}, warmup={warmup}")
    timings = []
    for _ in range(repeats):
        gc.collect()
        start = time.perf_counter()
        func()
        timings.append(time.perf_counter() - start)
    return TimingResult(tuple(timings[warmup:]))


def environment() -> dict[str, str]:
    """Hardware and library versions; the numbers are only meaningful alongside these."""
    import numpy
    import pyarrow

    return {
        "python": platform.python_version(),
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpus": str(os.cpu_count()),
        "pandas": pd.__version__,
        "numpy": numpy.__version__,
        "pyarrow": pyarrow.__version__,
    }


class FormatBenchmark:
    """Time every format in ``formats`` against the same DataFrame.

    Contract:
        * ``frame`` must be a non-empty DataFrame containing ``column``.
        * ``formats`` must be a non-empty sequence of :class:`StorageFormat`
          with unique labels.
        Otherwise ``ValueError``.

    Args:
        frame: The dataset to store.
        formats: Formats to compare.
        workdir: Folder for the benchmark files (overwritten each run).
        column: Column used for the single-column read test.
        repeats: Runs per timing, including the warm-up.
        warmup: Leading runs discarded.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        formats: Sequence[StorageFormat],
        workdir: Path | str,
        column: str = "runs_total",
        repeats: int = 6,
        warmup: int = 1,
    ) -> None:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            raise ValueError("frame must be a non-empty DataFrame")
        if column not in frame.columns:
            raise ValueError(f"column {column!r} is not in the frame")
        if not formats or not all(isinstance(fmt, StorageFormat) for fmt in formats):
            raise ValueError("formats must be a non-empty sequence of StorageFormat")
        labels = [fmt.label for fmt in formats]
        if len(labels) != len(set(labels)):
            raise ValueError(f"format labels must be unique, got {labels}")
        if warmup < 0 or repeats <= warmup:
            raise ValueError(f"need repeats > warmup >= 0, got repeats={repeats}, warmup={warmup}")
        self._frame = frame
        self._formats = tuple(formats)
        self._workdir = Path(workdir)
        self._column = column
        self._repeats = repeats
        self._warmup = warmup

    def run(self, progress: Callable[[str], None] = print) -> pd.DataFrame:
        """Run all measurements and return one row per (format, operation)."""
        self._workdir.mkdir(parents=True, exist_ok=True)
        rows = []
        for fmt in self._formats:
            path = fmt.path_in(self._workdir)
            progress(f"  {fmt.label}: write")
            write = time_callable(lambda: fmt.write(self._frame, path), self._repeats, self._warmup)
            size = path.stat().st_size
            progress(f"  {fmt.label}: read")
            read = time_callable(lambda: fmt.read(path), self._repeats, self._warmup)
            progress(f"  {fmt.label}: read one column")
            column = time_callable(lambda: fmt.read_column(path, self._column), self._repeats, self._warmup)
            # Did the round trip keep the dtypes? (CSV and JSON cannot keep dates or categories.)
            reread = fmt.read(path)
            dtypes_kept = bool((reread.dtypes.astype(str) == self._frame.dtypes.astype(str)).all())
            for operation, result in (("write", write), ("read", read), ("read_column", column)):
                rows.append({
                    "format": fmt.label,
                    "operation": operation,
                    "size_mb": round(size / 1_000_000, 2),
                    "median_s": round(result.median, 4),
                    "min_s": round(result.minimum, 4),
                    "max_s": round(result.maximum, 4),
                    "runs_kept": len(result.samples),
                    "dtypes_preserved": dtypes_kept,
                })
        return pd.DataFrame(rows)
