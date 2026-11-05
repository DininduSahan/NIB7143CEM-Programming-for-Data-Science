"""Task 4, route A: the phase run-rate stage in PySpark versus pandas.

The stage is :func:`cricket_project.analysis.phase_run_rates`: filter to
limited-overs deliveries, label each with its phase (narrow transformations),
then group by format/phase/side and sum (a wide transformation: Spark must
shuffle rows so all rows of one group meet in one partition).

To find where (if anywhere) Spark overtakes pandas on this machine, the input
is replicated 1x, 2x, 4x... and both engines are timed on each size. Spark is
run twice per size: with its default 200 shuffle partitions and with one
shuffle partition per CPU core.

PySpark is imported lazily, so the rest of the package works without Java.
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd

from . import analysis
from .benchmark import time_callable

STAGE_COLUMNS = ["match_type", "over", "batting_team", "runs_total", "is_legal_ball"]


def build_scaled_inputs(clean_parquet: Path, outdir: Path, scales: Sequence[int]) -> dict[int, Path]:
    """Write ``k`` stacked copies of the stage's input columns for each ``k`` in ``scales``.

    Raises:
        ValueError: a scale is not a positive integer.
        FileNotFoundError: ``clean_parquet`` does not exist.
    """
    if not scales or any(not isinstance(k, int) or k < 1 for k in scales):
        raise ValueError(f"scales must be positive integers, got {scales}")
    if not Path(clean_parquet).is_file():
        raise FileNotFoundError(clean_parquet)
    base = pd.read_parquet(clean_parquet, columns=STAGE_COLUMNS)
    # Plain types so pandas and Spark read exactly the same physical file.
    base = base.astype({"match_type": str, "batting_team": str, "over": "int32",
                        "runs_total": "int32", "is_legal_ball": bool})
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k in sorted(set(scales)):
        path = outdir / f"stage_input_x{k}.parquet"
        if not path.exists():
            pd.concat([base] * k, ignore_index=True).to_parquet(path, index=False, row_group_size=500_000)
        paths[k] = path
    return paths


def pandas_stage(path: Path) -> pd.DataFrame:
    """The pandas implementation, exactly as used in the Task 3 pipeline."""
    return analysis.phase_run_rates(pd.read_parquet(path, columns=STAGE_COLUMNS))


def start_spark(shuffle_partitions: int | None = None, driver_memory: str = "4g"):
    """Create a local-mode SparkSession using every core.

    Raises:
        ImportError: PySpark is not installed.
    """
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)
    from pyspark.sql import SparkSession

    builder = (
        SparkSession.builder.master("local[*]")
        .appName("cricket-phase-run-rates")
        .config("spark.driver.memory", driver_memory)
        .config("spark.ui.showConsoleProgress", "false")
        .config("spark.sql.adaptive.enabled", "false")  # keep the plan as written, so partitions are what we set
    )
    if shuffle_partitions is not None:
        builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def spark_stage_frame(spark, path: Path):
    """Build (lazily) the Spark DataFrame for the stage. Nothing runs until an action."""
    from pyspark.sql import functions as F

    def phase(first_power: int, first_death: int):
        return (
            F.when(F.col("over") < first_power, "Powerplay")
            .when(F.col("over") < first_death, "Middle")
            .otherwise("Death")
        )

    rows = spark.read.parquet(str(path))
    return (
        rows.where(F.col("match_type").isin(*analysis.LIMITED_OVERS))                 # narrow
        .withColumn("phase", F.when(F.col("match_type") == "ODI", phase(10, 40))     # narrow
                    .otherwise(phase(6, 15)))
        .withColumn("side", F.when(F.col("batting_team") == analysis.FOCUS_TEAM,    # narrow
                                   analysis.FOCUS_TEAM).otherwise("Opposition"))
        .groupBy("match_type", "phase", "side")                                     # wide: shuffle
        .agg(F.sum("runs_total").alias("runs"), F.sum(F.col("is_legal_ball").cast("int")).alias("balls"))
        .withColumn("run_rate", F.round(6 * F.col("runs") / F.col("balls"), 2))
    )


def spark_stage(spark, path: Path) -> pd.DataFrame:
    """Run the Spark stage and collect the (12-row) result to pandas."""
    rows = spark_stage_frame(spark, path).collect()
    return pd.DataFrame([row.asDict() for row in rows])


def explain_plan(spark, path: Path) -> str:
    """The formatted physical plan, showing the Exchange (shuffle) and stage boundaries."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        spark_stage_frame(spark, path).explain(mode="formatted")
    return buffer.getvalue()


def _same_result(pandas_result: pd.DataFrame, spark_result: pd.DataFrame) -> bool:
    keys = ["match_type", "phase", "side"]
    left = pandas_result.assign(phase=pandas_result["phase"].astype(str)).sort_values(keys).reset_index(drop=True)
    right = spark_result.sort_values(keys).reset_index(drop=True)
    return bool(
        (left[keys].astype(str).to_numpy() == right[keys].astype(str).to_numpy()).all()
        and (left[["runs", "balls"]].to_numpy() == right[["runs", "balls"]].to_numpy()).all()
    )


def run_comparison(
    clean_parquet: Path,
    workdir: Path,
    scales: Sequence[int] = (1, 2, 4, 8, 16),
    repeats: int = 6,
    progress: Callable[[str], None] = print,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Time pandas and Spark on every scale.

    Returns:
        ``(results, details)`` where ``results`` has one row per (engine, scale)
        and ``details`` holds the session start-up time, input partition counts,
        the physical plan and the correctness check.
    """
    paths = build_scaled_inputs(clean_parquet, workdir, scales)
    cores = os.cpu_count() or 1
    rows: list[dict[str, object]] = []
    details: dict[str, object] = {"cores": cores}

    for k, path in paths.items():
        n_rows = len(pd.read_parquet(path, columns=["over"]))
        progress(f"  pandas x{k} ({n_rows:,} rows)")
        timing = time_callable(lambda: pandas_stage(path), repeats)
        rows.append({"engine": "pandas", "scale": k, "rows": n_rows, "shuffle_partitions": None,
                     "median_s": timing.median, "min_s": timing.minimum, "max_s": timing.maximum})

    for label, shuffle_partitions in (("spark", 200), ("spark_tuned", cores)):
        started = time.perf_counter()
        spark = start_spark(shuffle_partitions)
        details[f"{label}_startup_s"] = round(time.perf_counter() - started, 2)
        try:
            for k, path in paths.items():
                n_rows = len(pd.read_parquet(path, columns=["over"]))
                progress(f"  {label} x{k} ({n_rows:,} rows, {shuffle_partitions} shuffle partitions)")
                timing = time_callable(lambda: spark_stage(spark, path), repeats)
                input_partitions = spark.read.parquet(str(path)).rdd.getNumPartitions()
                rows.append({"engine": label, "scale": k, "rows": n_rows,
                             "shuffle_partitions": shuffle_partitions, "input_partitions": input_partitions,
                             "median_s": timing.median, "min_s": timing.minimum, "max_s": timing.maximum})
            first = paths[min(paths)]
            details[f"{label}_matches_pandas"] = _same_result(pandas_stage(first), spark_stage(spark, first))
            if label == "spark":
                details["physical_plan"] = explain_plan(spark, first)
        finally:
            spark.stop()  # always release the JVM, even if a timing fails

    results = pd.DataFrame(rows)
    for column in ("median_s", "min_s", "max_s"):
        results[column] = results[column].round(4)
    return results, details
