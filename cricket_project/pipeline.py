"""Command-line stages that reproduce every number and figure in the report.

Run from the repository root::

    python main.py checksums --verify   # Obtain: prove the raw extract is unchanged
    python main.py pipeline             # Task 3: Obtain, Scrub, Explore, Model
    python main.py benchmark            # Task 2: storage formats
    python main.py spark                # Task 4 route A: pandas vs PySpark
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

from . import analysis, figures
from .benchmark import FormatBenchmark, environment
from .checksums import verify_manifest, write_manifest
from .cleaning import Scrubber, default_rules
from .data_loader import MatchLoader
from .exceptions import CricketDataError
from .privacy import pseudonymise
from .storage import ParquetFormat, compact_dtypes, get_format

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "Data" / "Raw"
MANIFEST = ROOT / "Data" / "raw_checksums.sha256"
OUTPUTS = ROOT / "outputs"
CLEAN_PARQUET = OUTPUTS / "data" / "deliveries_clean.parquet"
TABLES = OUTPUTS / "tables"
FIGURES = OUTPUTS / "figures"

# Formats and codecs compared in Task 2: (name, compression).
BENCHMARK_FORMATS: tuple[tuple[str, str | None], ...] = (
    ("csv", None), ("csv", "gzip"),
    ("json", None), ("json", "gzip"),
    ("pickle", None), ("pickle", "gzip"),
    ("parquet", None), ("parquet", "snappy"), ("parquet", "zstd"),
)


def _show(path: Path) -> str:
    """A path as printed to the user, starting at the repository folder."""
    return f"{ROOT.name}/{path.relative_to(ROOT).as_posix()}"


def _table(frame: pd.DataFrame, name: str) -> Path:
    TABLES.mkdir(parents=True, exist_ok=True)
    path = TABLES / f"{name}.csv"
    frame.to_csv(path, index=False)
    return path


def cmd_checksums(args: argparse.Namespace) -> None:
    """Write or verify the raw-data checksum manifest."""
    if args.write:
        count = write_manifest(RAW_DIR, MANIFEST)
        print(f"Wrote {count} checksums to {_show(MANIFEST)}")
    else:
        count = verify_manifest(RAW_DIR, MANIFEST)
        print(f"OK: {count} raw files match {_show(MANIFEST)}")


def cmd_pipeline(args: argparse.Namespace) -> None:
    """Obtain -> Scrub -> Explore -> Model, writing tables and figures."""
    print("[Obtain] verifying raw extract checksums")
    print(f"  {verify_manifest(RAW_DIR, MANIFEST)} files unchanged")
    loader = MatchLoader(RAW_DIR, strict=False)
    raw = loader.load_frame()
    print(f"  loaded {loader.loaded_count} matches, {len(raw):,} deliveries, {len(loader.failures)} files rejected")
    _table(pd.DataFrame(loader.failures, columns=["file", "reason"]), "load_failures")

    print("[Scrub]")
    scrubber = Scrubber(default_rules())
    clean, reports = scrubber.run(raw)
    report = Scrubber.report_frame(reports)
    _table(report, "scrub_report")
    for row in report.itertuples():
        print(f"  {row.rule:<24} rows={row.rows_affected:>7,}  matches={row.matches_affected:>4}")
    missing = pd.DataFrame({"missing_before": raw.isna().sum(), "missing_after": clean.reindex(columns=raw.columns).isna().sum()})
    _table(missing.rename_axis("column").reset_index(), "missing_values_before_after")
    _table(pd.DataFrame({
        "measure": ["rows", "columns", "distinct venues", "distinct cities"],
        "raw": [len(raw), raw.shape[1], raw["venue"].nunique(), raw["city"].nunique()],
        "clean": [len(clean), clean.shape[1], clean["venue"].nunique(), clean["city"].nunique()],
    }), "scrub_summary")

    processed = compact_dtypes(pseudonymise(clean))
    ParquetFormat("zstd").write(processed, CLEAN_PARQUET)
    print(f"  wrote {_show(CLEAN_PARQUET)} ({processed.shape[0]:,} x {processed.shape[1]}, names removed)")

    print("[Explore]")
    totals = analysis.innings_totals(processed)
    describe = totals.loc[totals["innings"] == 1].groupby("match_type")["runs"].describe().round(1)
    _table(describe.reset_index(), "first_innings_totals")
    rates = analysis.phase_run_rates(processed)
    _table(rates, "phase_run_rates")
    summary = analysis.match_summary(processed)
    _table(summary["outcome"].value_counts().rename_axis("outcome").reset_index(), "outcomes")
    print(f"  {figures.first_innings_totals(totals, FIGURES / 'fig1_first_innings_totals.png').name}")
    print(f"  {figures.phase_run_rates(rates, FIGURES / 'fig2_phase_run_rates.png').name}")

    print("[Model]")
    _table(analysis.win_rates(summary, ["match_type", "home"]), "win_rates_home_away")
    _table(analysis.win_rates(summary, ["match_type", "home", "batted_first"]), "win_rates_home_batting_first")
    tests = pd.DataFrame([
        analysis.independence_test(summary, factor, match_type)
        for factor in ("home", "won_toss", "batted_first")
        for match_type in ("ODI", "T20", "Test")
    ])
    _table(tests, "independence_tests")
    print(tests.to_string(index=False))
    print(f"  {figures.factor_win_rates(tests, FIGURES / 'fig3_factor_win_rates.png').name}")
    print(f"Done. Tables in {_show(TABLES)}, figures in {_show(FIGURES)}")


def _strip_local_paths(text: str) -> str:
    """Make absolute paths start at the repository folder, e.g.
    ``file:/C:/Users/.../NIB7143CEM-Programming-for-Data-Science/outputs/x`` becomes
    ``NIB7143CEM-Programming-for-Data-Science/outputs/x``, so committed outputs do
    not reveal the author's username or folder layout."""
    prefix = r"(?:file:/*)?" + re.escape(ROOT.parent.as_posix()) + "/"
    return re.sub(prefix, "", text, flags=re.IGNORECASE)


def _clean_frame() -> pd.DataFrame:
    if not CLEAN_PARQUET.exists():
        raise FileNotFoundError(f"{_show(CLEAN_PARQUET)} not found: run `python main.py pipeline` first")
    return pd.read_parquet(CLEAN_PARQUET)


def cmd_benchmark(args: argparse.Namespace) -> None:
    """Task 2: size and timings for every format in BENCHMARK_FORMATS."""
    if args.from_table:
        results = pd.read_csv(TABLES / "storage_benchmark.csv")
        print(f"  {figures.storage_results(results, FIGURES / 'fig4_storage_formats.png').name}")
        return
    frame = _clean_frame()
    formats = [get_format(name, codec) for name, codec in BENCHMARK_FORMATS]
    print(f"Benchmarking {len(formats)} formats on {len(frame):,} rows x {frame.shape[1]} columns, "
          f"{args.repeats} runs each (first discarded)")
    workdir = Path(args.workdir) if args.workdir else OUTPUTS / "benchmark_files"
    shown = _show(workdir.resolve()) if workdir.resolve().is_relative_to(ROOT) else workdir
    print(f"  benchmark files in {shown}")
    results = FormatBenchmark(frame, formats, workdir, repeats=args.repeats).run()
    _table(results, "storage_benchmark")
    env = environment()
    (TABLES / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
    print(results.to_string(index=False))
    print(f"  {figures.storage_results(results, FIGURES / 'fig4_storage_formats.png').name}")


def cmd_spark(args: argparse.Namespace) -> None:
    """Task 4 route A: pandas vs PySpark on growing copies of the data."""
    from .spark_compare import run_comparison

    _clean_frame()  # fail early with a clear message if the pipeline has not run
    results, details = run_comparison(CLEAN_PARQUET, OUTPUTS / "spark", args.scales, args.repeats)
    _table(results, "spark_vs_pandas")
    plan = _strip_local_paths(details.pop("physical_plan", ""))
    (TABLES / "spark_physical_plan.txt").write_text(plan, encoding="utf-8")
    (TABLES / "spark_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    print(results.to_string(index=False))
    print(json.dumps(details, indent=2))
    print(f"  {figures.spark_scaling(results, FIGURES / 'fig5_spark_vs_pandas.png').name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="main.py", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    checksums = sub.add_parser("checksums", help="write or verify the raw data checksum manifest")
    mode = checksums.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true", help="record checksums (do this once, after obtaining the data)")
    mode.add_argument("--verify", action="store_true", help="check the raw files against the manifest")
    checksums.set_defaults(func=cmd_checksums)

    sub.add_parser("pipeline", help="Task 3: Obtain, Scrub, Explore, Model").set_defaults(func=cmd_pipeline)

    bench = sub.add_parser("benchmark", help="Task 2: compare storage formats")
    bench.add_argument("--repeats", type=int, default=6, help="runs per timing incl. 1 warm-up (default 6)")
    bench.add_argument("--workdir", help="where to write the benchmark files (default outputs/benchmark_files)")
    bench.add_argument("--from-table", action="store_true",
                       help="only redraw the figure from outputs/tables/storage_benchmark.csv")
    bench.set_defaults(func=cmd_benchmark)

    spark = sub.add_parser("spark", help="Task 4: pandas vs PySpark")
    spark.add_argument("--scales", type=int, nargs="+", default=[1, 2, 4, 8, 16], help="data multipliers")
    spark.add_argument("--repeats", type=int, default=6, help="runs per timing incl. 1 warm-up (default 6)")
    spark.set_defaults(func=cmd_spark)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (CricketDataError, FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0
