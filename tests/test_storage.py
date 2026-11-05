"""Tests for storage formats, the benchmark timer, checksums and pseudonymisation."""

from __future__ import annotations

import pandas as pd
import pytest

from cricket_project.benchmark import FormatBenchmark, time_callable
from cricket_project.checksums import verify_manifest, write_manifest
from cricket_project.exceptions import RawDataIntegrityError, StorageFormatError
from cricket_project.privacy import pseudonymise
from cricket_project.storage import FORMATS, compact_dtypes, get_format


@pytest.fixture
def frame() -> pd.DataFrame:
    return pd.DataFrame({
        "match_id": ["a", "a", "b"],
        "venue": ["Galle", "Galle", "Kandy"],
        "runs_total": [1, 4, 6],
        "is_super_over": [False, False, True],
    })


@pytest.mark.parametrize("name", sorted(FORMATS))
@pytest.mark.parametrize("compression", [None, "gzip"])
def test_round_trip_every_format(tmp_path, frame, name, compression):
    fmt = get_format(name, compression)
    path = fmt.write(frame, fmt.path_in(tmp_path))

    back = fmt.read(path)
    assert back["runs_total"].tolist() == [1, 4, 6]
    assert fmt.read_column(path, "venue").tolist() == ["Galle", "Galle", "Kandy"]


def test_unknown_format_is_storage_format_error():
    with pytest.raises(StorageFormatError, match="unknown format 'excel'"):
        get_format("excel")


def test_unsupported_compression_is_value_error():
    with pytest.raises(ValueError, match="does not support compression 'zstd'"):
        get_format("csv", "zstd")


def test_read_missing_file_is_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_format("parquet").read(tmp_path / "missing.parquet")


def test_write_rejects_empty_frame(tmp_path):
    with pytest.raises(ValueError, match="non-empty DataFrame"):
        get_format("csv").write(pd.DataFrame(), tmp_path / "x.csv")


def test_compact_dtypes_uses_categories_and_small_ints(frame):
    compact = compact_dtypes(frame)
    assert compact["venue"].dtype == "category"
    assert compact["runs_total"].dtype == "int8"
    assert frame["venue"].dtype != "category"  # original untouched


def test_time_callable_discards_warmup():
    calls = []
    result = time_callable(lambda: calls.append(1), repeats=6, warmup=1)
    assert len(calls) == 6 and len(result.samples) == 5
    assert result.minimum <= result.median <= result.maximum


def test_time_callable_rejects_bad_repeats():
    with pytest.raises(ValueError, match="repeats > warmup"):
        time_callable(lambda: None, repeats=1, warmup=1)


def test_benchmark_reports_three_operations_per_format(tmp_path, frame):
    formats = [get_format("csv"), get_format("parquet", "snappy")]
    results = FormatBenchmark(frame, formats, tmp_path, column="runs_total", repeats=2).run(progress=lambda _: None)
    assert len(results) == 6
    assert set(results["operation"]) == {"write", "read", "read_column"}
    assert results.loc[results["format"] == "parquet+snappy", "dtypes_preserved"].all()


def test_checksum_manifest_detects_tampering(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "1.json").write_text("{}", encoding="utf-8")
    manifest = tmp_path / "sums.sha256"
    write_manifest(raw, manifest)
    assert verify_manifest(raw, manifest) == 1

    (raw / "1.json").write_text('{"edited": true}', encoding="utf-8")
    (raw / "2.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RawDataIntegrityError) as caught:
        verify_manifest(raw, manifest)
    assert caught.value.problems == ["modified: 1.json", "unexpected: 2.json"]


def test_pseudonymise_drops_names_and_refuses_partial_drop():
    names = pd.DataFrame({"batter": ["A"], "bowler": ["B"], "non_striker": ["C"], "player_out": [None],
                          "batter_id": ["id-a"]})
    assert list(pseudonymise(names).columns) == ["batter_id"]
    with pytest.raises(ValueError, match="not found"):
        pseudonymise(names.drop(columns="bowler"))
