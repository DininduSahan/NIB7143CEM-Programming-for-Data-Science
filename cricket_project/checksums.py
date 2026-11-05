"""Keeping the raw extract immutable (Obtain stage).

``write_manifest`` records a SHA-256 for every file in ``Data/Raw`` in the
standard ``sha256sum`` format, so it can also be checked with
``sha256sum -c`` on Linux/macOS. ``verify_manifest`` recomputes them and
raises :class:`~cricket_project.exceptions.RawDataIntegrityError` if any file
has changed, disappeared or been added.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from .exceptions import RawDataIntegrityError

_CHUNK = 1 << 20  # read 1 MiB at a time so large files never sit fully in memory


def sha256_of(path: Path | str) -> str:
    """Hex SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def compute_checksums(directory: Path | str) -> dict[str, str]:
    """``{file name: sha256}`` for every regular file directly inside ``directory``.

    Raises:
        ValueError: ``directory`` is not a directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")
    return {path.name: sha256_of(path) for path in sorted(directory.iterdir()) if path.is_file()}


def write_manifest(directory: Path | str, manifest: Path | str) -> int:
    """Write the checksum manifest and return how many files it covers."""
    checksums = compute_checksums(directory)
    lines = [f"{digest}  {name}\n" for name, digest in checksums.items()]
    Path(manifest).write_text("".join(lines), encoding="utf-8", newline="\n")
    return len(checksums)


def read_manifest(manifest: Path | str) -> dict[str, str]:
    """Parse a ``sha256sum``-style manifest.

    Raises:
        FileNotFoundError: the manifest does not exist.
        ValueError: a line is not ``<64 hex chars>  <name>``.
    """
    entries: dict[str, str] = {}
    for number, line in enumerate(Path(manifest).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        digest, sep, name = line.partition("  ")
        if not sep or len(digest) != 64 or not name:
            raise ValueError(f"{manifest}:{number}: malformed manifest line {line!r}")
        entries[name] = digest
    return entries


def verify_manifest(directory: Path | str, manifest: Path | str) -> int:
    """Check every raw file against the manifest and return how many were verified.

    Raises:
        RawDataIntegrityError: any file is modified, missing, or not in the manifest.
    """
    expected = read_manifest(manifest)
    actual = compute_checksums(directory)
    problems = [f"modified: {name}" for name in sorted(expected.keys() & actual.keys())
                if expected[name] != actual[name]]
    problems += [f"missing: {name}" for name in sorted(expected.keys() - actual.keys())]
    problems += [f"unexpected: {name}" for name in sorted(actual.keys() - expected.keys())]
    if problems:
        raise RawDataIntegrityError(problems)
    return len(actual)
