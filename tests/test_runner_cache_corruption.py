"""Regression test: `_load_cache` tolerates a corrupt parquet file.

The runner's `merged.write_parquet(cache_path)` is non-atomic; a
killed-mid-write cache leaves a truncated file on disk. Pre-fix,
the next `run()` would crash inside `pl.read_parquet` with a
polars ComputeError. Post-fix, the truncated file is detected via
`_file_present` (PAR1 magic check) and treated as missing —
the run rebuilds from scratch.
"""
from __future__ import annotations

import sys
from io import StringIO
from pathlib import Path

from corroborate.runner.runner import _load_cache


def _swap_stderr(buf: StringIO) -> StringIO:
    prev = sys.stderr
    sys.stderr = buf
    return prev


def test_load_cache_returns_empty_on_truncated_parquet(
    tmp_path: Path,
) -> None:
    """A 256-byte file with no PAR1 footer is the killed-mid-write
    failure mode. The runner should treat it as missing + warn,
    not crash."""
    bad = tmp_path / 'cache.parquet'
    # Random 256-byte content; no PAR1 footer (truncated).
    bad.write_bytes(b'X' * 256)

    buf = StringIO()
    prev = _swap_stderr(buf)
    try:
        df = _load_cache(bad)
    finally:
        sys.stderr = prev

    assert df.height == 0, (
        f'expected empty DataFrame on truncated parquet; '
        f'got {df.height} rows'
    )
    assert df.columns == [], (
        f'expected empty columns; got {df.columns}'
    )
    warning = buf.getvalue()
    assert 'WARNING' in warning, f'no warning emitted: {warning!r}'
    assert 'truncated' in warning or 'invalid' in warning


def test_load_cache_returns_empty_on_missing_path() -> None:
    """Pre-existing contract: nonexistent path → empty DataFrame.
    Pin against a regression that flipped the `not path.exists()`
    branch."""
    df = _load_cache(Path('/tmp/definitely_does_not_exist.parquet'))
    assert df.height == 0


def test_load_cache_returns_empty_on_none() -> None:
    """Pre-existing contract: None path → empty DataFrame."""
    df = _load_cache(None)
    assert df.height == 0
