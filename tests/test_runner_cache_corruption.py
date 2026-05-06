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


# ============ C2: atomic cache writes ============


def test_atomic_write_parquet_no_partial_remains_after_success(
    tmp_path: Path,
) -> None:
    """**C2 invariant** (CACHE_BUILD.md): the cache write goes
    through `<path>.partial` then `os.rename`. After a successful
    write, no `.partial` file remains.

    Pre-fix: `merged.write_parquet(cache_path)` direct write —
    a crashed mid-write left a torn parquet at the consumer's
    path. Post-fix: tmp+rename is atomic on POSIX; consumers
    either see the pre-write state or the new state."""
    import polars as pl
    from corroborate.runner.runner import _atomic_write_parquet

    df = pl.DataFrame({'id': ['a', 'b', 'c'], 'x': [1, 2, 3]})
    out = tmp_path / 'cache.parquet'
    _atomic_write_parquet(df, out)

    assert out.exists()
    loaded = pl.read_parquet(out)
    assert loaded.height == 3
    # No `.partial` sibling left over.
    assert not out.with_suffix(out.suffix + '.partial').exists()


def test_atomic_write_parquet_simulated_crash_leaves_no_torn_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Simulate a crash between write and rename. The `.partial`
    file is left in place (recovery breadcrumb); `<out>` is NOT
    created. Consumers see the pre-write state — empty in this
    test."""
    import polars as pl
    from corroborate.runner.runner import _atomic_write_parquet

    df = pl.DataFrame({'id': ['a', 'b', 'c']})
    out = tmp_path / 'cache.parquet'

    real_replace = Path.replace

    def fake_replace(self: Path, target: Path) -> Path:
        raise RuntimeError('simulated crash mid-rename')

    monkeypatch.setattr(Path, 'replace', fake_replace)
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match='simulated crash'):
        _atomic_write_parquet(df, out)
    monkeypatch.setattr(Path, 'replace', real_replace)

    assert not out.exists(), (
        'consumer-facing cache path must NOT exist on crash; '
        'it would expose a torn parquet to readers'
    )
    assert out.with_suffix(out.suffix + '.partial').exists()


def test_atomic_write_text_writes_via_partial(tmp_path: Path) -> None:
    """The sidecar JSON path uses the same atomicity helper."""
    from corroborate.runner.runner import _atomic_write_text

    out = tmp_path / 'cache.hashes.json'
    _atomic_write_text(out, '{"foo": "bar"}')
    assert out.exists()
    assert out.read_text() == '{"foo": "bar"}'
    assert not out.with_suffix(out.suffix + '.partial').exists()


def test_cache_write_path_uses_atomic_helper_end_to_end(
    tmp_path: Path,
) -> None:
    """End-to-end: `_ingest_and_compute` writes the cache via the
    atomic helper, so a `.partial` file does NOT linger after a
    successful build. Pin the integration so a future refactor
    that replaces `_atomic_write_parquet` with a direct write
    breaches this test."""
    import polars as pl
    from corroborate.runner.runner import _ingest_and_compute

    cache_path = tmp_path / 'h.parquet'
    new_data = pl.DataFrame({
        'id': ['cell-0', 'cell-1'],
        'arm_key': ['baseline', 'baseline'],
        'env_name': ['Test', 'Test'],
    })
    merged = _ingest_and_compute(
        bridges=(),
        data=new_data,
        cache_path=cache_path,
        write_cache=True,
        restore_from_cloud=False,
    )
    assert merged.height == 2
    assert cache_path.exists()
    assert not cache_path.with_suffix(
        cache_path.suffix + '.partial'
    ).exists()


# ============ C4: orphan eviction ============


def test_invalidate_drifted_drops_orphan_measurable_columns(
    tmp_path: Path,
) -> None:
    """**C4 invariant** (CACHE_BUILD.md): a registered measurable
    that's NO LONGER in the required set is an orphan — pre-fix
    it persisted forever, growing the cache. Post-fix
    `_invalidate_drifted` drops it on the next build.

    Construction: register a measurable, populate a cache with
    its column AS IF it had been required at the last build,
    then call `_invalidate_drifted` with an EMPTY required set.
    The orphan column should be dropped."""
    from collections.abc import Mapping

    import polars as pl

    from corroborate.measurables import measurable
    from corroborate.runner.runner import _invalidate_drifted

    @measurable(reads=('x',))
    def soon_to_be_orphan(record: Mapping[str, object]) -> float:
        del record
        return 1.0
    _ = soon_to_be_orphan   # @measurable auto-registers

    # Cache has the orphan's column; required set is empty
    # (the bridges no longer ask for it).
    cache = pl.DataFrame({
        'id': ['cell-0', 'cell-1'],
        'arm_key': ['baseline', 'baseline'],
        'soon_to_be_orphan': [1.0, 1.0],   # the orphan column
    })
    invalidated = _invalidate_drifted(
        cache, manifest={}, required=[],
    )
    assert 'soon_to_be_orphan' not in invalidated.columns, (
        f'expected orphan column dropped; got {invalidated.columns}'
    )
    # Provenance / lineage tags PRESERVED — the eviction must not
    # touch them.
    assert 'id' in invalidated.columns
    assert 'arm_key' in invalidated.columns


def test_invalidate_drifted_keeps_unregistered_columns(
    tmp_path: Path,
) -> None:
    """**Negative control**: columns that aren't registered
    measurables (raw record fields, lineage tags) are NEVER
    orphans — they stay in the cache regardless of the required
    set."""
    import polars as pl
    from corroborate.runner.runner import _invalidate_drifted

    cache = pl.DataFrame({
        'id': ['cell-0'],
        'arm_key': ['baseline'],
        'env_name': ['CartPole-v1'],
        'raw_record_field': [42.0],   # NOT a registered measurable
    })
    invalidated = _invalidate_drifted(
        cache, manifest={}, required=[],
    )
    assert invalidated.columns == cache.columns, (
        f'unregistered columns should be preserved; '
        f'got {invalidated.columns} vs {cache.columns}'
    )
