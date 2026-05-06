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

import pytest

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
    # No `.partial`-flavored sibling left over (unique-suffix tmp
    # files are cleaned up by `replace`; failure path unlinks them).
    leftover = list(tmp_path.glob('*.partial'))
    assert leftover == [], f'unexpected .partial files: {leftover}'


def test_atomic_write_parquet_simulated_crash_preserves_existing_content(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**C2 invariant** (CACHE_BUILD.md): a killed-mid-write
    consumer sees the PRE-WRITE state. The previous test created
    `out` for the first time, so "doesn't exist after crash" was
    the expected state regardless of atomicity — it never
    actually tested that an existing file's contents are
    preserved.

    Substrate-grounded probe: pre-stage `out` with content X,
    monkey-patch `Path.replace` to fail, attempt to write
    content Y, assert content X is still readable. The atomicity
    contract is "old state preserved on failed write," not "no
    partial file lingers."
    """
    import polars as pl
    from corroborate.runner.runner import _atomic_write_parquet

    out = tmp_path / 'cache.parquet'

    # Pre-stage with content X (3 rows).
    content_x = pl.DataFrame({'id': ['a', 'b', 'c'], 'val': [1, 2, 3]})
    content_x.write_parquet(out)
    assert out.exists()

    real_replace = Path.replace

    def fake_replace(self: Path, target: Path) -> Path:
        raise RuntimeError('simulated crash mid-rename')

    monkeypatch.setattr(Path, 'replace', fake_replace)

    # Attempt to overwrite with content Y (5 rows). Replace fails.
    content_y = pl.DataFrame({
        'id': ['p', 'q', 'r', 's', 't'], 'val': [10, 20, 30, 40, 50],
    })
    with pytest.raises(RuntimeError, match='simulated crash'):
        _atomic_write_parquet(content_y, out)
    monkeypatch.setattr(Path, 'replace', real_replace)

    # **The actual atomicity assertion**: `out` still has X.
    after_crash = pl.read_parquet(out)
    assert after_crash.height == 3, (
        f'expected pre-crash content X (3 rows) preserved; '
        f'got {after_crash.height} rows — content Y leaked through'
    )
    assert after_crash['id'].to_list() == ['a', 'b', 'c'], (
        f'pre-crash IDs not preserved: got {after_crash["id"].to_list()}'
    )
    assert after_crash['val'].to_list() == [1, 2, 3], (
        f'pre-crash values not preserved: got {after_crash["val"].to_list()}'
    )


def test_atomic_write_parquet_failure_cleans_up_tmp_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**Concurrency invariant** (post-#11 roast fix): two
    concurrent writers against the same destination must NOT
    collide on a fixed `.partial` suffix. The unique-suffix
    `tempfile.mkstemp` design eliminates the TOCTOU race where
    writer B unlinks writer A's in-progress partial. The
    failure path must unlink its OWN tmp file so it can't grow
    into stale clutter that re-enters via a future build's
    `partial.exists()` short-circuit (the bug-shape pre-#11).

    Probe: monkeypatch `Path.replace` to fail, attempt a write,
    verify NO `.partial`-flavored file remains in the directory.
    """
    import polars as pl
    from corroborate.runner.runner import _atomic_write_parquet

    df = pl.DataFrame({'id': ['a'], 'x': [1]})
    out = tmp_path / 'cache.parquet'

    real_replace = Path.replace

    def fake_replace(self: Path, target: Path) -> Path:
        raise RuntimeError('simulated rename failure')

    monkeypatch.setattr(Path, 'replace', fake_replace)
    with pytest.raises(RuntimeError, match='simulated rename failure'):
        _atomic_write_parquet(df, out)
    monkeypatch.setattr(Path, 'replace', real_replace)

    leftover = list(tmp_path.glob('*.partial'))
    assert leftover == [], (
        f'failure path leaked tmp files: {leftover} — concurrent '
        f'retries would accumulate clutter'
    )


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


# ============ Per-corpus parallelism + disk budget ============


def test_estimate_max_workers_caps_by_largest_trace_size(
    tmp_path: Path, monkeypatch,
) -> None:
    """**Disk-budget bound** (Phase 0 #5): worker count divides
    `(available_disk / safety_factor) // largest_trace_bytes`.
    A 4 GB trace + 12 GB available + safety_factor=4 → 12/4 = 3 GB
    safe_avail / 4 GB per worker = 0 → returned 1 (the floor)."""
    import json
    from corroborate.runner.runner import _estimate_max_workers

    sub = tmp_path / 'corpus_a'
    sub.mkdir()
    manifest = {
        'remote_root': 'file:///fake',
        'files': [
            {
                'relpath': 'tmp/cell001__traces.parquet',
                'size_bytes': 4 * 1024**3,   # 4 GB
                'sha256': 'a' * 64,
                'pushed_at': '2026-05-06T00:00:00Z',
            },
        ],
    }
    (sub / '_remote.json').write_text(json.dumps(manifest))

    # Fake `shutil.disk_usage` to return 12 GB free.
    import shutil
    real_disk_usage = shutil.disk_usage

    class _FakeUsage:
        def __init__(self, free: int) -> None:
            self.total = free * 2
            self.used = free
            self.free = free

    def fake_disk_usage(path: object) -> _FakeUsage:
        return _FakeUsage(12 * 1024**3)

    monkeypatch.setattr(shutil, 'disk_usage', fake_disk_usage)
    monkeypatch.delenv('CORROBORATE_CACHE_WORKERS', raising=False)
    n = _estimate_max_workers([sub], tmp_path, safety_factor=4.0)
    monkeypatch.setattr(shutil, 'disk_usage', real_disk_usage)
    # 12 GB / 4 = 3 GB safe; 3 GB / 4 GB per worker = 0; floor 1.
    assert n == 1, (
        f'expected n=1 (largest trace exceeds safe budget); got {n}'
    )


def test_estimate_max_workers_caps_at_hard_limit(
    tmp_path: Path, monkeypatch,
) -> None:
    """**Hard cap** (Phase 0 #5): even with infinite disk and tiny
    traces, the estimator returns at most `hard_cap` (default 4)
    — beyond that, GIL + fork overhead degrade returns."""
    import json
    from corroborate.runner.runner import _estimate_max_workers

    sub = tmp_path / 'corpus_a'
    sub.mkdir()
    (sub / '_remote.json').write_text(json.dumps({
        'remote_root': 'file:///fake',
        'files': [{
            'relpath': 'traces.parquet',
            'size_bytes': 1024,   # 1 KB
            'sha256': 'b' * 64,
            'pushed_at': '2026-05-06T00:00:00Z',
        }],
    }))

    # Plenty of disk: 1 TB free.
    import shutil

    class _FakeUsage:
        def __init__(self, free: int) -> None:
            self.total = free * 2
            self.used = free
            self.free = free

    monkeypatch.setattr(
        shutil, 'disk_usage',
        lambda path: _FakeUsage(1024**4),
    )
    monkeypatch.delenv('CORROBORATE_CACHE_WORKERS', raising=False)
    n = _estimate_max_workers([sub], tmp_path)
    assert n == 4, (
        f'expected hard cap n=4; got {n} (1 TB free / tiny traces '
        f'would suggest unbounded workers without the cap)'
    )


def test_estimate_max_workers_respects_env_override(
    tmp_path: Path, monkeypatch,
) -> None:
    """`CORROBORATE_CACHE_WORKERS` env var bypasses the disk-budget
    calculation. Useful when the user knows their environment
    better than the heuristic."""
    from corroborate.runner.runner import _estimate_max_workers
    monkeypatch.setenv('CORROBORATE_CACHE_WORKERS', '7')
    sub = tmp_path / 'corpus_a'
    sub.mkdir()
    n = _estimate_max_workers([sub], tmp_path)
    assert n == 7


def test_estimate_max_workers_no_manifests_returns_hard_cap(
    tmp_path: Path, monkeypatch,
) -> None:
    """No manifests → no traces to budget → returns hard_cap.
    Cache build is purely CPU-bound; disk-budget bound doesn't
    apply."""
    from corroborate.runner.runner import _estimate_max_workers
    monkeypatch.delenv('CORROBORATE_CACHE_WORKERS', raising=False)
    sub = tmp_path / 'no_manifest'
    sub.mkdir()
    n = _estimate_max_workers([sub], tmp_path)
    assert n == 4


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


# ============ Phase 2.2: directory walk skips legacy manifest ============


def test_ingest_and_compute_directory_path_writes_cache_and_unlinks_legacy_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**Phase 2.2 invariant** (CACHE_BUILD.md): when `data` is a
    directory, the per-corpus `measurements.parquet` stores are
    authoritative; the per-hypothesis cache becomes a backward-
    compat snapshot. The legacy `<cache>.hashes.json` is
    unlinked on every directory-path write so a stale snapshot
    never misleads a reader still consulting it.
    """
    # Force sequential walk — `fork`-based ProcessPoolExecutor
    # under pytest can deadlock on inherited file descriptors.
    monkeypatch.setenv('CORROBORATE_CACHE_WORKERS', '1')

    import polars as pl
    from corroborate.runner.runner import (
        _ingest_and_compute,
        _manifest_path,
    )

    # Two corpora as subdirs of `data_dir`. Each cell carries a
    # distinct `seed` so `_dedup_by_content` doesn't collapse
    # content-equal rows.
    data_dir = tmp_path / 'corpora'
    data_dir.mkdir()
    for name, env in (('corpA', 'EnvA'), ('corpB', 'EnvB')):
        sub = data_dir / name
        sub.mkdir()
        df = pl.DataFrame({
            'id': [f'{name}-cell-0', f'{name}-cell-1'],
            'arm_key': ['baseline', 'baseline'],
            'env_name': [env, env],
            'seed': [0, 1],
        })
        df.write_parquet(sub / 'runs.parquet')

    cache_path = tmp_path / 'h.parquet'
    legacy_manifest = _manifest_path(cache_path)
    legacy_manifest.write_text('{"stale": "hash"}')
    assert legacy_manifest.exists()

    merged = _ingest_and_compute(
        bridges=(),
        data=data_dir,
        cache_path=cache_path,
        write_cache=True,
        restore_from_cloud=False,
    )

    assert merged.height == 4, (
        f'expected 4 cells (2 corpora × 2 cells); got {merged.height}'
    )
    assert cache_path.exists()
    assert not legacy_manifest.exists(), (
        f'legacy `{legacy_manifest.name}` should be unlinked '
        f'on directory-path writes (Phase 2.2)'
    )
    # Atomic-write invariant still holds.
    assert not cache_path.with_suffix(
        cache_path.suffix + '.partial'
    ).exists()


def test_ingest_and_compute_directory_path_unlinks_legacy_manifest_even_when_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """**#5 roast fix**: the legacy-manifest unlink was previously
    gated on `merged.height > 0`. An empty-corpora directory
    walk (every corpus skipped, every restore failed, the
    directory genuinely empty) would leave a stale
    `<cache>.hashes.json` next to a stale cache parquet — any
    reader consulting the manifest would see "cache + manifest
    both present, looks consistent" but be reading drift hashes
    that no longer correspond to the (now-absent) data.
    Post-fix: unlink fires on every directory-path write,
    regardless of merge size."""
    monkeypatch.setenv('CORROBORATE_CACHE_WORKERS', '1')

    from corroborate.runner.runner import (
        _ingest_and_compute,
        _manifest_path,
    )

    # An empty `data_dir` — no subdirs, so the walk loads zero corpora.
    data_dir = tmp_path / 'corpora_empty'
    data_dir.mkdir()

    cache_path = tmp_path / 'h.parquet'
    legacy_manifest = _manifest_path(cache_path)
    legacy_manifest.write_text('{"stale": "hash"}')
    assert legacy_manifest.exists()

    merged = _ingest_and_compute(
        bridges=(),
        data=data_dir,
        cache_path=cache_path,
        write_cache=True,
        restore_from_cloud=False,
    )

    assert merged.height == 0
    assert not legacy_manifest.exists(), (
        f'empty-corpora directory walk left stale legacy manifest; '
        f'pre-fix: unlink was gated on `merged.height > 0`. '
        f'Post-fix: unlink should be unconditional.'
    )
