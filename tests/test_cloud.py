"""Tests for cloud archive — archive/restore/ls/purge round-trip.

Uses fsspec's `file://` backend with a per-test tmp_path so each
test gets an isolated remote root. `memory://` would also work
but `file://` is closer to the production S3/GCS path semantics
and makes failures easier to diagnose by inspecting the tmp
directory."""
from __future__ import annotations

from pathlib import Path

import pytest

from corroborate.corpus import cloud


# ============ Fixtures ============

def _write(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    _ = p.write_bytes(data)


@pytest.fixture
def sweep_dir(tmp_path: Path) -> Path:
    d = tmp_path / 'sweep'
    _write(d / 'runs.parquet', b'runs-payload')
    _write(d / 'traces.parquet', b'traces-payload-with-more-bytes')
    return d


@pytest.fixture
def remote_root(tmp_path: Path) -> str:
    return f'file://{tmp_path / "remote"}'


# ============ Round-trip ============

def test_archive_then_ls_returns_manifest(
    sweep_dir: Path, remote_root: str,
) -> None:
    manifest = cloud.archive(sweep_dir, remote_root)
    assert manifest.remote_root == remote_root
    assert {f.relpath for f in manifest.files} == {
        'runs.parquet', 'traces.parquet',
    }
    via_ls = cloud.ls(sweep_dir)
    assert via_ls == manifest


def test_archive_persists_manifest_atomically(
    sweep_dir: Path, remote_root: str,
) -> None:
    _ = cloud.archive(sweep_dir, remote_root)
    manifest_path = sweep_dir / cloud.MANIFEST_NAME
    assert manifest_path.exists()
    # No tmp leftover: atomic rename completed
    assert not manifest_path.with_suffix('.json.tmp').exists()


def test_archive_purge_local_removes_files(
    sweep_dir: Path, remote_root: str,
) -> None:
    _ = cloud.archive(sweep_dir, remote_root, purge_local=True)
    assert not (sweep_dir / 'runs.parquet').exists()
    assert not (sweep_dir / 'traces.parquet').exists()
    # Manifest preserved
    assert (sweep_dir / cloud.MANIFEST_NAME).exists()


def test_archive_then_purge_then_restore_round_trip(
    sweep_dir: Path, remote_root: str,
) -> None:
    original_runs = (sweep_dir / 'runs.parquet').read_bytes()
    original_traces = (sweep_dir / 'traces.parquet').read_bytes()

    _ = cloud.archive(sweep_dir, remote_root)
    deleted = cloud.purge(sweep_dir)
    assert set(deleted) == {'runs.parquet', 'traces.parquet'}
    assert not (sweep_dir / 'runs.parquet').exists()
    assert not (sweep_dir / 'traces.parquet').exists()

    restored = cloud.restore(sweep_dir)
    assert set(restored) == {'runs.parquet', 'traces.parquet'}
    assert (sweep_dir / 'runs.parquet').read_bytes() == original_runs
    assert (sweep_dir / 'traces.parquet').read_bytes() == original_traces


# ============ Idempotency ============

def test_archive_idempotent_same_content(
    sweep_dir: Path, remote_root: str,
) -> None:
    first = cloud.archive(sweep_dir, remote_root)
    pushed_at_before = {f.relpath: f.pushed_at for f in first.files}
    second = cloud.archive(sweep_dir, remote_root)
    pushed_at_after = {f.relpath: f.pushed_at for f in second.files}
    # Skipped → pushed_at unchanged for both files
    assert pushed_at_before == pushed_at_after


# ============ I2: ConflictingArchive on sha256 mismatch ============


def test_archive_raises_conflicting_archive_on_sha256_mismatch(
    sweep_dir: Path, remote_root: str,
) -> None:
    """**Invariant I2** (SWEEP_PERSISTENCY.md): when the local
    file's sha256 differs from the manifest's prior entry for the
    same relpath, archive() must raise ConflictingArchive instead
    of silently overwriting. The user opts into overwrite via
    `force=True`.

    Pre-fix: silent last-writer-wins. Post-fix: explicit error
    naming the relpath, both sha256s, and the remote URI.
    """
    # First archive populates the manifest.
    cloud.archive(sweep_dir, remote_root)

    # Modify the local file → different sha256 vs manifest.
    p = sweep_dir / 'runs.parquet'
    original = p.read_bytes()
    p.write_bytes(original + b'\x00')   # one-byte mutation

    import pytest
    with pytest.raises(cloud.ConflictingArchive) as exc_info:
        cloud.archive(sweep_dir, remote_root)

    err = exc_info.value
    assert err.relpath == 'runs.parquet'
    assert err.local_sha256 != err.prior_sha256
    assert remote_root in err.remote_uri


def test_archive_force_true_overwrites_on_sha256_mismatch(
    sweep_dir: Path, remote_root: str,
) -> None:
    """`force=True` is the explicit overwrite opt-in. Manifest's
    prior entry is replaced with the new sha256 + pushed_at."""
    first = cloud.archive(sweep_dir, remote_root)
    prior_sha = {f.relpath: f.sha256 for f in first.files}['runs.parquet']

    p = sweep_dir / 'runs.parquet'
    p.write_bytes(p.read_bytes() + b'\x00')

    second = cloud.archive(sweep_dir, remote_root, force=True)
    new_sha = {f.relpath: f.sha256 for f in second.files}['runs.parquet']

    assert new_sha != prior_sha, (
        'force=True should have replaced the manifest entry '
        'with the new content sha256.'
    )


# ============ I5: row_ids provenance breadcrumb ============


def test_archive_records_row_ids_for_runrow_parquet(
    tmp_path: Path,
) -> None:
    """**Invariant I5** (SWEEP_PERSISTENCY.md): when a parquet
    archived via `archive()` carries an `id` column (RunRow or
    TraceRow shards), the manifest entry records the per-shard
    list of IDs. Enables `id → shard → cell address` traceability
    when investigating anomalous rows in a merged corpus."""
    from corroborate.bridge.verdict import Verdict
    from corroborate.corpus.persistence import write_runrows
    from corroborate.corpus.schema import RunRow

    sweep_dir = tmp_path / 'sweep'
    sweep_dir.mkdir()
    rows = [
        RunRow(
            id=f'run-{i}',
            parent_id=None, cycle_id=None,
            timestamp='2026-05-06T00:00:00Z',
            verdict=Verdict.HELD,
            arm_key='baseline',
            measurements={},
        )
        for i in range(3)
    ]
    write_runrows(rows, sweep_dir / 'runs.parquet')
    remote_root = f'file://{tmp_path / "remote"}'

    manifest = cloud.archive(sweep_dir, remote_root)
    by_relpath = {f.relpath: f for f in manifest.files}
    assert 'runs.parquet' in by_relpath
    entry = by_relpath['runs.parquet']
    assert entry.row_ids == ('run-0', 'run-1', 'run-2'), (
        f'expected row_ids to record RunRow.id list; got '
        f'{entry.row_ids!r}'
    )


def test_archive_omits_row_ids_for_non_runrow_parquet(
    sweep_dir: Path, remote_root: str,
) -> None:
    """Negative control: for a parquet WITHOUT an `id` column,
    `row_ids` is the empty tuple — the I5 sniffer is robust to
    non-row parquets and doesn't fabricate IDs."""
    # The fixture's `runs.parquet` is fake bytes (not a real
    # parquet); the sniffer should return () via the
    # ColumnNotFoundError / ComputeError branch.
    manifest = cloud.archive(sweep_dir, remote_root)
    for f in manifest.files:
        assert f.row_ids == (), (
            f'fake-bytes parquet {f.relpath!r} produced spurious '
            f'row_ids = {f.row_ids!r}'
        )


def test_remotefile_round_trip_with_row_ids() -> None:
    """RemoteFile.from_dict / as_dict round-trip preserves
    row_ids when present."""
    f = cloud.RemoteFile(
        relpath='tmp/cell001__runs.parquet',
        size_bytes=100, sha256='deadbeef',
        pushed_at='2026-05-06T00:00:00Z',
        row_ids=('a', 'b', 'c'),
    )
    d = f.as_dict()
    assert d['row_ids'] == ['a', 'b', 'c']
    f2 = cloud.RemoteFile.from_dict(d)
    assert f2 == f


def test_remotefile_round_trip_legacy_manifest_without_row_ids() -> None:
    """Backward-compat: manifests written before I5 landed don't
    have a `row_ids` field. `from_dict` defaults to empty tuple."""
    legacy_dict = {
        'relpath': 'runs.parquet',
        'size_bytes': 100,
        'sha256': 'deadbeef',
        'pushed_at': '2026-05-06T00:00:00Z',
        # no row_ids
    }
    f = cloud.RemoteFile.from_dict(legacy_dict)
    assert f.row_ids == ()
    # Re-serializing omits the empty row_ids — manifests stay
    # byte-identical post-rewrite.
    assert 'row_ids' not in f.as_dict()


def test_restore_skips_files_already_present_with_matching_sha(
    sweep_dir: Path, remote_root: str,
) -> None:
    _ = cloud.archive(sweep_dir, remote_root)
    # restore without purge: both files already present, expect skip
    restored = cloud.restore(sweep_dir)
    assert restored == []


# ============ Selection ============

def test_archive_with_explicit_files_subset(
    sweep_dir: Path, remote_root: str,
) -> None:
    manifest = cloud.archive(
        sweep_dir, remote_root, files=['traces.parquet'],
    )
    assert [f.relpath for f in manifest.files] == ['traces.parquet']


def test_restore_with_explicit_files_subset(
    sweep_dir: Path, remote_root: str,
) -> None:
    _ = cloud.archive(sweep_dir, remote_root)
    _ = cloud.purge(sweep_dir)
    restored = cloud.restore(sweep_dir, files=['runs.parquet'])
    assert restored == ['runs.parquet']
    assert (sweep_dir / 'runs.parquet').exists()
    assert not (sweep_dir / 'traces.parquet').exists()


# ============ Error paths ============

def test_archive_rejects_retargeting_existing_manifest(
    sweep_dir: Path, remote_root: str, tmp_path: Path,
) -> None:
    _ = cloud.archive(sweep_dir, remote_root)
    other = f'file://{tmp_path / "other"}'
    with pytest.raises(ValueError, match='already pinned'):
        _ = cloud.archive(sweep_dir, other)


def test_restore_detects_sha256_mismatch_and_removes_partial(
    sweep_dir: Path, remote_root: str,
) -> None:
    _ = cloud.archive(sweep_dir, remote_root)
    _ = cloud.purge(sweep_dir)
    # Corrupt the remote file by overwriting it directly
    remote_path = Path(remote_root.removeprefix('file://')) / 'traces.parquet'
    _ = remote_path.write_bytes(b'corrupted!')
    with pytest.raises(RuntimeError, match='sha256 mismatch'):
        _ = cloud.restore(sweep_dir, files=['traces.parquet'])
    # Partial download was removed
    assert not (sweep_dir / 'traces.parquet').exists()


def test_restore_refuses_to_overwrite_drifted_local_file(
    sweep_dir: Path, remote_root: str,
) -> None:
    _ = cloud.archive(sweep_dir, remote_root)
    _ = (sweep_dir / 'runs.parquet').write_bytes(b'drifted-local')
    with pytest.raises(RuntimeError, match='manifest'):
        _ = cloud.restore(sweep_dir, files=['runs.parquet'])


def test_restore_overwrite_replaces_drifted_local(
    sweep_dir: Path, remote_root: str,
) -> None:
    original = (sweep_dir / 'runs.parquet').read_bytes()
    _ = cloud.archive(sweep_dir, remote_root)
    _ = (sweep_dir / 'runs.parquet').write_bytes(b'drifted-local')
    restored = cloud.restore(
        sweep_dir, files=['runs.parquet'], overwrite=True,
    )
    assert restored == ['runs.parquet']
    assert (sweep_dir / 'runs.parquet').read_bytes() == original


def test_ls_without_archive_raises(sweep_dir: Path) -> None:
    with pytest.raises(FileNotFoundError):
        _ = cloud.ls(sweep_dir)


def test_archive_empty_sweep_raises(tmp_path: Path) -> None:
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(ValueError, match='no files to archive'):
        _ = cloud.archive(empty, f'file://{tmp_path / "remote"}')


def test_archive_default_excludes_tmp_subdir(
    tmp_path: Path,
) -> None:
    sweep = tmp_path / 'sweep'
    _write(sweep / 'traces.parquet', b'merged')
    _write(sweep / 'tmp' / 'arm000.parquet', b'shard0')
    _write(sweep / 'tmp' / 'arm001.parquet', b'shard1')

    manifest = cloud.archive(sweep, f'file://{tmp_path / "remote"}')
    relpaths = {f.relpath for f in manifest.files}
    assert relpaths == {'traces.parquet'}


# ============ Manifest round-trip ============

def test_manifest_dataclass_round_trip(
    sweep_dir: Path, remote_root: str,
) -> None:
    written = cloud.archive(sweep_dir, remote_root)
    via_dict = cloud.RemoteManifest.from_dict(dict(written.as_dict()))
    assert via_dict == written
