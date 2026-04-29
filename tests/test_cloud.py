"""Tests for cloud archive — archive/restore/ls/purge round-trip.

Uses fsspec's `file://` backend with a per-test tmp_path so each
test gets an isolated remote root. `memory://` would also work
but `file://` is closer to the production S3/GCS path semantics
and makes failures easier to diagnose by inspecting the tmp
directory."""
from __future__ import annotations

from pathlib import Path

import pytest

from corroborate import cloud


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
