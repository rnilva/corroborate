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


def _write_real_parquet(p: Path, n_rows: int = 1000) -> None:
    """Write a real parquet file (PAR1 footer + > 1 KiB) so the
    cloud archive's CI5 precondition check passes. Tests that
    care about archive transport mechanics rather than parquet
    content shape use these fixtures."""
    import polars as pl
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({
        'id': [f'cell-{i}' for i in range(n_rows)],
        'x': list(range(n_rows)),
    })
    df.write_parquet(p)


@pytest.fixture
def sweep_dir(tmp_path: Path) -> Path:
    d = tmp_path / 'sweep'
    _write_real_parquet(d / 'runs.parquet')
    _write_real_parquet(d / 'traces.parquet', n_rows=500)
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
    # Rewrite as a different valid parquet (preserves CI5
    # archive-precondition pass; only sha256 changes).
    p = sweep_dir / 'runs.parquet'
    _write_real_parquet(p, n_rows=1500)  # different content, different sha

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
    _write_real_parquet(p, n_rows=1500)  # different content, different sha

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
    tmp_path: Path, remote_root: str,
) -> None:
    """Negative control: for a parquet WITHOUT an `id` column,
    `row_ids` is the empty tuple — the I5 sniffer is robust to
    non-row parquets and doesn't fabricate IDs."""
    import polars as pl
    sweep = tmp_path / 'sweep'
    sweep.mkdir()
    # Real parquet (passes CI5) but no `id` column → sniffer
    # should return () via the ColumnNotFoundError /
    # ComputeError branch.
    df = pl.DataFrame({'x': list(range(1000)), 'y': list(range(1000))})
    df.write_parquet(sweep / 'runs.parquet')

    manifest = cloud.archive(sweep, remote_root)
    for f in manifest.files:
        assert f.row_ids == (), (
            f'no-id parquet {f.relpath!r} produced spurious '
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


# ============ I5 backfill ============


def test_backfill_row_ids_populates_legacy_manifest_entries(
    tmp_path: Path,
) -> None:
    """**Backfill helper for I5**: a manifest written before I5
    landed has empty `row_ids` on every entry. `backfill_row_ids`
    reads the `id` column from each remote parquet and rewrites
    the manifest in place. Idempotent — running twice is a no-op."""
    from corroborate.bridge.verdict import Verdict
    from corroborate.corpus.persistence import write_runrows
    from corroborate.corpus.schema import RunRow

    sweep_dir = tmp_path / 'sweep'
    sweep_dir.mkdir()
    rows = [
        RunRow(
            id=f'legacy-{i}',
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

    # Archive normally, then SIMULATE a legacy manifest by
    # rewriting the manifest with row_ids stripped — this is the
    # state of corpora archived before I5 landed.
    cloud.archive(sweep_dir, remote_root)
    import json
    manifest_path = sweep_dir / cloud.MANIFEST_NAME
    raw = json.loads(manifest_path.read_text())
    for f in raw['files']:
        f.pop('row_ids', None)
    manifest_path.write_text(json.dumps(raw, indent=2))

    # Confirm manifest is now legacy-style.
    legacy = cloud.ls(sweep_dir)
    assert all(f.row_ids == () for f in legacy.files)

    # Run backfill.
    n_updated = cloud.backfill_row_ids(sweep_dir)
    assert n_updated == 1, (
        f'expected 1 entry updated (runs.parquet); got {n_updated}'
    )

    # Verify row_ids populated.
    upgraded = cloud.ls(sweep_dir)
    runs_entry = next(
        f for f in upgraded.files if f.relpath == 'runs.parquet'
    )
    assert runs_entry.row_ids == ('legacy-0', 'legacy-1', 'legacy-2')

    # Idempotent: running again returns 0 (nothing left to backfill).
    assert cloud.backfill_row_ids(sweep_dir) == 0


def test_backfill_row_ids_no_op_when_manifest_absent(
    tmp_path: Path,
) -> None:
    """Returns 0 cleanly when there's no manifest to backfill."""
    sweep_dir = tmp_path / 'empty'
    sweep_dir.mkdir()
    assert cloud.backfill_row_ids(sweep_dir) == 0


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


# ============ cloud-fallback purge (post-merge-cleanup orphans) ============


def _sweep_with_subcorpus_archived(
    sweep_dir: Path, remote_prefix: str, sub_name: str,
) -> str:
    """Build a sweep that mirrors the post-merge-cleanup orphan
    shape: cloud sub-archive intact, but the LOCAL sub-corpus dir
    has been wiped, and the local top-level parquets are what
    survived after the merge.

    For test simplicity: top-level parquets have the SAME content
    as sub-corpus parquets (the single-intervention merge case
    where merge copies are identity). Returns sub_remote_root used
    for the archive."""
    sub = sweep_dir / sub_name
    _write_real_parquet(sub / 'runs.parquet', n_rows=200)
    _write_real_parquet(sub / 'traces.parquet', n_rows=200)
    sub_remote_root = f'{remote_prefix.rstrip("/")}/{sweep_dir.name}/{sub_name}'
    _ = cloud.archive(sub, sub_remote_root)
    # Copy top-level parquets — same content as sub, simulating
    # the single-arm merge identity case.
    (sweep_dir / 'runs.parquet').write_bytes(
        (sub / 'runs.parquet').read_bytes(),
    )
    (sweep_dir / 'traces.parquet').write_bytes(
        (sub / 'traces.parquet').read_bytes(),
    )
    # Now wipe the local sub-dir to mirror the post-merge cleanup.
    import shutil
    shutil.rmtree(sub)
    return sub_remote_root


def test_purge_without_local_manifest_fails_without_fallback(
    tmp_path: Path,
) -> None:
    """Default behaviour preserved: no local manifest → refuse
    purge. The cloud-fallback path requires an explicit
    `cloud_fallback_prefix` argument."""
    sweep_dir = tmp_path / 'orphan_sweep'
    sweep_dir.mkdir()
    remote_prefix = f'file://{tmp_path / "remote"}'
    _ = _sweep_with_subcorpus_archived(sweep_dir, remote_prefix, 'canonical')

    # Local top-level parquets exist but no _remote.json.
    assert (sweep_dir / 'runs.parquet').exists()
    assert not (sweep_dir / cloud.MANIFEST_NAME).exists()

    with pytest.raises(FileNotFoundError, match='no manifest'):
        _ = cloud.purge(sweep_dir)


def test_purge_with_cloud_fallback_succeeds_for_orphan(
    tmp_path: Path,
) -> None:
    """The post-merge-cleanup orphan: local top-level parquets
    exist but their _remote.json was wiped along with the sub-
    corpus dir that originally held it. Cloud sub-archives are
    intact. With cloud_fallback_prefix, purge looks up the sub-
    archives, verifies row_id coverage, and permits deletion."""
    sweep_dir = tmp_path / 'orphan_sweep'
    sweep_dir.mkdir()
    remote_prefix = f'file://{tmp_path / "remote"}'
    _ = _sweep_with_subcorpus_archived(sweep_dir, remote_prefix, 'canonical')

    deleted = cloud.purge(
        sweep_dir, cloud_fallback_prefix=remote_prefix,
    )
    assert set(deleted) == {'runs.parquet', 'traces.parquet'}
    assert not (sweep_dir / 'runs.parquet').exists()
    assert not (sweep_dir / 'traces.parquet').exists()


def test_purge_cloud_fallback_refuses_uncovered_row_ids(
    tmp_path: Path,
) -> None:
    """Safety check: if the local top-level parquet contains
    row_ids NOT covered by any cloud sub-archive, purge refuses
    rather than risk silent data loss. Simulated by replacing
    the top-level runs.parquet with a parquet whose row_ids
    differ from the archived sub-corpus."""
    sweep_dir = tmp_path / 'orphan_sweep'
    sweep_dir.mkdir()
    remote_prefix = f'file://{tmp_path / "remote"}'
    _ = _sweep_with_subcorpus_archived(sweep_dir, remote_prefix, 'canonical')
    # Replace top-level runs.parquet with row_ids that don't
    # appear in the archived sub-corpus (cell-200 .. cell-399).
    import polars as pl
    pl.DataFrame({
        'id': [f'cell-{i}' for i in range(200, 400)],
        'x': list(range(200)),
    }).write_parquet(sweep_dir / 'runs.parquet')

    with pytest.raises(ValueError, match='not covered by sub-archives'):
        _ = cloud.purge(
            sweep_dir, cloud_fallback_prefix=remote_prefix,
            files=['runs.parquet'],
        )
    # Local file preserved (refused deletion).
    assert (sweep_dir / 'runs.parquet').exists()


def test_archive_picks_up_nested_sidecar_tree(
    sweep_dir: Path, remote_root: str,
) -> None:
    """SIDECAR_DIRS walks recurse: nested layouts like
    `q_checkpoints/<arm_name>/cell*.msgpack` (produced by multi-arm
    sweeps that namespace ckpts per intervention via the
    yaml_sweep.py post-merge lift) MUST be picked up by the default
    file selection, not just direct children of the sidecar dir.

    Regression for the 2026-05-26 substrate-coverage gap: the
    nested layout used to be silently skipped by `is_file()` on
    iterdir entries that landed on arm-named subdirs."""
    msgpack_a = sweep_dir / 'q_checkpoints' / 'arm_a' / 'cell000_0_burst00.msgpack'
    msgpack_b = sweep_dir / 'q_checkpoints' / 'arm_a' / 'cell000_0_burst01.msgpack'
    msgpack_c = sweep_dir / 'q_checkpoints' / 'arm_b' / 'cell001_0_final.msgpack'
    for p in (msgpack_a, msgpack_b, msgpack_c):
        p.parent.mkdir(parents=True, exist_ok=True)
        _ = p.write_bytes(b'msgpack-payload-' + p.name.encode())

    # validate=False — msgpacks are below the CI5 1 KiB floor and
    # have no PAR1 footer.
    manifest = cloud.archive(sweep_dir, remote_root, validate=False)
    relpaths = {f.relpath for f in manifest.files}
    assert 'runs.parquet' in relpaths
    assert 'traces.parquet' in relpaths
    assert 'q_checkpoints/arm_a/cell000_0_burst00.msgpack' in relpaths
    assert 'q_checkpoints/arm_a/cell000_0_burst01.msgpack' in relpaths
    assert 'q_checkpoints/arm_b/cell001_0_final.msgpack' in relpaths
    assert len(relpaths) == 5


def test_purge_deletes_both_parquets_and_nested_sidecars(
    sweep_dir: Path, remote_root: str,
) -> None:
    """End-to-end downstream of the recursing-sidecar archive:
    `corroborate purge <sweep_dir>` after archive deletes BOTH
    the parquets AND the nested-sidecar msgpacks via the unified
    manifest. This is what makes the substrate's
    `keep_q_checkpoint_*` opt-in self-cleaning at sweep end."""
    msgpack = sweep_dir / 'q_checkpoints' / 'arm_a' / 'cell000_0_burst00.msgpack'
    msgpack.parent.mkdir(parents=True, exist_ok=True)
    _ = msgpack.write_bytes(b'ckpt-bytes')

    _ = cloud.archive(sweep_dir, remote_root, validate=False)
    assert msgpack.is_file()
    deleted = cloud.purge(sweep_dir)
    # Both parquets + the nested msgpack land in the deletion set.
    assert set(deleted) == {
        'runs.parquet', 'traces.parquet',
        'q_checkpoints/arm_a/cell000_0_burst00.msgpack',
    }
    assert not (sweep_dir / 'runs.parquet').exists()
    assert not msgpack.exists()
    # Manifest preserved (for restore).
    assert (sweep_dir / cloud.MANIFEST_NAME).exists()


def test_purge_cloud_fallback_uses_direct_manifest_at_sweep_path(
    tmp_path: Path,
) -> None:
    """A sub-corpus dir has its own direct MANIFEST.json on the cloud
    (no nested children). Cloud-fallback should find that manifest
    directly via `fetch_remote_manifest` rather than walking children.

    Use case: purging an inner sub-corpus dir like
    `<root>/parent/<sub>/` where the cloud has archived JUST
    `<remote_prefix>/parent/<sub>/MANIFEST.json`. The user passes
    `--remote-prefix <prefix>/parent/`, and purge resolves to the
    sub's own manifest.
    """
    sweep_dir = tmp_path / 'sub_corpus'
    _write_real_parquet(sweep_dir / 'runs.parquet', n_rows=200)
    _write_real_parquet(sweep_dir / 'traces.parquet', n_rows=200)
    # Archive directly to a cloud path (no nested children).
    direct_remote = f'file://{tmp_path / "remote_parent" / "sub_corpus"}'
    _ = cloud.archive(sweep_dir, direct_remote)
    # Simulate the orphan case: wipe local manifest, keep parquets.
    (sweep_dir / cloud.MANIFEST_NAME).unlink()
    parent_prefix = f'file://{tmp_path / "remote_parent"}'

    deleted = cloud.purge(
        sweep_dir, cloud_fallback_prefix=parent_prefix,
    )
    assert set(deleted) == {'runs.parquet', 'traces.parquet'}


def test_purge_cloud_fallback_raises_when_no_subarchives_found(
    tmp_path: Path,
) -> None:
    """No sub-archives at the cloud prefix → purge refuses. The
    error message points at the prefix so the user can diagnose
    a misspelled remote or wrong cloud bucket."""
    sweep_dir = tmp_path / 'orphan_sweep'
    sweep_dir.mkdir()
    _write_real_parquet(sweep_dir / 'runs.parquet', n_rows=200)
    empty_remote = f'file://{tmp_path / "empty_remote"}'

    with pytest.raises(
        FileNotFoundError, match='neither a direct MANIFEST.json',
    ):
        _ = cloud.purge(
            sweep_dir, cloud_fallback_prefix=empty_remote,
        )


# ============ restore_columns: column-projected restore ============


def _write_multicol_parquet(p: Path, n_rows: int = 500) -> None:
    """Multi-column parquet so column projection has columns to drop.
    Fat columns are deliberately big (300-element lists × random
    payload) so parquet metadata doesn't dominate the size measurement."""
    import polars as pl
    import random
    random.seed(0)
    p.parent.mkdir(parents=True, exist_ok=True)
    fat = lambda: [random.random() for _ in range(300)]
    df = pl.DataFrame({
        'id': [f'cell-{i}' for i in range(n_rows)],
        'wanted_col': [random.random() for _ in range(n_rows)],
        'fat_col_a': [fat() for _ in range(n_rows)],
        'fat_col_b': [fat() for _ in range(n_rows)],
        'fat_col_c': [fat() for _ in range(n_rows)],
    })
    df.write_parquet(p)


def test_restore_columns_writes_thin_local(tmp_path: Path) -> None:
    """`restore_columns` materializes only the requested columns,
    producing a much smaller local file than the cloud original."""
    sweep = tmp_path / 'sweep'
    _write_multicol_parquet(sweep / 'traces.parquet')
    full_size = (sweep / 'traces.parquet').stat().st_size
    remote = f'file://{tmp_path / "remote"}'
    _ = cloud.archive(sweep, remote)

    # Purge local to force a fetch
    (sweep / 'traces.parquet').unlink()

    restored = cloud.restore_columns(
        sweep, file_columns={
            'traces.parquet': ['id', 'wanted_col'],
        },
    )
    assert restored == ['traces.parquet']

    thin = sweep / 'traces.parquet'
    assert thin.exists()
    thin_size = thin.stat().st_size
    # Projection should drop the three fat list columns (3× ~400KB
    # each); thin file should be a small fraction of the original.
    assert thin_size < full_size * 0.5, (
        f'thin {thin_size} not meaningfully smaller than full {full_size}'
    )

    import polars as pl
    df = pl.read_parquet(thin)
    assert set(df.columns) == {'id', 'wanted_col'}
    assert df.height == 500


def test_restore_columns_raises_on_unknown_relpath(tmp_path: Path) -> None:
    sweep = tmp_path / 'sweep'
    _write_real_parquet(sweep / 'runs.parquet')
    remote = f'file://{tmp_path / "remote"}'
    _ = cloud.archive(sweep, remote)
    with pytest.raises(KeyError, match='manifest does not contain'):
        _ = cloud.restore_columns(
            sweep, file_columns={'not_in_manifest.parquet': ['x']},
        )


def test_restore_columns_overwrite_false_skips_existing(
    tmp_path: Path,
) -> None:
    """When overwrite=False, an existing local file is left alone."""
    sweep = tmp_path / 'sweep'
    _write_multicol_parquet(sweep / 'traces.parquet')
    remote = f'file://{tmp_path / "remote"}'
    _ = cloud.archive(sweep, remote)
    # Replace local with stub
    sentinel = b'pre-existing-do-not-overwrite'
    _ = (sweep / 'traces.parquet').write_bytes(sentinel)
    restored = cloud.restore_columns(
        sweep, file_columns={'traces.parquet': ['id']},
        overwrite=False,
    )
    assert restored == []  # nothing rewritten
    assert (sweep / 'traces.parquet').read_bytes() == sentinel


def test_archive_empty_sweep_raises(tmp_path: Path) -> None:
    empty = tmp_path / 'empty'
    empty.mkdir()
    with pytest.raises(ValueError, match='no files to archive'):
        _ = cloud.archive(empty, f'file://{tmp_path / "remote"}')


def test_archive_default_excludes_tmp_subdir(
    tmp_path: Path,
) -> None:
    sweep = tmp_path / 'sweep'
    _write_real_parquet(sweep / 'traces.parquet')
    _write_real_parquet(sweep / 'tmp' / 'arm000.parquet')
    _write_real_parquet(sweep / 'tmp' / 'arm001.parquet')

    manifest = cloud.archive(sweep, f'file://{tmp_path / "remote"}')
    relpaths = {f.relpath for f in manifest.files}
    assert relpaths == {'traces.parquet'}


# ============ CI1 guard at archive boundary ============

def test_archive_refuses_hybrid_layout_with_nested_corpora(
    tmp_path: Path,
) -> None:
    """`cloud.archive` must refuse a parent corpus whose dir
    already contains a nested sub-corpus (each with its own
    `runs.parquet`). Mirrors the runner's CI1 ingest check;
    closes the asymmetry that previously let archive accept a
    layout the runner would reject. The raise must fire BEFORE
    any cloud I/O happens."""
    from corroborate.corpus.integrity import NestedCorpusError

    parent = tmp_path / 'parent'
    _write_real_parquet(parent / 'runs.parquet')
    _write_real_parquet(parent / 'child' / 'runs.parquet')

    remote_dir = tmp_path / 'remote'
    remote_uri = f'file://{remote_dir}'

    with pytest.raises(NestedCorpusError):
        _ = cloud.archive(parent, remote_uri)

    # No cloud writes happened: the remote dir is either absent
    # or empty (no MANIFEST.json, no parquets).
    if remote_dir.exists():
        assert list(remote_dir.iterdir()) == []
    # Parent dir still has no local manifest.
    assert not (parent / cloud.MANIFEST_NAME).exists()


# ============ Manifest round-trip ============

def test_manifest_dataclass_round_trip(
    sweep_dir: Path, remote_root: str,
) -> None:
    written = cloud.archive(sweep_dir, remote_root)
    via_dict = cloud.RemoteManifest.from_dict(dict(written.as_dict()))
    assert via_dict == written
