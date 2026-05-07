"""Tests for `corpus.integrity` — defensive primitives at the
corpus boundary (CORPUS_INTEGRITY.md).

Each invariant CI1-CI7 has a dedicated test. Currently:
- CI1 nested-corpus detection.

Pending (Phase 1 continuation):
- CI3 cloud-root collision.
- CI5 archive precondition.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from corroborate.corpus.cloud import RemoteFile, RemoteManifest
from corroborate.corpus.integrity import (
    ArchivePrecondition,
    NestedCorpusError,
    RemoteRootCollision,
    assert_archive_eligible,
    assert_no_nested_corpora,
    assert_unique_remote_root,
)


# ============ CI1 — corpora are leaves ============


def _make_corpus(d: Path) -> None:
    """Create a minimal valid corpus dir at `d`."""
    d.mkdir(parents=True, exist_ok=True)
    df = pl.DataFrame({'id': ['a', 'b'], 'env_name': ['Env', 'Env']})
    df.write_parquet(d / 'runs.parquet')


def test_assert_no_nested_corpora_clean_layout(tmp_path: Path) -> None:
    """Top-level corpora only: no violation, no exception."""
    root = tmp_path / 'data'
    root.mkdir()
    _make_corpus(root / 'corpA')
    _make_corpus(root / 'corpB')

    # Should NOT raise.
    assert_no_nested_corpora(root)


def test_assert_no_nested_corpora_pure_nested_raises(
    tmp_path: Path,
) -> None:
    """**CI1 violation, pure-nested shape**: parent dir has no
    `runs.parquet` of its own but contains sub-dirs that do.
    The runner would treat the parent as a "corpus" with no
    runs.parquet (skip), silently dropping all the nested
    corpora it contains."""
    root = tmp_path / 'data'
    root.mkdir()
    parent = root / 'parent_no_own_data'
    parent.mkdir()
    _make_corpus(parent / 'sub_a')
    _make_corpus(parent / 'sub_b')

    with pytest.raises(NestedCorpusError) as exc_info:
        assert_no_nested_corpora(root)

    msg = str(exc_info.value)
    assert 'CI1' in msg
    assert 'parent_no_own_data' in msg
    assert 'sub_a' in msg
    assert 'sub_b' in msg
    assert len(exc_info.value.violations) == 2


def test_assert_no_nested_corpora_hybrid_raises(tmp_path: Path) -> None:
    """**CI1 violation, hybrid shape**: parent has its own
    `runs.parquet` AND sub-directories with theirs. The runner
    ingests the parent's, silently drops the inner ones.
    Mirrors the in-the-wild `minatar_sync_curve_resume/` shape
    (top-level + nested ddqn_sync3k subdir)."""
    root = tmp_path / 'data'
    root.mkdir()
    parent = root / 'parent_with_own_data'
    _make_corpus(parent)            # parent's own runs.parquet
    _make_corpus(parent / 'inner')  # nested sub-corpus

    with pytest.raises(NestedCorpusError) as exc_info:
        assert_no_nested_corpora(root)

    msg = str(exc_info.value)
    assert 'parent_with_own_data' in msg
    assert 'inner' in msg
    assert len(exc_info.value.violations) == 1


def test_assert_no_nested_corpora_lists_all_violators(
    tmp_path: Path,
) -> None:
    """One scan should surface all violators so the user can fix
    them in a single pass — not one error at a time."""
    root = tmp_path / 'data'
    root.mkdir()
    p1 = root / 'p1'
    p2 = root / 'p2'
    _make_corpus(p1 / 'a')
    _make_corpus(p1 / 'b')
    _make_corpus(p2)
    _make_corpus(p2 / 'inner')

    with pytest.raises(NestedCorpusError) as exc_info:
        assert_no_nested_corpora(root)

    # Three violations total: p1/a, p1/b, p2/inner.
    assert len(exc_info.value.violations) == 3


def test_assert_no_nested_corpora_ignores_tmp_shards(
    tmp_path: Path,
) -> None:
    """`tmp/cell***__...__runs.parquet` are per-arm shards from
    the sweep, NOT corpora. CI1 must not flag a corpus dir
    just because it has un-merged shards in `tmp/`."""
    root = tmp_path / 'data'
    root.mkdir()
    corp = root / 'corp_with_shards'
    _make_corpus(corp)
    tmp_dir = corp / 'tmp'
    tmp_dir.mkdir()
    # Per-arm shard with characteristic naming.
    shard = tmp_dir / 'cell000__Env__baseline__runs.parquet'
    pl.DataFrame({'id': ['x'], 'env_name': ['Env']}).write_parquet(shard)

    # Should NOT raise — tmp/ shards aren't corpora.
    assert_no_nested_corpora(root)


def test_assert_no_nested_corpora_handles_missing_root(
    tmp_path: Path,
) -> None:
    """Non-existent root: silent no-op (caller's normal
    not-a-dir error path handles it)."""
    assert_no_nested_corpora(tmp_path / 'does_not_exist')
    # No exception expected.


# ============ CI3 — cloud-root uniqueness ============


def _stamp_manifest(sweep_dir: Path, remote_root: str) -> None:
    """Hand-write a `_remote.json` claiming `remote_root`."""
    sweep_dir.mkdir(parents=True, exist_ok=True)
    m = RemoteManifest(
        remote_root=remote_root,
        files=(RemoteFile(
            relpath='runs.parquet',
            sha256='0' * 64,
            size_bytes=100,
            pushed_at='2026-05-07T00:00:00+00:00',
            row_ids=(),
        ),),
    )
    from corroborate.corpus.cloud import _save_manifest
    _save_manifest(sweep_dir, m)


def test_assert_unique_remote_root_no_siblings(tmp_path: Path) -> None:
    """No sibling corpus exists: no collision possible."""
    sweep = tmp_path / 'data' / 'corpA'
    sweep.mkdir(parents=True)
    # Should NOT raise.
    assert_unique_remote_root(
        sweep, 's3://corroborate-archive/corpA',
    )


def test_assert_unique_remote_root_distinct_roots_pass(
    tmp_path: Path,
) -> None:
    """Sibling claims a DIFFERENT remote_root: fine."""
    root = tmp_path / 'data'
    sib = root / 'corpA'
    me = root / 'corpB'
    me.mkdir(parents=True)
    _stamp_manifest(sib, 's3://corroborate-archive/corpA')

    # Should NOT raise.
    assert_unique_remote_root(
        me, 's3://corroborate-archive/corpB',
    )


def test_assert_unique_remote_root_collision_raises(
    tmp_path: Path,
) -> None:
    """**CI3 violation**: a sibling corpus already claims this
    `remote_root`. Two corpora pushing to the same s3 prefix
    silently overwrite each other; refuse upfront.
    Mirrors the in-the-wild
    `minatar_sync_curve/{ddqn_sync1k, ddqn_sync3k, vanilla_sync1k,
    vanilla_sync3k}` quartet sharing
    `s3://corroborate-archive/minatar_sync_curve`."""
    root = tmp_path / 'data'
    sib = root / 'corpA'
    me = root / 'corpB'
    me.mkdir(parents=True)
    shared_root = 's3://corroborate-archive/minatar_sync_curve'
    _stamp_manifest(sib, shared_root)

    with pytest.raises(RemoteRootCollision) as exc_info:
        assert_unique_remote_root(me, shared_root)

    assert exc_info.value.remote_root == shared_root
    assert exc_info.value.existing_dir == sib
    msg = str(exc_info.value)
    assert 'CI3' in msg
    assert shared_root in msg


def test_assert_unique_remote_root_self_excluded(
    tmp_path: Path,
) -> None:
    """A corpus's OWN existing `_remote.json` doesn't count
    against itself — re-archiving to your own root is fine."""
    root = tmp_path / 'data'
    me = root / 'corpA'
    me.mkdir(parents=True)
    my_root = 's3://corroborate-archive/corpA'
    _stamp_manifest(me, my_root)

    # Should NOT raise — only OTHER corpora's claims are
    # collisions.
    assert_unique_remote_root(me, my_root)


# ============ CI5 — archive refuses trivial files ============


def test_assert_archive_eligible_valid_parquet(tmp_path: Path) -> None:
    """A real parquet (>1 KiB, valid PAR1 footer) is eligible."""
    p = tmp_path / 'good.parquet'
    df = pl.DataFrame({'x': list(range(1000))})  # ensures > 1 KiB
    df.write_parquet(p)
    assert p.stat().st_size > 1024
    # Should NOT raise.
    assert_archive_eligible(p)


def test_assert_archive_eligible_zero_byte_raises(tmp_path: Path) -> None:
    """**CI5 violation, 0-byte placeholder**: the canonical
    `action_dim_wide/traces.parquet` shape — empty file
    archived as authoritative cloud copy."""
    p = tmp_path / 'empty.parquet'
    p.touch()
    assert p.stat().st_size == 0

    with pytest.raises(ArchivePrecondition) as exc_info:
        assert_archive_eligible(p)
    msg = str(exc_info.value)
    assert 'CI5' in msg
    assert '0 bytes' in msg or 'size 0' in msg


def test_assert_archive_eligible_truncated_parquet_raises(
    tmp_path: Path,
) -> None:
    """**CI5 violation, missing PAR1 footer**: a truncated /
    killed-mid-write parquet passes the size check but lacks
    the magic footer. Reading it later would `ComputeError`."""
    p = tmp_path / 'truncated.parquet'
    p.write_bytes(b'X' * 4096)  # large enough to pass size check
    assert p.stat().st_size > 1024

    with pytest.raises(ArchivePrecondition) as exc_info:
        assert_archive_eligible(p)
    msg = str(exc_info.value)
    assert 'PAR1' in msg


def test_assert_archive_eligible_below_min_size_raises(
    tmp_path: Path,
) -> None:
    """A non-parquet file under 1 KiB is rejected. Most
    legitimate archive payloads (parquet, JSON sidecars) are
    well over 1 KiB; sub-KiB files are usually accidents."""
    p = tmp_path / 'tiny.json'
    p.write_text('{"foo":"bar"}')

    with pytest.raises(ArchivePrecondition):
        assert_archive_eligible(p)


def test_assert_archive_eligible_missing_file_raises(
    tmp_path: Path,
) -> None:
    """A non-existent path raises. (Caller's `local.is_file()`
    check usually catches this earlier, but the integrity
    function is robust to bad inputs.)"""
    with pytest.raises(ArchivePrecondition):
        assert_archive_eligible(tmp_path / 'nope.parquet')
