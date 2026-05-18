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
    IN_PROGRESS_SENTINEL,
    ArchivePrecondition,
    NestedCorpusError,
    RemoteRootCollision,
    TraceContaminationError,
    assert_archive_eligible,
    assert_no_nested_corpora,
    assert_traces_subset_of_runs,
    assert_unique_remote_root,
    audit_trace_contamination,
    is_in_progress,
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


def test_assert_no_nested_corpora_respects_in_progress_sentinel(
    tmp_path: Path,
) -> None:
    """**Sentinel escape hatch**: a corpus dir marked with
    `.in_progress` (sweep mid-flight) is skipped by CI1 even
    though it has nested per-arm sub-corpora. This is the
    legitimate "sweep dispatcher producing per-arm sub-dirs
    before the merge step" pattern — nesting is transient,
    user knows about it, the runner shouldn't refuse.
    """
    root = tmp_path / 'data'
    root.mkdir()
    parent = root / 'sweep_in_flight'
    _make_corpus(parent / 'arm_a')
    _make_corpus(parent / 'arm_b')
    # Without the sentinel: violation.
    with pytest.raises(NestedCorpusError):
        assert_no_nested_corpora(root)
    # With the sentinel: silent skip.
    (parent / IN_PROGRESS_SENTINEL).touch()
    assert_no_nested_corpora(root)


def test_is_in_progress_helper(tmp_path: Path) -> None:
    """Trivial cheap helper, but worth pinning so future
    refactors don't accidentally change the sentinel filename."""
    corpus = tmp_path / 'corp'
    corpus.mkdir()
    assert is_in_progress(corpus) is False
    (corpus / IN_PROGRESS_SENTINEL).touch()
    assert is_in_progress(corpus) is True
    assert IN_PROGRESS_SENTINEL == '.in_progress'


def test_assert_no_nested_corpora_respects_sub_corpora_only_sentinel(
    tmp_path: Path,
) -> None:
    """`.sub_corpora_only` sentinel: parent intentionally contains
    a flat list of sub-corpora with no own runs.parquet. Without
    the sentinel this triggers CI1 (pure-nested violation); with
    it, CI1 silently skips the parent. This is the
    `merge_top_level: false` opt-out path."""
    from corroborate.corpus.integrity import (
        SUB_CORPORA_ONLY_SENTINEL,
        is_sub_corpora_only,
    )
    root = tmp_path / 'data'
    root.mkdir()
    parent = root / 'sweep_no_merge'
    _make_corpus(parent / 'arm_a')
    _make_corpus(parent / 'arm_b')
    # Without the sentinel: pure-nested violation.
    with pytest.raises(NestedCorpusError):
        assert_no_nested_corpora(root)
    # With the sentinel: silent skip.
    (parent / SUB_CORPORA_ONLY_SENTINEL).touch()
    assert_no_nested_corpora(root)
    assert is_sub_corpora_only(parent) is True
    assert SUB_CORPORA_ONLY_SENTINEL == '.sub_corpora_only'


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


# ============ CI4 — content-dedup ignores volatile object reprs ============


def test_dedup_excludes_object_repr_string_columns() -> None:
    """**CI4**: a string column whose values look like Python
    object reprs (`<...\\sobject\\sat\\s0x[0-9a-f]+>`) carries
    process-volatile memory addresses and should be EXCLUDED
    from `_dedup_by_content` content comparison.

    Two cells with identical scientific content but different
    repr addresses should collapse to one row, not survive as
    two — that's the inflation pattern that hit the
    `vanilla_sync*` corpora pre-fix.
    """
    from corroborate.runner.runner import _dedup_by_content

    # Two rows with same content, different `claim_repr` repr
    # addresses (simulates `<Claim:foo object at 0x...>` reprs
    # that vary per Python session).
    df = pl.DataFrame({
        'id': ['a', 'b'],
        'arm_key': ['baseline', 'baseline'],
        'env_name': ['CartPole-v1', 'CartPole-v1'],
        'gamma': [0.99, 0.99],
        'seed': [0, 0],
        'claim_repr': [
            '<corroborate.Claim:double_greedify object at 0x77d49bbb2ea0>',
            '<corroborate.Claim:double_greedify object at 0x71a8845ddc10>',
        ],
    })

    out = _dedup_by_content(df, source='test')
    assert out.height == 1, (
        f'expected dedup to collapse 2 content-equal rows; got {out.height}. '
        f'CI4 dynamic-volatile-string exclusion did not fire — claim_repr '
        f'differing memory addresses kept the rows distinct.'
    )


def test_dedup_keeps_distinct_content_rows() -> None:
    """**Negative control**: rows with the SAME object-repr
    column but DIFFERENT non-volatile content stay distinct.
    The volatile exclusion must not over-collapse legitimate
    distinct cells."""
    from corroborate.runner.runner import _dedup_by_content

    df = pl.DataFrame({
        'id': ['a', 'b'],
        'arm_key': ['baseline', 'baseline'],
        'env_name': ['CartPole-v1', 'CartPole-v1'],
        'gamma': [0.99, 0.95],   # ← differs
        'seed': [0, 0],
        'claim_repr': [
            '<C object at 0x1>',
            '<C object at 0x2>',
        ],
    })

    out = _dedup_by_content(df, source='test')
    assert out.height == 2, (
        f'expected both distinct rows preserved; got {out.height}. '
        f'CI4 over-collapsed despite content difference (gamma).'
    )


def test_dedup_volatile_detection_skips_non_repr_strings() -> None:
    """A normal string column (e.g. `arm_key='baseline'`) is
    NOT treated as volatile — only the specific object-repr
    pattern triggers exclusion."""
    from corroborate.runner.runner import _volatile_object_repr_columns

    df = pl.DataFrame({
        'id': ['a', 'b'],
        'arm_key': ['baseline', 'baseline'],   # plain string — NOT a repr
        'env_name': ['CartPole-v1', 'CartPole-v1'],
        'env_repr': [
            '<gymnax.E object at 0x1>',
            '<gymnax.E object at 0x2>',
        ],
    })

    volatile = _volatile_object_repr_columns(df)
    assert 'env_repr' in volatile
    assert 'arm_key' not in volatile
    assert 'env_name' not in volatile


# ============ CI8 — traces.id ⊆ runs.id ============


def _write_runs(p: Path, ids: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({'id': ids, 'env_name': ['Env'] * len(ids)}).write_parquet(p)


def _write_traces(p: Path, ids: list[str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame({'id': ids, 'reward': [0.0] * len(ids)}).write_parquet(p)


def test_audit_trace_contamination_healthy(tmp_path: Path) -> None:
    """`traces.id == runs.id` is the canonical healthy state.
    No contamination."""
    _write_runs(tmp_path / 'runs.parquet', ['a', 'b', 'c'])
    _write_traces(tmp_path / 'traces.parquet', ['a', 'b', 'c'])
    stats = audit_trace_contamination(tmp_path)
    assert stats is not None
    assert not stats.is_contaminated
    assert stats.spurious_count == 0


def test_audit_trace_contamination_partial_coverage_ok(
    tmp_path: Path,
) -> None:
    """Partial trace coverage (`traces.id ⊂ runs.id`) is
    legitimate — some cells didn't get traces archived. Not a
    contamination."""
    _write_runs(tmp_path / 'runs.parquet', ['a', 'b', 'c', 'd'])
    _write_traces(tmp_path / 'traces.parquet', ['a', 'b'])  # only 2 of 4
    stats = audit_trace_contamination(tmp_path)
    assert stats is not None
    assert not stats.is_contaminated
    assert stats.spurious_count == 0
    assert stats.overlap_count == 2


def test_audit_trace_contamination_zero_overlap(tmp_path: Path) -> None:
    """**The canonical CI8 violation**: cloud-archive collision
    leaves traces.parquet with ids from a DIFFERENT sweep.
    `traces.id ∩ runs.id == ∅`. Mirrors the in-the-wild
    `minatar_sync_curve_pt2_ddqn_sync1k` shape."""
    _write_runs(tmp_path / 'runs.parquet', ['a', 'b', 'c'])
    _write_traces(tmp_path / 'traces.parquet', ['x', 'y', 'z'])
    stats = audit_trace_contamination(tmp_path)
    assert stats is not None
    assert stats.is_contaminated
    assert stats.spurious_count == 3
    assert stats.overlap_count == 0


def test_audit_trace_contamination_partial_overlap(
    tmp_path: Path,
) -> None:
    """Mixed: traces has some legitimate ids AND some spurious
    ones. Even one spurious id contaminates the file — we can't
    selectively trust the matching subset because there's no way
    to distinguish "valid-and-archived" from "happens-to-share-
    UUID with a different sweep."
    """
    _write_runs(tmp_path / 'runs.parquet', ['a', 'b', 'c'])
    _write_traces(tmp_path / 'traces.parquet', ['a', 'b', 'spurious'])
    stats = audit_trace_contamination(tmp_path)
    assert stats is not None
    assert stats.is_contaminated
    assert stats.spurious_count == 1
    assert stats.overlap_count == 2


def test_audit_trace_contamination_missing_files(
    tmp_path: Path,
) -> None:
    """Either file absent → silent None (caller decides). The
    audit doesn't fabricate a violation when there's nothing
    to compare."""
    # Neither file
    assert audit_trace_contamination(tmp_path) is None
    # Only runs
    _write_runs(tmp_path / 'runs.parquet', ['a'])
    assert audit_trace_contamination(tmp_path) is None


def test_assert_traces_subset_of_runs_raises_on_violation(
    tmp_path: Path,
) -> None:
    """`assert_traces_subset_of_runs` is the assert-form of the
    audit — raises `TraceContaminationError` when contaminated,
    silent otherwise."""
    _write_runs(tmp_path / 'runs.parquet', ['a', 'b'])
    _write_traces(tmp_path / 'traces.parquet', ['a', 'b'])
    # Healthy: no exception.
    assert_traces_subset_of_runs(tmp_path)

    # Contaminate.
    _write_traces(tmp_path / 'traces.parquet', ['a', 'b', 'spurious'])
    with pytest.raises(TraceContaminationError) as exc_info:
        assert_traces_subset_of_runs(tmp_path)
    msg = str(exc_info.value)
    assert 'CI8' in msg
    assert 'spurious' in msg.lower() or 'absent' in msg.lower()


# ============ CI7 — broader trace eviction ============


def test_trace_is_cloud_recoverable_with_matching_sha(
    tmp_path: Path,
) -> None:
    """**CI7**: a locally-cached trace file IS cloud-recoverable
    when the manifest's sha256 matches the local file's sha256.
    The runner can safely evict the local copy.
    """
    from corroborate.corpus.cloud import _sha256_file
    from corroborate.runner.runner import _trace_is_cloud_recoverable

    corpus = tmp_path / 'corp'
    corpus.mkdir()
    traces = corpus / 'traces.parquet'
    _write_traces(traces, ['a', 'b', 'c'])
    sha = _sha256_file(traces)

    # Stamp a manifest claiming the file at this sha256.
    m = RemoteManifest(
        remote_root='s3://bucket/corp',
        files=(RemoteFile(
            relpath='traces.parquet',
            sha256=sha, size_bytes=traces.stat().st_size,
            pushed_at='2026-05-07T00:00:00+00:00',
            row_ids=(),
        ),),
    )
    from corroborate.corpus.cloud import _save_manifest
    _save_manifest(corpus, m)

    assert _trace_is_cloud_recoverable(corpus, traces) is True


def test_trace_is_cloud_recoverable_when_no_manifest(
    tmp_path: Path,
) -> None:
    """**CI7**: local-only corpus (no `_remote.json`) is NOT
    cloud-recoverable. Eviction would lose the data
    permanently. The runner must skip eviction."""
    from corroborate.runner.runner import _trace_is_cloud_recoverable
    corpus = tmp_path / 'corp'
    corpus.mkdir()
    traces = corpus / 'traces.parquet'
    _write_traces(traces, ['a'])

    assert _trace_is_cloud_recoverable(corpus, traces) is False


def test_trace_is_cloud_recoverable_when_sha_mismatches(
    tmp_path: Path,
) -> None:
    """**CI7**: local file's sha256 differs from the manifest's
    (local drift — partial recovery, mid-write modification, or
    different upload state). NOT recoverable from cloud at the
    *current* local state — eviction would lose the divergent
    local content."""
    from corroborate.runner.runner import _trace_is_cloud_recoverable

    corpus = tmp_path / 'corp'
    corpus.mkdir()
    traces = corpus / 'traces.parquet'
    _write_traces(traces, ['a', 'b'])

    # Manifest claims a DIFFERENT sha256.
    m = RemoteManifest(
        remote_root='s3://bucket/corp',
        files=(RemoteFile(
            relpath='traces.parquet',
            sha256='different' + 'x' * 56,
            size_bytes=traces.stat().st_size,
            pushed_at='2026-05-07T00:00:00+00:00',
            row_ids=(),
        ),),
    )
    from corroborate.corpus.cloud import _save_manifest
    _save_manifest(corpus, m)

    assert _trace_is_cloud_recoverable(corpus, traces) is False
