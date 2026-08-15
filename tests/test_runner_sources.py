"""Tests for the cache-sources sidecar — input provenance for
`cache/<short>.parquet`.

This module covers the schema, atomic I/O, and the `evict()`
wire-in. The build path (`_ingest_and_compute`) and
`check_cache_sources` are covered in their own modules.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import polars as pl

from corroborate.bridge.bridge import Bridge
from corroborate.core.finding import Finding
from corroborate.core.intervention import DoEffect, Intervention
from corroborate.runner.runner import (
    CacheSourceEntry,
    CacheSources,
    _read_sources,  # pyright: ignore[reportPrivateUsage]
    _sources_path,  # pyright: ignore[reportPrivateUsage]
    _write_sources,  # pyright: ignore[reportPrivateUsage]
    evict,
)


def _trivial_doeffect() -> DoEffect:
    from corroborate.core.claim import claim

    @claim
    def _stub(x: int) -> int:
        return x

    return DoEffect(arms=(
        (),
        (Intervention(slot_path='stub', replacement=_stub),),
    ))


@dataclass(frozen=True)
class _StubHypothesis:
    """Smallest viable Hypothesis satisfying `_validate_hypothesis`."""
    INTERVENTION: ClassVar[DoEffect] = _trivial_doeffect()
    BRIDGES: ClassVar[tuple[Bridge, ...]] = ()
    FINDINGS: ClassVar[tuple[Finding, ...]] = ()


# ============ Fixtures ============

def _make_entry(
    corpus: str,
    data_root: str | None = '/repo/experiments/data',
    remote_root: str | None = None,
    ingested_at: tuple[str, ...] = ('2026-05-15T12:00:00+00:00',),
) -> CacheSourceEntry:
    return CacheSourceEntry(
        corpus=corpus,
        data_root=data_root,
        remote_root=remote_root,
        ingested_at=ingested_at,
    )


# ============ 1. Corruption tolerance ============

def test_read_sources_returns_none_when_absent(tmp_path: Path) -> None:
    p = tmp_path / 'ddqn.sources.json'
    assert _read_sources(p) is None


def test_read_sources_returns_none_on_malformed_json(tmp_path: Path) -> None:
    p = tmp_path / 'ddqn.sources.json'
    _ = p.write_text('not-json{{')
    assert _read_sources(p) is None


def test_read_sources_returns_none_on_wrong_shape(tmp_path: Path) -> None:
    """JSON parses but the top-level isn't a mapping. Mirrors
    `_read_manifest`'s tolerance."""
    p = tmp_path / 'ddqn.sources.json'
    _ = p.write_text('[1, 2, 3]')
    assert _read_sources(p) is None


def test_read_sources_drops_malformed_entries(tmp_path: Path) -> None:
    """A valid top-level with one well-formed entry and one
    missing-required-key entry should keep the good one and drop
    the bad — same robustness as the cloud manifest reader."""
    p = tmp_path / 'ddqn.sources.json'
    _ = p.write_text(
        '{"sources": ['
        '  {"corpus": "good", "data_root": null, "remote_root": null,'
        '   "ingested_at": ["2026-05-15T12:00:00+00:00"]},'
        '  {"data_root": null}'   # missing 'corpus' → drop
        ']}'
    )
    got = _read_sources(p)
    assert got is not None
    assert len(got.sources) == 1
    assert got.sources[0].corpus == 'good'


# ============ Round-trip ============

def test_write_then_read_round_trip(tmp_path: Path) -> None:
    p = tmp_path / 'ddqn.sources.json'
    original = CacheSources(sources=(
        _make_entry('cartpole_1M_postfix',
                    remote_root='s3://test-bucket/cartpole_1M_postfix'),
        _make_entry('fourrooms_100k_slice',
                    ingested_at=('2026-05-10T08:00:00+00:00',
                                 '2026-05-15T12:00:00+00:00')),
    ))
    _write_sources(p, original)
    got = _read_sources(p)
    assert got is not None
    # Set-equality on entries (order isn't part of the contract,
    # but the sort is — see byte-stable test below).
    assert set(got.sources) == set(original.sources)


# ============ 12. JSON stable diff ============

def test_write_sources_is_byte_stable(tmp_path: Path) -> None:
    """Same `CacheSources` written twice → byte-identical files.
    Requires sort_keys=True + sources sorted by `corpus`."""
    p = tmp_path / 'ddqn.sources.json'
    src = CacheSources(sources=(
        # Intentionally out-of-order corpus names; write must sort.
        _make_entry('zebra'),
        _make_entry('alpha',
                    remote_root='s3://test-bucket/alpha'),
    ))
    _write_sources(p, src)
    first = p.read_bytes()
    _write_sources(p, src)
    second = p.read_bytes()
    assert first == second


def test_write_sources_sorts_entries_by_corpus(tmp_path: Path) -> None:
    p = tmp_path / 'ddqn.sources.json'
    src = CacheSources(sources=(
        _make_entry('zebra'),
        _make_entry('alpha'),
        _make_entry('mango'),
    ))
    _write_sources(p, src)
    got = _read_sources(p)
    assert got is not None
    # Reading preserves write-order because we sort on write.
    assert [e.corpus for e in got.sources] == ['alpha', 'mango', 'zebra']


# ============ Path resolution ============

def test_sources_path_mirrors_hashes_path(tmp_path: Path) -> None:
    cp = tmp_path / 'ddqn.parquet'
    sp = _sources_path(cp)
    assert sp.name == 'ddqn.sources.json'
    assert sp.parent == cp.parent


# ============ 5. evict() drops sources entry ============

def _write_cache_parquet(
    path: Path, rows: list[dict[str, object]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_parquet(path)


def test_evict_drops_sources_entry(tmp_path: Path) -> None:
    """evict() filters the cache parquet; the sidecar should mirror
    the filter so it doesn't carry stale entries for evicted corpora."""
    cache_path = tmp_path / 'ddqn.parquet'
    sources_path = _sources_path(cache_path)

    # Two-corpus cache: A and B.
    _write_cache_parquet(cache_path, [
        {'id': 'cell-a-0', 'corpus': 'corpus_A', 'pad': 'x' * 200},
        {'id': 'cell-a-1', 'corpus': 'corpus_A', 'pad': 'x' * 200},
        {'id': 'cell-b-0', 'corpus': 'corpus_B', 'pad': 'x' * 200},
    ])
    # Sidecar lists both.
    _write_sources(sources_path, CacheSources(sources=(
        _make_entry('corpus_A'),
        _make_entry('corpus_B',
                    remote_root='s3://test-bucket/corpus_B'),
    )))

    # Use a minimal stub Hypothesis-like object (the runner accepts
    # either a Hypothesis or str; the str path would import a real
    # module). For tests we pass a dummy class object satisfying
    # Hypothesis's Protocol structurally.

    total, counts = evict(
        _StubHypothesis,
        ['corpus_A'],
        cache_path=cache_path,
    )

    # Parquet was filtered.
    assert total == 2
    assert counts == {'corpus_A': 2}
    df = pl.read_parquet(cache_path)
    assert set(df['corpus'].to_list()) == {'corpus_B'}

    # Sidecar mirrors.
    got = _read_sources(sources_path)
    assert got is not None
    assert [e.corpus for e in got.sources] == ['corpus_B']


def test_evict_removes_sidecar_when_all_entries_dropped(tmp_path: Path) -> None:
    """When evict drops the only entry, the sidecar file is removed
    (matches the parquet-removal-on-zero-rows behavior)."""
    cache_path = tmp_path / 'ddqn.parquet'
    sources_path = _sources_path(cache_path)

    _write_cache_parquet(cache_path, [
        {'id': 'cell-a-0', 'corpus': 'corpus_A', 'pad': 'x' * 200},
    ])
    _write_sources(sources_path, CacheSources(
        sources=(_make_entry('corpus_A'),),
    ))

    _ = evict(
        _StubHypothesis,
        ['corpus_A'],
        cache_path=cache_path,
    )
    assert not sources_path.exists()


def test_evict_no_op_when_no_sidecar(tmp_path: Path) -> None:
    """evict() must not fail when there's no sidecar to mirror."""
    cache_path = tmp_path / 'ddqn.parquet'
    _write_cache_parquet(cache_path, [
        {'id': 'cell-a-0', 'corpus': 'corpus_A', 'pad': 'x' * 200},
    ])

    total, _counts = evict(
        _StubHypothesis,
        ['corpus_A'],
        cache_path=cache_path,
    )
    assert total == 1  # parquet filtered fine, no sidecar touched
    assert not _sources_path(cache_path).exists()


# ============ Build wire-in: _update_sources_for_walk ============

# Tests #2, #3, #4, #13 from the plan. We call `_update_sources_for_walk`
# directly with a hand-built `new_walk` DataFrame to exercise the
# write logic without invoking the full `_ingest_and_compute` path
# (which needs JAX-compile-heavy fixtures). The integration shape
# is covered by the live smoke at the end of implementation.

from corroborate.runner.runner import (  # noqa: E402
    _update_sources_for_walk,  # pyright: ignore[reportPrivateUsage]
)


def _walk_df(corpora: list[str]) -> pl.DataFrame:
    """A minimal `new_walk`-shape DataFrame with just the `corpus`
    column the wire-in actually reads."""
    return pl.DataFrame({'corpus': corpora})


def test_build_writes_sources_for_walk(tmp_path: Path) -> None:
    """Test #2: A walk producing 'A' and 'B' creates a sidecar
    with one entry per corpus, each carrying the resolved
    `data_root` and a length-1 `ingested_at`."""
    cache_path = tmp_path / 'ddqn.parquet'
    walk_root = tmp_path / 'data'
    walk_root.mkdir(parents=True)
    (walk_root / 'A').mkdir()
    (walk_root / 'B').mkdir()

    _update_sources_for_walk(
        _sources_path(cache_path),
        new_walk=_walk_df(['A', 'B']),
        walk_root=walk_root,
    )

    got = _read_sources(_sources_path(cache_path))
    assert got is not None
    by_corpus = {e.corpus: e for e in got.sources}
    assert set(by_corpus) == {'A', 'B'}
    for name in ('A', 'B'):
        e = by_corpus[name]
        assert e.data_root == str(walk_root.resolve())
        assert len(e.ingested_at) == 1
        assert e.remote_root is None  # no _remote.json in fixture


def test_build_re_ingest_appends_timestamp(tmp_path: Path) -> None:
    """Test #3: Calling the build twice for the same corpus appends
    a second timestamp; the entry isn't duplicated."""
    cache_path = tmp_path / 'ddqn.parquet'
    walk_root = tmp_path / 'data'
    walk_root.mkdir(parents=True)
    (walk_root / 'A').mkdir()

    df = _walk_df(['A'])
    _update_sources_for_walk(
        _sources_path(cache_path), new_walk=df, walk_root=walk_root,
    )
    _update_sources_for_walk(
        _sources_path(cache_path), new_walk=df, walk_root=walk_root,
    )

    got = _read_sources(_sources_path(cache_path))
    assert got is not None
    assert len(got.sources) == 1
    assert len(got.sources[0].ingested_at) == 2


def test_build_append_new_corpus_preserves_existing(
    tmp_path: Path,
) -> None:
    """Test #4: Ingest A then B in separate calls. Both entries
    surface; A's `ingested_at` retains length 1 (not re-appended)."""
    cache_path = tmp_path / 'ddqn.parquet'
    walk_root = tmp_path / 'data'
    walk_root.mkdir(parents=True)
    (walk_root / 'A').mkdir()
    (walk_root / 'B').mkdir()

    _update_sources_for_walk(
        _sources_path(cache_path),
        new_walk=_walk_df(['A']), walk_root=walk_root,
    )
    _update_sources_for_walk(
        _sources_path(cache_path),
        new_walk=_walk_df(['B']), walk_root=walk_root,
    )

    got = _read_sources(_sources_path(cache_path))
    assert got is not None
    by_corpus = {e.corpus: e for e in got.sources}
    assert set(by_corpus) == {'A', 'B'}
    assert len(by_corpus['A'].ingested_at) == 1
    assert len(by_corpus['B'].ingested_at) == 1


def test_build_null_preserves_existing_values(tmp_path: Path) -> None:
    """Test #13: When a re-ingest call returns a `remote_root=None`
    (no _remote.json this time), a previously-recorded
    `remote_root` must NOT be overwritten with null. Same rule
    keeps `data_root` from getting clobbered by hypothetical
    mixed-mode ingest where one call has a known root and another
    doesn't."""
    cache_path = tmp_path / 'ddqn.parquet'
    walk_root = tmp_path / 'data'
    walk_root.mkdir(parents=True)
    (walk_root / 'A').mkdir()

    # Seed the sidecar with a prior entry carrying a non-null
    # remote_root (simulating a previous ingest that DID find a
    # `_remote.json`).
    sources_path = _sources_path(cache_path)
    _write_sources(sources_path, CacheSources(sources=(
        _make_entry('A',
                    remote_root='s3://test-bucket/A',
                    ingested_at=('2026-05-15T11:00:00+00:00',)),
    )))

    # Now re-ingest A. The walk_root has no `_remote.json` under
    # `A/`, so the new `remote_root` would be None.
    _update_sources_for_walk(
        sources_path,
        new_walk=_walk_df(['A']),
        walk_root=walk_root,
    )

    got = _read_sources(sources_path)
    assert got is not None
    assert len(got.sources) == 1
    e = got.sources[0]
    # Prior remote_root preserved (B2 fix: null-preserving update).
    assert e.remote_root == 's3://test-bucket/A'
    assert len(e.ingested_at) == 2


def test_build_no_op_on_empty_walk(tmp_path: Path) -> None:
    """An empty walk (height 0) must NOT touch the sidecar."""
    cache_path = tmp_path / 'ddqn.parquet'
    walk_root = tmp_path / 'data'
    walk_root.mkdir(parents=True)

    _update_sources_for_walk(
        _sources_path(cache_path),
        new_walk=pl.DataFrame({'corpus': []}, schema={'corpus': pl.String}),
        walk_root=walk_root,
    )
    # No sidecar created.
    assert not _sources_path(cache_path).exists()


# ============ check_cache_sources: drift detection ============

from corroborate.runner.runner import check_cache_sources  # noqa: E402


def _make_corpus_runs(
    parent: Path, name: str, n_rows: int, with_id: bool = True,
) -> Path:
    corpus_dir = parent / name
    corpus_dir.mkdir(parents=True, exist_ok=True)
    cols: dict[str, list[object]] = {
        'pad': ['x' * 200 for _ in range(n_rows)],
    }
    if with_id:
        cols['id'] = [f'{name}-{i}' for i in range(n_rows)]
    pl.DataFrame(cols).write_parquet(corpus_dir / 'runs.parquet')
    return corpus_dir / 'runs.parquet'


def test_check_returns_matched_when_counts_agree(tmp_path: Path) -> None:
    """Test #6 happy path: cache cells == on-disk rows → MATCHED."""
    cache_path = tmp_path / 'ddqn.parquet'
    data_root = tmp_path / 'data'

    _ = _make_corpus_runs(data_root, 'cartpole', n_rows=60)
    _write_cache_parquet(cache_path, [
        {'id': f'cell-{i}', 'corpus': 'cartpole', 'pad': 'x' * 200}
        for i in range(60)
    ])
    _write_sources(_sources_path(cache_path), CacheSources(sources=(
        _make_entry('cartpole', data_root=str(data_root.resolve())),
    )))

    drifts = check_cache_sources(cache_path)
    assert len(drifts) == 1
    d = drifts[0]
    assert d.status == 'MATCHED'
    assert d.cache_cell_count == 60
    assert d.current_cell_count == 60


def test_check_detects_drifted(tmp_path: Path) -> None:
    """Test #6: cache and on-disk counts differ → DRIFTED."""
    cache_path = tmp_path / 'ddqn.parquet'
    data_root = tmp_path / 'data'

    _ = _make_corpus_runs(data_root, 'cartpole', n_rows=50)  # less than cache
    _write_cache_parquet(cache_path, [
        {'id': f'cell-{i}', 'corpus': 'cartpole', 'pad': 'x' * 200}
        for i in range(60)
    ])
    _write_sources(_sources_path(cache_path), CacheSources(sources=(
        _make_entry('cartpole', data_root=str(data_root.resolve())),
    )))

    drifts = check_cache_sources(cache_path)
    assert len(drifts) == 1
    d = drifts[0]
    assert d.status == 'DRIFTED'
    assert d.cache_cell_count == 60
    assert d.current_cell_count == 50


def test_check_missing_local(tmp_path: Path) -> None:
    """Test #7: corpus dir deleted from disk → MISSING_LOCAL,
    current_cell_count is None (NOT 0 — distinguishes from real
    zero per the R3 fix)."""
    cache_path = tmp_path / 'ddqn.parquet'
    data_root = tmp_path / 'data'
    data_root.mkdir()

    _write_cache_parquet(cache_path, [
        {'id': 'c-0', 'corpus': 'gone', 'pad': 'x' * 200},
    ])
    _write_sources(_sources_path(cache_path), CacheSources(sources=(
        _make_entry('gone', data_root=str(data_root.resolve())),
    )))

    drifts = check_cache_sources(cache_path)
    assert len(drifts) == 1
    d = drifts[0]
    assert d.status == 'MISSING_LOCAL'
    assert d.current_cell_count is None  # critical: None vs 0


def test_check_zero_rows_is_real_zero(tmp_path: Path) -> None:
    """Test #8 (R3 fix): runs.parquet has 0 rows but the `id`
    column exists → current_cell_count is 0, NOT None. If cache
    also has 0 cells, status is MATCHED (real zero === real zero)."""
    cache_path = tmp_path / 'ddqn.parquet'
    data_root = tmp_path / 'data'

    # 0-row runs.parquet WITH id column.
    corpus_dir = data_root / 'empty'
    corpus_dir.mkdir(parents=True)
    pl.DataFrame(
        schema={'id': pl.String, 'pad': pl.String},
    ).write_parquet(corpus_dir / 'runs.parquet')

    # The cache has rows for some other corpus; the sidecar has
    # an entry for 'empty' but the cache doesn't list 'empty' →
    # STALE_SIDECAR_ENTRY with current_cell_count=0 (real zero).
    _write_cache_parquet(cache_path, [
        {'id': 'other-0', 'corpus': 'other', 'pad': 'x' * 200},
    ])
    _write_sources(_sources_path(cache_path), CacheSources(sources=(
        _make_entry('empty', data_root=str(data_root.resolve())),
    )))

    drifts = check_cache_sources(cache_path)
    by_corpus = {d.corpus: d for d in drifts}
    # 'empty' in sidecar, not in cache → STALE_SIDECAR_ENTRY.
    assert 'empty' in by_corpus
    assert by_corpus['empty'].status == 'STALE_SIDECAR_ENTRY'
    assert by_corpus['empty'].current_cell_count == 0  # real zero, NOT None
    # 'other' in cache, not in sidecar → NO_SIDECAR_RECORD.
    assert 'other' in by_corpus
    assert by_corpus['other'].status == 'NO_SIDECAR_RECORD'


def test_check_missing_id_column(tmp_path: Path) -> None:
    """Test #9 (R3 fix): runs.parquet exists but has no `id`
    column → MISSING_LOCAL, current_cell_count=None."""
    cache_path = tmp_path / 'ddqn.parquet'
    data_root = tmp_path / 'data'

    _ = _make_corpus_runs(data_root, 'no_id', n_rows=10, with_id=False)
    _write_cache_parquet(cache_path, [
        {'id': f'c-{i}', 'corpus': 'no_id', 'pad': 'x' * 200}
        for i in range(10)
    ])
    _write_sources(_sources_path(cache_path), CacheSources(sources=(
        _make_entry('no_id', data_root=str(data_root.resolve())),
    )))

    drifts = check_cache_sources(cache_path)
    assert len(drifts) == 1
    d = drifts[0]
    assert d.status == 'MISSING_LOCAL'
    assert d.current_cell_count is None


def test_check_pre_sidecar_cache(tmp_path: Path) -> None:
    """Test #10: cache parquet with corpus column, no sidecar →
    every distinct corpus reports NO_SIDECAR_RECORD."""
    cache_path = tmp_path / 'ddqn.parquet'
    _write_cache_parquet(cache_path, [
        {'id': 'a-0', 'corpus': 'A', 'pad': 'x' * 200},
        {'id': 'b-0', 'corpus': 'B', 'pad': 'x' * 200},
    ])
    # No sidecar.

    drifts = check_cache_sources(cache_path)
    assert {d.corpus for d in drifts} == {'A', 'B'}
    assert all(d.status == 'NO_SIDECAR_RECORD' for d in drifts)


def test_check_stale_sidecar_entry(tmp_path: Path) -> None:
    """Test #11 (B3 fix): sidecar lists a corpus the cache parquet
    has no rows for → STALE_SIDECAR_ENTRY. Surfaces the orphan."""
    cache_path = tmp_path / 'ddqn.parquet'
    data_root = tmp_path / 'data'
    _ = _make_corpus_runs(data_root, 'ghost', n_rows=5)

    # Cache has rows for someone else; sidecar lists 'ghost'.
    _write_cache_parquet(cache_path, [
        {'id': 'other-0', 'corpus': 'other', 'pad': 'x' * 200},
    ])
    _write_sources(_sources_path(cache_path), CacheSources(sources=(
        _make_entry('ghost', data_root=str(data_root.resolve())),
    )))

    drifts = check_cache_sources(cache_path)
    by_corpus = {d.corpus: d for d in drifts}
    assert by_corpus['ghost'].status == 'STALE_SIDECAR_ENTRY'
    assert by_corpus['ghost'].cache_cell_count == 0
    # current_cell_count populated via §R3 so the operator can see
    # whether the source is recoverable.
    assert by_corpus['ghost'].current_cell_count == 5
