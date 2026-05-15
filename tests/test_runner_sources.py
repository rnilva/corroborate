"""Tests for the cache-sources sidecar — input provenance for
`cache/<short>.parquet`. See the plan at
`/root/.claude/plans/cache-sources-sidecar.md`.

This module covers the schema, atomic I/O, and the `evict()`
wire-in (test #5 in the plan). The build path (`_ingest_and_compute`)
and `check_cache_sources` come in follow-up commits.
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
    data_root: str | None = '/workspace/corroborate/experiments/data',
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
                    remote_root='s3://corroborate-archive/cartpole_1M_postfix'),
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
                    remote_root='s3://corroborate-archive/alpha'),
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
                    remote_root='s3://corroborate-archive/corpus_B'),
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
