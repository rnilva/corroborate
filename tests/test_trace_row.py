"""Tests for `TraceRow` schema + persistence round-trip.

The trace store is the v9-`traces.parquet` analog: per-cell raw
observation persisted as a flat columnar parquet. Every leaf-path
becomes its own typed parquet column; mixed scalar (HPs, summary
scalars) and list (per-step trajectories) columns coexist.

Tests cover:

1. as_dict / from_row_dict are inverse on simple leaves.
2. Round-trip via parquet preserves typed columns.
3. Heterogeneous rows (different leaf paths per row) null-pad
   cleanly — rows that don't carry a path don't gain a spurious
   None entry on read.
4. List-typed leaves (trajectories) round-trip as Python lists.
5. signature.walk_paths surfaces nested HPs at dotted paths."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import pytest

from corroborate.persistence import read_tracerows, write_tracerows
from corroborate.schema import TraceRow


def test_as_dict_flattens_leaves_to_top_level() -> None:
    row = TraceRow(
        id='abc',
        cycle_id='c1',
        timestamp='2026-01-01T00:00:00+00:00',
        leaves={
            'gamma': 0.99,
            'optimizer.inner.lr': 0.001,
            'reward': [1.0, 0.0, 1.0],
        },
    )
    d = row.as_dict()
    assert d['id'] == 'abc'
    assert d['cycle_id'] == 'c1'
    assert d['timestamp'] == '2026-01-01T00:00:00+00:00'
    assert d['gamma'] == 0.99
    assert d['optimizer.inner.lr'] == 0.001
    assert d['reward'] == [1.0, 0.0, 1.0]


def test_from_row_dict_splits_provenance_from_leaves() -> None:
    d: Mapping[str, object] = {
        'id': 'abc',
        'cycle_id': None,
        'timestamp': '2026-01-01T00:00:00+00:00',
        'gamma': 0.99,
        'reward': [1.0, 0.0],
    }
    row = TraceRow.from_row_dict(d)
    assert row.id == 'abc'
    assert row.cycle_id is None
    assert row.leaves == {'gamma': 0.99, 'reward': [1.0, 0.0]}


def test_round_trip_via_parquet_preserves_types(tmp_path: Path) -> None:
    rows_in = [
        TraceRow(
            id='r1', cycle_id='c1',
            timestamp='2026-01-01T00:00:00+00:00',
            leaves={
                'gamma': 0.99,
                'optimizer.inner.lr': 0.001,
                'replay.batch_size': 64,
                'optimizer': 'dataclass:Adam(lr=0.001)',
                'reward': [1.0, 0.0, 1.0],
                'episode_length': [10, 20, 15],
            },
        ),
        TraceRow(
            id='r2', cycle_id='c1',
            timestamp='2026-01-01T00:00:01+00:00',
            leaves={
                'gamma': 0.99,
                'optimizer.inner.lr': 0.001,
                'replay.batch_size': 64,
                'optimizer': 'dataclass:Adam(lr=0.001)',
                'reward': [0.0, 1.0],
                'episode_length': [5, 25],
            },
        ),
    ]
    path = tmp_path / 'traces.parquet'
    write_tracerows(rows_in, path)
    rows_out = read_tracerows(path)

    assert len(rows_out) == 2
    assert rows_out[0].id == 'r1'
    assert rows_out[0].leaves['gamma'] == 0.99
    assert rows_out[0].leaves['optimizer.inner.lr'] == 0.001
    assert rows_out[0].leaves['replay.batch_size'] == 64
    assert rows_out[0].leaves['optimizer'] == 'dataclass:Adam(lr=0.001)'
    assert rows_out[0].leaves['reward'] == [1.0, 0.0, 1.0]
    assert rows_out[0].leaves['episode_length'] == [10, 20, 15]


def test_heterogeneous_leaf_paths_null_pad(tmp_path: Path) -> None:
    """When two rows carry different leaf paths, polars null-pads
    the missing columns. On read, those nulls are dropped — they
    don't appear in `leaves` of the row that didn't carry them."""
    rows_in = [
        TraceRow(
            id='r1', cycle_id=None,
            timestamp='2026-01-01T00:00:00+00:00',
            leaves={'gamma': 0.99, 'reward': [1.0, 0.0]},
        ),
        TraceRow(
            id='r2', cycle_id=None,
            timestamp='2026-01-01T00:00:01+00:00',
            # No 'gamma' or 'reward'; carries a different path.
            leaves={'optimizer.inner.lr': 0.001},
        ),
    ]
    path = tmp_path / 'traces.parquet'
    write_tracerows(rows_in, path)
    rows_out = read_tracerows(path)

    assert 'gamma' in rows_out[0].leaves
    assert 'reward' in rows_out[0].leaves
    assert 'optimizer.inner.lr' not in rows_out[0].leaves

    assert 'optimizer.inner.lr' in rows_out[1].leaves
    assert 'gamma' not in rows_out[1].leaves
    assert 'reward' not in rows_out[1].leaves


def test_walk_paths_surfaces_nested_hps_at_dotted_paths() -> None:
    """`signature.walk_paths` must produce dotted topology paths
    keyed at the leaf — `optimizer.inner.lr` (nested HP under
    WarmedUpdate(inner=Adam(...))) — not the flat `lr`."""
    from functools import partial

    from corroborate.rl.dqn.dqn import dqn
    from corroborate.signature import walk, walk_paths

    configured = partial(dqn, optimizer=_make_warmed_adam())
    paths = walk_paths(walk(configured), regime='hp')

    # Top-level HPs (gamma is dqn's direct kwarg).
    assert 'gamma' in paths
    # Nested: optimizer is a Module field, inner is its inner
    # Module, lr is Adam's leaf.
    assert 'optimizer.inner.lr' in paths
    # Sibling at the same depth resolves to its own path —
    # NOT colliding with optimizer.inner.lr.
    assert 'optimizer.warmup_steps' in paths


def _make_warmed_adam() -> object:
    from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
    return WarmedUpdate(inner=Adam(lr=1e-3), warmup_steps=100)


def test_unsupported_leaf_type_raises() -> None:
    """A None-valued leaf inside the leaves dict at construction
    time would normally indicate a bug — we don't write nulls.
    The from_row_dict path *does* tolerate None (null-padding),
    but in-memory leaves should be typed."""
    with pytest.raises(TypeError):
        TraceRow.from_row_dict({
            'id': 'r1', 'cycle_id': None,
            'timestamp': 't',
            'bad_leaf': object(),
        })
