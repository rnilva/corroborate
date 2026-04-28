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


def test_walk_paths_surfaces_nested_leaves_at_dotted_paths() -> None:
    """`signature.walk_paths` must produce dotted topology paths
    keyed at each leaf — `optimizer.inner.lr` (nested leaf under
    WarmedUpdate(inner=Adam(...))) — not the flat `lr`."""
    from functools import partial

    from corroborate.rl.dqn.dqn import dqn
    from corroborate.signature import walk, walk_paths

    configured = partial(dqn, optimizer=_make_warmed_adam())
    paths = walk_paths(walk(configured), regime='leaf')

    # Top-level leaves (gamma is dqn's direct kwarg).
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


# ============ apply_trace_reductions ============

def test_apply_trace_reductions_noop_when_empty() -> None:
    """No exprs + no drops → traces returned unchanged."""
    from corroborate.persistence import apply_trace_reductions
    rows = [
        TraceRow(
            id='r1', cycle_id=None,
            timestamp='2026-01-01T00:00:00+00:00',
            leaves={'reward': [1.0, 0.5, 0.0]},
        ),
    ]
    out = apply_trace_reductions(rows, add=(), drop=())
    assert out == rows


def test_apply_trace_reductions_adds_polars_expr_column() -> None:
    """An `add` expr produces a new column that lands as a
    TraceRow leaf."""
    import polars as pl
    from corroborate.persistence import apply_trace_reductions
    rows = [
        TraceRow(
            id='r1', cycle_id=None,
            timestamp='2026-01-01T00:00:00+00:00',
            leaves={'reward': [1.0, 0.5, 0.0]},
        ),
    ]
    [out] = apply_trace_reductions(
        rows,
        add=(pl.col('reward').list.max().alias('reward_max'),),
    )
    assert out.leaves['reward'] == [1.0, 0.5, 0.0]
    assert out.leaves['reward_max'] == 1.0


def test_apply_trace_reductions_drops_named_columns() -> None:
    """A `drop` list removes those columns from the trace.
    Provenance fields are NOT droppable (they're TraceRow's typed
    fields, not leaves) — but a leaf can be."""
    from corroborate.persistence import apply_trace_reductions
    rows = [
        TraceRow(
            id='r1', cycle_id=None,
            timestamp='2026-01-01T00:00:00+00:00',
            leaves={'reward': [1.0, 0.5, 0.0], 'gamma': 0.99},
        ),
    ]
    [out] = apply_trace_reductions(rows, drop=('reward',))
    assert 'reward' not in out.leaves
    assert out.leaves['gamma'] == 0.99


def test_apply_trace_reductions_subsample_via_polars_expr() -> None:
    """Subsampling expressed as `list.gather_every(N)` — the
    user's suggested unification of subsample + reduce under one
    declarative polars-expr surface."""
    import polars as pl
    from corroborate.persistence import apply_trace_reductions
    rows = [
        TraceRow(
            id='r1', cycle_id=None,
            timestamp='2026-01-01T00:00:00+00:00',
            leaves={'reward': [float(i) for i in range(10)]},
        ),
    ]
    every_other = (
        pl.col('reward')
        .list.gather_every(2)
        .alias('reward_every_2nd')
    )
    [out] = apply_trace_reductions(
        rows,
        add=(every_other,),
        drop=('reward',),
    )
    assert out.leaves['reward_every_2nd'] == [0.0, 2.0, 4.0, 6.0, 8.0]
    assert 'reward' not in out.leaves


def test_apply_trace_reductions_collapse_3d_to_1d() -> None:
    """The 3-D → 1-D reduction pattern that's the actual
    motivation. Polars's `list.eval` has known limitations with
    deeply nested lists; `map_elements` with a Python UDF handles
    arbitrary nesting cleanly. Either is a polars expression and
    works through `apply_trace_reductions`."""
    import polars as pl
    from corroborate.persistence import apply_trace_reductions
    # 3-D shape: (2 outer, 2 inner-batch, 3 actions)
    online_q = [
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],   # step 0
        [[0.0, 0.5, 1.0], [2.0, 1.5, 0.5]],   # step 1
    ]
    rows = [
        TraceRow(
            id='r1', cycle_id=None,
            timestamp='2026-01-01T00:00:00+00:00',
            leaves={'online_q_values': online_q},
        ),
    ]
    # Per-step max-Q over (batch × actions). map_elements + Python
    # comprehension is the most readable form for nested-list
    # aggregation; runs once per cell at write time, so the UDF
    # overhead is negligible.
    expr = pl.col('online_q_values').map_elements(
        lambda nested: [max(max(act) for act in batch)
                        for batch in nested.to_list()],
        return_dtype=pl.List(pl.Float64),
    ).alias('online_max_q_per_step')
    [out] = apply_trace_reductions(
        rows,
        add=(expr,),
        drop=('online_q_values',),
    )
    # Step 0: max of [3.0, 6.0] = 6.0; Step 1: max of [1.0, 2.0] = 2.0.
    assert out.leaves['online_max_q_per_step'] == [6.0, 2.0]
    assert 'online_q_values' not in out.leaves


# ============ Zarr round-trip for multi-dim arrays ============

def test_arrays_field_round_trip_via_zarr(tmp_path: Path) -> None:
    """`TraceRow.arrays` round-trips through zarr keyed by cell_id.
    `leaves` go to parquet as before; `arrays` to zarr."""
    import numpy as np
    rows_in = [
        TraceRow(
            id='cell-1', cycle_id='c1',
            timestamp='2026-01-01T00:00:00+00:00',
            leaves={'gamma': 0.99, 'reward': [1.0, 0.0, 0.5]},
            arrays={
                'predicted_q': np.arange(12, dtype=np.float64).reshape(3, 4),
                'pearson_stats': np.zeros((10, 5), dtype=np.float64),
            },
        ),
        TraceRow(
            id='cell-2', cycle_id='c1',
            timestamp='2026-01-01T00:00:01+00:00',
            leaves={'gamma': 0.99, 'reward': [0.5, 1.0]},
            arrays={
                'predicted_q': np.ones((2, 4), dtype=np.float64),
            },
        ),
    ]
    parquet_path = tmp_path / 'traces.parquet'
    zarr_path = tmp_path / 'arrays.zarr'

    write_tracerows(rows_in, parquet_path, zarr_path=zarr_path)
    rows_out = read_tracerows(parquet_path, zarr_path=zarr_path)

    assert len(rows_out) == 2
    by_id = {r.id: r for r in rows_out}

    # Leaves preserved.
    assert by_id['cell-1'].leaves['gamma'] == 0.99
    assert by_id['cell-1'].leaves['reward'] == [1.0, 0.0, 0.5]

    # Arrays preserved with shape + values.
    p1 = by_id['cell-1'].arrays['predicted_q']
    assert p1.shape == (3, 4)
    assert p1.tolist() == np.arange(12).reshape(3, 4).tolist()
    p2 = by_id['cell-1'].arrays['pearson_stats']
    assert p2.shape == (10, 5)

    # Cell-2's arrays are independent (different shapes per cell).
    assert by_id['cell-2'].arrays['predicted_q'].shape == (2, 4)
    # cell-2 has no pearson_stats — heterogeneous-array support.
    assert 'pearson_stats' not in by_id['cell-2'].arrays


def test_write_tracerows_rejects_arrays_without_zarr_path(tmp_path: Path) -> None:
    """If a TraceRow has non-empty arrays but no zarr_path is given,
    write fails — silent data loss is the wrong default."""
    import numpy as np
    rows = [
        TraceRow(
            id='cell-1', cycle_id=None,
            timestamp='2026-01-01T00:00:00+00:00',
            leaves={'gamma': 0.99},
            arrays={'q': np.zeros((3, 4))},
        ),
    ]
    parquet_path = tmp_path / 'traces.parquet'
    with pytest.raises(ValueError, match='arrays present'):
        write_tracerows(rows, parquet_path)


def test_read_tracerows_without_zarr_returns_empty_arrays(
    tmp_path: Path,
) -> None:
    """Reading parquet without a zarr_path yields rows with empty
    `arrays`. Useful for scalar-only analyses that don't need
    multi-dim data."""
    rows_in = [
        TraceRow(
            id='cell-1', cycle_id=None,
            timestamp='2026-01-01T00:00:00+00:00',
            leaves={'gamma': 0.99, 'reward': [1.0, 0.5]},
        ),
    ]
    parquet_path = tmp_path / 'traces.parquet'
    write_tracerows(rows_in, parquet_path)
    rows_out = read_tracerows(parquet_path)
    assert len(rows_out) == 1
    assert rows_out[0].arrays == {}
    assert rows_out[0].leaves['gamma'] == 0.99


def test_zarr_overwrite_on_rewrite(tmp_path: Path) -> None:
    """Re-writing to the same zarr_path replaces existing cell
    groups instead of appending or corrupting."""
    import numpy as np
    parquet_path = tmp_path / 'traces.parquet'
    zarr_path = tmp_path / 'arrays.zarr'

    # First write.
    rows_v1 = [
        TraceRow(
            id='cell-1', cycle_id=None,
            timestamp='2026-01-01T00:00:00+00:00',
            leaves={'gamma': 0.99},
            arrays={'q': np.zeros((3, 4), dtype=np.float64)},
        ),
    ]
    write_tracerows(rows_v1, parquet_path, zarr_path=zarr_path)

    # Second write: same cell_id, different array values.
    rows_v2 = [
        TraceRow(
            id='cell-1', cycle_id=None,
            timestamp='2026-01-01T00:00:00+00:00',
            leaves={'gamma': 0.99},
            arrays={'q': np.ones((3, 4), dtype=np.float64) * 7.0},
        ),
    ]
    write_tracerows(rows_v2, parquet_path, zarr_path=zarr_path)

    rows_out = read_tracerows(parquet_path, zarr_path=zarr_path)
    assert rows_out[0].arrays['q'].tolist() == (
        np.ones((3, 4)) * 7.0
    ).tolist()
