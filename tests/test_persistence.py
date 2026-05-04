"""Tests for parquet round-trip across the row types.

Each row type has paired write/read functions; the test pattern
is: construct row → write to tmp parquet → read back → assert
equality. The persistence layer is flat columnar — every
provenance field and every measurement entry becomes its own
typed parquet column. Heterogeneous rows null-pad cleanly."""
from __future__ import annotations

from pathlib import Path

from corroborate.persistence import (
    read_runrows,
    write_runrows,
)
from corroborate.schema import RunRow
from corroborate.verdict import Verdict


# ============ Fixtures ============

def _sample_runrow() -> RunRow:
    return RunRow(
        id='run-1',
        parent_id=None,
        cycle_id='cycle-7',
        timestamp='2026-04-27T10:00:00Z',
        verdict=Verdict.HELD,
        measurements={
            'env_name': 'CartPole-v1',
            'seed': 42,
            'total_steps': 30_000,
            'intervention_name': 'dqn_with_double_greedify',
            'gamma': 0.99,
            'optimizer.inner.lr': 0.001,
            'late_window_mean': 120.5,
            'bridge.some_bridge.verdict': 'held',
            'bridge.some_bridge.stats.rho': 0.85,
        },
    )


# ============ RunRow round-trip ============

def test_runrow_parquet_round_trip_single(tmp_path: Path) -> None:
    path = tmp_path / 'runs.parquet'
    rows = [_sample_runrow()]
    write_runrows(rows, path)
    assert path.exists()

    loaded = read_runrows(path)
    assert len(loaded) == 1
    assert loaded[0] == rows[0]


def test_runrow_parquet_round_trip_multiple_homogeneous(tmp_path: Path) -> None:
    """Multiple rows with the SAME measurement keys round-trip."""
    rows = [
        RunRow(
            id=f'run-{i}', parent_id=None,
            cycle_id=None, timestamp='t', verdict=Verdict.HELD,
            measurements={
                'env_name': 'CartPole-v1', 'seed': i,
                'gamma': 0.99,
                'late_window_mean': float(i),
            },
        )
        for i in range(3)
    ]
    path = tmp_path / 'runs.parquet'
    write_runrows(rows, path)
    loaded = read_runrows(path)
    assert loaded == rows


def test_runrow_parquet_with_no_measurements(tmp_path: Path) -> None:
    """Empty measurements round-trip losslessly."""
    row = RunRow(
        id='no-meas', parent_id=None,
        cycle_id=None, timestamp='t',
        verdict=Verdict.HELD,
    )
    path = tmp_path / 'runs.parquet'
    write_runrows([row], path)
    assert read_runrows(path) == [row]


def test_runrow_parquet_arm_key_round_trip(tmp_path: Path) -> None:
    """`arm_key` is a typed framework-surface column; explicit
    values round-trip and the field stays distinct from the
    `intervention_name` measurement."""
    row = RunRow(
        id='r-1', parent_id=None,
        cycle_id=None, timestamp='t',
        verdict=Verdict.HELD,
        arm_key='bootstrap=Claim:double_greedify',
        measurements={'intervention_name': 'ddqn'},
    )
    path = tmp_path / 'runs.parquet'
    write_runrows([row], path)
    loaded = read_runrows(path)
    assert len(loaded) == 1
    assert loaded[0].arm_key == 'bootstrap=Claim:double_greedify'
    assert loaded[0].measurements['intervention_name'] == 'ddqn'
    assert loaded[0] == row


def test_runrow_arm_key_defaults_to_baseline(tmp_path: Path) -> None:
    """RunRows constructed without `arm_key` default to
    `'baseline'`; backward-compat for fixtures + old parquets."""
    row = RunRow(
        id='r-default', parent_id=None,
        cycle_id=None, timestamp='t',
        verdict=Verdict.HELD,
    )
    assert row.arm_key == 'baseline'
    path = tmp_path / 'runs.parquet'
    write_runrows([row], path)
    loaded = read_runrows(path)
    assert loaded[0].arm_key == 'baseline'


def test_runrow_parquet_heterogeneous_keys_null_pad(tmp_path: Path) -> None:
    """When two rows carry different measurement paths, polars
    null-pads the missing columns. On read, those nulls are
    skipped — they don't appear in `measurements` of the row that
    didn't carry them."""
    rows_in = [
        RunRow(
            id='r1', parent_id=None,
            cycle_id=None, timestamp='t', verdict=Verdict.HELD,
            measurements={'gamma': 0.99, 'env_name': 'CartPole-v1'},
        ),
        RunRow(
            id='r2', parent_id=None,
            cycle_id=None, timestamp='t', verdict=Verdict.HELD,
            # No 'gamma' or 'env_name'; carries a different path.
            measurements={'optimizer.inner.lr': 0.001},
        ),
    ]
    path = tmp_path / 'runs.parquet'
    write_runrows(rows_in, path)
    rows_out = read_runrows(path)

    assert 'gamma' in rows_out[0].measurements
    assert 'env_name' in rows_out[0].measurements
    assert 'optimizer.inner.lr' not in rows_out[0].measurements

    assert 'optimizer.inner.lr' in rows_out[1].measurements
    assert 'gamma' not in rows_out[1].measurements
    assert 'env_name' not in rows_out[1].measurements


# ============ Empty collections ============

def test_empty_measurements_via_parquet(tmp_path: Path) -> None:
    """A row with empty measurements must round-trip without
    losing the empty-vs-None distinction."""
    row = RunRow(
        id='r', parent_id=None, cycle_id=None,
        timestamp='t', verdict=Verdict.HELD,
    )
    path = tmp_path / 'runs.parquet'
    write_runrows([row], path)
    loaded = read_runrows(path)
    assert loaded == [row]


# ============ Dtype tightening + streaming reader ============

def test_tighten_trace_dtypes_casts_list_columns(tmp_path: Path) -> None:
    """`tighten_trace_dtypes` casts List(Float64) → List(Float32) and
    List(Int64) → List(Int32). Other column dtypes pass through
    unchanged."""
    import polars as pl

    from corroborate.persistence import tighten_trace_dtypes

    src = pl.DataFrame({
        'id': ['a', 'b'],
        'series_f64': [[1.0, 2.0], [3.0, 4.0]],
        'series_i64': [[1, 2], [3, 4]],
        'scalar_int': [10, 20],
    })
    src_path = tmp_path / 'src.parquet'
    src.write_parquet(src_path)

    tightened = tighten_trace_dtypes(pl.scan_parquet(src_path)).collect()
    assert tightened['series_f64'].dtype == pl.List(pl.Float32)
    assert tightened['series_i64'].dtype == pl.List(pl.Int32)
    # Scalar columns unchanged.
    assert tightened['scalar_int'].dtype == pl.Int64
    assert tightened['id'].dtype == pl.String
    # Values preserved across the cast.
    assert tightened['series_f64'].to_list() == [[1.0, 2.0], [3.0, 4.0]]
    assert tightened['series_i64'].to_list() == [[1, 2], [3, 4]]


def test_iter_trace_records_streams_one_dict_per_cell(
    tmp_path: Path,
) -> None:
    """`iter_trace_records` yields one polars row dict per cell.
    With column projection, only the named columns + 'id' surface.
    Memory-bounded by batch_size, not corpus size."""
    import polars as pl

    from corroborate.persistence import iter_trace_records

    src = pl.DataFrame({
        'id': ['a', 'b', 'c', 'd', 'e'],
        'series_a': [[1.0, 2.0]] * 5,
        'series_b': [[10, 20, 30]] * 5,
        'extra': ['x', 'y', 'z', 'q', 'w'],
    })
    src_path = tmp_path / 'src.parquet'
    src.write_parquet(src_path)

    # Stream all rows; verify ids in order.
    seen_ids: list[str] = []
    for record in iter_trace_records(src_path, batch_size=2):
        cid = record['id']
        assert isinstance(cid, str)
        seen_ids.append(cid)
    assert seen_ids == ['a', 'b', 'c', 'd', 'e']

    # Stream with projection — only named columns + 'id' surface.
    for record in iter_trace_records(
        src_path, columns=('series_a',), batch_size=10,
    ):
        assert set(record.keys()) == {'id', 'series_a'}


# ============ ComputationGraph sidecar round-trip ============

def test_graphs_sidecar_round_trips_topology(tmp_path: Path) -> None:
    """`write_graphs_sidecar` + `read_graphs_sidecar` recovers
    the same nodes + edges + edge metadata. Provenance survives
    a sweep — post-hoc consumers reconstruct the static call
    topology without re-running the trace pass."""
    from corroborate.graph.computation import (
        ComputationEdge, ComputationGraph,
    )
    from corroborate.graph import Graph
    from corroborate.persistence import (
        read_graphs_sidecar, write_graphs_sidecar,
    )

    g: ComputationGraph = Graph()
    g = g.with_node('claim_a')
    g = g.with_node('claim_b')
    g = g.with_node('claim_c')
    g = g.with_edge(
        'claim_a', 'claim_b',
        ComputationEdge(reader_arg='x', source_path=''),
    )
    g = g.with_edge(
        'claim_b', 'claim_c',
        ComputationEdge(reader_arg='y', source_path='value'),
    )

    p = tmp_path / 'graphs.json'
    write_graphs_sidecar({'arm_one': g, 'arm_two': g}, p)
    out = read_graphs_sidecar(p)
    assert set(out.keys()) == {'arm_one', 'arm_two'}
    for arm_key in out:
        recovered = out[arm_key]
        assert sorted(recovered.nodes) == ['claim_a', 'claim_b', 'claim_c']
        edges = sorted(
            (
                e.source, e.target,
                e.metadata.reader_arg, e.metadata.source_path,
            )
            for e in recovered.edges
        )
        assert edges == [
            ('claim_a', 'claim_b', 'x', ''),
            ('claim_b', 'claim_c', 'y', 'value'),
        ]


def test_graphs_sidecar_absent_file_returns_empty(
    tmp_path: Path,
) -> None:
    """Missing sidecar isn't an error — substrates that don't
    capture a graph (or didn't persist one) should return an
    empty mapping cleanly."""
    from corroborate.persistence import read_graphs_sidecar
    out = read_graphs_sidecar(tmp_path / 'absent.json')
    assert out == {}
