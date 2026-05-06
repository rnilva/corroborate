"""Tests for parquet round-trip across the row types.

Each row type has paired write/read functions; the test pattern
is: construct row → write to tmp parquet → read back → assert
equality. The persistence layer is flat columnar — every
provenance field and every measurement entry becomes its own
typed parquet column. Heterogeneous rows null-pad cleanly."""
from __future__ import annotations

from pathlib import Path

from corroborate.corpus.persistence import (
    read_runrows,
    write_runrows,
)
from corroborate.corpus.schema import RunRow
from corroborate.bridge.verdict import Verdict


# ============ Fixtures ============

def _sample_runrow() -> RunRow:
    return RunRow(
        id='run-1',
        parent_id=None,
        cycle_id='cycle-7',
        timestamp='2026-04-27T10:00:00Z',
        verdict=Verdict.HELD,
        arm_key='dqn_with_double_greedify',
        measurements={
            'env_name': 'CartPole-v1',
            'seed': 42,
            'total_steps': 30_000,
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
    values round-trip distinct from any open-surface measurement."""
    row = RunRow(
        id='r-1', parent_id=None,
        cycle_id=None, timestamp='t',
        verdict=Verdict.HELD,
        arm_key='bootstrap=Claim:double_greedify',
        measurements={'env_name': 'TestEnv'},
    )
    path = tmp_path / 'runs.parquet'
    write_runrows([row], path)
    loaded = read_runrows(path)
    assert len(loaded) == 1
    assert loaded[0].arm_key == 'bootstrap=Claim:double_greedify'
    assert loaded[0].measurements['env_name'] == 'TestEnv'
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

    from corroborate.corpus.persistence import tighten_trace_dtypes

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

    from corroborate.corpus.persistence import iter_trace_records

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
    from corroborate.corpus.persistence import (
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
    from corroborate.corpus.persistence import read_graphs_sidecar
    out = read_graphs_sidecar(tmp_path / 'absent.json')
    assert out == {}


# ============ stream_concat_parquets scratch-dir placement ============

def _write_tiny_parquet(path: Path, n_rows: int, salt: int) -> None:
    import polars as pl
    pl.DataFrame({
        'id': [f'r-{salt}-{i}' for i in range(n_rows)],
        'value': [float(salt + i) for i in range(n_rows)],
    }).write_parquet(str(path))


def test_stream_concat_scratch_defaults_to_out_parent(
    tmp_path: Path,
) -> None:
    """Regression for the merge-bug where the dispatcher's
    `stream_concat_parquets` silently failed with `ENOSPC` on
    `/tmp` (small overlay fs). The fix routes the chunked
    scratch to `out.parent` by default, putting it on the same
    filesystem the output is provisioned on. This test checks
    the placement contract: the scratch dir is created inside
    `out.parent`, not in the system tempfile dir."""
    from corroborate.corpus.persistence import stream_concat_parquets

    src_dir = tmp_path / 'src'
    out_dir = tmp_path / 'out'
    src_dir.mkdir()
    out_dir.mkdir()
    inputs = [src_dir / f'shard_{i:02d}.parquet' for i in range(8)]
    for i, p in enumerate(inputs):
        _write_tiny_parquet(p, n_rows=3, salt=i)

    seen_dirs: list[str] = []
    real_mkdtemp = __import__('tempfile').mkdtemp

    def spy_mkdtemp(prefix: str = 'tmp', dir: str | None = None) -> str:
        # Capture the dir kwarg so we can assert placement without
        # hooking the resulting Path itself (which gets cleaned up).
        seen_dirs.append(str(dir))
        return real_mkdtemp(prefix=prefix, dir=dir)

    import tempfile
    monkey = tempfile
    orig = monkey.mkdtemp
    monkey.mkdtemp = spy_mkdtemp
    try:
        stream_concat_parquets(
            inputs, out_dir / 'merged.parquet', chunk_size=3,
        )
    finally:
        monkey.mkdtemp = orig

    # 8 inputs at chunk_size=3 → ceil(8/3) = 3 chunks → recursive
    # call with 3 chunks ≤ 3 hits the small case (no temp dir).
    # So mkdtemp should have been called exactly once at the
    # outer level, with `dir=str(out_dir)`.
    assert len(seen_dirs) == 1, seen_dirs
    assert seen_dirs[0] == str(out_dir), (
        f'expected scratch in out.parent={out_dir!r}, got {seen_dirs[0]!r}'
    )
    assert (out_dir / 'merged.parquet').exists()


def test_stream_concat_explicit_scratch_dir_honored(
    tmp_path: Path,
) -> None:
    """Caller can override the scratch placement (e.g. point at a
    fast SSD even when the output lands on a slow archive disk)."""
    import tempfile
    from corroborate.corpus.persistence import stream_concat_parquets

    src_dir = tmp_path / 'src'
    out_dir = tmp_path / 'out'
    scratch_dir = tmp_path / 'fast_scratch'
    src_dir.mkdir()
    out_dir.mkdir()
    inputs = [src_dir / f'shard_{i:02d}.parquet' for i in range(8)]
    for i, p in enumerate(inputs):
        _write_tiny_parquet(p, n_rows=3, salt=i)

    seen_dirs: list[str] = []
    orig = tempfile.mkdtemp

    def spy_mkdtemp(prefix: str = 'tmp', dir: str | None = None) -> str:
        seen_dirs.append(str(dir))
        return orig(prefix=prefix, dir=dir)

    tempfile.mkdtemp = spy_mkdtemp
    try:
        stream_concat_parquets(
            inputs, out_dir / 'merged.parquet',
            chunk_size=3, scratch_dir=scratch_dir,
        )
    finally:
        tempfile.mkdtemp = orig

    assert seen_dirs == [str(scratch_dir)], seen_dirs
    assert scratch_dir.exists(), 'scratch_dir auto-created if missing'
    assert (out_dir / 'merged.parquet').exists()


# ============ I4: tmp+rename atomicity ============


def test_stream_concat_atomicity_partial_file_absent_after_success(
    tmp_path: Path,
) -> None:
    """**Invariant I4** (SWEEP_PERSISTENCY.md): writes to a
    `<out>.partial` sibling and rename atomically to `<out>` after
    the parquet write completes. After a successful merge, no
    `.partial` should remain at the consumer's path.

    Pre-fix: `stream_concat_parquets` wrote directly to `out`. A
    process crash mid-write left a torn parquet that downstream
    `pl.read_parquet` would fail on with `ComputeError`. Post-fix:
    the rename is filesystem-atomic; consumers either see the
    pre-merge state (`out` absent / pre-existing) or the
    post-merge state (`out` present, fully written), never the
    half-written intermediate."""
    import polars as pl
    from corroborate.corpus.persistence import stream_concat_parquets

    inputs = []
    for i in range(3):
        p = tmp_path / f'input_{i}.parquet'
        pl.DataFrame({'x': [i, i + 1]}).write_parquet(p)
        inputs.append(p)
    out = tmp_path / 'merged.parquet'
    stream_concat_parquets(inputs, out)

    # Final output exists and is readable.
    assert out.exists()
    df = pl.read_parquet(out)
    assert df.height == 6
    # No `.partial` sibling left over.
    assert not out.with_suffix(out.suffix + '.partial').exists(), (
        'Stale .partial file present after successful concat — '
        'tmp+rename cleanup broken.'
    )


def test_stream_concat_atomicity_simulated_crash_leaves_no_torn_parquet(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Simulate a crash mid-write by monkeypatching `replace` to
    raise BEFORE the rename. The `.partial` file is left in place;
    `out` is NOT created. A subsequent reader of `out` sees the
    pre-merge state (in this test: nothing), not a torn parquet.
    """
    import polars as pl
    from corroborate.corpus.persistence import stream_concat_parquets

    inputs = []
    for i in range(3):
        p = tmp_path / f'input_{i}.parquet'
        pl.DataFrame({'x': [i, i + 1]}).write_parquet(p)
        inputs.append(p)
    out = tmp_path / 'merged.parquet'

    # Stub the rename to raise — simulates a crash between write
    # and rename (e.g., process killed by oom-killer).
    real_replace = Path.replace

    def fake_replace(self: Path, target: Path) -> Path:
        raise RuntimeError('simulated crash mid-write')

    monkeypatch.setattr(Path, 'replace', fake_replace)
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match='simulated crash'):
        stream_concat_parquets(inputs, out)

    monkeypatch.setattr(Path, 'replace', real_replace)
    # Consumer's path is empty — no torn parquet visible.
    assert not out.exists(), (
        'Consumer-facing `out` should not exist when rename '
        'failed; it would expose a torn parquet to readers.'
    )
    # The .partial file IS present (recovery breadcrumb): a future
    # tooling pass can detect it and either retry or clean up. The
    # framework guarantees consumers don t see torn data, NOT that
    # disk is left tidy on crash.
    assert out.with_suffix(out.suffix + '.partial').exists()
