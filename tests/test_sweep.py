"""Tests for `sweep` — substrate-agnostic exogenous-grid runner.

The sweep primitive iterates the Cartesian product of an
exogenous grid (substrate-named keys × value lists) and calls a
runner per grid point. The runner returns a SweepCellResult
(runs + traces + graph); exceptions become CellFailures.

The framework knows nothing about RL concepts. These tests use
generic exogenous keys (`group_id`, `replicate`) and synthetic
runners that build minimal SweepCellResults."""
from __future__ import annotations

import uuid
from collections.abc import Mapping

from corroborate.hypothesis import Hypothesis
from corroborate.schema import RunRow, TraceRow
from corroborate.sweep import (
    SweepCellResult,
    empty_graph,
    sweep,
)
from corroborate.verdict import Verdict


def _make_run(name: str, **measurements: object) -> RunRow:
    leaf_measurements: dict[str, str | int | float | bool] = {
        'intervention_name': name,
    }
    for k, v in measurements.items():
        if isinstance(v, (str, int, float, bool)):
            leaf_measurements[k] = v
    return RunRow(
        id=str(uuid.uuid4()), parent_id=None, cycle_id=None,
        timestamp='2026-04-28T00:00:00Z',
        verdict=Verdict.HELD, measurements=leaf_measurements,
    )


def _make_trace(rid: str) -> TraceRow:
    return TraceRow(
        id=rid, cycle_id=None,
        timestamp='2026-04-28T00:00:00Z',
        leaves={},
    )


# ============ Synthetic substrate runners ============

def _trivial_runner(
    h: Hypothesis[Mapping[str, object]],
    grid_point: Mapping[str, object],
) -> SweepCellResult:
    """One run + matching trace per grid point."""
    run = _make_run(h.name, **dict(grid_point))
    return SweepCellResult(
        runs=(run,), traces=(_make_trace(run.id),), graph=empty_graph(),
    )


def _batched_runner(
    h: Hypothesis[Mapping[str, object]],
    grid_point: Mapping[str, object],
) -> SweepCellResult:
    """N (run, trace) pairs per grid point — simulates a substrate
    that vmap-batches over a sub-axis."""
    n_inner = 3
    runs: list[RunRow] = []
    traces: list[TraceRow] = []
    for inner in range(n_inner):
        run = _make_run(h.name, inner_index=inner, **dict(grid_point))
        runs.append(run)
        traces.append(_make_trace(run.id))
    return SweepCellResult(
        runs=tuple(runs), traces=tuple(traces), graph=empty_graph(),
    )


def _failing_runner(
    h: Hypothesis[Mapping[str, object]],
    grid_point: Mapping[str, object],
) -> SweepCellResult:
    """Raises on a specific grid point; otherwise behaves like
    `_trivial_runner`."""
    if grid_point.get('replicate') == 0:
        raise RuntimeError('runner blew up on replicate 0')
    return _trivial_runner(h, grid_point)


# ============ Tests ============

def test_sweep_iterates_cartesian_product() -> None:
    """A 2 × 3 grid produces 6 cell calls; with one row per call
    that's 6 runs across 6 SweepCellResults."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={},
    )
    result = sweep(
        h,
        exogenous_grid={
            'group_id': ['a', 'b'],
            'replicate': [0, 1, 2],
        },
        runner=_trivial_runner,
    )
    assert len(result.cell_results) == 6
    assert len(result.all_runs) == 6
    assert len(result.failures) == 0


def test_sweep_runner_can_emit_multiple_rows() -> None:
    """Substrates that batch internally return multiple rows per
    grid point. all_runs flattens across cell_results."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={},
    )
    result = sweep(
        h,
        exogenous_grid={'env_label': ['x', 'y']},
        runner=_batched_runner,
    )
    # 2 grid points × 3 inner rows = 6 runs.
    assert len(result.cell_results) == 2
    assert len(result.all_runs) == 6
    assert len(result.all_traces) == 6
    inner_indices = sorted(
        int(r.measurements['inner_index']) for r in result.all_runs
    )
    assert inner_indices == [0, 0, 1, 1, 2, 2]


def test_sweep_empty_grid_runs_runner_once() -> None:
    """An empty grid means one cell with empty grid_point."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={},
    )
    result = sweep(
        h, exogenous_grid={}, runner=_trivial_runner,
    )
    assert len(result.cell_results) == 1


def test_sweep_failures_captured_with_grid_point() -> None:
    """Runner exceptions become CellFailures with the grid_point
    that caused them. Successful cells continue."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h_fail', intervention={},
    )
    result = sweep(
        h,
        exogenous_grid={'replicate': [0, 1, 2]},
        runner=_failing_runner,
    )
    # Replicate 0 fails; 1, 2 succeed.
    assert len(result.cell_results) == 2
    assert len(result.failures) == 1
    assert result.failures[0].grid_point == {'replicate': 0}
    assert result.failures[0].intervention_name == 'h_fail'
    assert 'RuntimeError' in result.failures[0].error


def test_sweep_runner_stamps_grid_point_into_measurements() -> None:
    """The framework does NOT auto-stamp grid values onto rows.
    Whether they appear is up to the substrate's runner."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={},
    )
    result = sweep(
        h,
        exogenous_grid={'group_id': ['alpha'], 'replicate': [42]},
        runner=_trivial_runner,
    )
    assert len(result.all_runs) == 1
    [row] = result.all_runs
    assert row.measurements['group_id'] == 'alpha'
    assert row.measurements['replicate'] == 42
    assert row.measurements['intervention_name'] == 'h'


# ============ Runner-as-Protocol: class-based runner satisfies ============

class _StatefulRunner:
    """Runner implemented as a class with init state. Satisfies
    the Runner Protocol via __call__."""
    def __init__(self, prefix: str) -> None:
        self._prefix = prefix
        self._calls = 0

    def __call__(
        self,
        h: Hypothesis[Mapping[str, object]],
        grid_point: Mapping[str, object],
    ) -> SweepCellResult:
        self._calls += 1
        run = _make_run(
            f'{self._prefix}_{h.name}', call_index=self._calls,
            **dict(grid_point),
        )
        return SweepCellResult(
            runs=(run,), traces=(_make_trace(run.id),),
            graph=empty_graph(),
        )


def test_sweep_accepts_class_based_runner_with_state() -> None:
    """A class implementing __call__(h, grid_point) -> SweepCellResult
    satisfies the Runner Protocol structurally; sweep accepts it."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={},
    )
    runner = _StatefulRunner(prefix='X')
    result = sweep(
        h,
        exogenous_grid={'replicate': [0, 1, 2]},
        runner=runner,
    )
    assert len(result.all_runs) == 3
    # The runner accumulated state across calls.
    assert runner._calls == 3
    # Calls 1, 2, 3 reflected in the runs (order matches grid iteration).
    indices = [int(r.measurements['call_index']) for r in result.all_runs]
    assert indices == [1, 2, 3]
