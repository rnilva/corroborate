"""Tests for `sweep` — substrate-agnostic exogenous-grid runner.

The sweep primitive iterates the Cartesian product of an
exogenous grid (substrate-named keys × value lists) and calls a
runner per grid point. The runner returns one or more RunRows;
exceptions become CellFailures.

The framework knows nothing about RL concepts. These tests use
generic exogenous keys (`group_id`, `replicate`) and a synthetic
runner that builds minimal RunRows."""
from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence

from corroborate.hypothesis import Hypothesis
from corroborate.schema import RunRow
from corroborate.sweep import sweep
from corroborate.verdict import Verdict


# ============ Synthetic substrate runners ============

def _trivial_runner(
    h: Hypothesis[Mapping[str, object]],
    grid_point: Mapping[str, object],
) -> Sequence[RunRow]:
    """Returns one RunRow per grid point, stamping the grid values
    into measurements. Substrate-side concern — the framework does
    not pre-stamp anything."""
    measurements: dict[str, str | int | float | bool] = {
        'intervention_name': h.name,
    }
    for k, v in grid_point.items():
        if isinstance(v, (str, int, float, bool)):
            measurements[k] = v
    return [
        RunRow(
            id=str(uuid.uuid4()), parent_id=None, cycle_id=None,
            timestamp='2026-04-28T00:00:00Z',
            verdict=Verdict.HELD, measurements=measurements,
        ),
    ]


def _batched_runner(
    h: Hypothesis[Mapping[str, object]],
    grid_point: Mapping[str, object],
) -> Sequence[RunRow]:
    """Returns N RunRows per grid point — simulates a substrate
    that vmap-batches over a sub-axis (e.g. RL substrate batching
    seeds within one env)."""
    n_inner = 3
    rows: list[RunRow] = []
    for inner in range(n_inner):
        measurements: dict[str, str | int | float | bool] = {
            'intervention_name': h.name,
            'inner_index': inner,
        }
        for k, v in grid_point.items():
            if isinstance(v, (str, int, float, bool)):
                measurements[k] = v
        rows.append(RunRow(
            id=str(uuid.uuid4()), parent_id=None, cycle_id=None,
            timestamp='2026-04-28T00:00:00Z',
            verdict=Verdict.HELD, measurements=measurements,
        ))
    return rows


def _failing_runner(
    h: Hypothesis[Mapping[str, object]],
    grid_point: Mapping[str, object],
) -> Sequence[RunRow]:
    """Raises on a specific grid point; otherwise behaves like
    `_trivial_runner`. Tests failure capture + provenance."""
    if grid_point.get('replicate') == 0:
        raise RuntimeError('runner blew up on replicate 0')
    return _trivial_runner(h, grid_point)


# ============ Tests ============

def test_sweep_iterates_cartesian_product() -> None:
    """A 2 × 3 grid produces 6 cell calls; with one row per call
    that's 6 rows."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(),
    )
    rows, failures = sweep(
        h,
        exogenous_grid={
            'group_id': ['a', 'b'],
            'replicate': [0, 1, 2],
        },
        runner=_trivial_runner,
    )
    assert len(rows) == 6
    assert len(failures) == 0


def test_sweep_runner_can_emit_multiple_rows() -> None:
    """Substrates that batch internally (vmap over a sub-axis)
    return multiple RunRows per grid point. The sweep flattens."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(),
    )
    rows, _ = sweep(
        h,
        exogenous_grid={'env_label': ['x', 'y']},
        runner=_batched_runner,
    )
    # 2 grid points × 3 inner rows = 6 rows.
    assert len(rows) == 6
    inner_indices = sorted(int(r.measurements['inner_index']) for r in rows)
    assert inner_indices == [0, 0, 1, 1, 2, 2]


def test_sweep_empty_grid_runs_runner_once() -> None:
    """An empty grid means one cell with empty grid_point."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(),
    )
    rows, _ = sweep(
        h, exogenous_grid={}, runner=_trivial_runner,
    )
    assert len(rows) == 1


def test_sweep_failures_captured_with_grid_point() -> None:
    """Runner exceptions become CellFailures with the grid_point
    that caused them."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h_fail', intervention={}, bridges=(),
    )
    rows, failures = sweep(
        h,
        exogenous_grid={'replicate': [0, 1, 2]},
        runner=_failing_runner,
    )
    # Replicate 0 fails; 1, 2 succeed.
    assert len(rows) == 2
    assert len(failures) == 1
    assert failures[0].grid_point == {'replicate': 0}
    assert failures[0].intervention_name == 'h_fail'
    assert 'RuntimeError' in failures[0].error


def test_sweep_runner_stamps_grid_point_into_measurements() -> None:
    """The framework does NOT auto-stamp grid values onto rows.
    Whether they appear is up to the substrate's runner. Verify
    the synthetic runner's stamping works as expected."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='h', intervention={}, bridges=(),
    )
    rows, _ = sweep(
        h,
        exogenous_grid={'group_id': ['alpha'], 'replicate': [42]},
        runner=_trivial_runner,
    )
    [row] = rows
    assert row.measurements['group_id'] == 'alpha'
    assert row.measurements['replicate'] == 42
    assert row.measurements['intervention_name'] == 'h'
