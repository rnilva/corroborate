"""Sweep — exogenous-grid runner.

Substrate-agnostic primitive: takes a `Hypothesis[R]`, an
exogenous-variable grid (substrate-named keys × value lists), and
a `Runner[R]` that knows how to execute one cell. The framework
iterates the Cartesian product of the grid, collects per-cell
results from each runner call, and returns them with any failures.

The framework knows nothing about RL concepts (`env`, `seed`,
`total_steps`). Those are exogenous *names the substrate chose*.
A non-RL substrate sweeping over (`patient_id`, `dose`,
`measurement_day`) uses the same primitive — it just authors a
different `exogenous_grid` and a different `Runner`.

The contract:

- `exogenous_grid: Mapping[str, Sequence[object]]` — each key is
  a name the substrate chose; values are the levels to vary across
  cells. Cartesian product produces the cell list.
- `Runner[R]` — Protocol with `__call__(h, grid_point) ->
  SweepCellResult`. The substrate may implement Runner as a class
  with init state (e.g. RL substrate caches an env catalogue) or
  as a bare function. Both satisfy the Protocol.

`Runner.__call__` returns a `SweepCellResult` carrying the
per-seed RunRows + TraceRows + the captured ComputationGraph.
The graph is structurally constant across seeds in a vmap-batched
substrate and is the substrate's contribution to the
mechanism_key / redundancy primitives downstream.

Subprocess isolation (one process per cell) is deferred. v0 runs
in-process; large sweeps that need isolation can wrap this
primitive without changing the contract."""
from __future__ import annotations

import itertools
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from corroborate.computation_graph import ComputationGraph
from corroborate.graph import Graph
from corroborate.hypothesis import Hypothesis
from corroborate.schema import RunRow, TraceRow


@dataclass(frozen=True, slots=True)
class CellFailure:
    """One cell that raised during execution. Captures the
    grid-point values + the exception's str representation. Sweeps
    return failures alongside successful results so callers see
    the gap explicitly (no silent drops)."""
    intervention_name: str
    grid_point: Mapping[str, object]
    error: str
    duration_s: float


@dataclass(frozen=True, slots=True)
class SweepCellResult:
    """One runner-call's output: per-seed records + the
    Hypothesis-level graph captured during the call.

    Substrates that don't capture a graph (non-RL, or substrates
    without `@claim` records) emit an empty `Graph()`; the
    optionality is 'graph has nodes', not 'graph is None'."""
    runs: tuple[RunRow, ...]
    traces: tuple[TraceRow, ...]
    graph: ComputationGraph


@dataclass(frozen=True, slots=True)
class SweepResult:
    """Aggregated sweep output across all grid points. Failures
    captured alongside successes — the framework never silently
    drops a failed cell.

    `cell_results` preserves per-grid-point grouping; the
    `all_runs` / `all_traces` properties flatten when the consumer
    just wants the row collection."""
    cell_results: tuple[SweepCellResult, ...]
    failures: tuple[CellFailure, ...] = field(default_factory=tuple)

    @property
    def all_runs(self) -> tuple[RunRow, ...]:
        return tuple(r for cr in self.cell_results for r in cr.runs)

    @property
    def all_traces(self) -> tuple[TraceRow, ...]:
        return tuple(t for cr in self.cell_results for t in cr.traces)


class Runner[R: Mapping[str, object]](Protocol):
    """Substrate's bridge into corroborate.sweep. Receives one
    Hypothesis + one exogenous-grid point; returns a
    `SweepCellResult` with per-cell records + captured graph.

    Protocol (not a bare Callable alias) so substrates can hold
    init state — e.g. the RL runner caches the env catalogue and
    JIT-compiles once per arm, not once per grid point. Bare
    functions still satisfy via their implicit `__call__`.

    The grid_point is `Mapping[str, object]` rather than a typed
    dict — substrate runners type-narrow each key at the boundary.
    Parameterizing Runner over a typed grid shape would ripple
    through `sweep` and force users to declare a dataclass per
    substrate; not worth the cost."""
    def __call__(
        self,
        h: Hypothesis[R],
        grid_point: Mapping[str, object],
    ) -> SweepCellResult: ...


def sweep[R: Mapping[str, object]](
    h: Hypothesis[R],
    *,
    exogenous_grid: Mapping[str, Sequence[object]],
    runner: Runner[R],
) -> SweepResult:
    """Run `h` on each Cartesian point of `exogenous_grid`.
    Returns a `SweepResult` with `cell_results` (per-grid-point
    SweepCellResults) and `failures` (per-grid-point CellFailures
    for cells that raised).

    `exogenous_grid` keys are substrate-chosen. Iteration order
    follows `dict.items()` order; values are zipped via
    `itertools.product`. An empty grid (`{}`) runs the runner
    exactly once with an empty grid_point.

    Each `runner(h, grid_point)` call is wrapped in try/except —
    exceptions become `CellFailure` entries with the offending
    grid_point and the exception's string. The runner is
    responsible for the structure of the returned SweepCellResult
    (e.g. how many seeds it batches over)."""
    keys = list(exogenous_grid.keys())
    value_lists = [list(exogenous_grid[k]) for k in keys]
    if not keys:
        # Empty grid → one cell with empty grid_point.
        grid_points: list[dict[str, object]] = [{}]
    else:
        grid_points = [
            {k: v for k, v in zip(keys, point, strict=True)}
            for point in itertools.product(*value_lists)
        ]

    cell_results: list[SweepCellResult] = []
    failures: list[CellFailure] = []
    for grid_point in grid_points:
        t0 = time.monotonic()
        try:
            result = runner(h, grid_point)
        except Exception as exc:  # noqa: BLE001
            failures.append(CellFailure(
                intervention_name=h.name,
                grid_point=dict(grid_point),
                error=f'{type(exc).__name__}: {exc}',
                duration_s=time.monotonic() - t0,
            ))
            continue
        cell_results.append(result)
    return SweepResult(
        cell_results=tuple(cell_results),
        failures=tuple(failures),
    )


def empty_graph() -> ComputationGraph:
    """Convenience for substrates that don't capture a graph.
    Returns a fresh empty `Graph[str, ComputationEdge]`."""
    return Graph()
