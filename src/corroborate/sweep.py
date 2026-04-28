"""Sweep — exogenous-grid runner.

Substrate-agnostic primitive: takes a `Hypothesis[R]`, an
exogenous-variable grid (substrate-named keys × value lists), and
a `runner` that knows how to execute one cell. The framework
iterates the Cartesian product of the grid, collects RunRows from
each runner call, and returns them with any failures.

The framework knows nothing about RL concepts (`env`, `seed`,
`total_steps`). Those are exogenous *names the substrate chose*.
A non-RL substrate sweeping over (`patient_id`, `dose`,
`measurement_day`) uses the same primitive — it just authors a
different `exogenous_grid` and a different `runner`.

The contract:

- `exogenous_grid: Mapping[str, Sequence[object]]` — each key is
  a name the substrate chose; values are the levels to vary across
  cells. Cartesian product produces the cell list.
- `runner: Callable[[Hypothesis[R], Mapping[str, object]],
   Sequence[RunRow]]` — receives one grid point and the
  hypothesis. Returns one or more RunRows (substrates that
  vmap-batch internally over a sub-axis emit multiple rows per
  call). Exceptions are caught and recorded as `CellFailure`.

The sweep does NOT add measurements to the rows: the runner is
responsible for stamping the grid-point values onto its emitted
RunRows. This keeps the framework agnostic to which exogenous
keys exist, AND lets the substrate decide which keys belong on
provenance vs. measurements.

Subprocess isolation (one process per cell) is deferred. v0 runs
in-process; large sweeps that need isolation can wrap this
primitive without changing the contract."""
from __future__ import annotations

import itertools
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from corroborate.hypothesis import Hypothesis
from corroborate.schema import RunRow


@dataclass(frozen=True, slots=True)
class CellFailure:
    """One cell that raised during execution. Captures the
    grid-point values + the exception's str representation. Sweeps
    return failures alongside successful rows so callers see the
    gap explicitly (no silent drops)."""
    intervention_name: str
    grid_point: Mapping[str, object]
    error: str
    duration_s: float


type Runner[R] = Callable[
    [Hypothesis[R], Mapping[str, object]],
    Sequence[RunRow],
]


def sweep[R: Mapping[str, object]](
    h: Hypothesis[R],
    *,
    exogenous_grid: Mapping[str, Sequence[object]],
    runner: Runner[R],
) -> tuple[list[RunRow], list[CellFailure]]:
    """Run `h` on each Cartesian point of `exogenous_grid`. Returns
    `(successful_rows, failures)`.

    `exogenous_grid` keys are substrate-chosen. Iteration order
    follows `dict.items()` order; values are zipped via
    `itertools.product`. An empty grid (`{}`) runs the runner
    exactly once with an empty grid_point.

    Each runner call may return MULTIPLE RunRows (substrates that
    vmap a batch axis internally — e.g. RL substrate batching over
    seeds). All rows are flattened into the returned list."""
    keys = list(exogenous_grid.keys())
    value_lists = [list(exogenous_grid[k]) for k in keys]
    if not keys:
        # Empty grid → one cell with empty grid_point.
        cells = [({}, )]
    else:
        cells = [
            ({k: v for k, v in zip(keys, point, strict=True)},)
            for point in itertools.product(*value_lists)
        ]

    rows: list[RunRow] = []
    failures: list[CellFailure] = []
    for (grid_point,) in cells:
        t0 = time.monotonic()
        try:
            cell_rows = runner(h, grid_point)
        except Exception as exc:  # noqa: BLE001
            failures.append(CellFailure(
                intervention_name=h.name,
                grid_point=dict(grid_point),
                error=f'{type(exc).__name__}: {exc}',
                duration_s=time.monotonic() - t0,
            ))
            continue
        rows.extend(cell_rows)
    return rows, failures
