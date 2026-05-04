"""RL substrate's bridge into `corroborate.sweep`.

Provides `DQNRunner` (a stateful Runner[DQNTrajectoryRecord]
caching the env catalogue) and `run_dqn_sweep` (a convenience
that iterates one or more hypotheses through the framework's
sweep primitive).

The Hypothesis is the experiment specification — including any
HP-grid expansion. Substrate authors who want to sweep an HP
grid construct N hypotheses (one per grid point) at hypothesis-
construction time, where they own the mapping from "HP names"
(e.g. `capacity`) to dqn kwargs (`'replay': Replay(capacity=...)`).
The runner's contract is just (Hypothesis, env, seeds) →
SweepCellResult.

Exogenous-grid contract:

- `env_name: str` — the gymnax env to instantiate (resolved
  against the runner's catalogue).
- `seeds: tuple[int, ...]` — the seeds vmap-batched within one
  call. Wrapped as a single-element list when given to the
  framework's grid (`exogenous_grid['seeds'] = [seeds]`) so a
  single Cartesian point covers all seeds at once.

Returned `SweepCellResult.runs` carries one row per seed; the
graph is captured once per (hypothesis, env) call and is the
same across seeds (vmap traces the body once).

Per-arm parquet idempotency (skip-if-done) is the experiment
script's concern — not the runner's. `collect_ddqn_runs.py`
wraps `run_dqn_sweep` for that."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TypeIs

from corroborate.core.hypothesis import Hypothesis
from corroborate.rl.cell_runner import run_dqn_arm
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import EnvSpec
from corroborate.runner.sweep import (
    CellFailure,
    Runner,
    SweepCellResult,
    SweepResult,
    sweep,
)


def _is_tuple_of_int(v: object) -> TypeIs[tuple[int, ...]]:
    """TypeIs narrowing `object` to `tuple[int, ...]`. Excludes
    `bool` per `int`-vs-`bool` subclass relationship — `True`/
    `False` would otherwise pass an `isinstance(_, int)` check
    and corrupt seed-int consumers."""
    return (
        isinstance(v, tuple)
        and all(isinstance(s, int) and not isinstance(s, bool) for s in v)
    )


class DQNRunner:
    """RL substrate's `Runner[DQNTrajectoryRecord]`. Holds the
    env catalogue so each call doesn't re-resolve env_specs from
    strings, and so the runner's identity is shared across grid
    points (a future scheduler can introspect cache state via
    the runner instance).

    Reads exactly two keys from `grid_point`:

    - `env_name: str` — required. Looked up in the catalogue.
    - `seeds: tuple[int, ...]` — required. Vmap-batched inside
      `run_dqn_arm`.

    Other keys are an error — HP variation belongs in the
    Hypothesis (where the substrate author owns the
    HP-name→dqn-kwarg mapping)."""

    def __init__(self, env_catalogue: Mapping[str, EnvSpec]) -> None:
        self._envs = env_catalogue

    def __call__(
        self,
        h: Hypothesis[DQNTrajectoryRecord],
        grid_point: Mapping[str, object],
    ) -> SweepCellResult:
        env_name = grid_point['env_name']
        if not isinstance(env_name, str):
            raise TypeError(
                f"DQNRunner: grid_point['env_name'] must be str, "
                f"got {type(env_name).__name__}",
            )
        seeds_v = grid_point['seeds']
        if not _is_tuple_of_int(seeds_v):
            raise TypeError(
                f"DQNRunner: grid_point['seeds'] must be tuple[int, ...], "
                f"got {type(seeds_v).__name__}",
            )
        seeds = seeds_v
        unexpected = set(grid_point) - {'env_name', 'seeds', 'wrappers'}
        if unexpected:
            raise ValueError(
                f"DQNRunner: unexpected grid_point keys {sorted(unexpected)} "
                f"— HP variation belongs in the Hypothesis, not the grid",
            )
        wrappers_v = grid_point.get('wrappers', ())
        if not isinstance(wrappers_v, tuple):
            raise TypeError(
                f"DQNRunner: grid_point['wrappers'] must be tuple; "
                f"got {type(wrappers_v).__name__}",
            )
        env_spec = self._envs[env_name]

        arm = run_dqn_arm(env_spec, seeds, h, wrappers=wrappers_v)
        return SweepCellResult(
            runs=tuple(c.run for c in arm.cells),
            traces=tuple(c.trace for c in arm.cells),
            graph=arm.graph,
        )


def run_dqn_sweep(
    hypotheses: Sequence[Hypothesis[DQNTrajectoryRecord]],
    *,
    env_specs: Mapping[str, EnvSpec],
    seeds: tuple[int, ...],
) -> SweepResult:
    """Convenience: build a DQNRunner, sweep each hypothesis
    through the framework's `corroborate.sweep.sweep` primitive
    once per env, concatenate cell_results + failures.

    HP variation: the caller authors the full Cartesian product
    of hypotheses upfront. `hypotheses` carries one entry per
    (HP-grid point × intervention arm) combination."""
    runner: Runner[DQNTrajectoryRecord] = DQNRunner(env_specs)
    grid: dict[str, Sequence[object]] = {
        'env_name': list(env_specs.keys()),
        'seeds': [seeds],  # single grid value; all seeds vmap in run_dqn_arm
    }
    all_cells: list[SweepCellResult] = []
    all_failures: list[CellFailure] = []
    for h in hypotheses:
        result = sweep(h, exogenous_grid=grid, runner=runner)
        all_cells.extend(result.cell_results)
        all_failures.extend(result.failures)
    return SweepResult(
        cell_results=tuple(all_cells),
        failures=tuple(all_failures),
    )


__all__ = ['DQNRunner', 'run_dqn_sweep']
