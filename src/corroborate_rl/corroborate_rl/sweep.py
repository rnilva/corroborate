"""RL substrate's bridge into `corroborate.sweep`.

Provides `DQNRunner` — a stateful `Runner[DQNTrajectoryRecord]`
caching the env catalogue. Each `__call__` receives a composed
Claim (the substrate's theory pre-bound with HPs and overlaid
with one arm's Intervention tuple), the framework-derived
`arm_key`, the typed Measurables, and one exogenous-grid point.

The runner reads exactly two keys from `grid_point`:
- `env_name: str` — required. Looked up in the catalogue.
- `seeds: tuple[int, ...]` — required. Vmap-batched inside
  `run_dqn_arm`.

Optional `wrappers: tuple[EnvWrapper, ...]` — env-augmentation
wrappers applied in order (e.g. ActionDuplicate for |A|
inflation experiments).

Other keys are an error: HP variation lives in the substrate's
`base` (substrate's outer loop iterates HP regimes by building
distinct Hypothesis objects); cell-level exogenous variation
goes through env_name + seeds.

Per-arm parquet idempotency (skip-if-done) is handled by the
framework's `run_intervention` driver — not the runner's."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeIs

from corroborate_rl.cell_runner import run_dqn_arm
from corroborate_rl.dqn.invariants import DQNTrajectoryRecord
from corroborate_rl.env_catalogue import EnvSpec
from corroborate.measurables import Measurable
from corroborate.runner.sweep import SweepCellResult


def _is_tuple_of_int(v: object) -> TypeIs[tuple[int, ...]]:
    """TypeIs narrowing `object` to `tuple[int, ...]`. Excludes
    `bool` per `int`-vs-`bool` subclass relationship."""
    return (
        isinstance(v, tuple)
        and all(isinstance(s, int) and not isinstance(s, bool) for s in v)
    )


class DQNRunner:
    """RL substrate's `Runner[DQNTrajectoryRecord]`. Holds the
    env catalogue so each call doesn't re-resolve env_specs from
    strings, and so the runner's identity is shared across grid
    points."""

    def __init__(self, env_catalogue: Mapping[str, EnvSpec]) -> None:
        self._envs = env_catalogue

    def __call__(
        self,
        claim: Callable[..., DQNTrajectoryRecord],
        arm_key: str,
        measurables: tuple[
            Measurable[DQNTrajectoryRecord, object], ...,
        ],
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
                f"DQNRunner: grid_point['seeds'] must be "
                f"tuple[int, ...], got {type(seeds_v).__name__}",
            )
        seeds = seeds_v
        unexpected = (
            set(grid_point) - {'env_name', 'seeds', 'wrappers'}
        )
        if unexpected:
            raise ValueError(
                f"DQNRunner: unexpected grid_point keys "
                f"{sorted(unexpected)} — HP variation belongs in the "
                f"substrate's `base`, not the grid",
            )
        wrappers_v = grid_point.get('wrappers', ())
        if not isinstance(wrappers_v, tuple):
            raise TypeError(
                f"DQNRunner: grid_point['wrappers'] must be tuple; "
                f"got {type(wrappers_v).__name__}",
            )
        env_spec = self._envs[env_name]

        arm = run_dqn_arm(
            env_spec, seeds, claim, arm_key, measurables,
            wrappers=wrappers_v,
        )
        return SweepCellResult(
            runs=tuple(c.run for c in arm.cells),
            traces=tuple(c.trace for c in arm.cells),
            graph=arm.graph,
        )


__all__ = ['DQNRunner']
