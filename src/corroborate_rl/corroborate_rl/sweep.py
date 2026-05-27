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

Optional `init_online_params_batched: Params` — per-seed
stacked online-param pytree consumed by `dqn`'s
`init_online_params` kwarg. Materialised at sweep-dispatch time
by `dispatch_sweep` when the sweep's
`init_q_checkpoint_path_template` is set.

Other keys are an error: HP variation lives in the substrate's
`base` (substrate's outer loop iterates HP regimes by building
distinct Hypothesis objects); cell-level exogenous variation
goes through env_name + seeds.

Per-arm parquet idempotency (skip-if-done) is handled by the
framework's `run_intervention` driver — not the runner's."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TypeIs

import jax

from corroborate_rl.cell_runner import run_dqn_arm
from corroborate_rl.dqn.claims.q_network import Params
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


def _is_params(v: object) -> TypeIs[Params]:
    """TypeIs narrowing `object` to `Params` (dict[str, jax.Array]).
    Used to validate `grid_point['init_online_params_batched']`
    before threading into the vmap. Empty-dict permitted (the
    sweep dispatcher only emits the key when ckpts are actually
    loaded; a defensive False at the boundary is enough)."""
    if not isinstance(v, dict):
        return False
    return all(
        isinstance(k, str) and isinstance(val, jax.Array)
        for k, val in v.items()
    )


class DQNRunner:
    """RL substrate's `Runner[DQNTrajectoryRecord]`. Holds the
    env catalogue so each call doesn't re-resolve env_specs from
    strings, and so the runner's identity is shared across grid
    points.

    `q_checkpoint_dir`: when set, every `__q_checkpoint__*` payload
    in the per-cell record gets persisted to a msgpack sidecar
    under this dir. The substrate's `dispatch_sweep` constructs the
    runner with this set to `<out_dir>/q_checkpoints/` per arm-
    config; library callers can leave it None (default).

    The runner maintains a `_call_count` to mirror the framework's
    `run_intervention` cell-index counter — each `__call__` is one
    `(grid_point, arm)` pair, the same granularity at which the
    parquet shards are named `cell{NNN}__...`. Cells inside one
    call (multi-seed vmap) share `cell_idx` and are disambiguated
    by `seed` in the checkpoint filename."""

    def __init__(
        self,
        env_catalogue: Mapping[str, EnvSpec],
        *,
        q_checkpoint_dir: Path | None = None,
    ) -> None:
        self._envs = env_catalogue
        self._q_checkpoint_dir = q_checkpoint_dir
        self._call_count = 0

    def reset_for_intervention(
        self, *, q_checkpoint_dir: Path | None = None,
    ) -> None:
        """Re-arm the runner for a fresh `run_intervention` call.

        The framework's `run_intervention` restarts its `cell_idx`
        counter from 0 per call (one call = one arm-config in
        `dispatch_sweep`'s loop). The runner mirrors that by
        zero-ing `_call_count` so the `cell_idx` the runner stamps
        on checkpoint filenames aligns with the framework's
        parquet-shard numbering.

        `q_checkpoint_dir` is rebound per call so each arm-config
        writes to its own `<out_dir>/<cfg.name>/q_checkpoints/`
        subdir — without this, two arm-configs would collide on
        `cell000_<seed>_*.msgpack`."""
        self._q_checkpoint_dir = q_checkpoint_dir
        self._call_count = 0

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
            set(grid_point)
            - {'env_name', 'seeds', 'wrappers',
               'init_online_params_batched'}
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
        init_params_raw = grid_point.get('init_online_params_batched')
        init_online_params_batched: Params | None
        if init_params_raw is None:
            init_online_params_batched = None
        elif _is_params(init_params_raw):
            init_online_params_batched = init_params_raw
        else:
            raise TypeError(
                f"DQNRunner: grid_point['init_online_params_batched'] "
                f"must be Params (dict[str, jax.Array]) or absent; "
                f"got {type(init_params_raw).__name__}",
            )
        env_spec = self._envs[env_name]

        cell_idx = self._call_count
        self._call_count += 1
        arm = run_dqn_arm(
            env_spec, seeds, claim, arm_key, measurables,
            wrappers=wrappers_v,
            q_checkpoint_dir=self._q_checkpoint_dir,
            cell_idx=cell_idx,
            init_online_params_batched=init_online_params_batched,
        )
        return SweepCellResult(
            runs=tuple(c.run for c in arm.cells),
            traces=tuple(c.trace for c in arm.cells),
            graph=arm.graph,
        )


__all__ = ['DQNRunner']
