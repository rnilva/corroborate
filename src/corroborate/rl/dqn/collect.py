"""DQN convenience wrapper over the framework's `run_hypotheses`.

`run_hypotheses` (in `corroborate.sweep`) is the framework-level
`do()` operator at the corpus level: takes hypotheses + an
exogenous grid + a Runner and materializes a corpus.

This module's `collect_sweep_to_parquet` is a thin DQN-specific
wrapper that:

- Builds a chunked exogenous grid from `EnvConfig`s (each chunk
  of `chunk_size` seeds becomes one grid point).
- Wires up `DQNRunner` automatically.
- Defaults `trace_reductions` to the canonical `Q_TRACE_REDUCTIONS`
  (per-step Q vector max/min/mean/std/argmax) and `trace_drops` to
  the high-cardinality 2-D Q-tensor columns.

Each (hypothesis, env_config-chunk) is one arm = one runner call =
one parquet pair. The framework's `run_hypotheses` handles the
per-arm loop, optional R2 archive, and final corpus merge."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from corroborate.hypothesis import Hypothesis
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.dqn.trace_reductions import (
    Q_TRACE_DROPS, Q_TRACE_REDUCTIONS,
)
from corroborate.rl.env_catalogue import EnvSpec
from corroborate.rl.sweep import DQNRunner
from corroborate.sweep import run_hypotheses


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Per-env sweep config: env name, total seed count, and the
    vmap chunk size for memory management.

    `chunk_size = n_seeds` runs the whole arm in one vmap; smaller
    chunks split the arm into multiple grid points (each becomes
    its own arm at the framework level). Used when an env's
    obs-shape × capacity blows up the f32[cap, n_seeds, obs]
    replay tensor on the GPU."""
    env_name: str
    n_seeds: int = 30
    chunk_size: int = 30


def collect_sweep_to_parquet(
    hypotheses: Sequence[Hypothesis[DQNTrajectoryRecord]],
    env_configs: Sequence[EnvConfig],
    *,
    out_dir: Path,
    env_specs: Mapping[str, EnvSpec] | None = None,
    trace_reductions: Sequence[pl.Expr] = Q_TRACE_REDUCTIONS,
    trace_drops: Sequence[str] = Q_TRACE_DROPS,
    arm_tag: Callable[[Hypothesis[DQNTrajectoryRecord], Mapping[str, object]], str] | None = None,
    archive_remote: str | None = None,
) -> tuple[Path, Path]:
    """Run all (hypothesis × env_config) arms, persist per-arm,
    concatenate to `out_dir/{runs,traces}.parquet`. Returns the
    two final paths.

    `env_specs` defaults to looking up each env_config.env_name via
    `corroborate.rl.env_catalogue.get`; pass explicitly to share a
    pre-resolved catalogue across multiple `collect_*` calls.

    `trace_reductions` / `trace_drops` default to the canonical
    `Q_TRACE_REDUCTIONS` / `Q_TRACE_DROPS` (per-step Q reductions
    + drop the 2-D `online_q_per_action` / `target_q_per_action`).
    Override for substrates with different traces.

    `arm_tag` produces the filename suffix for each arm's parquets.
    Default: `{env_name}__{hypothesis.name}` — uniquely identifies
    arms within a sweep but not across HP grid points; callers
    that vary HPs through hypothesis_count should override to
    encode the grid point in the tag.

    `archive_remote`, `arm_tag`, and `out_dir` are forwarded to
    `corroborate.sweep.run_hypotheses`.

    Each `EnvConfig` is exploded into one or more grid points by
    chunking its seeds: `chunk_size = 15` on `n_seeds = 30` yields
    two grid points per env, each becoming its own arm at the
    framework level."""
    from corroborate.rl.env_catalogue import get as _get_spec

    if env_specs is None:
        env_specs = {
            ec.env_name: _get_spec(ec.env_name)
            for ec in env_configs
        }
    runner = DQNRunner(env_specs)

    grid_per_arm: list[Mapping[str, object]] = []
    for ec in env_configs:
        seeds = tuple(range(ec.n_seeds))
        for i in range(0, len(seeds), ec.chunk_size):
            grid_per_arm.append({
                'env_name': ec.env_name,
                'seeds': seeds[i:i + ec.chunk_size],
            })

    if arm_tag is None:
        def arm_tag_default(
            h: Hypothesis[DQNTrajectoryRecord],
            grid_point: Mapping[str, object],
        ) -> str:
            env_name = grid_point.get('env_name', '')
            return f'{env_name}__{h.name}'
        effective_arm_tag = arm_tag_default
    else:
        effective_arm_tag = arm_tag

    return run_hypotheses(
        hypotheses,
        grid_per_arm=grid_per_arm,
        runner=runner,
        out_dir=out_dir,
        archive_remote=archive_remote,
        arm_tag=effective_arm_tag,
        trace_reductions=trace_reductions,
        trace_drops=trace_drops,
    )


__all__ = ['EnvConfig', 'collect_sweep_to_parquet']
