"""DQN substrate's per-env sweep config + arm authoring helpers.

`EnvConfig` captures "how to sweep this env" (name + seeds +
chunk size). `chunked_arms` builds the Cartesian list of
`(Hypothesis, grid_point)` pairs from `(hypotheses, env_configs)`
with seeds chunked per env's chunk_size. `paired_arms` builds
`zip`-paired (h, env_config) for the case where each hypothesis
is bound to one env (e.g. CNN.obs_shape varies).

Sweep orchestration + persistence + R2 archival is at the
framework level (`corroborate.sweep.run_hypotheses`); this
module is just authoring helpers.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from corroborate.core.hypothesis import Hypothesis
from corroborate_rl.dqn.invariants import DQNTrajectoryRecord
from corroborate_rl.env_catalogue import EnvWrapper


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Per-env sweep config: env name, total seed count, and the
    vmap chunk size for memory management.

    `chunk_size = n_seeds` runs the whole arm in one vmap; smaller
    chunks split the arm into multiple grid points (each becomes
    its own arm at the framework level). Used when an env's
    obs-shape × capacity blows up the f32[cap, n_seeds, obs]
    replay tensor on the GPU.

    `wrappers` is a tuple of `EnvWrapper` instances applied in
    order to the gymnax env at sweep time. Replaces ad-hoc
    per-wrapper fields (reward_scale, reward_clip_min, etc.):
    each new env transformation just registers a wrapper class
    and gets a YAML entry — no 7-place plumbing per intervention.
    Use `RewardScale(scale=0.1)`, `RewardClip(clip_min=0.0)`,
    or any other registered wrapper."""
    env_name: str
    n_seeds: int = 30
    chunk_size: int = 30
    wrappers: tuple[EnvWrapper, ...] = ()


def _chunks(ec: EnvConfig) -> list[tuple[int, ...]]:
    """Seed range split into chunk_size-sized tuples."""
    seeds = tuple(range(ec.n_seeds))
    return [
        seeds[i:i + ec.chunk_size]
        for i in range(0, ec.n_seeds, ec.chunk_size)
    ]


def chunked_arms(
    hypotheses: Sequence[Hypothesis[DQNTrajectoryRecord]],
    env_configs: Sequence[EnvConfig],
) -> list[tuple[Hypothesis[DQNTrajectoryRecord], Mapping[str, object]]]:
    """Cartesian arms × chunked seeds. Produces one arm per
    (h, env, seed_chunk) triple, suitable for
    `run_hypotheses(arms=...)`."""
    return [
        (h, {'env_name': ec.env_name, 'seeds': chunk,
             'wrappers': ec.wrappers})
        for h in hypotheses
        for ec in env_configs
        for chunk in _chunks(ec)
    ]


def paired_arms(
    hypotheses: Sequence[Hypothesis[DQNTrajectoryRecord]],
    env_configs_aligned: Sequence[EnvConfig],
) -> list[tuple[Hypothesis[DQNTrajectoryRecord], Mapping[str, object]]]:
    """Zip-paired arms × chunked seeds. Each hypothesis pairs
    with one env_config (e.g. a CNN configured for that env's
    obs_shape). `len(hypotheses) == len(env_configs_aligned)`
    required; mismatch raises ValueError."""
    if len(hypotheses) != len(env_configs_aligned):
        raise ValueError(
            f'paired_arms: hypotheses ({len(hypotheses)}) and '
            f'env_configs_aligned ({len(env_configs_aligned)}) '
            f'must match length.',
        )
    return [
        (h, {'env_name': ec.env_name, 'seeds': chunk,
             'wrappers': ec.wrappers})
        for h, ec in zip(hypotheses, env_configs_aligned, strict=True)
        for chunk in _chunks(ec)
    ]


def env_arm_tag(
    h: Hypothesis[DQNTrajectoryRecord],
    grid_point: Mapping[str, object],
) -> str:
    """Default arm_tag for DQN sweeps: `{env_name}__{h.name}` with
    a `__wrap[<canonical>]` suffix when env wrappers are
    configured. The suffix keeps arm identity unique when the
    same env runs under different wrapper compositions
    (causal-probe sweeps)."""
    from corroborate_rl.env_catalogue import wrappers_canonical_str
    env_name = grid_point.get('env_name', '')
    wrappers = grid_point.get('wrappers', ())
    suffix = (
        f'__wrap[{wrappers_canonical_str(wrappers)}]'
        if isinstance(wrappers, tuple) and wrappers
        else ''
    )
    return f'{env_name}__{h.name}{suffix}'


__all__ = [
    'EnvConfig', 'chunked_arms', 'paired_arms', 'env_arm_tag',
]
