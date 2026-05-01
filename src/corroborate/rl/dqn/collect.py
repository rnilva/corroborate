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

from corroborate.hypothesis import Hypothesis
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Per-env sweep config: env name, total seed count, and the
    vmap chunk size for memory management.

    `chunk_size = n_seeds` runs the whole arm in one vmap; smaller
    chunks split the arm into multiple grid points (each becomes
    its own arm at the framework level). Used when an env's
    obs-shape × capacity blows up the f32[cap, n_seeds, obs]
    replay tensor on the GPU.

    `reward_scale != 1.0` wraps the env in `RewardScaledEnv` —
    multiplies env reward by `scale` at every step, so MC
    variance scales by `scale²` while dynamics, |A|, obs_dim, and
    optimal policy are unchanged. Causal-probe lever for
    moderator hypotheses (the env-level `log_mc_variance →
    g_link` attenuation in particular).

    `reward_clip_min` / `reward_clip_max` (default None for both)
    wrap in `RewardClippedEnv` — clips step reward to the given
    bounds. Different intervention from scaling: clipping
    CHANGES the optimal policy (no longer needs to weigh
    clipped-side outcomes), so this is a probe of whether DDQN's
    behavioral pattern depends on the unclipped reward
    structure (e.g. SpaceInvaders' negative hit-penalty)."""
    env_name: str
    n_seeds: int = 30
    chunk_size: int = 30
    reward_scale: float = 1.0
    reward_clip_min: float | None = None
    reward_clip_max: float | None = None


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
             'reward_scale': ec.reward_scale,
             'reward_clip_min': ec.reward_clip_min,
             'reward_clip_max': ec.reward_clip_max})
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
             'reward_scale': ec.reward_scale,
             'reward_clip_min': ec.reward_clip_min,
             'reward_clip_max': ec.reward_clip_max})
        for h, ec in zip(hypotheses, env_configs_aligned, strict=True)
        for chunk in _chunks(ec)
    ]


def env_arm_tag(
    h: Hypothesis[DQNTrajectoryRecord],
    grid_point: Mapping[str, object],
) -> str:
    """Default arm_tag for DQN sweeps: `{env_name}__{h.name}` with
    a `__rs={scale}` suffix when `reward_scale != 1.0` and
    `__rclip[{min},{max}]` suffix when reward clipping is set.
    The suffixes keep arm identity unique when the same env runs
    at multiple reward configurations (causal-probe sweeps)."""
    env_name = grid_point.get('env_name', '')
    reward_scale = grid_point.get('reward_scale', 1.0)
    rs_suffix = (
        f'__rs={reward_scale}'
        if isinstance(reward_scale, (int, float))
            and float(reward_scale) != 1.0
        else ''
    )
    clip_min = grid_point.get('reward_clip_min')
    clip_max = grid_point.get('reward_clip_max')
    if clip_min is not None or clip_max is not None:
        clip_suffix = f'__rclip[{clip_min},{clip_max}]'
    else:
        clip_suffix = ''
    return f'{env_name}__{h.name}{rs_suffix}{clip_suffix}'


__all__ = [
    'EnvConfig', 'chunked_arms', 'paired_arms', 'env_arm_tag',
]
