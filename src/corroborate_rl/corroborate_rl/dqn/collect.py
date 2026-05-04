"""DQN substrate's per-env sweep config.

`EnvConfig` captures "how to sweep this env" (name + seeds +
chunk size + env wrappers). `_chunks` splits an EnvConfig's seed
range into vmap-able chunks; the substrate's `dispatch_sweep`
reads it.

Sweep orchestration + persistence + R2 archival is at the
framework level (`corroborate.runner.sweep.run_intervention`);
this module is just authoring helpers."""
from __future__ import annotations

from dataclasses import dataclass

from corroborate_rl.env_catalogue import EnvWrapper


@dataclass(frozen=True, slots=True)
class EnvConfig:
    """Per-env sweep config: env name, total seed count, and the
    vmap chunk size for memory management.

    `chunk_size = n_seeds` runs the whole arm in one vmap; smaller
    chunks split the arm into multiple grid points. Used when an
    env's obs-shape × capacity blows up the f32[cap, n_seeds, obs]
    replay tensor on the GPU.

    `wrappers` is a tuple of `EnvWrapper` instances applied in
    order to the gymnax env at sweep time."""
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


__all__ = ['EnvConfig']
