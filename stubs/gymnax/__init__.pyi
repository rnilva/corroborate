"""Minimal gymnax stubs — opaque Env / EnvParams surface.

Why this stub exists: gymnax does not ship type stubs upstream. Its
real signatures use a giant `EnvParams` union and Box/Discrete
unions for spaces; both leak `Unknown` / `Any` through the framework
boundary even though the corroborate-side contract is small. Rather
than relax pyright globally for `corroborate.rl`, we narrow at the
boundary with this stub: only the surface `corroborate.rl.dqn`
actually touches.

Mirrors poc_v9's `poc_v8/stubs/gymnax/__init__.pyi` — same shape,
same scope. Expand only when a new gymnax function is reached for."""
from __future__ import annotations

import jax
import jax.numpy as jnp


class EnvParams:
    """Opaque environment-parameter container."""
    max_steps_in_episode: int
    ...


class Space:
    """Opaque action / observation space.

    Real gymnax has Box / Discrete subclasses; we collapse to a
    single duck-typed surface (.shape for obs, .n for discrete
    actions) because that's all the framework consumes."""
    shape: tuple[int, ...]
    n: int


class Env:
    """Opaque gymnax environment."""
    def observation_space(self, params: EnvParams) -> Space: ...
    def action_space(self, params: EnvParams) -> Space: ...
    def reset(
        self,
        rng: jax.Array,
        params: EnvParams,
    ) -> tuple[jnp.ndarray, EnvParams]: ...
    def step(
        self,
        rng: jax.Array,
        state: EnvParams,
        action: jnp.ndarray,
        params: EnvParams,
    ) -> tuple[
        jnp.ndarray,        # next_obs
        EnvParams,          # next_state
        jnp.ndarray,        # reward
        jnp.ndarray,        # done
        dict[str, object],  # info
    ]: ...


def make(env_id: str, **env_kwargs: object) -> tuple[Env, EnvParams]: ...
