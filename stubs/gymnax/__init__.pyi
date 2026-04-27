"""Minimal gymnax stubs — opaque Env / EnvParams surface.

Why this stub exists: gymnax does not ship type stubs upstream. Its
real signatures use a giant `EnvParams` union and Box/Discrete
unions for spaces; both leak `Unknown` / `Any` through the framework
boundary even though the corroborate-side contract is small. Rather
than relax pyright globally for `corroborate.rl`, we narrow at the
boundary with this stub: only the surface `corroborate.rl.dqn`
actually touches.

`Env` methods accept `params: object` and return `state: object`
(rather than the runtime `EnvParams` subclass) so the stub's `Env`
satisfies corroborate's `GymnaxEnvLike` Protocol contravariantly.
The runtime `env_params` is still an `EnvParams` instance — the
loose stub just doesn't constrain that distinction at the type
layer (which the framework treats as opaque anyway)."""
from __future__ import annotations

import jax
import jax.numpy as jnp


class EnvParams:
    """Opaque environment-parameter container. No attrs declared
    here — `max_steps_in_episode` exists on most env params but
    not all (bandits lack it), so corroborate's
    `MaxStepsParams` Protocol does the runtime check via
    `isinstance` to narrow honestly."""
    ...


class Space:
    """Opaque action / observation space. No attrs declared —
    real gymnax has Box (`.shape`) and Discrete (`.n`) subclasses;
    corroborate's `HasShape` / `HasN` Protocols narrow at the
    use site rather than the stub conflating both surfaces."""
    ...


class Env:
    """Opaque gymnax environment. Methods take `params: object`
    and return opaque `state: object` so this class satisfies
    corroborate's `GymnaxEnvLike` structural Protocol."""
    def observation_space(self, params: object) -> Space: ...
    def action_space(self, params: object) -> Space: ...
    def reset(
        self,
        rng: jax.Array,
        params: object,
    ) -> tuple[jax.Array, object]: ...
    def step(
        self,
        rng: jax.Array,
        state: object,
        action: jax.Array,
        params: object,
    ) -> tuple[
        jax.Array,           # next_obs
        object,              # next_state
        jax.Array,           # reward
        jax.Array,           # done
        dict[str, object],   # info
    ]: ...


def make(env_id: str, **env_kwargs: object) -> tuple[Env, EnvParams]: ...
