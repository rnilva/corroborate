"""Jumanji → gymnax env-interface adapter.

Jumanji uses a dm_env-style API (`env.reset(key) → (state, timestep)`,
`env.step(state, action) → (state, timestep)` with no key on step
because state carries the RNG); the corroborate_rl substrate calls
the gymnax-style API
(`env.reset(rng, params) → (obs, state)`,
`env.step(rng, state, action, params) → (obs, state, reward, done, info)`).
This module bridges the two so jumanji envs flow through `cell_runner`
unchanged. No type erasure — each registered jumanji env declares
its own typed obs extractor (closure typed by jumanji's per-env
`Observation` NamedTuple).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Generic, TypeVar

import jax
import jax.numpy as jnp
from gymnax.environments import environment as gymnax_env
from gymnax.environments import spaces as gymnax_spaces

if TYPE_CHECKING:
    from jumanji.env import Environment as JumanjiEnvironment


# Per-env Observation / State are jumanji-defined NamedTuples /
# `flax.struct.dataclass`es; we keep them opaque via TypeVars and
# carry the typed obs-extractor as a closure parameter.
ObsT = TypeVar('ObsT')
StateT = TypeVar('StateT')


@dataclass(frozen=True, slots=True)
class JumanjiEnv(Generic[ObsT, StateT]):
    """Adapt a jumanji Environment to the gymnax Env protocol surface.

    `inner` is the wrapped jumanji env. `obs_extract` projects
    jumanji's structured Observation NamedTuple to a flat
    `jax.Array` (typically `lambda obs: obs.grid` for grid-shaped
    envs). `n_actions` and `obs_shape` are pre-computed at
    registration time so `action_space` / `observation_space` don't
    need to re-introspect the jumanji spec on every call.

    The `rng` argument to `step` is unused (jumanji's State carries
    its own key, advanced internally by `inner.step`); the `params`
    argument is unused (jumanji bakes config in at env construction
    time). Both are accepted to satisfy the gymnax `Env` Protocol.
    """
    inner: JumanjiEnvironment
    obs_extract: Callable[[ObsT], jax.Array]
    obs_shape: tuple[int, ...]
    n_actions: int

    def reset(
        self, rng: jax.Array, params: gymnax_env.EnvParams,
    ) -> tuple[jax.Array, StateT]:
        del params  # jumanji bakes params into env construction
        state, ts = self.inner.reset(rng)
        return self.obs_extract(ts.observation), state

    def step(
        self,
        rng: jax.Array,
        state: StateT,
        action: jax.Array,
        params: gymnax_env.EnvParams,
    ) -> tuple[
        jax.Array, StateT, jax.Array, jax.Array, dict[str, object],
    ]:
        del params
        next_state, ts = self.inner.step(state, action.astype(jnp.int32))
        # dm_env step_type==2 is LAST (terminal). Jumanji also sets
        # discount to 0 on terminal — either is canonical; step_type
        # is the documented invariant.
        done = ts.step_type == 2
        # Auto-reset on done to match gymnax env contract — jumanji
        # itself doesn't auto-reset, so without this the env stays in
        # a permanently-terminal state for the rest of training (the
        # `done` flag stays 1 forever after the first episode ends,
        # producing degenerate TD targets).
        reset_state, reset_ts = self.inner.reset(rng)
        final_state = jax.tree.map(
            lambda r, n: jnp.where(done, r, n),
            reset_state, next_state,
        )
        next_obs = self.obs_extract(ts.observation)
        reset_obs = self.obs_extract(reset_ts.observation)
        final_obs = jnp.where(done, reset_obs, next_obs)
        return (
            final_obs,
            final_state,
            ts.reward,
            done,
            {},
        )

    def action_space(
        self, params: gymnax_env.EnvParams,
    ) -> gymnax_spaces.Discrete:
        del params
        return gymnax_spaces.Discrete(num_categories=self.n_actions)

    def observation_space(
        self, params: gymnax_env.EnvParams,
    ) -> gymnax_spaces.Box:
        del params
        return gymnax_spaces.Box(
            low=0.0, high=1.0,
            shape=self.obs_shape, dtype=jnp.float32,
        )
