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

    def reset_env(
        self, rng: jax.Array, params: gymnax_env.EnvParams,
    ) -> tuple[jax.Array, StateT]:
        # Jumanji's `reset` is the no-auto-reset reset (jumanji
        # doesn't auto-reset elsewhere); the gymnax-side
        # `reset_env` Protocol method matches `reset` here.
        return self.reset(rng, params)

    def _classify_done(
        self, ts_step_type: jax.Array, ts_discount: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """Jumanji distinguishes natural termination from
        truncation by the `(step_type, discount)` pair (see
        `jumanji.types.termination` vs `jumanji.types.truncation`):
        - termination: step_type == LAST (==2), discount == 0
        - truncation:  step_type == LAST (==2), discount > 0
        Returns `(done, truncated)` both as float32 in {0.0, 1.0}."""
        is_last = ts_step_type == 2
        # discount > 0 alongside LAST → truncation. Termination has
        # discount=0; mid-episode has step_type != LAST.
        truncated_bool = jnp.logical_and(is_last, ts_discount > 0.0)
        return is_last.astype(jnp.float32), truncated_bool.astype(jnp.float32)

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
        done, truncated = self._classify_done(ts.step_type, ts.discount)
        info: dict[str, object] = {'truncated': truncated}
        # Auto-reset on done to match gymnax env contract — jumanji
        # itself doesn't auto-reset, so without this the env stays in
        # a permanently-terminal state for the rest of training (the
        # `done` flag stays 1 forever after the first episode ends,
        # producing degenerate TD targets).
        reset_state, reset_ts = self.inner.reset(rng)
        done_bool = done > 0.5
        final_state = jax.tree.map(
            lambda r, n: jnp.where(done_bool, r, n),
            reset_state, next_state,
        )
        next_obs = self.obs_extract(ts.observation)
        reset_obs = self.obs_extract(reset_ts.observation)
        final_obs = jnp.where(done_bool, reset_obs, next_obs)
        return (
            final_obs,
            final_state,
            ts.reward,
            done_bool,
            info,
        )

    def step_env(
        self,
        rng: jax.Array,
        state: StateT,
        action: jax.Array,
        params: gymnax_env.EnvParams,
    ) -> tuple[
        jax.Array, StateT, jax.Array, jax.Array, dict[str, object],
    ]:
        """No-auto-reset step. Returns the pre-reset
        `(next_obs, next_state)` so the rollout-phase stores the
        physical-continuation state in replay (load-bearing for the
        truncation-aware Bellman target — bootstraps against
        `v(s_pre_reset)` at truncations, not
        `v(s_reset_initial)`).

        `info['truncated']` is set from jumanji's
        `(step_type, discount)` pair via `_classify_done` —
        truncation (LAST + nonzero discount) propagates to the
        Bellman target so it continues bootstrap rather than zeroing."""
        del rng, params
        next_state, ts = self.inner.step(state, action.astype(jnp.int32))
        done, truncated = self._classify_done(ts.step_type, ts.discount)
        info: dict[str, object] = {'truncated': truncated}
        next_obs = self.obs_extract(ts.observation)
        return next_obs, next_state, ts.reward, done > 0.5, info

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
