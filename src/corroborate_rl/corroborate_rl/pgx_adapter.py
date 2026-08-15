"""Pgx → gymnax env-interface adapter.

Pgx uses an api-style `state = env.init(key)` / `state =
env.step(state, action, key)` where `State` is a typed dataclass
bundling observation, rewards, terminated, truncated, and
_step_count; the corroborate_rl implementation calls the gymnax-style
API (`env.reset(rng, params) → (obs, state)`,
`env.step(rng, state, action, params)
  → (obs, state, reward, done, info)`).

This adapter mirrors `jumanji_adapter.py`: bake the pgx env at
registration time, route reset/step calls through the gymnax
Protocol surface. Pgx doesn't auto-reset on terminal, so we
manually swap in a freshly-reset state when `terminated` fires
(same pattern as the jumanji adapter — without this, `done`
stays 1 forever after the first terminal step and the TD
target becomes degenerate).

Pgx v2.0+ requires the RNG key as the third positional argument
to `step` (not the second, as in v1.x).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, TypeVar

import jax
import jax.numpy as jnp
from gymnax.environments import environment as gymnax_env
from gymnax.environments import spaces as gymnax_spaces

if TYPE_CHECKING:
    from pgx import Env as PgxEnvT
    from pgx.core import State as PgxState


StateT = TypeVar('StateT')


@dataclass(frozen=True, slots=True)
class PgxEnv(Generic[StateT]):
    """Adapt a pgx `Env` to the gymnax Env Protocol surface.

    `inner` is the wrapped pgx env. `obs_shape` and `n_actions`
    are pre-computed at registration time. The pgx observation is
    a `jax.Array` of dtype bool (MinAtar) or float; downstream
    expects float32, so we cast in the adapter.

    The `params` argument to reset/step is unused (pgx bakes its
    config into the env constructor). Both are accepted to satisfy
    the gymnax Env Protocol.
    """
    inner: PgxEnvT
    obs_shape: tuple[int, ...]
    n_actions: int

    def _cast_obs(self, obs: jax.Array) -> jax.Array:
        # MinAtar pgx obs is bool; cast to float32 for the Q-network input.
        return obs.astype(jnp.float32)

    def reset(
        self, rng: jax.Array, params: gymnax_env.EnvParams,
    ) -> tuple[jax.Array, StateT]:
        del params  # pgx bakes config at env construction time
        state = self.inner.init(rng)
        return self._cast_obs(state.observation), state

    def reset_env(
        self, rng: jax.Array, params: gymnax_env.EnvParams,
    ) -> tuple[jax.Array, StateT]:
        # Pgx's `init` is the no-auto-reset reset (there's no
        # separate auto-resetting `reset`); the gymnax-side
        # `reset_env` Protocol method matches `reset` here.
        return self.reset(rng, params)

    def _classify_truncated(
        self, done: jax.Array, raw_truncated: jax.Array,
    ) -> jax.Array:
        """Enforce the `truncated=1 ⇒ done=1` invariant at the
        wrapper boundary. Pgx's `terminated | truncated → done`
        already implies the invariant, but the explicit
        `jnp.where(done, raw, 0)` form makes the contract
        self-evident and immune to downstream pgx semantic drift.
        Rollout-phase consumers read `truncated` as a sibling of
        `done` without re-narrowing."""
        return jnp.where(
            done.astype(jnp.bool_),
            raw_truncated.astype(jnp.float32),
            jnp.zeros_like(raw_truncated, dtype=jnp.float32),
        )

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
        step_key, reset_key = jax.random.split(rng)
        next_state = self.inner.step(state, action.astype(jnp.int32), step_key)
        done = next_state.terminated | next_state.truncated
        # Pgx natively distinguishes terminated (genuine terminal)
        # from truncated (artificial cap, e.g. pgx's
        # `_step_count >= max_termination_steps`). Surface
        # truncated through `info['truncated']` so the substrate's
        # `bootstrap` claim continues bootstrap on truncations.
        # See `pgx/core.py:105` for the upstream split.
        info: dict[str, object] = {
            'truncated': self._classify_truncated(done, next_state.truncated),
        }
        # Auto-reset on done — pgx itself doesn't auto-reset; without
        # this, `done` stays high forever after the first episode ends.
        reset_state = self.inner.init(reset_key)
        final_state = jax.tree.map(
            lambda r, n: jnp.where(done, r, n),
            reset_state, next_state,
        )
        final_obs = jnp.where(
            done,
            self._cast_obs(reset_state.observation),
            self._cast_obs(next_state.observation),
        )
        # pgx's rewards field is a 1-element array (single-agent envs)
        reward = next_state.rewards.squeeze()
        return (final_obs, final_state, reward, done, info)

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

        `info['truncated']` carries the pgx-native truncation flag
        (`_step_count >= max_termination_steps`) so the rollout's
        Bellman target masks correctly at the cap."""
        del params
        next_state = self.inner.step(state, action.astype(jnp.int32), rng)
        done = next_state.terminated | next_state.truncated
        info: dict[str, object] = {
            'truncated': self._classify_truncated(done, next_state.truncated),
        }
        next_obs = self._cast_obs(next_state.observation)
        reward = next_state.rewards.squeeze()
        return next_obs, next_state, reward, done, info

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
