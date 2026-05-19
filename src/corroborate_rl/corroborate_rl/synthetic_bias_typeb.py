"""Synthetic bias Type-A/B chain MDP — controlled-substrate for
the bias-asymmetry causal classifier test.

The empirical n=1 limitation of the natural-env Type-A/B panel
(only Asterix γ=0.999 shows DDQN harm among 8 envs) blocks
isolating which env-feature is causal: K, true Var_a[Q*],
reward sparsity, effective horizon, argmax-margin fragility?

This env lets us SWEEP those features independently. The MDP is:

- States: chain s ∈ {0, ..., L-1}; transitions advance cyclically
  s → (s+1) mod L.
- Actions: K-armed at every state.
- Reward: at each step, with probability `reward_sparsity`,
  sample r ~ Normal(mean[action], reward_noise_scale); else r=0.
  Where `mean[a] = reward_variance_scale × (2a/(K-1) - 1)`
  spans [-rvs, +rvs] uniformly across K actions.
- Termination: after `max_steps_in_episode` steps.

The TRUE optimal Q at state s, action a, given the deterministic
cyclic transition, is

  Q*(s, a) = reward_sparsity × mean[a]
           + γ · max_b Q*(next(s), b)
         (= reward_sparsity × mean[a]
            + γ · V*(s+1 mod L))

so cross-action variance at any state equals (reward_sparsity ×
reward_variance_scale)² — INDEPENDENT of state, by construction.
This is the load-bearing property: by setting
`reward_variance_scale` we control Var_a[Q*] cleanly without
confounding it with anything else.

Hasselt 2010's per-step max-bias is c · σ_action with
c = √(2 ln K / π). Vanilla's symmetric max-of-K bias accumulates
through the chain via 1/(1-γ); DDQN's clip prevents it. The
expected pattern:

- High `reward_variance_scale`: true action heterogeneity is
  large; the bias asymmetry is policy-informative; DDQN's
  symmetric clip removes informative variance → harm.
- Low `reward_variance_scale`: action-quality is uniform; the
  bias asymmetry is pure noise; DDQN's clip cleans it up → help.

API matches the gymnax `Env` Protocol structurally:
`reset(rng, params) → (obs, state)`, `step(rng, state, action,
params) → (obs, state, reward, done, info)`. Per the substrate
convention, the env is config-free (no class fields) and all
per-cell configuration flows through `BiasTypeBParams`. The
chain length `n_states` and action count `n_actions` are
fixed-per-name at registration time (see
`SYNTHETIC_REGISTRATIONS` in `env_catalogue.py`); other knobs
(`reward_variance_scale`, `reward_sparsity`,
`reward_noise_scale`) vary via `make_synthetic_bias_typeb`'s
keyword args.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import struct
from gymnax.environments import spaces

if TYPE_CHECKING:
    from gymnax import Box, Discrete


@struct.dataclass
class BiasTypeBParams:
    """Per-cell parameters for the synthetic bias Type-B env.

    `n_states`, `n_actions` carried here are READ-ONLY metadata
    matching the catalogue registration; the runtime env reads
    them via attribute and they don't vary per cell.

    `reward_variance_scale` sets Var_a[mean] = rvs². Together
    with `reward_sparsity` ∈ (0, 1] (per-step nonzero reward
    probability) and `reward_noise_scale` (Hasselt σ_action
    source), these are the load-bearing knobs.

    `max_steps_in_episode` matches gymnax convention; substrate
    consumers (`eval_episode_cap`) read it via this field name."""
    n_states: int = struct.field(pytree_node=False, default=4)
    n_actions: int = struct.field(pytree_node=False, default=4)
    reward_variance_scale: float = 1.0
    reward_sparsity: float = 1.0
    reward_noise_scale: float = 0.1
    max_steps_in_episode: int = struct.field(pytree_node=False, default=64)


@struct.dataclass
class BiasTypeBState:
    """Per-step env state. Step counter + chain position."""
    step: jax.Array  # int32 scalar
    state: jax.Array  # int32 scalar in [0, n_states)


@dataclass(frozen=True, slots=True)
class BiasTypeBEnv:
    """Synthetic K-action chain MDP for bias Type-A/B causal
    testing.

    Construction is config-free; per-cell config flows through
    `BiasTypeBParams`. Mirrors `LunarLanderEnv`'s class shape."""

    def reset(
        self, rng: jax.Array, params: BiasTypeBParams,
    ) -> tuple[jax.Array, BiasTypeBState]:
        del rng
        state = BiasTypeBState(
            step=jnp.int32(0), state=jnp.int32(0),
        )
        return self._obs(state.state, params.n_states), state

    def step(
        self,
        rng: jax.Array,
        state: BiasTypeBState,
        action: jax.Array,
        params: BiasTypeBParams,
    ) -> tuple[
        jax.Array, BiasTypeBState, jax.Array, jax.Array,
        dict[str, jax.Array],
    ]:
        # Action-specific mean: linear in action index, spread by
        # reward_variance_scale across [-rvs, +rvs].
        denom = jnp.maximum(params.n_actions - 1, 1)
        a_normalized = action.astype(jnp.float32) / denom.astype(jnp.float32)
        # a_normalized ∈ [0, 1] → mean ∈ [-rvs, +rvs]
        mean_a = params.reward_variance_scale * (2.0 * a_normalized - 1.0)

        # Gaussian per-step reward noise + sparsity gate.
        key_noise, key_sparse = jax.random.split(rng, 2)
        eps = (
            jax.random.normal(key_noise) * params.reward_noise_scale
        )
        nonzero = (
            jax.random.uniform(key_sparse) < params.reward_sparsity
        )
        reward = jnp.where(
            nonzero,
            mean_a + eps,
            jnp.float32(0.0),
        )

        # Deterministic cyclic transition.
        new_state_idx = (state.state + 1) % params.n_states
        new_step = state.step + 1
        done = new_step >= params.max_steps_in_episode

        new_state = BiasTypeBState(step=new_step, state=new_state_idx)
        obs = self._obs(new_state_idx, params.n_states)
        return obs, new_state, reward, done, {}

    def _obs(
        self, state_idx: jax.Array, n_states: int,
    ) -> jax.Array:
        """One-hot encoding of chain position."""
        return jax.nn.one_hot(
            state_idx, num_classes=n_states, dtype=jnp.float32,
        )

    def action_space(self, params: BiasTypeBParams) -> 'Discrete':
        return spaces.Discrete(params.n_actions)

    def observation_space(self, params: BiasTypeBParams) -> 'Box':
        return spaces.Box(
            low=0.0, high=1.0,
            shape=(params.n_states,),
            dtype=jnp.float32,
        )


def make_synthetic_bias_typeb(
    *,
    n_states: int = 4,
    n_actions: int = 4,
    reward_variance_scale: float = 1.0,
    reward_sparsity: float = 1.0,
    reward_noise_scale: float = 0.1,
    max_steps_in_episode: int = 64,
) -> tuple[BiasTypeBEnv, BiasTypeBParams]:
    """Factory matching the lunar_lander pattern. Builds an env
    instance + its default params; substrate's cell_runner consumes
    the pair via the gymnax-style API.

    For the parametric Type-A/B sweep, register one factory closure
    per named config (see `env_catalogue.py`)."""
    env = BiasTypeBEnv()
    params = BiasTypeBParams(
        n_states=n_states,
        n_actions=n_actions,
        reward_variance_scale=float(reward_variance_scale),
        reward_sparsity=float(reward_sparsity),
        reward_noise_scale=float(reward_noise_scale),
        max_steps_in_episode=int(max_steps_in_episode),
    )
    return env, params
