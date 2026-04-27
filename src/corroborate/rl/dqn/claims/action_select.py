"""Action selection — rollout policy claims.

`epsilon_greedy` is the canonical exploratory rollout. The
schedule (`linear_epsilon`) is a separate claim because it
swaps independently — exponential / cosine / piecewise
schedules are future alternatives."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from corroborate.claim import claim


@claim
def epsilon_greedy(
    q_values: jax.Array,
    rng_key: jax.Array,
    epsilon: jax.Array,
    n_actions: int,
) -> jax.Array:
    """ε-greedy action selection. With probability ε, sample
    uniformly from the action space; else argmax over Q-values."""
    explore_key, action_key = jax.random.split(rng_key)
    explore = jax.random.uniform(explore_key) < epsilon
    random_action = jax.random.randint(action_key, (), 0, n_actions)
    greedy_action = jnp.argmax(q_values).astype(jnp.int32)
    return jnp.where(explore, random_action, greedy_action)


@claim
def linear_epsilon(
    step: jax.Array,
    *,
    eps_init: float = 1.0,
    eps_final: float = 0.05,
    anneal_steps: int = 10_000,
) -> jax.Array:
    """Linear ε schedule: anneal from `eps_init` at step 0 to
    `eps_final` at `anneal_steps`, constant afterwards."""
    progress = jnp.minimum(step / anneal_steps, 1.0)
    return eps_init + (eps_final - eps_init) * progress
