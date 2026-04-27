"""Replay buffer — uniform FIFO ring.

The buffer is six parallel arrays (obs, action, reward, next_obs,
done, size) rather than a struct of per-transition records —
keeps the pytree threaded through `jax.lax.scan` flat. v0 ships
uniform sampling; PrioritisedReplay is a future @claim swap on
the sampling-side, with side-car priority arrays.

**Theorem reference.** Lin 1992: uniform i.i.d. resampling from
a buffer breaks the strong correlation between consecutive
SGD updates and reduces gradient-estimator variance. Convergence
of tabular Q-learning + replay holds iff every transition is
eventually replayed (Singh-Sutton 1996). Uniform sampling is
*not* Bellman-consistent — old transitions reflect a stale
behaviour distribution, biasing the bootstrap target. This is
the bias prioritised-replay (Schaul 2016) addresses. The
`lin_iid_gap(capacity)` measurable in `invariants.py` reports
KL(empirical sampling ‖ uniform) — quantifies how far the
realised sampling distribution deviates from the i.i.d.
assumption."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from corroborate.claim import claim
from corroborate.rl.dqn.state import DQNState


@claim
def buffer_init(capacity: int, obs_dim: int) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
]:
    """Initialise FIFO replay buffer. Returns
    `(obs, action, reward, next_obs, done, size)`. Size starts at 0."""
    obs = jnp.zeros((capacity, obs_dim))
    action = jnp.zeros((capacity,), dtype=jnp.int32)
    reward = jnp.zeros((capacity,))
    next_obs = jnp.zeros((capacity, obs_dim))
    done = jnp.zeros((capacity,))
    size = jnp.int32(0)
    return obs, action, reward, next_obs, done, size


@claim
def buffer_add(
    *,
    state: DQNState,
    capacity: int,
    obs: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    next_obs: jax.Array,
    done: jax.Array,
) -> tuple[
    jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array,
]:
    """Append one transition to the FIFO ring. Returns the six
    updated buffer arrays + new size."""
    idx = state.buf_size % capacity
    new_obs = state.buf_obs.at[idx].set(obs)
    new_action = state.buf_action.at[idx].set(action.astype(jnp.int32))
    new_reward = state.buf_reward.at[idx].set(reward)
    new_next_obs = state.buf_next_obs.at[idx].set(next_obs)
    new_done = state.buf_done.at[idx].set(done.astype(jnp.float32))
    new_size = jnp.minimum(state.buf_size + 1, capacity)
    return new_obs, new_action, new_reward, new_next_obs, new_done, new_size


@claim
def buffer_sample(
    *,
    state: DQNState,
    rng_key: jax.Array,
    batch_size: int,
    capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Uniform-random sample of `batch_size` transitions from the
    populated portion of the buffer. Returns batched
    `(obs, action, reward, next_obs, done, indices)`.

    `indices` is exposed so the Lin-coverage invariant
    (`buffer_coverage`) can measure how much of the buffer the
    sampler actually visits — uniform replay is *not* Bellman-
    consistent, and the sampling distribution's diversity is the
    empirical proxy for the Lin 1992 i.i.d. assumption."""
    valid_size = jnp.minimum(state.buf_size, capacity)
    indices = jax.random.randint(rng_key, (batch_size,), 0, valid_size)
    return (
        state.buf_obs[indices],
        state.buf_action[indices],
        state.buf_reward[indices],
        state.buf_next_obs[indices],
        state.buf_done[indices],
        indices.astype(jnp.int32),
    )
