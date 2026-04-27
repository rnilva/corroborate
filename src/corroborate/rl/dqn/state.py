"""DQN state — single threaded structure read by `dqn_step`.

A `NamedTuple` so it threads through `jax.lax.scan` cleanly. The
state carries everything `dqn_step` needs across iterations: the
two parameter sets (online + target), optimizer state, replay
buffer, env state, current observation, step counter, RNG, and
running episode return.

The replay buffer is FLAT (separate fields per transition
component) rather than a struct of arrays — keeps `jax.lax.scan`'s
pytree handling shallow."""
from __future__ import annotations

from typing import NamedTuple

import jax

from corroborate.rl.dqn.types import EnvState, OptState, Params


class DQNState(NamedTuple):
    """Per-step DQN state. Threaded through `jax.lax.scan`.

    `env_state` and `opt_state` are typed `object` (via
    `types.EnvState` / `types.OptState`) — the framework can't
    constrain third-party (gymnax / optax) pytree shapes. Bridge
    bodies that consume these narrow at use site."""

    # Parameter sets — both same pytree shape (MLP weights/biases).
    online_params: Params
    target_params: Params

    # optax optimizer state — opaque to the framework.
    opt_state: OptState

    # Replay buffer (FIFO ring): fixed capacity, indexed by step % capacity.
    buf_obs: jax.Array         # (capacity, obs_dim)
    buf_action: jax.Array      # (capacity,) int32
    buf_reward: jax.Array      # (capacity,)
    buf_next_obs: jax.Array    # (capacity, obs_dim)
    buf_done: jax.Array        # (capacity,) float32 (0/1)
    buf_size: jax.Array        # () int32 — number of transitions stored

    # Env state pytree (gymnax-specific) and current observation.
    env_state: EnvState
    obs: jax.Array             # (obs_dim,)

    # Bookkeeping.
    step: jax.Array            # () int32 — total steps elapsed
    rng_key: jax.Array         # PRNGKey

    # Running per-episode return; resets on `done`.
    ep_return: jax.Array       # () float32
