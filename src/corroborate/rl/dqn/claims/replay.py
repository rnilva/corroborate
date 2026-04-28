"""Replay buffer — uniform FIFO ring as a `Replay` Module.

The buffer is six parallel arrays (obs, action, reward, next_obs,
done, size) bundled in `ReplayState` — keeps the pytree threaded
through `jax.lax.scan` shallow while letting authors swap the
sampling strategy via `Replay(sample=PrioritisedSample())`.

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

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp

from corroborate.claim import ClaimBase, claim, record_call


# ============ Replay-state shape ============

class ReplayState(NamedTuple):
    """The opaque state owned by `Replay`. dqn threads it through
    `lax.scan` as a single field on `DQNState.replay`. Authors
    swapping `Replay` for a different implementation (e.g. PER)
    define their own `ReplayState`-shaped substate.

    Six parallel arrays + a fill counter — kept flat so the
    pytree leaves are individually `jax.Array` (vmap-friendly).
    Indexed by `step % capacity` for FIFO replacement."""
    obs: jax.Array          # (capacity, obs_dim)
    action: jax.Array       # (capacity,) int32
    reward: jax.Array       # (capacity,)
    next_obs: jax.Array     # (capacity, obs_dim)
    done: jax.Array         # (capacity,) float32 (0/1)
    size: jax.Array         # () int32 — number of transitions stored


class Transition(NamedTuple):
    """One transition's worth of data passed to `Replay.add`."""
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    done: jax.Array


class Batch(NamedTuple):
    """Sampled batch returned by `Replay.sample_batch`. `indices`
    exposes which buffer slots were drawn — measurables like
    `lin_iid_gap` consume this to quantify sampling-distribution
    diversity."""
    obs: jax.Array          # (batch_size, obs_dim)
    action: jax.Array       # (batch_size,) int32
    reward: jax.Array       # (batch_size,)
    next_obs: jax.Array     # (batch_size, obs_dim)
    done: jax.Array         # (batch_size,)
    indices: jax.Array      # (batch_size,) int32


# ============ Sampling strategy (BufferSample slot) ============

@claim
def uniform_sample(
    *,
    state: ReplayState,
    rng_key: jax.Array,
    batch_size: int,
    capacity: int,
) -> Batch:
    """Uniform-random sample of `batch_size` transitions from the
    populated portion of the buffer.

    Indices are drawn from `[0, valid_size)` where `valid_size =
    min(state.size, capacity)` — avoids reading uninitialised
    pre-fill slots before the buffer is warm."""
    valid_size = jnp.minimum(state.size, capacity)
    indices = jax.random.randint(rng_key, (batch_size,), 0, valid_size)
    return Batch(
        obs=state.obs[indices],
        action=state.action[indices],
        reward=state.reward[indices],
        next_obs=state.next_obs[indices],
        done=state.done[indices],
        indices=indices.astype(jnp.int32),
    )


# ============ Replay Module ============

@dataclass(frozen=True, slots=True)
class Replay(ClaimBase):
    """FIFO replay buffer Module.

    Construction-time HPs:
    - `capacity` — total buffer size; FIFO replacement.
    - `batch_size` — sample size per `sample_batch` call.
    - `sample` — sampling-strategy slot (default `uniform_sample`).
      Future: PrioritisedSample with side-car priority arrays.

    Three operations: `init(obs_dim) → ReplayState`,
    `add(state, transition) → ReplayState`, `sample_batch(state,
    rng_key) → Batch`. dqn threads `ReplayState` as a single
    sub-state field on `DQNState`.

    `__call__` is not defined — `Replay` is a stateful component
    accessed via its named methods, not a single forward
    operation."""
    capacity: int = 10_000
    batch_size: int = 64
    sample: 'BufferSample' = field(default=uniform_sample)

    def init(self, obs_dim: int) -> ReplayState:
        """Allocate empty buffer arrays. `size` starts at 0."""
        return ReplayState(
            obs=jnp.zeros((self.capacity, obs_dim)),
            action=jnp.zeros((self.capacity,), dtype=jnp.int32),
            reward=jnp.zeros((self.capacity,)),
            next_obs=jnp.zeros((self.capacity, obs_dim)),
            done=jnp.zeros((self.capacity,)),
            size=jnp.int32(0),
        )

    def add(self, state: ReplayState, transition: Transition) -> ReplayState:
        """Append `transition` to the FIFO ring at index
        `state.size % capacity`. Increments `size` (capped at
        capacity)."""
        idx = state.size % self.capacity
        new_state = ReplayState(
            obs=state.obs.at[idx].set(transition.obs),
            action=state.action.at[idx].set(
                transition.action.astype(jnp.int32),
            ),
            reward=state.reward.at[idx].set(transition.reward),
            next_obs=state.next_obs.at[idx].set(transition.next_obs),
            done=state.done.at[idx].set(transition.done.astype(jnp.float32)),
            size=jnp.minimum(state.size + 1, self.capacity),
        )
        record_call(self, (state, transition), {}, new_state)
        return new_state

    def sample_batch(self, state: ReplayState, rng_key: jax.Array) -> Batch:
        """Draw `batch_size` transitions via `self.sample`. The
        sampling strategy is the slot — uniform by default; PER
        swap is `replace(Replay(), sample=prioritised_sample)`."""
        return self.sample(
            state=state, rng_key=rng_key,
            batch_size=self.batch_size, capacity=self.capacity,
        )


# Late import to satisfy the ForwardRef in `Replay.sample`'s default
# annotation — `BufferSample` Protocol lives in `types.py`, which
# would create a circular import if pulled at module-top.
from corroborate.rl.dqn.types import BufferSample  # noqa: E402  pyright: ignore[reportUnusedImport]
