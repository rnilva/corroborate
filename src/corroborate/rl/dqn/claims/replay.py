"""Replay buffer — uniform FIFO ring as a `Replay` config bundle.

The buffer is six parallel arrays (obs, action, reward, next_obs,
done, size) bundled in `ReplayState` — keeps the pytree threaded
through `jax.lax.scan` shallow while letting authors swap the
sampling strategy via `Replay(sample=prioritised_sample)`.

**`Replay` is a config bundle, NOT a framework Claim.** Lin
1992's theoretical claim is about the *sampling distribution* —
which lives in the `sample` slot (an `@claim` free function:
`uniform_sample`, `prioritised_sample`, ...). The slot's FnClaim
records itself via `record_call`. The `Replay` dataclass owns
HPs (`capacity`, `batch_size`) and the slot, plus the bookkeeping
methods (`init`, `add`, `sample_batch`); none of those methods
are theoretical claims, so the dataclass doesn't need to satisfy
the Claim Protocol. The walker still surfaces `replay.capacity`,
`replay.batch_size`, `replay.sample` as topology leaves
regardless of Claim status.

**Theorem reference (on the `sample` slot, not on Replay itself).**
Lin 1992: uniform i.i.d. resampling from a buffer breaks the
strong correlation between consecutive SGD updates and reduces
gradient-estimator variance. Convergence of tabular Q-learning +
replay holds iff every transition is eventually replayed
(Singh-Sutton 1996). Uniform sampling is *not* Bellman-consistent
— old transitions reflect a stale behaviour distribution, biasing
the bootstrap target. This is the bias prioritised-replay (Schaul
2016) addresses. Bridges that want to falsify the i.i.d.
assumption directly need access to the per-step `sample_indices`
trace; that data isn't currently emitted (it cost ~12 GB / 56% of
the §3 corpus and had no consumer wired in), but `Batch.indices`
is still surfaced so a future bridge can opt in by extending the
trace emission set."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import jax
import jax.numpy as jnp

from corroborate.claim import claim


# ============ Replay-state shape ============

class ReplayState(NamedTuple):
    """The opaque state owned by `Replay`. dqn threads it through
    `lax.scan` as a single field on `DQNState.replay`. Authors
    swapping `Replay` for a different implementation (e.g. PER)
    define their own `ReplayState`-shaped substate.

    Six parallel arrays + a fill counter — kept flat so the
    pytree leaves are individually `jax.Array` (vmap-friendly).
    Indexed by `step % capacity` for FIFO replacement.

    For n-step accumulation, six pending-window scalars/arrays
    track the in-progress aggregate (head obs+action, accumulated
    γ-weighted reward, latest next_obs, accumulated done, count of
    raw transitions consumed). For n_step=1 the window emits on
    every add and these fields go through identity-shaped
    transitions only — observable behavior matches the pre-n-step
    version exactly."""
    obs: jax.Array          # (capacity, *obs_shape)
    action: jax.Array       # (capacity,) int32
    reward: jax.Array       # (capacity,)
    next_obs: jax.Array     # (capacity, *obs_shape)
    done: jax.Array         # (capacity,) float32 (0/1)
    size: jax.Array         # () int32 — number of transitions stored
    # N-step pending window. `pending_count` tells us how many raw
    # transitions have been folded into the in-progress aggregate.
    # When `pending_count == n_step` or `pending_acc_done == 1`,
    # the next `add` emits the aggregate and resets the window.
    pending_head_obs: jax.Array     # (*obs_shape,)
    pending_head_action: jax.Array  # () int32
    pending_acc_reward: jax.Array   # () float32 — Σ γ^k r_{t+k}
    pending_next_obs: jax.Array     # (*obs_shape,) — most recent s'
    pending_acc_done: jax.Array     # () float32 — max done in window
    pending_count: jax.Array        # () int32 — raw transitions in window


class Transition(NamedTuple):
    """One transition's worth of data passed to `Replay.add`."""
    obs: jax.Array
    action: jax.Array
    reward: jax.Array
    next_obs: jax.Array
    done: jax.Array


class Batch(NamedTuple):
    """Sampled batch returned by `Replay.sample_batch`. `indices`
    exposes which buffer slots were drawn — surfaced for bridges
    that need to inspect the realised sampling distribution
    (none currently consume it, but the field is part of the
    Batch contract so opting back in only requires extending the
    diagnostic dict in `train_phase`)."""
    obs: jax.Array          # (batch_size, *obs_shape)
    action: jax.Array       # (batch_size,) int32
    reward: jax.Array       # (batch_size,)
    next_obs: jax.Array     # (batch_size, *obs_shape)
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
class Replay:
    """FIFO replay buffer config bundle.

    Construction-time HPs:
    - `capacity` — total buffer size; FIFO replacement.
    - `batch_size` — sample size per `sample_batch` call.
    - `n_step` — number of raw transitions folded into each
      stored "n-step transition". `n_step=1` (default) recovers
      single-step DQN exactly. `n_step=k` accumulates Σ γ^j r_{t+j}
      over the window and stores `(s_t, a_t, R^k_t, s_{t+k},
      done_within_k)` as one buffer entry.
    - `gamma` — discount used for the in-window reward sum.
      Required when n_step > 1; for n_step=1 it's ignored
      (single-step `add` doesn't multiply rewards).
    - `sample` — sampling-strategy slot (default `uniform_sample`).
      The slot IS the theoretical claim (Lin 1992); swapping the
      slot is how PER intervenes (`Replay(sample=prioritised_sample)`).

    Three methods (mechanics, not framework Claims):
    - `init(obs_shape) → ReplayState` — allocate empty buffer arrays.
    - `add(state, transition) → ReplayState` — FIFO ring append.
      For n_step > 1: accumulate into the pending window; emit
      aggregate to the main buffer when window is full or the
      transition is terminal.
    - `sample_batch(state, rng_key) → Batch` — binding wrapper that
      delegates to `self.sample` with `batch_size`/`capacity` from
      the bundle. The slot's FnClaim records itself via `record_call`;
      this wrapper is just glue.

    `Replay` is NOT a framework Claim — it has no single end-to-end
    operation; it bundles config + mechanics + a slot Claim. The
    walker surfaces `replay.capacity`, `replay.batch_size`,
    `replay.n_step`, `replay.gamma`, `replay.sample` as topology
    leaves regardless.

    **Theorem reference (n-step).** Sutton 1988 / Sutton-Barto §7:
    n-step returns trade off bias (longer horizon → more bootstrap
    bias compounding) against variance (longer rollout → more
    Monte Carlo noise). For DQN, multi-step accelerates credit
    assignment on sparse-reward envs by reducing the chain of
    bootstraps the algorithm relies on."""
    capacity: int = 10_000
    batch_size: int = 64
    n_step: int = 1
    gamma: float = 0.99
    sample: 'BufferSample' = field(default=uniform_sample)

    def init(self, obs_shape: tuple[int, ...]) -> ReplayState:
        """Allocate empty buffer arrays at the env's native obs
        shape. `size` starts at 0. Mechanics — not a Claim.

        `obs_shape` is the env's `observation_shape` tuple — e.g.
        `(4,)` for CartPole, `(10, 10, 4)` for MinAtar. The
        buffer stores at native shape so CNN q-networks see the
        spatial structure directly; MLP q-networks flatten
        trailing dims internally (greedy match to weight `w0`'s
        input dim)."""
        return ReplayState(
            obs=jnp.zeros((self.capacity, *obs_shape)),
            action=jnp.zeros((self.capacity,), dtype=jnp.int32),
            reward=jnp.zeros((self.capacity,)),
            next_obs=jnp.zeros((self.capacity, *obs_shape)),
            done=jnp.zeros((self.capacity,)),
            size=jnp.int32(0),
            pending_head_obs=jnp.zeros(obs_shape),
            pending_head_action=jnp.int32(0),
            pending_acc_reward=jnp.float32(0.0),
            pending_next_obs=jnp.zeros(obs_shape),
            pending_acc_done=jnp.float32(0.0),
            pending_count=jnp.int32(0),
        )

    def add(self, state: ReplayState, transition: Transition) -> ReplayState:
        """Fold `transition` into the in-progress n-step pending
        window. Emit an aggregated (head_obs, head_action,
        n_step_reward, latest_next_obs, any_done) transition to the
        main buffer when the window is full or the transition is
        terminal. Mechanics — not a Claim.

        For `n_step=1` this short-circuits to the original single-
        step semantics: count goes 0→1, emit immediately, write
        the raw transition to the buffer slot at `size % capacity`."""
        # Cast to the pending-window's init dtypes (jnp.zeros
        # defaults to float32 for floats; obs in some envs is
        # int32 — without the cast, scan's carry-input/output
        # type check fails: float32 in, int32 out, "carry types
        # differ").
        obs_dtype = state.pending_head_obs.dtype
        starting_new = state.pending_count == 0
        new_head_obs = jnp.where(
            starting_new,
            transition.obs.astype(obs_dtype),
            state.pending_head_obs,
        )
        new_head_action = jnp.where(
            starting_new,
            transition.action.astype(jnp.int32),
            state.pending_head_action,
        )
        gamma_k = self.gamma ** state.pending_count.astype(jnp.float32)
        new_acc_reward = (
            state.pending_acc_reward
            + gamma_k * transition.reward.astype(jnp.float32)
        )
        new_acc_done = jnp.maximum(
            state.pending_acc_done, transition.done.astype(jnp.float32),
        )
        new_next_obs = transition.next_obs.astype(obs_dtype)
        new_count = state.pending_count + 1

        should_emit = (new_count >= self.n_step) | (new_acc_done > 0.5)
        emit = should_emit.astype(jnp.float32)

        idx = state.size % self.capacity
        new_buffer_obs = state.obs.at[idx].set(new_head_obs)
        new_buffer_action = state.action.at[idx].set(new_head_action)
        new_buffer_reward = state.reward.at[idx].set(new_acc_reward)
        new_buffer_next_obs = state.next_obs.at[idx].set(new_next_obs)
        new_buffer_done = state.done.at[idx].set(new_acc_done)
        new_size = jnp.minimum(
            state.size + should_emit.astype(jnp.int32), self.capacity,
        )

        # Reset pending on emit; otherwise carry forward the in-
        # progress aggregate for the next add to extend.
        zero_obs = jnp.zeros_like(new_head_obs)
        new_pending_head_obs = jnp.where(should_emit, zero_obs, new_head_obs)
        new_pending_head_action = jnp.where(
            should_emit, jnp.int32(0), new_head_action,
        )
        new_pending_acc_reward = jnp.where(
            should_emit, jnp.float32(0.0), new_acc_reward,
        )
        new_pending_next_obs = jnp.where(
            should_emit, jnp.zeros_like(new_next_obs), new_next_obs,
        )
        new_pending_acc_done = jnp.where(
            should_emit, jnp.float32(0.0), new_acc_done,
        )
        new_pending_count = jnp.where(
            should_emit, jnp.int32(0), new_count,
        )
        del emit

        return ReplayState(
            obs=new_buffer_obs,
            action=new_buffer_action,
            reward=new_buffer_reward,
            next_obs=new_buffer_next_obs,
            done=new_buffer_done,
            size=new_size,
            pending_head_obs=new_pending_head_obs,
            pending_head_action=new_pending_head_action,
            pending_acc_reward=new_pending_acc_reward,
            pending_next_obs=new_pending_next_obs,
            pending_acc_done=new_pending_acc_done,
            pending_count=new_pending_count,
        )

    def sample_batch(self, state: ReplayState, rng_key: jax.Array) -> Batch:
        """Draw `batch_size` transitions via `self.sample`. The
        sampling strategy is the slot — uniform by default; PER
        swap is `replace(Replay(), sample=prioritised_sample)`.

        Binding wrapper, not a Claim itself: the slot's FnClaim
        records the actual call (Lin 1992's claim-side)."""
        return self.sample(
            state=state, rng_key=rng_key,
            batch_size=self.batch_size, capacity=self.capacity,
        )


# Late import to satisfy the ForwardRef in `Replay.sample`'s default
# annotation — `BufferSample` Protocol lives in `types.py`, which
# would create a circular import if pulled at module-top.
from corroborate.rl.dqn.types import BufferSample  # noqa: E402  pyright: ignore[reportUnusedImport]
