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

**N-step targets do NOT live on `Replay`.** The Σγᵏrₖ
accumulation that produces an n-step return is theoretical
content (Sutton 1988 / Sutton-Barto §7.6), not buffer mechanics.
It lives in the `n_step_return` Free Claim in this module,
called from `rollout_phase` between the env step and
`Replay.add`. `Replay` itself stores raw transitions whatever
their semantic — single-step or pre-aggregated n-step.

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

from corroborate import claim


# ============ Replay-state shape ============

class ReplayState(NamedTuple):
    """The opaque state owned by `Replay`. dqn threads it through
    `lax.scan` as a single field on `DQNState.replay`. Authors
    swapping `Replay` for a different implementation (e.g. PER)
    define their own `ReplayState`-shaped substate.

    Five parallel arrays + a fill counter — kept flat so the
    pytree leaves are individually `jax.Array` (vmap-friendly).
    Indexed by `step % capacity` for FIFO replacement."""
    obs: jax.Array          # (capacity, *obs_shape)
    action: jax.Array       # (capacity,) int32
    reward: jax.Array       # (capacity,)
    next_obs: jax.Array     # (capacity, *obs_shape)
    done: jax.Array         # (capacity,) float32 (0/1)
    size: jax.Array         # () int32 — number of transitions stored


class PendingNStepState(NamedTuple):
    """In-progress n-step return aggregate. Sub-state of
    `DQNState`. `count` tells how many raw transitions have been
    folded into the running aggregate. When `count == n_step` or
    `acc_done == 1`, `n_step_return` emits an aggregated
    transition for `Replay.add` and resets the window.

    For `n_step=1` the window emits on every step and these
    fields turn over once per call — observable behavior matches
    pre-n-step DQN exactly."""
    head_obs: jax.Array         # (*obs_shape,) — sₜ at start of window
    head_action: jax.Array      # () int32 — aₜ
    acc_reward: jax.Array       # () float32 — Σ γᵏ r_{t+k}
    next_obs: jax.Array         # (*obs_shape,) — most recent s'
    acc_done: jax.Array         # () float32 — max done in window
    n_in_window: jax.Array      # () int32 — raw transitions in window


def init_pending_n_step(obs_shape: tuple[int, ...]) -> PendingNStepState:
    """Allocate an empty pending-window state. Called by
    `init_state` alongside `Replay.init`. Mechanics — not a Claim
    (no theorem; just zero-filled arrays at the right shapes)."""
    return PendingNStepState(
        head_obs=jnp.zeros(obs_shape),
        head_action=jnp.int32(0),
        acc_reward=jnp.float32(0.0),
        next_obs=jnp.zeros(obs_shape),
        acc_done=jnp.float32(0.0),
        n_in_window=jnp.int32(0),
    )


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

# ============ N-step return: theoretical claim ============

@claim
def n_step_return(
    *,
    pending: PendingNStepState,
    transition: Transition,
    n_step: int,
    gamma: float,
) -> tuple[PendingNStepState, Transition, jax.Array]:
    """Sutton-Barto §7.6: n-step return as the bootstrap target.

    Folds `transition` into the in-progress window. Returns the
    updated `pending` state, an emitted `Transition`, and a
    scalar `should_emit` mask (1.0 when the emitted transition
    should be appended to Replay; 0.0 otherwise — the emitted
    transition's contents are stale in that case and must be
    masked out at the call site).

    The emit decision: window full (`count + 1 == n_step`) OR the
    new transition is terminal. On terminal mid-window, the
    emitted transition has `done=1` so the bootstrap discount
    `γⁿ·(1-done)·v(s')` zeroes its bootstrap term — semantically
    equivalent to "n-step return up to the terminal state, no
    bootstrap past it" (Sutton-Barto Eq 7.5).

    The `n_step` parameter at call time is the AUTHORED window
    length. For terminal transitions where the window emits early
    at `count = k < n_step`, the bootstrap discount γⁿ is wrong
    in principle (should be γᵏ), but `done=1` zeroes the v(s')
    term so the discrepancy is invisible — the target reduces to
    pure accumulated reward.

    **Off-policy bias (NOT corrected).** Under ε-greedy rollout,
    the actions a_{t+1},…,a_{t+n-1} composing the in-window sum
    are not all greedy w.r.t. the target Q. The standard n-step
    Q-learning target this claim emits assumes they are; the
    resulting bias is small in practice (Hessel et al 2018 §5)
    and uncorrected here to match Rainbow / standard DQN+n-step
    code. An IS-corrected alternative would be a separate Claim
    occupying the same role in `rollout_phase`.

    **Single-step special case (`n_step=1`).** count goes 0→1
    every call, emit always fires, the emitted Transition is
    identical to the input. Observable behavior matches plain
    DQN; the claim graph still records the call so the
    intervention surface is uniform across n_step values."""
    # Cast to the pending state's init dtypes — env-emitted
    # int32 obs (e.g. FourRooms) would otherwise mismatch the
    # zero-initialised float32 pending fields, and scan's carry-
    # type check fails.
    obs_dtype = pending.head_obs.dtype
    starting_new = pending.n_in_window == 0
    head_obs = jnp.where(
        starting_new,
        transition.obs.astype(obs_dtype),
        pending.head_obs,
    )
    head_action = jnp.where(
        starting_new,
        transition.action.astype(jnp.int32),
        pending.head_action,
    )
    gamma_k = gamma ** pending.n_in_window.astype(jnp.float32)
    acc_reward = (
        pending.acc_reward
        + gamma_k * transition.reward.astype(jnp.float32)
    )
    acc_done = jnp.maximum(
        pending.acc_done, transition.done.astype(jnp.float32),
    )
    next_obs = transition.next_obs.astype(obs_dtype)
    n_in_window = pending.n_in_window + 1

    should_emit_bool = (n_in_window >= n_step) | (acc_done > 0.5)
    should_emit = should_emit_bool.astype(jnp.float32)

    emitted = Transition(
        obs=head_obs, action=head_action, reward=acc_reward,
        next_obs=next_obs, done=acc_done,
    )
    new_pending = PendingNStepState(
        head_obs=jnp.where(should_emit_bool, jnp.zeros_like(head_obs), head_obs),
        head_action=jnp.where(should_emit_bool, jnp.int32(0), head_action),
        acc_reward=jnp.where(should_emit_bool, jnp.float32(0.0), acc_reward),
        next_obs=jnp.where(should_emit_bool, jnp.zeros_like(next_obs), next_obs),
        acc_done=jnp.where(should_emit_bool, jnp.float32(0.0), acc_done),
        n_in_window=jnp.where(should_emit_bool, jnp.int32(0), n_in_window),
    )
    return new_pending, emitted, should_emit


# ============ Replay Module ============

@dataclass(frozen=True, slots=True)
class Replay:
    """FIFO replay buffer config bundle.

    Construction-time HPs:
    - `capacity` — total buffer size; FIFO replacement.
    - `batch_size` — sample size per `sample_batch` call.
    - `sample` — sampling-strategy slot (default `uniform_sample`).
      The slot IS the theoretical claim (Lin 1992); swapping the
      slot is how PER intervenes (`Replay(sample=prioritised_sample)`).

    Three methods (mechanics, not framework Claims):
    - `init(obs_shape) → ReplayState` — allocate empty buffer arrays.
    - `add(state, transition, mask=1) → ReplayState` — FIFO ring
      append. `mask=0` no-ops the append (size + 1*0 unchanged,
      buffer slot stays the previous value); rollout passes the
      n-step `should_emit` mask so adds are skipped when the
      n-step window hasn't filled yet.
    - `sample_batch(state, rng_key) → Batch` — binding wrapper that
      delegates to `self.sample` with `batch_size`/`capacity` from
      the bundle. The slot's FnClaim records itself via `record_call`;
      this wrapper is just glue.

    `Replay` is NOT a framework Claim — it has no single end-to-end
    operation; it bundles config + mechanics + a slot Claim. The
    walker surfaces `replay.capacity`, `replay.batch_size`,
    `replay.sample` as topology leaves regardless of Claim status.

    N-step semantics live OUTSIDE Replay: see `n_step_return`
    Free Claim. Replay stores raw transitions; the n-step return
    is computed in rollout (between env step and `Replay.add`)
    by `n_step_return`, which masks the add when the window
    hasn't yet emitted."""
    capacity: int = 10_000
    batch_size: int = 64
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
        )

    def add(
        self,
        state: ReplayState,
        transition: Transition,
        mask: jax.Array | float = 1.0,
    ) -> ReplayState:
        """Append `transition` to the FIFO ring at index
        `state.size % capacity`, gated by `mask` ∈ {0.0, 1.0}.

        `mask=1` (default) appends and increments size by 1. `mask=0`
        leaves the buffer slot at its previous value and does NOT
        increment size. The mask is the n-step window's
        `should_emit` flag — when the rollout's `n_step_return`
        hasn't yet emitted, we still call `add` (to keep scan's
        carry shape stable) but with mask=0.

        Mechanics — not a Claim, no theoretical content beyond
        data-structure correctness."""
        idx = state.size % self.capacity
        emit = jnp.asarray(mask, dtype=jnp.float32)
        keep = 1.0 - emit
        # Blend new value vs existing at the index.
        old_obs = state.obs[idx]
        old_action = state.action[idx]
        old_reward = state.reward[idx]
        old_next_obs = state.next_obs[idx]
        old_done = state.done[idx]
        new_obs_val = (
            emit.reshape((1,) * old_obs.ndim) * transition.obs.astype(old_obs.dtype)
            + keep.reshape((1,) * old_obs.ndim) * old_obs
        )
        new_action_val = (
            emit * transition.action.astype(jnp.float32)
            + keep * old_action.astype(jnp.float32)
        ).astype(jnp.int32)
        new_reward_val = (
            emit * transition.reward.astype(jnp.float32) + keep * old_reward
        )
        new_next_obs_val = (
            emit.reshape((1,) * old_next_obs.ndim)
            * transition.next_obs.astype(old_next_obs.dtype)
            + keep.reshape((1,) * old_next_obs.ndim) * old_next_obs
        )
        new_done_val = (
            emit * transition.done.astype(jnp.float32) + keep * old_done
        )
        return ReplayState(
            obs=state.obs.at[idx].set(new_obs_val),
            action=state.action.at[idx].set(new_action_val),
            reward=state.reward.at[idx].set(new_reward_val),
            next_obs=state.next_obs.at[idx].set(new_next_obs_val),
            done=state.done.at[idx].set(new_done_val),
            size=jnp.minimum(
                state.size + emit.astype(jnp.int32), self.capacity,
            ),
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
