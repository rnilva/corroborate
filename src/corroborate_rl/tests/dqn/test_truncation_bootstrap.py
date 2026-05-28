"""Closed-form Bellman-target tests under the truncation flag.

Sutton-Barto §6.6 / Gymnasium-API distinction: `done` is the
env-reset signal (any reason the episode ended); `truncated=1`
flags the subset of dones that were artificial time-limit cutoffs
(experiment chose to stop, trajectory continues physically) vs
`truncated=0` genuine terminal absorbing states. The bootstrap
target consumes `terminated = done * (1 - truncated)` so the
discount mask continues for truncations, zeros for terminations.

These tests pin the contract by constructing scalar transitions
with hand-controlled Q(s', a) values, then asserting the
bootstrap claim returns the closed-form target. The Q-net is a
single linear layer with weights set so that for the test obs the
output's `max` equals a controlled scalar — closed-form, no
sampling-distribution slack.

Replay round-trip and n-step accumulation are covered as well —
the truncation column has to survive the buffer write/sample
cycle for end-to-end correctness."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from corroborate_rl.dqn.claims.bootstrap import (
    bootstrap as bootstrap_claim,
    max_greedify,
)
from corroborate_rl.dqn.claims.q_network import Params, QFunction
from corroborate_rl.dqn.claims.replay import (
    Replay,
    Transition,
    init_pending_n_step,
    n_step_return,
)


def _constant_q_network(value: float, n_actions: int = 2) -> QFunction:
    """Build a typed `QFunction` whose forward returns the constant
    vector `[value, value, ..., value]` of length `n_actions`,
    independent of params or obs. Closed-form control: `max_a Q(s',
    a) == value` for any `next_obs`.

    The framework's `QFunction` Protocol requires `(params, obs) ->
    jax.Array`. Implemented as a plain function (no params used)
    that returns a vector at the action axis with the same leading
    batch shape as `obs` (so batched / single-obs inputs both
    work)."""
    def fn(params: Params, obs: jax.Array) -> jax.Array:
        del params
        # Batch shape = obs.shape minus the trailing obs_dim. For a
        # single obs (1-D), the result is `(n_actions,)`; for
        # batched obs `(B, obs_dim)` the result is `(B, n_actions)`.
        if obs.ndim == 1:
            return jnp.full((n_actions,), value, dtype=jnp.float32)
        batch_shape = obs.shape[:-1]
        return jnp.full((*batch_shape, n_actions), value, dtype=jnp.float32)
    return fn


def test_bellman_target_terminal_zeroes_bootstrap() -> None:
    """`r=1, gamma=0.9, Q(s')=10, done=1, truncated=0` → target=1.
    True termination zeroes the discount: trajectory ended, no
    future."""
    q_network = _constant_q_network(value=10.0)
    # Single-cell batch — scalar batch dim 1 so all per-sample
    # ops stay vector-shaped.
    next_obs = jnp.zeros((1, 4), dtype=jnp.float32)
    reward = jnp.asarray([1.0], dtype=jnp.float32)
    done = jnp.asarray([1.0], dtype=jnp.float32)
    truncated = jnp.asarray([0.0], dtype=jnp.float32)
    target = bootstrap_claim(
        online_params={}, target_params={}, q_network=q_network,
        next_obs=next_obs, reward=reward, done=done,
        truncated=truncated, gamma=0.9,
        greedification=max_greedify,
    )
    # terminated = done * (1 - truncated) = 1 * 1 = 1
    # target = r + γ · (1 − 1) · v(s') = 1
    assert jnp.allclose(target, jnp.asarray([1.0], dtype=jnp.float32)), (
        f'terminal target expected 1.0, got {float(target[0])}'
    )


def test_bellman_target_truncated_continues_bootstrap() -> None:
    """`r=1, gamma=0.9, Q(s')=10, done=1, truncated=1` → target=10.
    Truncation (artificial time-limit cutoff) continues the
    bootstrap as if the trajectory had not ended."""
    q_network = _constant_q_network(value=10.0)
    next_obs = jnp.zeros((1, 4), dtype=jnp.float32)
    reward = jnp.asarray([1.0], dtype=jnp.float32)
    done = jnp.asarray([1.0], dtype=jnp.float32)
    truncated = jnp.asarray([1.0], dtype=jnp.float32)
    target = bootstrap_claim(
        online_params={}, target_params={}, q_network=q_network,
        next_obs=next_obs, reward=reward, done=done,
        truncated=truncated, gamma=0.9,
        greedification=max_greedify,
    )
    # terminated = 1 * (1 - 1) = 0
    # target = 1 + 0.9 · (1 − 0) · 10 = 10
    assert jnp.allclose(target, jnp.asarray([10.0], dtype=jnp.float32)), (
        f'truncated target expected 10.0, got {float(target[0])}'
    )


def test_bellman_target_nonterminal_full_bootstrap() -> None:
    """`r=1, gamma=0.9, Q(s')=10, done=0, truncated=0` → target=10.
    Standard mid-episode step: bootstrap fully active."""
    q_network = _constant_q_network(value=10.0)
    next_obs = jnp.zeros((1, 4), dtype=jnp.float32)
    reward = jnp.asarray([1.0], dtype=jnp.float32)
    done = jnp.asarray([0.0], dtype=jnp.float32)
    truncated = jnp.asarray([0.0], dtype=jnp.float32)
    target = bootstrap_claim(
        online_params={}, target_params={}, q_network=q_network,
        next_obs=next_obs, reward=reward, done=done,
        truncated=truncated, gamma=0.9,
        greedification=max_greedify,
    )
    # terminated = 0 * (1 - 0) = 0
    # target = 1 + 0.9 · 1 · 10 = 10
    assert jnp.allclose(target, jnp.asarray([10.0], dtype=jnp.float32)), (
        f'nonterminal target expected 10.0, got {float(target[0])}'
    )


def test_bellman_target_per_sample_mask() -> None:
    """Batch with mixed flags: each sample's target obeys its own
    (done, truncated) pair independently. Asserts per-row formula
    in a single call (catches reductions across the batch axis
    that would short-circuit the mask)."""
    q_network = _constant_q_network(value=10.0)
    next_obs = jnp.zeros((3, 4), dtype=jnp.float32)
    reward = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32)
    done = jnp.asarray([1.0, 1.0, 0.0], dtype=jnp.float32)
    truncated = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32)
    target = bootstrap_claim(
        online_params={}, target_params={}, q_network=q_network,
        next_obs=next_obs, reward=reward, done=done,
        truncated=truncated, gamma=0.9,
        greedification=max_greedify,
    )
    # row 0: terminated=1 → 1
    # row 1: terminated=0 (truncated) → 10
    # row 2: terminated=0 (mid-episode) → 10
    expected = jnp.asarray([1.0, 10.0, 10.0], dtype=jnp.float32)
    assert jnp.allclose(target, expected), (
        f'per-sample mask broken: expected {expected.tolist()}, '
        f'got {target.tolist()}'
    )


def test_replay_round_trip_preserves_truncated() -> None:
    """`Replay.add(...)` writes truncated; `uniform_sample(...)`
    reads it back. Closed-form: a transition added with
    `truncated=1.0` round-trips to a `Batch.truncated[i] == 1.0`."""
    replay = Replay(capacity=4, batch_size=4)
    state = replay.init(obs_shape=(1,))
    # Add one transition with truncated=1, one with truncated=0.
    txn_trunc = Transition(
        obs=jnp.float32([1.0]),
        action=jnp.int32(0),
        reward=jnp.float32(1.0),
        next_obs=jnp.float32([2.0]),
        done=jnp.float32(1.0),
        truncated=jnp.float32(1.0),
    )
    txn_term = Transition(
        obs=jnp.float32([3.0]),
        action=jnp.int32(0),
        reward=jnp.float32(1.0),
        next_obs=jnp.float32([4.0]),
        done=jnp.float32(1.0),
        truncated=jnp.float32(0.0),
    )
    state = replay.add(state, txn_trunc)
    state = replay.add(state, txn_term)
    # The capacity-4 buffer has two slots populated; valid_size = 2.
    # uniform_sample draws 4 from [0, 2); both indices present in
    # the sample by birthday-style guarantee under 4 draws.
    batch = replay.sample_batch(state, jax.random.PRNGKey(0))
    # For every drawn index i, batch.truncated[i] matches the
    # buffered value at that slot. We assert the per-slot
    # round-trip directly from the replay state.
    assert float(state.truncated[0]) == 1.0, (
        f'slot 0 truncated expected 1.0, got {float(state.truncated[0])}'
    )
    assert float(state.truncated[1]) == 0.0, (
        f'slot 1 truncated expected 0.0, got {float(state.truncated[1])}'
    )
    # Batch must carry the field per-row.
    assert batch.truncated.shape == (4,), (
        f'Batch.truncated shape expected (4,), got {batch.truncated.shape}'
    )
    # Each batch row's truncated value matches the buffer entry at
    # batch.indices[row].
    expected_per_row = state.truncated[batch.indices]
    assert jnp.allclose(batch.truncated, expected_per_row), (
        f'Batch.truncated diverged from buffer round-trip: '
        f'batch={batch.truncated.tolist()} vs '
        f'expected={expected_per_row.tolist()}'
    )


def test_n_step_return_propagates_truncated_within_window() -> None:
    """n_step=3 window: a mid-window truncated step propagates the
    flag through to the emitted aggregated transition. acc_truncated
    is the OR-aggregate (via max) — at most one done/truncated step
    per window (window emits on done), so `max` matches the OR
    semantic exactly."""
    pending = init_pending_n_step(obs_shape=(1,))
    txn1 = Transition(
        obs=jnp.float32([1.0]), action=jnp.int32(0),
        reward=jnp.float32(1.0), next_obs=jnp.float32([2.0]),
        done=jnp.float32(0.0), truncated=jnp.float32(0.0),
    )
    txn2 = Transition(
        obs=jnp.float32([2.0]), action=jnp.int32(1),
        reward=jnp.float32(2.0), next_obs=jnp.float32([3.0]),
        done=jnp.float32(1.0), truncated=jnp.float32(1.0),
    )
    pending, emitted_1, should_emit_1 = n_step_return(
        pending=pending, transition=txn1, n_step=3, gamma=0.9,
    )
    # First step: window has 1 of 3, no terminal → no emit.
    assert float(should_emit_1) == 0.0
    pending, emitted_2, should_emit_2 = n_step_return(
        pending=pending, transition=txn2, n_step=3, gamma=0.9,
    )
    # Second step: terminal mid-window → emit early.
    assert float(should_emit_2) == 1.0
    # The emitted transition carries acc_done=1 AND acc_truncated=1.
    assert float(emitted_2.done) == 1.0
    assert float(emitted_2.truncated) == 1.0, (
        'n-step emit lost the truncated flag from a mid-window '
        'truncation; bootstrap will incorrectly zero its target'
    )
    # Closed-form n-step return: r0 + γ r1 = 1 + 0.9·2 = 2.8.
    assert jnp.allclose(emitted_2.reward, jnp.float32(2.8)), (
        f'n-step accumulated reward off: expected 2.8, '
        f'got {float(emitted_2.reward)}'
    )
