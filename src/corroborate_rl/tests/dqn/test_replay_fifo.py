"""Replay FIFO ring regression tests.

Pre-fix bug: `Replay.add` capped `state.size` at `capacity` while
also using it as the write index (`idx = state.size % capacity`).
Once the buffer filled, `idx ≡ 0` permanently — every subsequent
add overwrote slot 0 only, freezing slots 1..capacity-1 with the
fill-time contents. Confirmed empirically with capacity=4, 8 adds
of distinct rewards 1..8: pre-fix produced `[8, 2, 3, 4]` instead
of the correct rotation `[5, 6, 7, 8]`.

Post-fix: `state.size` is unbounded; `uniform_sample` clips the
valid range via `min(size, capacity)`; `phases.rollout_phase` clips
the `buf_size` diagnostic so `fill_ratio_late` keeps its 0..1
semantics.

These tests run under `jax.lax.scan` because that is the production
path (the dqn_step composition is jit/scan'd); pre-fix the bug
appeared identically under both eager and scan execution, but
testing scan ensures no future jit-only regression.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from corroborate_rl.dqn.claims.replay import (
    Replay,
    Transition,
    n_step_return,
    init_pending_n_step,
)


def _scalar_transition(reward_value: jax.Array) -> Transition:
    """Distinct-reward transition for FIFO content checks. Accepts
    either a concrete float or a traced jax.Array (so the helper
    works both eagerly and inside `jax.lax.scan`)."""
    r = jnp.asarray(reward_value, dtype=jnp.float32)
    return Transition(
        obs=r.reshape((1,)),
        action=jnp.int32(0),
        reward=r,
        next_obs=(r + 0.5).reshape((1,)),
        done=jnp.float32(0.0),
        truncated=jnp.float32(0.0),
    )


def _scan_adds(
    capacity: int, n_adds: int, mask: float = 1.0,
) -> jnp.ndarray:
    """Run `n_adds` Replay.add calls under jax.lax.scan with rewards
    1..n_adds and constant `mask`. Returns the final reward buffer."""
    r = Replay(capacity=capacity, batch_size=1)
    init = r.init(obs_shape=(1,))

    def body(state, val):
        return r.add(state, _scalar_transition(val), mask=mask), None

    final, _ = jax.lax.scan(
        body, init, jnp.arange(1, n_adds + 1, dtype=jnp.float32),
    )
    return final.reward


def test_fifo_wraps_correctly_after_fill() -> None:
    """capacity=4, 8 adds → buffer holds the latest 4 in a rotation
    indexed by `add_count % capacity`. Pre-fix this was [8, 2, 3, 4]
    (slots 1..3 frozen at fill-time values); post-fix is [5, 6, 7, 8]
    (every slot overwritten in turn)."""
    rewards = _scan_adds(capacity=4, n_adds=8)
    expected = jnp.array([5.0, 6.0, 7.0, 8.0], dtype=jnp.float32)
    assert jnp.allclose(rewards, expected), (
        f'FIFO ring is broken: expected {expected.tolist()}, '
        f'got {rewards.tolist()}'
    )


def test_fifo_wraps_correctly_at_2x_capacity() -> None:
    """Two full cycles. capacity=3, 6 adds → final = [4, 5, 6]
    (the last full cycle overwrites everything from the first)."""
    rewards = _scan_adds(capacity=3, n_adds=6)
    expected = jnp.array([4.0, 5.0, 6.0], dtype=jnp.float32)
    assert jnp.allclose(rewards, expected)


def test_fifo_partial_fill_does_not_wrap() -> None:
    """Below capacity the buffer just fills slot-by-slot; the unused
    tail keeps its zero init."""
    rewards = _scan_adds(capacity=5, n_adds=3)
    expected = jnp.array(
        [1.0, 2.0, 3.0, 0.0, 0.0], dtype=jnp.float32,
    )
    assert jnp.allclose(rewards, expected)


def test_size_grows_unbounded_under_add() -> None:
    """Post-fix: `state.size` is the monotonic add-counter, not a
    capped slot count. This is what makes the FIFO index wrap
    correctly — pre-fix `size` was `min(size+1, capacity)` and the
    bug followed from the cap.

    Sample-time validity is bounded separately by
    `min(size, capacity)` inside `uniform_sample`; tested in
    `test_sampling_only_draws_populated_slots` below."""
    r = Replay(capacity=4, batch_size=1)
    init = r.init(obs_shape=(1,))

    def body(state, val):
        return r.add(state, _scalar_transition(val), mask=1.0), None

    final, _ = jax.lax.scan(
        body, init, jnp.arange(1, 11, dtype=jnp.float32),
    )
    assert int(final.size) == 10, (
        f'Expected size=10 after 10 adds (unbounded counter); '
        f'got {int(final.size)} (cap-at-capacity regression?)'
    )


def test_mask_zero_does_not_advance_size_or_overwrite() -> None:
    """`should_emit=0` (n-step window not yet emitting) must leave
    both the slot at the current write head AND the size counter
    unchanged. Pre-fix this was correct; the regression risk is
    that someone collapses `size + emit` to `size + 1`."""
    r = Replay(capacity=4, batch_size=1)
    init = r.init(obs_shape=(1,))
    state = r.add(init, _scalar_transition(7.0), mask=1.0)
    assert int(state.size) == 1
    assert float(state.reward[0]) == pytest.approx(7.0)

    # mask=0 add: size unchanged, slot 0 unchanged.
    state = r.add(state, _scalar_transition(99.0), mask=0.0)
    assert int(state.size) == 1
    assert float(state.reward[0]) == pytest.approx(7.0)


def test_sampling_only_draws_populated_slots_during_fill() -> None:
    """Below-capacity: `uniform_sample` must restrict to populated
    slots only (`min(size, capacity) = size`). A draw that read
    past the populated tail would surface zero-init garbage."""
    r = Replay(capacity=10, batch_size=64)
    state = r.init(obs_shape=(1,))
    for v in (1.0, 2.0, 3.0):
        state = r.add(state, _scalar_transition(v), mask=1.0)

    batch = r.sample_batch(state, jax.random.PRNGKey(0))
    # All 64 sampled rewards must be in {1, 2, 3} — no zero-init reads.
    sampled = set(float(x) for x in batch.reward.tolist())
    assert sampled.issubset({1.0, 2.0, 3.0}), (
        f'Sampler read past populated tail: got values {sampled}'
    )


def test_n_step_emit_gating_keeps_fifo_correct() -> None:
    """End-to-end check that n_step_return + Replay.add together
    produce the expected post-emit FIFO contents. With n_step=2,
    every other raw transition emits; the emitted transition's
    reward is r_t + γ·r_{t+1}, written to slot
    `(emit_count) % capacity`."""
    r = Replay(capacity=3, batch_size=1)
    state = r.init(obs_shape=(1,))
    pending = init_pending_n_step(obs_shape=(1,))

    gamma = 0.5  # easy arithmetic
    n_step = 2
    rewards_seq = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0]

    def body(carry, raw_r):
        s, p = carry
        t = _scalar_transition(raw_r)
        new_p, emitted, should_emit = n_step_return(
            pending=p, transition=t, n_step=n_step, gamma=gamma,
        )
        new_s = r.add(s, emitted, mask=should_emit)
        return (new_s, new_p), None

    (final_state, _final_pending), _ = jax.lax.scan(
        body, (state, pending), jnp.array(rewards_seq, dtype=jnp.float32),
    )

    # 8 raw transitions, n_step=2 → 4 emits, with accumulated rewards
    # [1+0.5·2, 4+0.5·8, 16+0.5·32, 64+0.5·128]
    # = [2.0, 8.0, 32.0, 128.0].
    # capacity=3 means after 4 emits the buffer holds the latest 3
    # in rotation: emit 1 → slot 0; emit 2 → slot 1; emit 3 → slot 2;
    # emit 4 → slot 0 (overwrites). Final reward = [128, 8, 32].
    expected = jnp.array([128.0, 8.0, 32.0], dtype=jnp.float32)
    assert jnp.allclose(final_state.reward, expected), (
        f'n_step+FIFO interaction broken: expected '
        f'{expected.tolist()}, got {final_state.reward.tolist()}'
    )
    # 4 emits, post-fix size is the unbounded add-counter of EMITS.
    assert int(final_state.size) == 4
