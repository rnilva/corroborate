"""Smoke test for DQN theorem-condition invariants.

Verifies:
- All 6 invariants in `DQN_INVARIANTS` return HELD on a real
  CartPole trajectory — confirms the run sits inside every
  theorem's domain.
- A tampered record (Q exploded to 1e6) triggers
  INVARIANT_VIOLATION on `q_bounded`, demonstrating the
  divergence-detector contract.
- `buffer_coverage(capacity, fraction)` factory returns HELD on
  a normal run that explores the buffer.
- The reads-set fingerprint flows from `from_key` through the
  reductions to the bridge's `targets`."""
from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import optax

from corroborate.rl.dqn.dqn import dqn_step, init_state
from corroborate.rl.dqn.invariants import (
    DQN_INVARIANTS,
    action_coverage,
    buffer_coverage,
    loss_bounded,
    max_q_overestimation_bounded,
    online_target_disagreement,
    q_bounded,
    td_error_bounded,
)
from corroborate.rl.loop import python_loop
from corroborate.verdict import Verdict


def _run_short_trajectory() -> Mapping[str, jnp.ndarray]:
    """Run DQN on CartPole for 100 steps and return the stacked
    record. Short enough for a smoke test, long enough to span
    warmup + a few training steps."""
    import gymnax
    env, env_params = gymnax.make('CartPole-v1')
    obs_dim = int(env.observation_space(env_params).shape[0])
    n_actions = int(env.action_space(env_params).n)

    optimizer = optax.adam(1e-3)
    state = init_state(
        env=env, env_params=env_params,
        obs_dim=obs_dim, n_actions=n_actions,
        seed=0, optimizer=optimizer, buffer_capacity=200,
    )

    from functools import partial
    step_fn = partial(
        dqn_step,
        env=env, env_params=env_params, n_actions=n_actions,
        optimizer=optimizer,
        warmup_steps=10, sync_period=10,
        buffer_capacity=200, batch_size=16,
    )
    _, record = python_loop(step_fn, state, length=100)
    return record


# ============ All invariants HELD on a real run ============

def test_all_invariants_held_on_cartpole_run() -> None:
    """A normal DQN run on CartPole should sit inside every
    theorem's domain — all 5 invariants HELD."""
    record = _run_short_trajectory()
    for inv in DQN_INVARIANTS:
        result = inv(record)
        assert result.verdict is Verdict.HELD, (
            f'invariant {inv.name} returned {result.verdict.name}: '
            f'{result.reason}'
        )
        # The tautological tag must always be present.
        assert result.stats['kind'] == 'tautological'
        # And the theorem reference must be recorded.
        assert 'theorem' in result.stats


# ============ Per-invariant: targets propagate from leaf reads ============

def test_q_bounded_targets_q_max() -> None:
    assert q_bounded.targets == ('max_q',)


def test_max_q_overestimation_bounded_targets_q_max() -> None:
    # Composition: bounded(growth_window(from_key('max_q'))).
    # Reads propagate: ('max_q',).
    assert max_q_overestimation_bounded.targets == ('max_q',)


def test_loss_bounded_targets_loss() -> None:
    assert loss_bounded.targets == ('loss',)


def test_td_error_bounded_targets_td_error() -> None:
    assert td_error_bounded.targets == ('td_error',)


def test_action_coverage_targets_action() -> None:
    assert action_coverage.targets == ('action',)


def test_online_target_disagreement_targets_argmax_keys() -> None:
    # Multi-input reduction propagates BOTH leaf keys.
    assert online_target_disagreement.targets == ('online_argmax', 'target_argmax')


# ============ buffer_coverage factory ============

def test_buffer_coverage_held_on_real_trajectory() -> None:
    record = _run_short_trajectory()
    inv = buffer_coverage(capacity=200, fraction=0.1)
    result = inv(record)
    assert result.verdict is Verdict.HELD


def test_buffer_coverage_violates_when_indices_too_local() -> None:
    """If `sample_indices` only ever picks 5 distinct positions,
    a fraction-of-buffer threshold of 50% violates."""
    record: Mapping[str, jnp.ndarray] = {
        # Buffer of 200, but sampler only ever drew indices [0..4]
        'sample_indices': jnp.zeros((50, 16), dtype=jnp.int32),  # all zeros
        'epsilon': jnp.asarray([0.5] * 50),
        'reward': jnp.asarray([1.0] * 50),
        'done': jnp.asarray([0.0] * 50),
        'max_q': jnp.asarray([1.0] * 50),
        'ep_return': jnp.asarray([1.0] * 50),
        'action': jnp.asarray([0] * 50),
        'loss': jnp.asarray([0.0] * 50),
        'td_error': jnp.asarray([0.0] * 50),
        'online_argmax': jnp.zeros((50, 16), dtype=jnp.int32),
        'target_argmax': jnp.zeros((50, 16), dtype=jnp.int32),
    }
    inv = buffer_coverage(capacity=200, fraction=0.5)
    result = inv(record)
    # 1 unique index < 100 threshold → invariant violation.
    assert result.verdict is Verdict.INVARIANT_VIOLATION


# ============ Tampered record triggers INVARIANT_VIOLATION ============

def test_q_exploded_record_violates_invariant() -> None:
    """A fabricated record with max_q at 1e6 should trip the
    Banach-contraction-bound invariant — divergence detector
    contract."""
    record: Mapping[str, jnp.ndarray] = {
        'epsilon': jnp.asarray([0.5] * 100),
        'reward': jnp.asarray([1.0] * 100),
        'done': jnp.asarray([0.0] * 100),
        'max_q': jnp.asarray([1.0] * 50 + [1e6] * 50),  # exploded
        'ep_return': jnp.asarray([10.0] * 100),
        'loss': jnp.asarray([0.1] * 100),
        'td_error': jnp.asarray([0.1] * 100),
    }
    result = q_bounded(record)
    assert result.verdict is Verdict.INVARIANT_VIOLATION
    assert result.stats['value'] == 1e6
    assert result.stats['threshold'] == 1e3


def test_loss_exploded_record_violates_invariant() -> None:
    """Fabricated loss explosion → loss_bounded INVARIANT_VIOLATION.
    Demonstrates the semi-gradient-divergence detector path."""
    record: Mapping[str, jnp.ndarray] = {
        'epsilon': jnp.asarray([0.5] * 50),
        'reward': jnp.asarray([1.0] * 50),
        'done': jnp.asarray([0.0] * 50),
        'max_q': jnp.asarray([1.0] * 50),
        'ep_return': jnp.asarray([10.0] * 50),
        'loss': jnp.asarray([1e7] * 50),  # exploded
        'td_error': jnp.asarray([0.1] * 50),
    }
    result = loss_bounded(record)
    assert result.verdict is Verdict.INVARIANT_VIOLATION


