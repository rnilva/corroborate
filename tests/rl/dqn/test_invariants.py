"""Tests for the DQN theorem-gap measurables.

Each gap is a `Measurable[DQNTrajectoryRecord, float]` returning
a magnitude. Three roles consume gaps differently:

- Intervention: read the scalar directly (`gap(record)`).
- Falsification / scope: wrap with `at_most(gap, threshold,
  of_claim=...)` to get a `Bridge[R]` that returns
  HELD or INVARIANT_VIOLATION.

These tests exercise:
1. The gap measurables compute the documented quantity on a real
   trajectory and on synthetic-failure trajectories.
2. The `at_most` wrap correctly maps gap to verdict for a
   committed threshold."""
from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import optax

from corroborate.invariant import at_most
from corroborate.rl.dqn.claims.bootstrap import vanilla_bootstrap
from corroborate.rl.dqn.claims.q_network import mlp_q
from corroborate.rl.dqn.claims.target_sync import periodic_copy
from corroborate.rl.dqn.dqn import dqn_step, init_state
from corroborate.rl.dqn.invariants import (
    fqi_decay_gap,
    hasselt_covariance_gap,
    jensen_overestimation_gap,
    lin_iid_gap,
    state_action_coverage_gap,
)
from corroborate.loop import python_loop
from corroborate.verdict import Verdict


def _run_short_trajectory() -> Mapping[str, jnp.ndarray]:
    """Run DQN on CartPole for 100 steps and return the stacked
    record."""
    import gymnax
    from corroborate.rl.env_catalogue import HasN, HasShape
    env, env_params = gymnax.make('CartPole-v1')
    obs_space = env.observation_space(env_params)
    act_space = env.action_space(env_params)
    assert isinstance(obs_space, HasShape)
    assert isinstance(act_space, HasN)
    obs_dim = int(obs_space.shape[0])
    n_actions = int(act_space.n)

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


# ============ FQI decay gap ============

def test_fqi_decay_gap_returns_finite_scalar_on_real_run() -> None:
    record = _run_short_trajectory()
    gap = fqi_decay_gap(sync_period=10, gamma=0.99)
    val = gap(record)
    assert isinstance(val, float)
    assert val >= 0.0  # gap is always non-negative


def test_fqi_decay_gap_nan_when_fewer_than_two_windows() -> None:
    """Need at least 2 windows for an across-window ratio; NaN
    no-data sentinel otherwise (distinguishes 'no data' from
    'data confirmed gap=0')."""
    import math
    record: Mapping[str, jnp.ndarray] = {
        'td_error': jnp.asarray([0.5] * 5),  # < one full window
    }
    gap = fqi_decay_gap(sync_period=10, gamma=0.99)
    assert math.isnan(gap(record))


def test_fqi_decay_gap_zero_for_geometric_decay_at_gamma() -> None:
    """Across-window sup-norm decays at exactly γ → gap = 0
    (the theorem's bound is met)."""
    gamma = 0.5
    sync_period = 10
    # Per-window sup norms: 1.0, 0.5, 0.25, 0.125 → ratio = 0.5 = γ
    window_sup_norms = [1.0, 0.5, 0.25, 0.125]
    parts: list[jnp.ndarray] = []
    for sup in window_sup_norms:
        # Window's max abs is `sup`; pad rest with smaller values.
        w = jnp.full(sync_period, sup * 0.9, dtype=jnp.float32)
        w = w.at[0].set(sup)  # ensure max is exactly `sup`
        parts.append(w)
    td_error = jnp.concatenate(parts)
    record: Mapping[str, jnp.ndarray] = {'td_error': td_error}
    gap = fqi_decay_gap(sync_period=sync_period, gamma=gamma)
    val = gap(record)
    # avg_ratio ≈ 0.5; gap = max(0, 0.5 - 0.5) = 0
    assert val < 1e-3


def test_fqi_decay_gap_positive_when_decay_slower_than_gamma() -> None:
    """Across-window sup-norm flat (no decay) → ratio = 1 →
    gap = 1 - γ > 0."""
    gamma = 0.99
    sync_period = 10
    n_windows = 4
    # Constant td_error → all windows have same sup norm → ratio=1
    td_error = jnp.full(sync_period * n_windows, 1.0, dtype=jnp.float32)
    record: Mapping[str, jnp.ndarray] = {'td_error': td_error}
    gap = fqi_decay_gap(sync_period=sync_period, gamma=gamma)
    val = gap(record)
    # avg_ratio = 1.0; gap = 1.0 - 0.99 = 0.01
    assert abs(val - 0.01) < 1e-3


def test_fqi_decay_gap_carries_reads() -> None:
    gap = fqi_decay_gap(sync_period=10)
    assert gap.reads == ('td_error',)


# ============ Lin i.i.d. gap ============

def test_lin_iid_gap_zero_for_uniform_sampling_post_fill() -> None:
    """Indices uniformly drawn from a fully-filled buffer →
    KL(empirical || uniform) ≈ 0."""
    n_steps = 100
    batch = 16
    capacity = 50
    rng = jnp.arange(n_steps * batch) % capacity
    indices = rng.reshape((n_steps, batch))
    # Buffer is fully filled for the entire trajectory.
    buf_size = jnp.full((n_steps,), capacity, dtype=jnp.int32)
    record: Mapping[str, jnp.ndarray] = {
        'sample_indices': indices, 'buf_size': buf_size,
    }
    gap = lin_iid_gap(capacity=capacity)
    val = gap(record)
    assert val < 0.1


def test_lin_iid_gap_large_for_concentrated_sampling_post_fill() -> None:
    """All samples concentrated on one index after the buffer
    fills → very biased sampling → large KL."""
    capacity = 200
    record: Mapping[str, jnp.ndarray] = {
        'sample_indices': jnp.zeros((50, 16), dtype=jnp.int32),
        # Buffer fills early then stays full.
        'buf_size': jnp.full((50,), capacity, dtype=jnp.int32),
    }
    gap = lin_iid_gap(capacity=capacity)
    val = gap(record)
    # KL(δ_0 || uniform_200) = log(200) ≈ 5.3.
    assert val > 4.0


def test_lin_iid_gap_nan_when_buffer_never_fills() -> None:
    """Buffer never fills (always under capacity) → NaN no-data
    sentinel, NOT a misleading high value from buffer-size
    confound nor a `0.0` collision with a perfectly-uniform
    sampling distribution."""
    import math
    capacity = 1000  # huge capacity buffer never reaches
    record: Mapping[str, jnp.ndarray] = {
        'sample_indices': jnp.zeros((50, 16), dtype=jnp.int32),
        'buf_size': jnp.arange(50, dtype=jnp.int32),  # 0, 1, 2, ..., 49
    }
    gap = lin_iid_gap(capacity=capacity)
    assert math.isnan(gap(record))


def test_lin_iid_gap_filters_out_pre_fill_steps() -> None:
    """Pre-fill steps are EXCLUDED from the KL computation —
    avoids the structural confound where small buffer makes
    sampling look biased toward low indices."""
    capacity = 100
    n_pre_fill = 30
    n_post_fill = 70
    # Pre-fill: indices concentrated low (because buffer is small).
    pre_indices = jnp.zeros((n_pre_fill, 16), dtype=jnp.int32)
    # Post-fill: indices uniform across capacity.
    post_indices = (
        jnp.arange(n_post_fill * 16) % capacity
    ).reshape((n_post_fill, 16))
    indices = jnp.concatenate([pre_indices, post_indices], axis=0)

    pre_buf = jnp.arange(n_pre_fill, dtype=jnp.int32)
    post_buf = jnp.full((n_post_fill,), capacity, dtype=jnp.int32)
    buf_size = jnp.concatenate([pre_buf, post_buf])

    record: Mapping[str, jnp.ndarray] = {
        'sample_indices': indices, 'buf_size': buf_size,
    }
    gap = lin_iid_gap(capacity=capacity)
    val = gap(record)
    # Without filtering, the pre-fill bias would dominate the KL.
    # With filtering, only the uniform post-fill segment counts.
    assert val < 0.5


def test_lin_iid_gap_carries_both_reads() -> None:
    gap = lin_iid_gap(capacity=200)
    assert gap.reads == ('sample_indices', 'buf_size')


# ============ Hasselt covariance gap (Pearson r) ============

def test_hasselt_gap_one_when_q_values_perfectly_correlated() -> None:
    """Online and target Q-values identical (varying across
    samples) → Pearson r = 1 → gap = 1 (DDQN reduces to vanilla,
    theorem doesn't bite)."""
    arr = jnp.arange(50 * 16 * 2, dtype=jnp.float32).reshape((50, 16, 2))
    record: Mapping[str, jnp.ndarray] = {
        'online_q_values': arr,
        'target_q_values': arr,
    }
    gap = hasselt_covariance_gap()
    assert abs(gap(record) - 1.0) < 1e-5


def test_hasselt_gap_zero_when_q_values_perfectly_anti_correlated() -> None:
    """Anti-correlation (r = -1) is FAVOURABLE for DDQN —
    estimators cancel rather than reinforce. Gap is asymmetric:
    `max(0, r)`, so anti-correlation ⇒ gap = 0 (not 1)."""
    arr = jnp.arange(50 * 16 * 2, dtype=jnp.float32).reshape((50, 16, 2))
    record: Mapping[str, jnp.ndarray] = {
        'online_q_values': arr,
        'target_q_values': -arr,
    }
    gap = hasselt_covariance_gap()
    assert gap(record) == 0.0


def test_hasselt_gap_nan_when_q_values_constant() -> None:
    """Zero variance on either side ⇒ correlation undefined ⇒
    NaN no-data sentinel (no information about independence
    either way)."""
    import math
    record: Mapping[str, jnp.ndarray] = {
        'online_q_values': jnp.ones((50, 16, 2)),
        'target_q_values': jnp.arange(50 * 16 * 2, dtype=jnp.float32).reshape((50, 16, 2)),
    }
    gap = hasselt_covariance_gap()
    assert math.isnan(gap(record))


def test_hasselt_gap_carries_both_reads() -> None:
    gap = hasselt_covariance_gap()
    assert gap.reads == ('online_q_values', 'target_q_values')


# ============ at_most wrap: scope commitment → verdict ============

def test_at_most_wrap_held_when_gap_under_threshold() -> None:
    """Author commits scope: 'periodic_copy's mechanism operates
    when fqi_decay_gap ≤ 0.05'. Synthetic decay-at-γ trajectory
    has gap=0 → HELD."""
    gamma = 0.5
    sync_period = 10
    # Per-window sup norms 1.0, 0.5, 0.25, 0.125 → ratio = γ
    parts: list[jnp.ndarray] = []
    for sup in (1.0, 0.5, 0.25, 0.125):
        w = jnp.full(sync_period, sup * 0.9, dtype=jnp.float32)
        w = w.at[0].set(sup)
        parts.append(w)
    record: Mapping[str, jnp.ndarray] = {
        'td_error': jnp.concatenate(parts),
    }
    gap = fqi_decay_gap(sync_period=sync_period, gamma=gamma)
    bridge = at_most(gap, threshold=0.05, of_claim=periodic_copy)
    result = bridge(record)
    assert result.verdict is Verdict.HELD
    assert result.stats['kind'] == 'tautological'
    assert result.stats['of_claim'] == 'periodic_copy'
    assert 'gap_value' in result.stats
    assert result.stats['threshold'] == 0.05


def test_at_most_wrap_invariant_violation_when_gap_over_threshold() -> None:
    """Synthetic concentrated sampling on a fully-filled buffer
    → large lin_iid_gap → over threshold → INVARIANT_VIOLATION
    (theorem out of scope)."""
    capacity = 200
    record: Mapping[str, jnp.ndarray] = {
        'sample_indices': jnp.zeros((50, 16), dtype=jnp.int32),
        'buf_size': jnp.full((50,), capacity, dtype=jnp.int32),
    }
    gap = lin_iid_gap(capacity=capacity)
    bridge = at_most(gap, threshold=1.0, of_claim=periodic_copy)
    result = bridge(record)
    assert result.verdict is Verdict.INVARIANT_VIOLATION


def test_at_most_wrap_targets_propagate_from_gap_reads() -> None:
    gap = hasselt_covariance_gap()
    bridge = at_most(gap, threshold=0.5, of_claim=vanilla_bootstrap)
    assert bridge.targets == ('online_q_values', 'target_q_values')


def test_at_most_wrap_default_name_includes_gap_and_threshold() -> None:
    gap = fqi_decay_gap(sync_period=10)
    bridge = at_most(gap, threshold=0.5, of_claim=mlp_q)
    assert 'fqi_decay_gap' in bridge.name
    assert '0.5' in bridge.name


# ============ Jensen overestimation gap ============

def test_jensen_gap_zero_when_predicted_equals_actual() -> None:
    """No bias → gap = 0."""
    record: Mapping[str, jnp.ndarray] = {
        'predicted_q_at_start': jnp.full((5, 4), 10.0),
        'mc_return': jnp.full((5, 4), 10.0),
    }
    gap = jensen_overestimation_gap()
    assert gap(record) == 0.0


def test_jensen_gap_positive_when_predicted_exceeds_actual() -> None:
    """Predicted > actual on average → positive overestimation
    bias → gap = mean bias."""
    record: Mapping[str, jnp.ndarray] = {
        'predicted_q_at_start': jnp.full((5, 4), 12.0),
        'mc_return': jnp.full((5, 4), 10.0),
    }
    gap = jensen_overestimation_gap()
    assert abs(gap(record) - 2.0) < 1e-5


def test_jensen_gap_zero_when_predicted_under_actual() -> None:
    """Underestimation isn't the Jensen signature; clip to 0."""
    record: Mapping[str, jnp.ndarray] = {
        'predicted_q_at_start': jnp.full((5, 4), 5.0),
        'mc_return': jnp.full((5, 4), 10.0),
    }
    gap = jensen_overestimation_gap()
    assert gap(record) == 0.0


def test_jensen_gap_carries_eval_record_reads() -> None:
    gap = jensen_overestimation_gap()
    assert gap.reads == ('predicted_q_at_start', 'mc_return')


# ============ State-action coverage gap ============

def test_sa_coverage_gap_zero_for_perfect_coverage() -> None:
    """Every (s, a) pair visited exactly once → coverage = 1 →
    gap = 0."""
    n_buckets = 4
    n_actions = 2
    pairs = jnp.arange(n_buckets * n_actions, dtype=jnp.int32)
    state_hashes = pairs // n_actions
    actions = pairs % n_actions
    record: Mapping[str, jnp.ndarray] = {
        'state_hash': state_hashes,
        'action': actions,
    }
    gap = state_action_coverage_gap(
        state_hash_cardinality=n_buckets, n_actions=n_actions,
    )
    assert gap(record) == 0.0


def test_sa_coverage_gap_one_for_zero_coverage_against_huge_card() -> None:
    """Few unique pairs vs huge cardinality → near-1 gap."""
    record: Mapping[str, jnp.ndarray] = {
        'state_hash': jnp.zeros((50,), dtype=jnp.int32),  # all bucket 0
        'action': jnp.zeros((50,), dtype=jnp.int32),       # all action 0
    }
    gap = state_action_coverage_gap(
        state_hash_cardinality=10000, n_actions=4,
    )
    val = gap(record)
    # 1 unique pair / 40000 max ≈ 0.999975
    assert val > 0.999


def test_sa_coverage_gap_nan_when_cardinality_none() -> None:
    """env_spec.state_hash=None → cardinality=None → no-data
    gap measurable returning NaN. NaN distinguishes 'no data'
    from 'gap = 0' (perfect coverage)."""
    import math
    record: Mapping[str, jnp.ndarray] = {
        'state_hash': jnp.zeros((50,), dtype=jnp.int32),
        'action': jnp.zeros((50,), dtype=jnp.int32),
    }
    gap = state_action_coverage_gap(
        state_hash_cardinality=None, n_actions=4,
    )
    assert math.isnan(gap(record))
    # And the reads-set is empty for the no-data variant.
    assert gap.reads == ()
