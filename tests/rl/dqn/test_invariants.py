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
import pytest

from corroborate.invariant import at_most
# Side-effect import: registers pearson_r_online_target etc. in
# the measurable registry so resolved gap evaluation works.
import corroborate.rl.dqn.measurables  # noqa: F401
from corroborate.rl.dqn.claims.bootstrap import bootstrap
from corroborate.rl.dqn.claims.q_network import mlp_q
from corroborate.rl.dqn.claims.target_sync import periodic_copy
from corroborate.rl.dqn.dqn import dqn_step, init_state
from corroborate.rl.dqn.invariants import (
    fqi_decay_gap,
    hasselt_covariance_gap,
    jensen_dormancy_gap,
    jensen_floor_late,
    jensen_overestimation_gap,
    state_action_coverage_gap,
)
from corroborate.rl.loop import python_loop
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
    obs_shape = tuple(int(d) for d in obs_space.shape)
    n_actions = int(act_space.n)

    from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
    from corroborate.rl.dqn.claims.replay import Replay

    optimizer = WarmedUpdate(inner=Adam(), warmup_steps=10)()
    replay = Replay(capacity=200, batch_size=16)
    import jax
    state = init_state(
        env=env, env_params=env_params,
        obs_shape=obs_shape, n_actions=n_actions,
        rng_key=jax.random.PRNGKey(0),
        optimizer=optimizer, replay=replay,
    )

    from functools import partial
    step_fn = partial(
        dqn_step,
        env=env, env_params=env_params, n_actions=n_actions,
        optimizer=optimizer,
        sync_period=10,
        replay=replay,
    )
    _, record = python_loop(step_fn, state, length=100)
    return record


# ============ FQI decay gap ============

@pytest.mark.slow
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


# ============ Hasselt covariance gap (Pearson r) ============

def _pearson_stats_from_arrays(
    on: jnp.ndarray, tg: jnp.ndarray,
) -> jnp.ndarray:
    """Build the per-step `pearson_stats` series train_phase
    emits, given full Q-tensors (T, batch, n_actions). Helper so
    tests can use natural array fixtures without rewriting in
    sufficient-statistics form."""
    on_flat = on.reshape(on.shape[0], -1)  # (T, batch*n_actions)
    tg_flat = tg.reshape(tg.shape[0], -1)
    return jnp.stack([
        on_flat.mean(axis=-1),
        tg_flat.mean(axis=-1),
        (on_flat ** 2).mean(axis=-1),
        (tg_flat ** 2).mean(axis=-1),
        (on_flat * tg_flat).mean(axis=-1),
    ], axis=-1)  # (T, 5)


def _eval_gap(gap, record: Mapping[str, jnp.ndarray]) -> float:
    """Helper: evaluate a gap measurable through the resolver so
    its declared measurable-deps (e.g. `pearson_r_online_target`)
    auto-resolve from the registry."""
    from corroborate.measurable import evaluate_with_measurables
    return float(evaluate_with_measurables(gap.fn, dict(record)))


def test_hasselt_gap_one_when_q_values_perfectly_correlated() -> None:
    """Online and target Q-values identical (varying across
    samples) → Pearson r = 1 → gap = 1 (DDQN reduces to vanilla,
    theorem doesn't bite)."""
    arr = jnp.arange(50 * 16 * 2, dtype=jnp.float32).reshape((50, 16, 2))
    record: Mapping[str, jnp.ndarray] = {
        'pearson_stats': _pearson_stats_from_arrays(arr, arr),
    }
    gap = hasselt_covariance_gap()
    assert abs(_eval_gap(gap, record) - 1.0) < 1e-5


def test_hasselt_gap_zero_when_q_values_perfectly_anti_correlated() -> None:
    """Anti-correlation (r = -1) is FAVOURABLE for DDQN —
    estimators cancel rather than reinforce. Gap is asymmetric:
    `max(0, r)`, so anti-correlation ⇒ gap = 0 (not 1)."""
    arr = jnp.arange(50 * 16 * 2, dtype=jnp.float32).reshape((50, 16, 2))
    record: Mapping[str, jnp.ndarray] = {
        'pearson_stats': _pearson_stats_from_arrays(arr, -arr),
    }
    gap = hasselt_covariance_gap()
    assert _eval_gap(gap, record) == 0.0


def test_hasselt_gap_nan_when_q_values_constant() -> None:
    """Zero variance on either side ⇒ correlation undefined ⇒
    NaN no-data sentinel (no information about independence
    either way)."""
    import math
    on = jnp.ones((50, 16, 2))
    tg = jnp.arange(50 * 16 * 2, dtype=jnp.float32).reshape((50, 16, 2))
    record: Mapping[str, jnp.ndarray] = {
        'pearson_stats': _pearson_stats_from_arrays(on, tg),
    }
    gap = hasselt_covariance_gap()
    assert math.isnan(_eval_gap(gap, record))


def test_hasselt_gap_carries_pearson_stats_read() -> None:
    gap = hasselt_covariance_gap()
    assert gap.reads == ('pearson_stats',)


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
    """Synthetic flat-then-rising td_error → fqi_decay_gap large
    (window-norm ratios well above gamma) → over threshold →
    INVARIANT_VIOLATION (theorem out of scope)."""
    gamma = 0.5
    sync_period = 10
    # Per-window sup norms 1.0, 2.0, 4.0, 8.0 → ratio = 2.0 ≫ γ = 0.5.
    parts: list[jnp.ndarray] = []
    for sup in (1.0, 2.0, 4.0, 8.0):
        w = jnp.full(sync_period, sup * 0.5, dtype=jnp.float32)
        w = w.at[0].set(sup)
        parts.append(w)
    record: Mapping[str, jnp.ndarray] = {
        'td_error': jnp.concatenate(parts),
    }
    gap = fqi_decay_gap(sync_period=sync_period, gamma=gamma)
    bridge = at_most(gap, threshold=0.5, of_claim=periodic_copy)
    result = bridge(record)
    assert result.verdict is Verdict.INVARIANT_VIOLATION


def test_at_most_wrap_targets_propagate_from_gap_reads() -> None:
    gap = hasselt_covariance_gap()
    bridge = at_most(gap, threshold=0.5, of_claim=bootstrap)
    assert bridge.targets == ('pearson_stats',)


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


def test_jensen_floor_scales_with_log_action_dim() -> None:
    """σ × √(2 log |A|): doubling |A| from 2 to 4 increases the
    floor by √(log 4 / log 2) = √2 ≈ 1.414. Action dim emerges
    from `q.shape[-1]`; no separate plumbing."""
    import math
    sigma = 1.0
    # |A|=2 with σ=1: floor = √(2·log 2) ≈ 1.1774
    q2 = jnp.tile(jnp.array([[-sigma, sigma]]), (10, 1))
    # |A|=4 with σ=1 (achieved by symmetric ±σ split): floor =
    # √(2·log 4) ≈ 1.6651
    q4 = jnp.tile(jnp.array([[-sigma, -sigma, sigma, sigma]]), (10, 1))
    floor = jensen_floor_late()
    f2 = floor({'online_q_per_action': q2})
    f4 = floor({'online_q_per_action': q4})
    assert math.isclose(f2, math.sqrt(2.0 * math.log(2.0)), rel_tol=1e-3)
    assert math.isclose(f4, math.sqrt(2.0 * math.log(4.0)), rel_tol=1e-3)
    # Ratio is √(log 4 / log 2) = √2.
    assert math.isclose(f4 / f2, math.sqrt(2.0), rel_tol=1e-3)


def test_jensen_floor_late_uses_late_half_only() -> None:
    """Early half ignored: floor reflects late-training σ,
    matching where DDQN's correction is supposed to bite."""
    import math
    n_steps = 20
    early = jnp.tile(jnp.array([[-3.0, 3.0]]), (n_steps // 2, 1))   # σ=3
    late = jnp.tile(jnp.array([[-1.0, 1.0]]), (n_steps // 2, 1))    # σ=1
    q = jnp.concatenate([early, late], axis=0)
    floor = jensen_floor_late()
    val = floor({'online_q_per_action': q})
    # Expected: σ_late=1, |A|=2 → 1 × √(2 log 2)
    assert math.isclose(val, math.sqrt(2.0 * math.log(2.0)), rel_tol=1e-3)


def test_jensen_floor_nan_for_too_few_steps_or_actions() -> None:
    """Defensive: degenerate shapes (≤1 action or ≤1 step) yield
    NaN — the formula's domain doesn't apply."""
    import math
    floor = jensen_floor_late()
    assert math.isnan(floor({
        'online_q_per_action': jnp.array([[1.0]]),  # |A|=1
    }))
    assert math.isnan(floor({
        'online_q_per_action': jnp.array([[1.0, 2.0]]),  # 1 step
    }))


def test_jensen_dormancy_gap_zero_when_observed_above_floor() -> None:
    """Premise active: observed bias ≥ structural floor. dormancy
    gap = max(0, floor − observed) = 0."""
    sigma = 1.0
    q = jnp.tile(jnp.array([[-sigma, sigma]]), (10, 1))
    record: Mapping[str, jnp.ndarray] = {
        'predicted_q_at_start': jnp.full((5, 4), 11.0),
        'mc_return': jnp.full((5, 4), 10.0),    # observed_gap = 1.0
        'online_q_per_action': q,                # floor ≈ 1.18
    }
    # Wait — with observed=1.0 < floor≈1.18, premise is dormant.
    # Use a larger observed gap to demonstrate the active case.
    record = {
        **record,
        'predicted_q_at_start': jnp.full((5, 4), 12.0),  # gap=2.0 > 1.18
    }
    gap = jensen_dormancy_gap()
    assert gap(record) == 0.0


def test_jensen_dormancy_gap_positive_when_observed_below_floor() -> None:
    """Premise dormant: observed bias < structural floor → dormancy
    gap = floor − observed > 0."""
    sigma = 1.0
    q = jnp.tile(jnp.array([[-sigma, sigma]]), (10, 1))
    record: Mapping[str, jnp.ndarray] = {
        'predicted_q_at_start': jnp.full((5, 4), 10.5),
        'mc_return': jnp.full((5, 4), 10.0),    # observed_gap = 0.5
        'online_q_per_action': q,                # floor ≈ 1.1774
    }
    import math
    val = jensen_dormancy_gap()(record)
    expected_floor = math.sqrt(2.0 * math.log(2.0))
    assert math.isclose(val, expected_floor - 0.5, rel_tol=1e-3)


def test_jensen_dormancy_gap_carries_three_reads() -> None:
    """Reads-set is the union of jensen_overestimation_gap +
    jensen_floor_late."""
    gap = jensen_dormancy_gap()
    assert set(gap.reads) == {
        'predicted_q_at_start', 'mc_return', 'online_q_per_action',
    }


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
