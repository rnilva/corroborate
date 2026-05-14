"""Tests for `corroborate_rl.dqn.trace_reductions` — the trace-write-time
reductions that compress per-action Q tensors into per-step scalars
BEFORE `Q_TRACE_DROPS` removes the raw column. The most behaviorally
distinct of these is `_q_action_temporal_corr_at_state_late`, which can ONLY
be a trace reduction (not a post-hoc `@measurable`) because its
source column is dropped at persistence."""
from __future__ import annotations

import math
import statistics

import numpy as np
import pytest

from corroborate_rl.dqn.trace_reductions import (
    _online_max_q_m,
    _online_mean_q_m,
    _online_min_q_m,
    _online_std_q_m,
    _q_action_temporal_corr_at_state_late,
    _target_max_q_m,
)


def test_q_action_temporal_corr_perfect_co_motion_returns_one() -> None:
    """Synthetic high-intra-coupling cell: all actions co-move at
    every state visit. Per-state pairwise Pearson r → 1."""
    # 20 timesteps, 4 actions, 2 distinct states each revisited 10×
    # Per-state Q-vectors at consecutive visits: actions move together
    visits_state_0 = [[t, t + 0.1, t + 0.2, t + 0.3] for t in range(10)]
    visits_state_1 = [[t + 5.0, t + 5.1, t + 5.2, t + 5.3] for t in range(10)]
    q_per_step = visits_state_0 + visits_state_1
    state_hash = [0] * 10 + [1] * 10
    record = {'online_q_per_action': q_per_step, 'state_hash': state_hash}
    r = _q_action_temporal_corr_at_state_late(record)
    # Late half: state 0 t=5..9 and state 1 t=0..4. Each state has 5
    # visits — exactly at the min_state_visits threshold.
    # Per state actions move in lock step → pairwise r = +1.
    assert math.isclose(r, 1.0, abs_tol=1e-9)


def test_q_action_temporal_corr_independent_actions_returns_near_zero() -> None:
    """Synthetic zero-intra-coupling: independent random sequences
    per action at each state visit. Mean pairwise r → 0 in expectation."""
    import random
    rng = random.Random(0)
    # Same state visited 100 times, 3 actions with independent draws
    visits = [[rng.gauss(0, 1) for _ in range(3)] for _ in range(100)]
    record = {
        'online_q_per_action': visits,
        'state_hash': [42] * 100,
    }
    r = _q_action_temporal_corr_at_state_late(record)
    # Empirical r over 50 samples × 3 pairs has SE ≈ 0.14; |r| < 0.4
    # is a very generous bound that the closed-form null comfortably
    # satisfies. The test catches the structural failure (returning
    # 1.0 or NaN) rather than asserting tightness at the SE level.
    assert math.isfinite(r)
    assert abs(r) < 0.4


def test_q_action_temporal_corr_missing_keys_returns_nan() -> None:
    assert math.isnan(_q_action_temporal_corr_at_state_late({}))
    assert math.isnan(_q_action_temporal_corr_at_state_late({'online_q_per_action': [[0.0, 1.0]]}))
    assert math.isnan(_q_action_temporal_corr_at_state_late({'state_hash': [0]}))


def test_q_action_temporal_corr_too_few_state_visits_returns_nan() -> None:
    """Fewer than 5 visits to ANY state in the late window → NaN
    (insufficient power to estimate per-state pairwise correlations)."""
    record = {
        # 10 steps, 10 distinct states (no repeats) → no state has ≥5 visits
        'online_q_per_action': [[1.0, 2.0, 3.0]] * 10,
        'state_hash': list(range(10)),
    }
    assert math.isnan(_q_action_temporal_corr_at_state_late(record))


def test_q_action_temporal_corr_zero_variance_state_dropped() -> None:
    """A state where actions never vary across visits has zero
    pairwise variance — that state contributes nothing, but other
    valid states still average normally."""
    # State 0: 6 visits, all actions identical (zero variance, skipped)
    # State 1: 6 visits, perfect co-motion (r=1 for all action pairs)
    zero_var_visits = [[1.0, 2.0, 3.0]] * 6
    co_motion_visits = [[t, t + 0.5, t + 1.0] for t in range(6)]
    record = {
        'online_q_per_action': (
            zero_var_visits + co_motion_visits
            + zero_var_visits + co_motion_visits  # ×2 so late-half spans both
        ),
        'state_hash': [0] * 6 + [1] * 6 + [0] * 6 + [1] * 6,
    }
    r = _q_action_temporal_corr_at_state_late(record)
    # State 0 (zero-var) contributes nothing; state 1 (co-motion) → r=1
    assert math.isclose(r, 1.0, abs_tol=1e-9)


def test_q_action_temporal_corr_partial_anticorr_yields_negative() -> None:
    """One action pair perfectly anti-correlated, other pairs neutral.
    The averaged off-diagonal should be moderately negative."""
    # 3 actions, 6 visits to one state:
    # action 0 = ramp up, action 1 = ramp down (perfectly anti-corr),
    # action 2 = constant (no variance — its pairs are dropped).
    visits = []
    for t in range(6):
        visits.append([float(t), float(5 - t), 0.0])
    record = {
        'online_q_per_action': visits + visits,  # 12 visits so late-half = 6
        'state_hash': [99] * 12,
    }
    r = _q_action_temporal_corr_at_state_late(record)
    # Action 2 (constant) pairs with anyone gives zero-variance → skip.
    # Only the (0, 1) pair contributes → r = -1.0.
    assert math.isclose(r, -1.0, abs_tol=1e-9)


# ============ Framework-primitive trace-time Measurable parity ============
#
# The 5 simple `reduce_axis(SOURCE, axis=-1, op=...)` Measurables in
# `Q_TRACE_REDUCTIONS` replace the deleted `_per_step_max_q` /
# `_per_step_min_q` / `_per_step_mean_q` / `_per_step_std_q`
# closures. Verify that for a synthetic per-row struct, each
# Measurable produces output equivalent to the closure's original
# behavior on the same input.


@pytest.fixture
def synthetic_q_struct() -> dict[str, list[list[float]]]:
    """Per-row struct with 2-D shape `(T=5 steps, A=4 actions)` for
    both online and target Q. Distinct values per (t, a) so the
    max/min/mean/std are all non-trivial."""
    online = [
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.5, 2.5, 3.5],
        [-1.0, 0.0, 1.0, 2.0],
        [5.0, 4.0, 3.0, 2.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    target = [
        [0.9, 1.9, 2.9, 3.9],
        [0.4, 1.4, 2.4, 3.4],
        [-1.1, -0.1, 0.9, 1.9],
        [4.9, 3.9, 2.9, 1.9],
        [0.1, 0.1, 0.1, 0.1],
    ]
    return {
        'online_q_per_action': online,
        'target_q_per_action': target,
    }


def test_online_max_q_measurable_matches_per_step_max(
    synthetic_q_struct: dict[str, list[list[float]]],
) -> None:
    """`reduce_axis(ONLINE_Q_PER_ACTION, axis=-1, op='max')` on a
    `(T, A)` row produces the same per-step max-over-actions list
    that the deleted `_per_step_max_q` closure produced."""
    result = _online_max_q_m(synthetic_q_struct)
    expected = [max(p) for p in synthetic_q_struct['online_q_per_action']]
    assert np.allclose(np.asarray(result), np.asarray(expected))


def test_target_max_q_measurable_matches_per_step_max(
    synthetic_q_struct: dict[str, list[list[float]]],
) -> None:
    result = _target_max_q_m(synthetic_q_struct)
    expected = [max(p) for p in synthetic_q_struct['target_q_per_action']]
    assert np.allclose(np.asarray(result), np.asarray(expected))


def test_online_min_q_measurable_matches_per_step_min(
    synthetic_q_struct: dict[str, list[list[float]]],
) -> None:
    result = _online_min_q_m(synthetic_q_struct)
    expected = [min(p) for p in synthetic_q_struct['online_q_per_action']]
    assert np.allclose(np.asarray(result), np.asarray(expected))


def test_online_mean_q_measurable_matches_per_step_mean(
    synthetic_q_struct: dict[str, list[list[float]]],
) -> None:
    result = _online_mean_q_m(synthetic_q_struct)
    expected = [
        sum(p) / len(p)
        for p in synthetic_q_struct['online_q_per_action']
    ]
    assert np.allclose(np.asarray(result), np.asarray(expected))


def test_online_std_q_measurable_matches_per_step_std(
    synthetic_q_struct: dict[str, list[list[float]]],
) -> None:
    """`reduce_axis(SOURCE, axis=-1, op='std')` uses `np.std` (ddof=0,
    population std) — same as the deleted closure's
    `statistics.pstdev`."""
    result = _online_std_q_m(synthetic_q_struct)
    expected = [
        statistics.pstdev(p)
        for p in synthetic_q_struct['online_q_per_action']
    ]
    assert np.allclose(np.asarray(result), np.asarray(expected))


def test_trace_measurables_carry_compose_of() -> None:
    """The trace-time Measurables thread `compose_of` through
    `register_as` to the shared `ONLINE_Q_PER_ACTION` /
    `TARGET_Q_PER_ACTION` leaves — structural lineage preserved so
    `transitive_reads` (via `.reads`) returns the correct trace
    column names."""
    assert _online_max_q_m.reads == ('online_q_per_action',)
    assert _target_max_q_m.reads == ('target_q_per_action',)
    # compose_of carries the reduce_axis operand chain.
    assert len(_online_max_q_m.compose_of) == 1
    assert _online_max_q_m.compose_of[0].reads == ('online_q_per_action',)
