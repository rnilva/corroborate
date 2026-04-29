"""Tests for the DDQN measurables — declared scalar derivations
of the per-step record. Verifies values, shapes, registration,
and transitive resolution via `evaluate_with_measurables`."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from corroborate.measurable import (
    evaluate_with_measurables,
    get_registered,
)
from corroborate.rl.dqn.measurables import (
    pearson_r_online_target,
    q_gap_per_step,
    q_max_per_step,
    q_mean_per_step,
    q_std_per_step,
    target_q_max_per_step,
    target_q_mean_per_step,
    td_error_norm_per_step,
)


# ============ Registration ============

def test_all_dqn_measurables_registered() -> None:
    """Every measurable defined in `rl.dqn.measurables` is
    indexed in the global registry by its function name."""
    for m in (
        q_mean_per_step, q_max_per_step, q_std_per_step,
        q_gap_per_step, target_q_mean_per_step,
        target_q_max_per_step, td_error_norm_per_step,
        pearson_r_online_target,
    ):
        assert get_registered(m.name) is m


# ============ Q-distribution measurables ============

def _q_record() -> Mapping[str, object]:
    """Synthetic record with per-step Q-vectors over 3 actions."""
    online = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 4.0, 5.0],
        [0.0, 1.0, 2.0],
    ])
    target = np.array([
        [0.5, 1.5, 2.5],
        [3.0, 3.5, 4.5],
        [0.5, 0.5, 1.5],
    ])
    return {
        'online_q_per_action': online,
        'target_q_per_action': target,
        'td_error': np.array([0.1, 0.2, -0.3]),
        'pearson_stats': np.zeros((3, 5)),  # placeholder
    }


def test_q_mean_per_step_value() -> None:
    rec = _q_record()
    out = q_mean_per_step(rec)
    assert out.tolist() == [2.0, (4 + 4 + 5) / 3, 1.0]


def test_q_max_per_step_value() -> None:
    rec = _q_record()
    out = q_max_per_step(rec)
    assert out.tolist() == [3.0, 5.0, 2.0]


def test_q_std_per_step_finite() -> None:
    rec = _q_record()
    out = q_std_per_step(rec)
    assert all(math.isfinite(v) for v in out.tolist())


def test_q_gap_per_step_value() -> None:
    """Gap = max - second_max. For [1, 2, 3], gap = 3 - 2 = 1.
    For [4, 4, 5], gap = 5 - 4 = 1."""
    rec = _q_record()
    out = q_gap_per_step(rec)
    assert out.tolist() == [1.0, 1.0, 1.0]


def test_target_q_mean_per_step_value() -> None:
    rec = _q_record()
    out = target_q_mean_per_step(rec)
    expected = [1.5, (3.0 + 3.5 + 4.5) / 3, (0.5 + 0.5 + 1.5) / 3]
    for got, want in zip(out.tolist(), expected, strict=True):
        assert math.isclose(got, want)


def test_target_q_max_per_step_value() -> None:
    rec = _q_record()
    out = target_q_max_per_step(rec)
    assert out.tolist() == [2.5, 4.5, 1.5]


def test_td_error_norm_per_step_value() -> None:
    """td_error 1-D: per-step magnitude is just abs(td_error)."""
    rec = _q_record()
    out = td_error_norm_per_step(rec)
    assert out.tolist() == [0.1, 0.2, 0.3]


# ============ Transitive resolution via evaluate_with_measurables ============

def test_consumer_can_declare_measurables_as_params() -> None:
    """A consumer fn that declares q_max_per_step and
    q_mean_per_step as parameters has both auto-injected by
    `evaluate_with_measurables`."""
    def consumer(
        record: Mapping[str, object],
        q_max_per_step: np.ndarray,
        q_mean_per_step: np.ndarray,
    ) -> float:
        del record
        # Average gap between max and mean across steps.
        return float((q_max_per_step - q_mean_per_step).mean())

    out = evaluate_with_measurables(consumer, _q_record())
    # Per-step (max - mean) for the 3 steps in _q_record:
    # (3 - 2) = 1; (5 - 13/3) ≈ 0.667; (2 - 1) = 1. Mean ≈ 0.889.
    assert math.isclose(out, (1.0 + (5.0 - 13/3) + 1.0) / 3, abs_tol=1e-9)


def test_pearson_r_online_target_via_resolver() -> None:
    """Pearson-r measurable computed through the resolver. Inputs
    designed so r ≈ 1 (online and target have the same shape)."""
    online = np.arange(3 * 4 * 2, dtype=np.float64).reshape((3, 4, 2))
    on_flat = online.reshape(online.shape[0], -1)
    target_flat = on_flat.copy()
    pearson_stats = np.stack([
        on_flat.mean(axis=-1),
        target_flat.mean(axis=-1),
        (on_flat ** 2).mean(axis=-1),
        (target_flat ** 2).mean(axis=-1),
        (on_flat * target_flat).mean(axis=-1),
    ], axis=-1)
    rec: Mapping[str, object] = {'pearson_stats': pearson_stats}

    def consumer(
        record: Mapping[str, object],
        pearson_r_online_target: float,
    ) -> float:
        del record
        return pearson_r_online_target

    r = evaluate_with_measurables(consumer, rec)
    assert math.isclose(r, 1.0, abs_tol=1e-5)


# ============ Value-curve mediators (D3) ============

def _learning_curve_record(
    burst_means: list[float],
) -> Mapping[str, object]:
    """Build a synthetic record where `mc_return` is shaped
    `(n_bursts, K)` and every burst column is the same value
    (so per-burst means equal `burst_means[i]` exactly).
    """
    mc = np.array([[v] * 3 for v in burst_means], dtype=np.float64)
    return {'mc_return': mc}


def test_learning_curve_auc_linear_curve() -> None:
    """Linearly increasing returns 0..10 over 11 bursts → AUC of
    a triangle = 5.0 (mean of 0..10)."""
    from corroborate.rl.dqn.measurables import learning_curve_auc
    rec = _learning_curve_record([float(i) for i in range(11)])
    assert math.isclose(learning_curve_auc(rec), 5.0, abs_tol=1e-6)


def test_learning_curve_auc_constant_curve() -> None:
    """Constant return c → AUC = c."""
    from corroborate.rl.dqn.measurables import learning_curve_auc
    rec = _learning_curve_record([3.0] * 10)
    assert math.isclose(learning_curve_auc(rec), 3.0, abs_tol=1e-6)


def test_learning_curve_auc_missing_key_returns_nan() -> None:
    from corroborate.rl.dqn.measurables import learning_curve_auc
    assert math.isnan(learning_curve_auc({}))


def test_time_to_threshold_step_function() -> None:
    """Returns rise from 0 to 10 at burst index 5 → time-to-50%
    threshold (i.e. >= 5.0) is index 5 / (11-1) = 0.5."""
    from corroborate.rl.dqn.measurables import time_to_threshold
    burst_means = [0.0] * 5 + [10.0] * 6
    rec = _learning_curve_record(burst_means)
    t = time_to_threshold(rec, target_frac=0.5)
    assert math.isclose(t, 0.5, abs_tol=1e-6)


def test_time_to_threshold_never_crosses_returns_one() -> None:
    """Peak never reached → return value 1.0 (sentinel)."""
    from corroborate.rl.dqn.measurables import time_to_threshold
    rec = _learning_curve_record([1.0, 1.0, 1.0])
    t = time_to_threshold(rec, target_frac=2.0)  # 2× peak unreachable
    assert math.isclose(t, 1.0, abs_tol=1e-6)


def test_return_at_25pct_steps() -> None:
    """16-burst run, 25%-of-burst-axis index = 4. Return at burst
    4 = 4.0 in this synthetic."""
    from corroborate.rl.dqn.measurables import return_at_25pct_steps
    rec = _learning_curve_record([float(i) for i in range(16)])
    assert math.isclose(
        return_at_25pct_steps(rec), 4.0, abs_tol=1e-6,
    )


def test_plateau_slope_late_positive_climb() -> None:
    """Final-25% bursts increase by 1 per burst → slope 1.0."""
    from corroborate.rl.dqn.measurables import plateau_slope_late
    burst_means = [5.0] * 12 + [
        5.0, 6.0, 7.0, 8.0,
    ]  # last 4 of 16 are climbing 5→8
    rec = _learning_curve_record(burst_means)
    slope = plateau_slope_late(rec, frac=0.25)
    assert math.isclose(slope, 1.0, abs_tol=1e-6)


def test_plateau_slope_late_zero_on_constant() -> None:
    """Constant late tail → slope ≈ 0."""
    from corroborate.rl.dqn.measurables import plateau_slope_late
    rec = _learning_curve_record([5.0] * 16)
    assert math.isclose(
        plateau_slope_late(rec, frac=0.25), 0.0, abs_tol=1e-6,
    )
