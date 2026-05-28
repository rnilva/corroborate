"""Tests for the DDQN measurables — declared scalar derivations
of the per-step record. Verifies values, shapes, registration,
and transitive resolution via `evaluate_with_measurables`."""
from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from corroborate.measurables import (
    evaluate_with_measurables,
    get_registered,
)
from corroborate_rl.dqn.measurables import pearson_r_online_target


# ============ Registration ============

def test_pearson_r_online_target_registered() -> None:
    """The measurable is indexed in the global registry under its
    function name."""
    assert get_registered(pearson_r_online_target.name) is pearson_r_online_target


# ============ Pearson r — online vs target ============

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
    from corroborate_rl.dqn.measurables import learning_curve_auc
    rec = _learning_curve_record([float(i) for i in range(11)])
    assert math.isclose(learning_curve_auc(rec), 5.0, abs_tol=1e-6)


def test_learning_curve_auc_constant_curve() -> None:
    """Constant return c → AUC = c."""
    from corroborate_rl.dqn.measurables import learning_curve_auc
    rec = _learning_curve_record([3.0] * 10)
    assert math.isclose(learning_curve_auc(rec), 3.0, abs_tol=1e-6)


def test_learning_curve_auc_missing_key_returns_nan() -> None:
    from corroborate_rl.dqn.measurables import learning_curve_auc
    assert math.isnan(learning_curve_auc({}))


def test_time_to_threshold_step_function() -> None:
    """Returns rise from 0 to 10 at burst index 5 → time-to-50%
    threshold (i.e. >= 5.0) is index 5 / (11-1) = 0.5."""
    from corroborate_rl.dqn.measurables import time_to_threshold
    burst_means = [0.0] * 5 + [10.0] * 6
    rec = _learning_curve_record(burst_means)
    t = time_to_threshold(rec, target_frac=0.5)
    assert math.isclose(t, 0.5, abs_tol=1e-6)


def test_time_to_threshold_never_crosses_returns_one() -> None:
    """Peak never reached → return value 1.0 (sentinel)."""
    from corroborate_rl.dqn.measurables import time_to_threshold
    rec = _learning_curve_record([1.0, 1.0, 1.0])
    t = time_to_threshold(rec, target_frac=2.0)  # 2× peak unreachable
    assert math.isclose(t, 1.0, abs_tol=1e-6)


def test_return_at_25pct_steps() -> None:
    """16-burst run, 25%-of-burst-axis index = 4. Return at burst
    4 = 4.0 in this synthetic."""
    from corroborate_rl.dqn.measurables import return_at_25pct_steps
    rec = _learning_curve_record([float(i) for i in range(16)])
    assert math.isclose(
        return_at_25pct_steps(rec), 4.0, abs_tol=1e-6,
    )


def test_plateau_slope_late_positive_climb() -> None:
    """Final-25% bursts increase by 1 per burst → slope 1.0."""
    from corroborate_rl.dqn.measurables import plateau_slope_late
    burst_means = [5.0] * 12 + [
        5.0, 6.0, 7.0, 8.0,
    ]  # last 4 of 16 are climbing 5→8
    rec = _learning_curve_record(burst_means)
    slope = plateau_slope_late(rec, frac=0.25)
    assert math.isclose(slope, 1.0, abs_tol=1e-6)


def test_plateau_slope_late_zero_on_constant() -> None:
    """Constant late tail → slope ≈ 0."""
    from corroborate_rl.dqn.measurables import plateau_slope_late
    rec = _learning_curve_record([5.0] * 16)
    assert math.isclose(
        plateau_slope_late(rec, frac=0.25), 0.0, abs_tol=1e-6,
    )


# ============ bootstrap_fraction (truncation-aware) ============

def test_bootstrap_fraction_truncation_aware_closed_form() -> None:
    """Closed-form: `done=[1,0,1,0], truncated=[0,0,1,0]` →
    `terminated=[1,0,0,0]` → `bootstrap_fraction = 1 - 0.25 = 0.75`.
    The corrected semantic distinguishes a truncated done (where
    the trajectory physically continues; bootstrap fires) from a
    genuine terminated done (bootstrap zeros)."""
    from corroborate_rl.dqn.measurables import bootstrap_fraction
    rec: Mapping[str, object] = {
        'done': np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
        'truncated': np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float64),
    }
    result = bootstrap_fraction(rec)
    # terminated = done * (1 - truncated) = [1, 0, 0, 0]
    # bootstrap_fraction = 1 - 0.25 = 0.75
    assert math.isclose(result, 0.75, abs_tol=1e-9), (
        f'truncation-aware bootstrap_fraction expected 0.75 (one '
        f'genuine terminal out of 4 steps); got {result}'
    )


def test_bootstrap_fraction_back_compat_no_truncated_column() -> None:
    """Pre-refactor corpus: no `truncated` column. The measurable
    defaults `truncated=0`, collapsing the formula to the
    pre-refactor `1 - mean(done)` semantic. Same record without
    `truncated` → `1 - mean([1,0,1,0]) = 1 - 0.5 = 0.5`. Confirms
    legacy cells don't suffer verdict drift."""
    from corroborate_rl.dqn.measurables import bootstrap_fraction
    rec: Mapping[str, object] = {
        'done': np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
        # Note: NO 'truncated' key.
    }
    result = bootstrap_fraction(rec)
    # Back-compat: truncated default = 0 → terminated = done → 0.5.
    assert math.isclose(result, 0.5, abs_tol=1e-9), (
        f'pre-refactor bootstrap_fraction expected 0.5 (legacy '
        f'done-only semantic); got {result}'
    )


def test_bootstrap_fraction_all_truncated_returns_one() -> None:
    """Cap-only env where every "done" is a timeout truncation:
    `done=[1,1,1,1], truncated=[1,1,1,1]` → terminated = 0
    everywhere → bootstrap fires on every step → fraction = 1.0.
    The trajectory physically continued past every cap; the
    Bellman target should never zero on these cells."""
    from corroborate_rl.dqn.measurables import bootstrap_fraction
    rec: Mapping[str, object] = {
        'done': np.ones(4, dtype=np.float64),
        'truncated': np.ones(4, dtype=np.float64),
    }
    assert math.isclose(bootstrap_fraction(rec), 1.0, abs_tol=1e-9)


def test_bootstrap_fraction_nan_on_missing_done() -> None:
    """No `done` key → NaN (the column-erasure-safe path the
    measurable's `reads=('done', 'truncated')` advertises)."""
    from corroborate_rl.dqn.measurables import bootstrap_fraction
    assert math.isnan(bootstrap_fraction({}))


def test_bootstrap_fraction_nan_on_empty_done() -> None:
    """Empty trace → NaN."""
    from corroborate_rl.dqn.measurables import bootstrap_fraction
    rec: Mapping[str, object] = {'done': np.array([], dtype=np.float64)}
    assert math.isnan(bootstrap_fraction(rec))


def test_bootstrap_fraction_shape_mismatch_falls_back() -> None:
    """If `truncated` is present but shape-mismatched with `done`
    (corrupted trace), fall back to done-only semantics rather
    than silently masking. The fall-back keeps the value
    interpretable even on schema drift."""
    from corroborate_rl.dqn.measurables import bootstrap_fraction
    rec: Mapping[str, object] = {
        'done': np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64),
        'truncated': np.array([1.0, 0.0], dtype=np.float64),  # wrong size
    }
    # Fall-back to 1 - mean(done) = 0.5.
    assert math.isclose(bootstrap_fraction(rec), 0.5, abs_tol=1e-9)
