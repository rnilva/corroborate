"""Tests for `corroborate.statistics` — paired Hedges' g + MDE +
verdict-decision tree.

Synthetic-data tests with known answers:
1. Hedges' g formula matches the v9 reference.
2. Strong effect at adequate n → HELD.
3. Null effect at adequate n → NO_EFFECT (NULL_EFFECT or HELD-with-no-direction).
4. Small n with any effect → POWER_INSUFFICIENT.
5. Sign-flip (predicted positive, observed negative) → SIGN_FLIP."""
from __future__ import annotations

import math
import pytest

from corroborate.stats import (
    adequately_powered_paired,
    delta_i_from_q,
    derived_q_from_g_se,
    hedges_g_paired,
    mde_paired,
    verdict_from_paired_stats,
)
from corroborate.bridge.verdict import RefutationClass, Verdict


# ============ hedges_g_paired ============

def test_hedges_g_paired_zero_variance_returns_zero_g_nan_se() -> None:
    """All-equal deltas → g = 0, SE = NaN (no variation to estimate)."""
    g, se = hedges_g_paired([1.0, 1.0, 1.0, 1.0, 1.0])
    assert g == 0.0
    assert math.isnan(se)


def test_hedges_g_paired_too_small_n_returns_nan() -> None:
    """n < 2 → both NaN (no variance estimable)."""
    g, se = hedges_g_paired([5.0])
    assert math.isnan(g)
    assert math.isnan(se)


def test_hedges_g_paired_strong_positive_effect() -> None:
    """A clear positive effect (deltas ≈ 1.5 ± 0.25) gives a large
    positive g. SE grows with g (variance formula has a g²/(2n)
    term), so just check finite + positive."""
    deltas = [1.0, 1.5, 2.0, 1.2, 1.8, 1.4, 1.6, 1.3, 1.7, 1.5]
    g, se = hedges_g_paired(deltas)
    assert g > 1.5  # large effect
    assert se > 0.0  # finite, positive
    assert math.isfinite(se)


def test_hedges_g_paired_textbook_value() -> None:
    """Hedges' g on `[1.0, 2.0, 3.0]` should match the formula:
        mean = 2.0, stdev (ddof=1) = 1.0
        d = 2.0 / 1.0 = 2.0
        c_4 = 1 - 3 / (4*3 - 5) = 1 - 3/7 ≈ 0.5714
        g = 2.0 * 0.5714 ≈ 1.1429
        var = 1/3 + 1.1429^2 / 6 ≈ 0.5510
        se ≈ 0.7423"""
    g, se = hedges_g_paired([1.0, 2.0, 3.0])
    assert g == pytest.approx(2.0 * (1 - 3 / 7), rel=1e-6)
    expected_var = 1.0 / 3 + g * g / 6
    assert se == pytest.approx(math.sqrt(expected_var), rel=1e-6)


# ============ mde_paired ============

def test_mde_paired_decreases_with_n() -> None:
    """MDE shrinks as n grows — same α + power."""
    mde_10 = mde_paired(10, alpha=0.05, power=0.8)
    mde_100 = mde_paired(100, alpha=0.05, power=0.8)
    assert mde_100 < mde_10


def test_mde_paired_too_small_n_returns_nan() -> None:
    assert math.isnan(mde_paired(1))


# ============ derived_q + delta_i ============

def test_derived_q_zero_when_g_zero() -> None:
    """g=0, se>0 → Φ(0) = 0.5."""
    q = derived_q_from_g_se(0.0, 0.5)
    assert q == pytest.approx(0.5)


def test_derived_q_one_for_large_g_over_se() -> None:
    """Strong positive signal → q close to 1."""
    q = derived_q_from_g_se(5.0, 0.5)  # z = 10
    assert q > 0.99


def test_delta_i_zero_at_q_half() -> None:
    """q=0.5 → H_2(0.5) = 1 → ΔI = 0 (no information)."""
    assert delta_i_from_q(0.5) == pytest.approx(0.0)


def test_delta_i_one_at_perfect_signal() -> None:
    """q approaches 1 → ΔI approaches 1."""
    assert delta_i_from_q(0.99999) > 0.99


def test_delta_i_zero_at_boundary() -> None:
    """q = 0 or 1 returns 0 (boundary convention)."""
    assert delta_i_from_q(0.0) == 0.0
    assert delta_i_from_q(1.0) == 0.0


# ============ adequately_powered + verdict_from_paired_stats ============

def test_adequately_powered_strong_effect_n10() -> None:
    """g ≈ 1.5 at n=10 should be adequately powered."""
    assert adequately_powered_paired(1.5, 10, alpha=0.05, power=0.8)


def test_adequately_powered_weak_effect_n10() -> None:
    """g ≈ 0.1 at n=10 should NOT be adequately powered."""
    assert not adequately_powered_paired(0.1, 10, alpha=0.05, power=0.8)


def test_verdict_held_for_strong_positive_with_predicted_positive() -> None:
    """Strong positive g + predicted_direction='a_gt_b' → HELD."""
    g, se = 1.5, 0.4
    verdict, refutation, is_powered = verdict_from_paired_stats(
        g, se, n=10, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.HELD
    assert refutation is None
    assert is_powered


def test_verdict_sign_flip_for_negative_when_positive_predicted() -> None:
    """Strong negative g + predicted positive → NO_EFFECT/SIGN_FLIP."""
    g, se = -1.5, 0.4
    verdict, refutation, is_powered = verdict_from_paired_stats(
        g, se, n=10, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.NO_EFFECT
    assert refutation is RefutationClass.SIGN_FLIP
    assert is_powered


def test_verdict_underpowered_for_small_effect_at_n10() -> None:
    """Effect below MDE → POWER_INSUFFICIENT/UNDERPOWERED."""
    g, se = 0.05, 0.4
    verdict, refutation, is_powered = verdict_from_paired_stats(
        g, se, n=10, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.POWER_INSUFFICIENT
    assert refutation is RefutationClass.UNDERPOWERED
    assert not is_powered


def test_verdict_held_for_two_sided_with_either_sign() -> None:
    """predicted='two_sided' admits either sign at adequate power."""
    for g in (1.5, -1.5):
        verdict, refutation, is_powered = verdict_from_paired_stats(
            g, 0.4, n=10, predicted_direction='two_sided',
        )
        assert verdict is Verdict.HELD
        assert refutation is None
        assert is_powered


def test_verdict_held_for_no_predicted_direction() -> None:
    """predicted=None admits either sign at adequate power."""
    verdict, _, is_powered = verdict_from_paired_stats(
        1.5, 0.4, n=10, predicted_direction=None,
    )
    assert verdict is Verdict.HELD
    assert is_powered


# ============ random_effects_summary ============

def test_random_effects_summary_too_few_cells_returns_nan() -> None:
    """n<2 → all NaN."""
    from corroborate.stats import random_effects_summary
    p = random_effects_summary([(0.5, 0.2)])
    assert math.isnan(p.pooled_g)
    assert p.n_cells == 1


def test_random_effects_summary_homogeneous_cells_zero_tau2() -> None:
    """All cells with same g + same SE → tau² = 0 (no
    between-cell heterogeneity), pooled_g = common value."""
    from corroborate.stats import random_effects_summary
    p = random_effects_summary([(0.5, 0.2)] * 5)
    assert p.n_cells == 5
    assert p.pooled_g == pytest.approx(0.5, abs=1e-6)
    assert p.tau2 == pytest.approx(0.0, abs=1e-6)
    # PI brackets the common value at low heterogeneity.
    assert p.pi_lo < 0.5 < p.pi_hi


def test_random_effects_summary_heterogeneous_cells_positive_tau2() -> None:
    """Wildly different g across cells → tau² > 0; pooled estimate
    near the mean of the cells."""
    from corroborate.stats import random_effects_summary
    g_se = [(-0.5, 0.2), (1.5, 0.2), (0.0, 0.2), (2.0, 0.2)]
    p = random_effects_summary(g_se)
    assert p.n_cells == 4
    assert p.tau2 > 0.0
    assert p.I2 > 0.0
    # Pooled near simple average of g's (since SEs are equal).
    assert p.pooled_g == pytest.approx(0.75, abs=0.1)


# ============ random_effects_verdict ============

def test_random_effects_verdict_underpowered_below_three_cells() -> None:
    from corroborate.stats import (
        random_effects_summary,
        random_effects_verdict,
    )
    p = random_effects_summary([(0.5, 0.1), (0.5, 0.1)])
    verdict, refutation = random_effects_verdict(
        p, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.POWER_INSUFFICIENT
    assert refutation is RefutationClass.UNDERPOWERED


def test_random_effects_verdict_held_for_pi_strictly_positive() -> None:
    from corroborate.stats import (
        random_effects_summary,
        random_effects_verdict,
    )
    # Strong consistent effect; tight SE → narrow PI excluding zero.
    p = random_effects_summary([(1.5, 0.05)] * 5)
    verdict, _ = random_effects_verdict(
        p, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.HELD


def test_random_effects_verdict_no_effect_when_pi_brackets_zero() -> None:
    from corroborate.stats import (
        random_effects_summary,
        random_effects_verdict,
    )
    # High heterogeneity around zero → PI brackets zero.
    p = random_effects_summary([
        (-1.0, 0.3), (1.0, 0.3), (-0.5, 0.3), (0.5, 0.3), (0.0, 0.3),
    ])
    verdict, refutation = random_effects_verdict(
        p, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.NO_EFFECT
    assert refutation is RefutationClass.NULL_EFFECT


def test_random_effects_verdict_held_with_scope_flag_high_heterogeneity() -> None:
    """Heterogeneous-but-positive: PI excludes zero in predicted
    direction AND I² is high → HELD_WITH_SCOPE_FLAG. Discovery's
    natural input — meta-regression should identify cleavages.

    Constructed via direct PooledStats so the test asserts on the
    verdict logic, not on the synthesis-via-`random_effects_summary`
    path (which couples PI width to τ²)."""
    from corroborate.stats import (
        I2_THRESHOLD,
        PooledStats,
        random_effects_verdict,
    )
    p = PooledStats(
        pooled_g=4.0, se_pooled=0.05,
        tau2=0.5, I2=0.6, Q=10.0,
        pi_lo=2.0, pi_hi=6.0,
        empirical_min_g=3.0, empirical_max_g=5.0,
        n_cells=5,
    )
    assert p.I2 >= I2_THRESHOLD
    verdict, _ = random_effects_verdict(
        p, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.HELD_WITH_SCOPE_FLAG
    assert verdict.is_corroboration()
    assert not verdict.is_uniform()


def test_random_effects_verdict_held_uniform_low_heterogeneity() -> None:
    """Same effect across strata → low I² → plain HELD."""
    from corroborate.stats import (
        I2_THRESHOLD,
        random_effects_summary,
        random_effects_verdict,
    )
    # Five identical-effect strata → I² ≈ 0.
    p = random_effects_summary([(1.5, 0.05)] * 5)
    assert p.I2 < I2_THRESHOLD
    verdict, _ = random_effects_verdict(
        p, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.HELD
    assert verdict.is_uniform()


def test_random_effects_verdict_held_with_scope_flag_predicted_negative() -> None:
    """Symmetric case: a_lt_b with PI strictly negative + high I²
    → HELD_WITH_SCOPE_FLAG (the corroborated direction is
    negative, but heterogeneous in magnitude)."""
    from corroborate.stats import (
        PooledStats,
        random_effects_verdict,
    )
    p = PooledStats(
        pooled_g=-4.0, se_pooled=0.05,
        tau2=0.5, I2=0.7, Q=12.0,
        pi_lo=-6.0, pi_hi=-2.0,
        empirical_min_g=-5.0, empirical_max_g=-3.0,
        n_cells=5,
    )
    verdict, _ = random_effects_verdict(
        p, predicted_direction='a_lt_b',
    )
    assert verdict is Verdict.HELD_WITH_SCOPE_FLAG


# ============ recommended_n_paired ============

def test_recommended_n_paired_inverts_mde_paired() -> None:
    """If observed g equals the MDE at n=10, the recommended n
    should be ~10 (round-trip consistency)."""
    from corroborate.stats import (
        mde_paired,
        recommended_n_paired,
    )
    mde_at_10 = mde_paired(10, alpha=0.05, power=0.8)
    rec_n = recommended_n_paired(mde_at_10, alpha=0.05, power=0.8)
    assert rec_n == pytest.approx(10.0, rel=0.05)


def test_recommended_n_paired_zero_g_returns_nan() -> None:
    """Detecting a true zero effect with positive power is
    impossible — no finite n works."""
    from corroborate.stats import recommended_n_paired
    assert math.isnan(recommended_n_paired(0.0))


def test_recommended_n_paired_smaller_g_needs_larger_n() -> None:
    """Detecting a smaller effect needs a larger sample."""
    from corroborate.stats import recommended_n_paired
    n_big_effect = recommended_n_paired(1.0)
    n_small_effect = recommended_n_paired(0.2)
    assert n_small_effect > n_big_effect
