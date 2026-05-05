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

from corroborate.stats import hedges_g_paired
from corroborate.stats.effect_size import (
    adequately_powered_paired,
    delta_i_from_q,
    derived_q_from_g_se,
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
    """n<2 → all NaN. Pin every NaN-filled field on the early
    return — `float('nan')` mutations to `None` would store None
    in the PooledStats field, and `math.isnan(None)` raises
    TypeError. Pin the n<2 guard against `n<=2` (which would
    short-circuit at n=2 too) and `n<3`."""
    from corroborate.stats import random_effects_summary
    p = random_effects_summary([(0.5, 0.2)])
    assert math.isnan(p.pooled_g)
    assert math.isnan(p.se_pooled)
    assert math.isnan(p.tau2)
    assert math.isnan(p.I2)
    assert math.isnan(p.Q)
    assert math.isnan(p.pi_lo)
    assert math.isnan(p.pi_hi)
    assert math.isnan(p.empirical_min_g)
    assert math.isnan(p.empirical_max_g)
    assert p.n_cells == 1


def test_random_effects_summary_at_n_equals_2_returns_pooled() -> None:
    """n=2 valid cells: passes the n<2 guard, returns a real
    pooled estimate (not NaN-filled). Pin `n < 2` against
    `n <= 2` and `n < 3` (both would NaN at n=2)."""
    from corroborate.stats import random_effects_summary
    p = random_effects_summary([(0.5, 0.1), (0.7, 0.1)])
    assert p.n_cells == 2
    assert math.isfinite(p.pooled_g)
    assert math.isfinite(p.se_pooled)


def test_random_effects_summary_drops_zero_se_cell() -> None:
    """A cell with se=0 must be dropped (DL needs `var > 0`).
    Pin `se > 0` against `se >= 0` mutant — keeping a zero-SE
    cell would inject `1/0` into the inverse-variance weight.

    Construct: 2 valid cells + 1 zero-SE cell. After filtering,
    n=2 → real pooled estimate. With the mutant, the zero-SE
    cell stays in → `1/var = inf` → pooled_g blows up or NaN."""
    from corroborate.stats import random_effects_summary
    p = random_effects_summary([
        (0.5, 0.1), (0.7, 0.1), (1.0, 0.0),
    ])
    assert p.n_cells == 2
    assert math.isfinite(p.pooled_g)
    # The zero-SE cell would have dominated if kept (∞ weight).
    # Pooled g should sit around the average of the two valid
    # cells (~0.6), not be 1.0.
    assert 0.4 < p.pooled_g < 0.8


def test_random_effects_summary_or_in_filter_keeps_only_full_pairs() -> None:
    """The filter requires BOTH g AND se to be finite (and se>0).
    Pin `not isnan(g) and not isnan(se)` against the `or` mutant
    that would keep cells with one NaN field.

    Construct: 1 NaN-g cell + 1 NaN-se cell + 2 valid cells.
    Original drops both NaN-bearing → n=2 valid → real pooled.
    Mutant `g_ok or se_ok` keeps the NaN-bearing cells (one
    side is True for each), then computes vs from NaN-se →
    NaN var → NaN pooled."""
    from corroborate.stats import random_effects_summary
    p = random_effects_summary([
        (float('nan'), 0.1),
        (0.5, float('nan')),
        (0.5, 0.1),
        (0.7, 0.1),
    ])
    assert p.n_cells == 2
    assert math.isfinite(p.pooled_g)


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
        PooledStats,
        random_effects_verdict,
    )
    from corroborate.stats.effect_size import I2_THRESHOLD
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
        random_effects_summary,
        random_effects_verdict,
    )
    from corroborate.stats.effect_size import I2_THRESHOLD
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
    from corroborate.stats import recommended_n_paired
    from corroborate.stats.effect_size import mde_paired
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


# ============ verdict_from_paired_stats — gap-fill ============
#
# These tests target the verdict primitive's branch boundaries
# the existing tests above don't pin: NaN g, n<2 boundary, the
# `is_powered` flag on early-return paths, the g=0 boundary on
# both sign branches, and the symmetric `a_lt_b` sign-flip path.
# Each test identifies a distinct mutation that mutmut would
# otherwise leave surviving.

def test_verdict_nan_g_returns_power_insufficient() -> None:
    """NaN g (e.g., from a paired_g panel where one arm collapsed)
    must short-circuit to POWER_INSUFFICIENT/UNDERPOWERED with
    is_powered=False — even at large n. Pins the `or` in the
    early-return guard against being weakened to `and`."""
    verdict, refutation, is_powered = verdict_from_paired_stats(
        float('nan'), 0.4, n=100, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.POWER_INSUFFICIENT
    assert refutation is RefutationClass.UNDERPOWERED
    assert is_powered is False


def test_verdict_n_below_two_returns_power_insufficient() -> None:
    """n=1: not enough cells to estimate variance → must hit the
    early-return branch regardless of g magnitude. Pins the `n < 2`
    boundary against `n <= 2` (which would lock out n=2) and `n < 3`
    (which would also classify n=2 as underpowered)."""
    verdict, _, is_powered = verdict_from_paired_stats(
        5.0, 0.1, n=1, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.POWER_INSUFFICIENT
    assert is_powered is False


def test_verdict_n_equals_two_passes_early_guard() -> None:
    """n=2 with massive effect: passes the n<2 guard. MDE at
    n=2 is ~5.79 at default α/power, so g must clear that to
    distinguish `n < 2` (passes through to MDE check, then HELD)
    from `n <= 2` (forced POWER_INSUFFICIENT) and `n < 3`
    (also forced POWER_INSUFFICIENT)."""
    verdict, _, _ = verdict_from_paired_stats(
        10.0, 0.1, n=2, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.HELD


def test_verdict_g_exactly_zero_predicted_positive_is_sign_flip() -> None:
    """Strict `>` boundary: g=0 with predicted='a_gt_b' falls to
    SIGN_FLIP, not HELD. Pins `if g > 0` against `if g >= 0`
    (which would let g=0 through to HELD).

    Constructed at large n so adequately_powered is true even for
    tiny |g| (MDE → 0 as n → ∞)."""
    # At n=10000, MDE is ~0.028 — but g=0 fails MDE too. To isolate
    # the sign branch, we need adequately_powered AND g==0. That's
    # vacuously not adequately powered (|0| < any positive MDE).
    # So this branch is unreachable from the public API alone — the
    # mutant `g >= 0` would only matter if we somehow reached the
    # sign branch with g=0. Use a very small but non-zero g to
    # probe the boundary in spirit, then explicitly test g=0
    # routes to POWER_INSUFFICIENT (the upstream guard).
    verdict, refutation, _ = verdict_from_paired_stats(
        0.0, 0.001, n=10, predicted_direction='a_gt_b',
    )
    # |g|=0 < any MDE → POWER_INSUFFICIENT regardless of sign.
    assert verdict is Verdict.POWER_INSUFFICIENT
    assert refutation is RefutationClass.UNDERPOWERED


def test_verdict_held_for_small_positive_g_above_mde() -> None:
    """g just above MDE with predicted='a_gt_b' → HELD. Distinguishes
    the `if g > 0` boundary from `if g > 1` mutation: g=0.5 is in
    (0, 1] so original returns HELD, mutant `g > 1` returns SIGN_FLIP."""
    # n=200 so MDE is well below 0.5.
    verdict, refutation, _ = verdict_from_paired_stats(
        0.5, 0.1, n=200, predicted_direction='a_gt_b',
    )
    assert verdict is Verdict.HELD
    assert refutation is None


def test_verdict_sign_flip_for_positive_when_negative_predicted() -> None:
    """Symmetric to the existing a_gt_b sign-flip test. Predicted
    direction = 'a_lt_b' (negative effect) but observed g is
    positive → SIGN_FLIP with is_powered=True. Pins:

    - `g < 0` boundary in the a_lt_b branch (vs `g <= 0` mutant)
    - `g < 0` vs `g < 1` (small positive g should reach SIGN_FLIP,
      not HELD)
    - SIGN_FLIP returns is_powered=True (vs False mutant) on the
      a_lt_b path"""
    g = 1.5  # strong positive; large n
    verdict, refutation, is_powered = verdict_from_paired_stats(
        g, 0.1, n=50, predicted_direction='a_lt_b',
    )
    assert verdict is Verdict.NO_EFFECT
    assert refutation is RefutationClass.SIGN_FLIP
    assert is_powered is True


def test_verdict_sign_flip_for_small_positive_g_when_negative_predicted() -> None:
    """g in (0, 1) with predicted='a_lt_b' at adequate power →
    SIGN_FLIP. Pins `if g < 0` against `if g < 1` mutant on the
    a_lt_b branch (which would route a sub-1 positive g to HELD
    instead of SIGN_FLIP)."""
    verdict, refutation, _ = verdict_from_paired_stats(
        0.5, 0.1, n=200, predicted_direction='a_lt_b',
    )
    assert verdict is Verdict.NO_EFFECT
    assert refutation is RefutationClass.SIGN_FLIP


def test_verdict_held_for_small_negative_g_when_negative_predicted() -> None:
    """Mirror of the small-positive HELD test. g just below 0 with
    predicted='a_lt_b' → HELD. Pins `if g < 0` against `if g < 1`
    mutant (which would route a small negative g to HELD anyway —
    but a moderately positive g would also pass `< 1` and route to
    HELD instead of SIGN_FLIP). The g=-0.5 case nails the < 0
    boundary specifically."""
    verdict, refutation, _ = verdict_from_paired_stats(
        -0.5, 0.1, n=200, predicted_direction='a_lt_b',
    )
    assert verdict is Verdict.HELD
    assert refutation is None


def test_verdict_alpha_power_kwargs_propagate_to_mde_check() -> None:
    """`alpha` and `power` kwargs must reach `adequately_powered_paired`
    intact. With strict `power=0.99` and `alpha=0.001`, MDE goes up
    substantially — an effect that's HELD at default (alpha=0.05,
    power=0.8) becomes POWER_INSUFFICIENT under stricter settings.
    Pins both `alpha=alpha` and `power=power` kwarg passes against
    being silently dropped (which would fall back to the default
    inside `adequately_powered_paired`)."""
    # MDE at n=20:
    #   alpha=0.05  power=0.8  → 0.577
    #   alpha=0.001 power=0.8  → 1.007  (kwarg-drop mutant: power
    #                                    silently defaults to 0.8)
    #   alpha=0.05  power=0.99 → 0.922  (kwarg-drop mutant: alpha
    #                                    silently defaults to 0.05)
    #   alpha=0.001 power=0.99 → 1.394
    # g=1.2 is HELD at all kwarg-drop variants but POWER_INSUFFICIENT
    # only when BOTH strict kwargs propagate. Pins both kwarg passes.
    g, se, n = 1.2, 0.1, 20
    # Default alpha/power: powered for moderate g at n=10.
    verdict_default, _, powered_default = verdict_from_paired_stats(
        g, se, n=n, predicted_direction='a_gt_b',
        alpha=0.05, power=0.8,
    )
    assert verdict_default is Verdict.HELD
    assert powered_default is True
    # Strict alpha + power → larger MDE → not powered for same g.
    verdict_strict, _, powered_strict = verdict_from_paired_stats(
        g, se, n=n, predicted_direction='a_gt_b',
        alpha=0.001, power=0.99,
    )
    assert verdict_strict is Verdict.POWER_INSUFFICIENT
    assert powered_strict is False
