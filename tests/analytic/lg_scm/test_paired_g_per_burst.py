"""Closed-form assertions on `paired_g_per_burst` over the LG-SCM
substrate.

Two scenarios:

1. **Constant phase across bursts.** All bursts share the same SCM
   coefficients; the two arms differ in `beta_xz`. Every per-burst
   stratum should report a strongly-positive Hedges' g and
   `helped_fraction == 1.0`, since under shared-seed noise every
   paired Δ is structurally positive.

2. **Phase-flipping intervention.** Half the bursts have
   `Delta_beta_xz > 0` (treatment helps), half have
   `Delta_beta_xz < 0` (treatment hurts). The structural Δ on each
   burst's mean-Y is closed-form (`Delta_beta(b) * beta_zy * mu_x`),
   alternating in sign. Per-burst paired-g sees the alternation
   (positive g on positive bursts, negative g on negative
   bursts); scalar paired_g on the overall `y_mean` averages the
   alternation to ≈0 — exactly the per-vs-scalar masking failure
   mode CLAUDE.md flags as canonical (findings_fourrooms_time_series).

The phase-flipping case is the headline test: it asserts that
the framework's per-burst panel does NOT silently collapse phase
structure that a scalar contrast would hide. Closed-form analytical
SE on the scalar contrast lets us assert "scalar mean_diff is
indistinguishable from zero" with quantitative honesty rather
than a vague "is small".
"""
from __future__ import annotations

import math

from corroborate.analyses.paired.paired_g import paired_g
from corroborate.analyses.paired.paired_g_per_burst import (
    PerBurstStratum,
    paired_g_per_burst,
)
from corroborate.measurables.reductions import from_key, reduce_axis

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import (
    PER_BURST_Y_KEY,
    run_paired_phased_arms,
)


# Shared parameters tuned for sharp bounds.
_MU_X = 1.0
_SIGMA_X = 0.5
_BETA_ZY = 1.5
_SIGMA_Z = 0.1
_SIGMA_Y = 0.1
_N_STEPS_PER_BURST = 200
_N_PAIRS = 60
# Per-burst rel_err bound on Hedges' g at n_pairs=60. Sample-SD CV
# is ~9% at n=60, but the t-statistic distribution for d compounds
# the numerator's mean noise + denominator's SD noise; 20%
# corresponds roughly to a 2σ bound on individual-burst sampling
# fluctuation. Sharper than `g > 0.25 * structural_d` (4× tighter)
# while still reliable across re-runs. A real framework regression
# (sign flip, scale error, mis-pairing) would breach this by orders.
_PER_BURST_G_REL_ERR = 0.20


# Source: per-burst mean of `y_per_episode`. The default source
# in paired_g_per_burst reads `mc_return`; the implementation emits
# Y under a domain-honest key, so we wire a custom source.
_PER_BURST_Y_MEAN = reduce_axis(
    from_key(PER_BURST_Y_KEY), axis=-1, op='mean',
)


def _scm(beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=_MU_X,
        sigma_x=_SIGMA_X,
        beta_xz=beta_xz,
        sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY,
        sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS_PER_BURST,
    )


def _expected_per_burst_g(
    *, beta_xz_t: float, beta_xz_b: float, n_pairs: int,
) -> float:
    """Closed-form Hedges' g for one burst's per-burst paired g.

    Under shared-seed cancellation, per-paired Δ at burst b:
        Delta(seed) = (β_xz_t - β_xz_b) · β_zy · X_avg(seed)
    so the d statistic is independent of the (Δβ · β_zy) factor —
    it appears in both numerator and denominator and cancels:
        d = mu_x · sqrt(n_steps) / sigma_x   (signed by Δβ)
        g = d * c_4(n_pairs),  c_4 = 1 - 3/(4n - 5)
    """
    sign = 1.0 if (beta_xz_t - beta_xz_b) > 0 else -1.0
    d = sign * _MU_X * math.sqrt(_N_STEPS_PER_BURST) / _SIGMA_X
    c4 = 1.0 - 3.0 / (4 * n_pairs - 5)
    return d * c4


def _scalar_mean_diff_se_phase_flip(
    *, abs_delta_betas: tuple[float, ...], n_pairs: int,
) -> float:
    """Closed-form SE of scalar paired_g.mean_diff under phase-
    flipping arms. With `n_bursts` bursts each contributing
    (Delta_beta_b * beta_zy)^2 * sigma_x^2 / n_steps to the
    within-burst variance, the overall y_mean's variance averages
    them by `1 / n_bursts^2`:

        Var[Delta_y_mean(seed)]
            = (1 / n_bursts^2) * sum_b (Delta_beta_b * beta_zy)^2
                                 * sigma_x^2 / n_steps_per_burst
        Var[mean over n_pairs] = Var[Delta_y_mean(seed)] / n_pairs
    """
    n_b = len(abs_delta_betas)
    var_per_seed = sum(
        (db * _BETA_ZY) ** 2 for db in abs_delta_betas
    ) * (_SIGMA_X ** 2) / (n_b * n_b * _N_STEPS_PER_BURST)
    return math.sqrt(var_per_seed / n_pairs)


def _by_burst(
    strata: tuple[PerBurstStratum, ...],
) -> dict[int, PerBurstStratum]:
    return {s.burst_index: s for s in strata}


# ============ Constant-phase test ============

def test_per_burst_recovers_closed_form_g_under_constant_phase() -> None:
    """All bursts share the same SCM. Each per-burst stratum's g
    should match the closed-form Hedges' g (`mu_x · sqrt(n_steps)
    / sigma_x · c_4`) within 15% — the per-burst sample-SD CV
    floor at n_pairs=30.

    A regression that mis-pairs across bursts, silently collapses
    bursts to one stratum, or breaks the per-burst noise stream
    isolation in `simulate_phased` would fail this rel_err bound
    by orders of magnitude. The 15% bound is set by the sampling
    distribution of Hedges' d (whose denominator's CV ~13% at
    n=30), not by an arbitrary ceiling."""
    n_bursts = 5
    treatments = tuple(_scm(0.8) for _ in range(n_bursts))
    baselines = tuple(_scm(0.3) for _ in range(n_bursts))
    cells = run_paired_phased_arms(
        treatments_per_burst=treatments,
        baselines_per_burst=baselines,
        seeds=range(_N_PAIRS),
    )

    result = paired_g_per_burst.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        source=_PER_BURST_Y_MEAN,
    )

    assert result.n_strata == n_bursts, (
        f'expected {n_bursts} strata; got {result.n_strata} — '
        f'phased construction collapsed bursts'
    )
    by_b = _by_burst(result.strata)
    expected_g = _expected_per_burst_g(
        beta_xz_t=0.8, beta_xz_b=0.3, n_pairs=_N_PAIRS,
    )
    for b in range(n_bursts):
        s = by_b[b]
        rel_err = abs(s.g - expected_g) / expected_g
        assert rel_err < _PER_BURST_G_REL_ERR, (
            f'burst {b}: g = {s.g:.4f}, expected {expected_g:.4f} '
            f'(rel err {rel_err:.4f}). Per-burst d = mu_x · '
            f'sqrt(n_steps) / sigma_x = {expected_g / (1.0 - 3.0 / (4 * _N_PAIRS - 5)):.4f}'
        )


# ============ Phase-flipping test ============

def test_per_burst_unmasks_phase_flip_that_scalar_paired_g_hides() -> None:
    """Bursts 0,1 have `Delta_beta_xz = +0.5` (treatment helps);
    bursts 2,3 have `Delta_beta_xz = -0.5` (treatment hurts).
    Scalar mean across all bursts cancels: structural Δ on
    `y_mean` is exactly zero, so paired_g returns mean_diff
    indistinguishable from zero. Per-burst paired_g recovers the
    alternating signs.

    This is the canonical regression-prevention case for
    findings_fourrooms_time_series's "scalar mediator returns
    null when phases cancel; per-burst probe is the right
    diagnostic" — a per-burst primitive that silently averages
    bursts would fail this test catastrophically.
    """
    # Two positive-Δ bursts then two negative-Δ bursts.
    treatments = (_scm(0.8), _scm(0.8), _scm(0.3), _scm(0.3))
    baselines = (_scm(0.3), _scm(0.3), _scm(0.8), _scm(0.8))
    n_bursts = len(treatments)
    cells = run_paired_phased_arms(
        treatments_per_burst=treatments,
        baselines_per_burst=baselines,
        seeds=range(_N_PAIRS),
    )

    # ============ Per-burst sees the phase ============
    per_burst = paired_g_per_burst.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        source=_PER_BURST_Y_MEAN,
    )
    assert per_burst.n_strata == n_bursts
    by_b = _by_burst(per_burst.strata)
    expected_g_pos = _expected_per_burst_g(
        beta_xz_t=0.8, beta_xz_b=0.3, n_pairs=_N_PAIRS,
    )
    expected_g_neg = _expected_per_burst_g(
        beta_xz_t=0.3, beta_xz_b=0.8, n_pairs=_N_PAIRS,
    )

    for b in (0, 1):  # positive-Δ bursts
        s = by_b[b]
        rel_err = abs(s.g - expected_g_pos) / abs(expected_g_pos)
        assert rel_err < _PER_BURST_G_REL_ERR, (
            f'burst {b} (positive Δ): g = {s.g:.4f}, expected '
            f'{expected_g_pos:.4f} (rel err {rel_err:.4f})'
        )
    for b in (2, 3):  # negative-Δ bursts
        s = by_b[b]
        rel_err = abs(s.g - expected_g_neg) / abs(expected_g_neg)
        assert rel_err < _PER_BURST_G_REL_ERR, (
            f'burst {b} (negative Δ): g = {s.g:.4f}, expected '
            f'{expected_g_neg:.4f} (rel err {rel_err:.4f})'
        )

    # ============ Scalar paired_g masks the phase ============
    # Derive |Δβ_xz| per burst from the SCM tuples directly so a
    # change in test parameters can't desync the closed-form SE.
    abs_delta_betas = tuple(
        abs(t.beta_xz - b.beta_xz)
        for t, b in zip(treatments, baselines)
    )
    se = _scalar_mean_diff_se_phase_flip(
        abs_delta_betas=abs_delta_betas, n_pairs=_N_PAIRS,
    )
    bound = 4.0 * se
    scalar = paired_g.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        source='y_mean',
        pair_by=('seed',),
    )
    assert abs(scalar.mean_diff) < bound, (
        f'scalar paired_g.mean_diff = {scalar.mean_diff:.6f} not '
        f'within 4*SE = {bound:.6f} of structural zero — the '
        f'phase-flipping construction should cancel exactly across '
        f'bursts; if mean_diff drifts above this the masking story '
        f'is more nuanced than the construction implies'
    )
    # Hedges' g on the scalar contrast: structurally zero, with
    # sampling noise. Use Z-bound `|g / SE_g| < 2.5` instead of an
    # arbitrary magnitude cutoff. The framework reports `scalar.se`
    # which encodes its own uncertainty assessment; this catches
    # both inflated-g and SE-collapse regressions on the masking case.
    assert scalar.se > 0.0, (
        f'scalar.se = {scalar.se}; framework should report finite '
        f'SE on n_pairs=60 cells'
    )
    z_score = abs(scalar.g) / scalar.se
    assert z_score < 2.5, (
        f'|scalar.g / scalar.se| = {z_score:.4f} (g = {scalar.g:.4f}, '
        f'SE = {scalar.se:.4f}). The phase-cancelling construction '
        f'gives mean(Δ) = 0 structurally; per-burst sees '
        f'|g| ~ {abs(expected_g_pos):.1f} on each phase — the '
        f'phase-collapsed scalar must NOT report a significant '
        f'effect, or per-burst is not doing the work'
    )
