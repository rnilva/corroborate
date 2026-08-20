"""Closed-form assertions on `stratum_effect_panel_per_burst`
over the LG-SCM substrate.

The independent-samples counterpart to `paired_g_per_burst`'s
tests — same phased-arm scenarios, different statistic. Under
the LG-SCM, the per-burst y is `β_xz · β_zy · X_avg(seed) + ε`,
so the per-arm distribution across seeds at one burst is:

    y_t ~ Normal(β_xz_t · β_zy · μ_x, (β_xz_t · β_zy)² · σ_x² / n_steps)
    y_b ~ Normal(β_xz_b · β_zy · μ_x, (β_xz_b · β_zy)² · σ_x² / n_steps)

Independent-samples Cohen's d (simple-mean-variance form):

    d = (μ_t − μ_b) / sqrt((σ_t² + σ_b²) / 2)
      = (β_xz_t − β_xz_b) · μ_x · sqrt(n_steps)
        / (σ_x · sqrt((β_xz_t² + β_xz_b²) / 2))

That closed form is what the constant-phase test compares
against. The phase-flip test corroborates that the panel
preserves alternating signs (the canonical regression-prevention
case — a per-burst primitive that silently averages bursts
would fail catastrophically), the same way the paired form's
test does.

Two scenarios mirror `test_paired_g_per_burst.py`:

1. **Constant phase** — all bursts share SCM coefficients;
   every per-burst d matches the closed form within a sampling-
   distribution-derived bound.

2. **Phase-flipping intervention** — alternating Δβ signs;
   per-burst panel recovers alternating d signs.
"""
from __future__ import annotations

import math

from corroborate.analyses.panel.stratum_effect_panel_per_burst import (
    PerBurstStratumD,
    stratum_effect_panel_per_burst,
)
from corroborate.measurables.reductions import from_key, reduce_axis
from corroborate.data import cells_to_dataframe

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
_N_SEEDS_PER_ARM = 60

# Per-burst rel_err bound on Cohen's d at n_t = n_b = 60.
# Independent-samples d's denominator pools σ_t² and σ_b² — both
# are sample SDs with CV ≈ 1/sqrt(2(n-1)) ≈ 9% at n=60. The
# pooled-SD CV is similar. The numerator's mean-diff has SE ≈
# sqrt(σ_t²/n_t + σ_b²/n_b) — on the order of (μ_t − μ_b) / 10
# at n=60 for the parameters here. Combined sampling fluctuation
# on d ≈ 15%; 25% is a 1.7× slack accommodating modest re-run
# noise while still catching any structural breakage (sign flip,
# scale error, wrong pooling formula) by orders of magnitude.
_PER_BURST_D_REL_ERR = 0.25


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


def _expected_per_burst_d(
    *, beta_xz_t: float, beta_xz_b: float,
) -> float:
    """Closed-form independent-samples Cohen's d for one burst.

    Under the LG-SCM, per-burst y_seed = β_xz · β_zy · X_avg(seed)
    (dropping the tiny additive noise σ_y at n_steps=200). At one
    burst:
        μ_y_arm = β_xz_arm · β_zy · μ_x
        σ_y_arm = (β_xz_arm · β_zy) · σ_x / sqrt(n_steps)

    Simple-mean-variance pooled SD:
        σ_pool = sqrt((σ_y_t² + σ_y_b²) / 2)
               = β_zy · σ_x / sqrt(n_steps) · sqrt((β_xz_t² + β_xz_b²) / 2)

    d = (μ_y_t − μ_y_b) / σ_pool
      = (β_xz_t − β_xz_b) · μ_x · sqrt(n_steps)
        / (σ_x · sqrt((β_xz_t² + β_xz_b²) / 2))

    β_zy cancels (appears in both numerator's mean-diff and
    denominator's pooled SD)."""
    delta = beta_xz_t - beta_xz_b
    rms_beta = math.sqrt((beta_xz_t ** 2 + beta_xz_b ** 2) / 2.0)
    return (
        delta * _MU_X * math.sqrt(_N_STEPS_PER_BURST)
        / (_SIGMA_X * rms_beta)
    )


def _by_burst(
    strata: tuple[PerBurstStratumD, ...],
) -> dict[int, PerBurstStratumD]:
    return {s.burst_index: s for s in strata}


# ============ Constant-phase test ============

def test_per_burst_d_recovers_closed_form_under_constant_phase() -> None:
    """All bursts share the same SCM. Each per-burst stratum's
    independent-samples Cohen's d should match the closed-form
    `(β_xz_t − β_xz_b) · μ_x · sqrt(n_steps) / (σ_x · sqrt((β² +
    β²)/2))` within 25% — the per-arm sample-SD CV floor at
    n_seeds=60 compounded with the mean-diff sampling SE.

    A regression that mis-pools (paired SD), forgets the pooled
    denominator, or uses the wrong sign convention would breach
    the bound by orders. The 25% bound is set by the sampling
    distribution of independent-samples d, not by an arbitrary
    ceiling."""
    n_bursts = 5
    beta_t = 0.8
    beta_b = 0.3
    treatments = tuple(_scm(beta_t) for _ in range(n_bursts))
    baselines = tuple(_scm(beta_b) for _ in range(n_bursts))
    cells = run_paired_phased_arms(
        treatments_per_burst=treatments,
        baselines_per_burst=baselines,
        seeds=range(_N_SEEDS_PER_ARM),
    )

    result = stratum_effect_panel_per_burst.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        source=_PER_BURST_Y_MEAN,
    )

    assert result.n_strata == n_bursts, (
        f'expected {n_bursts} strata; got {result.n_strata} — '
        f'phased construction collapsed bursts'
    )
    by_b = _by_burst(result.strata)
    expected_d = _expected_per_burst_d(
        beta_xz_t=beta_t, beta_xz_b=beta_b,
    )
    for b in range(n_bursts):
        s = by_b[b]
        assert s.n_treatment == _N_SEEDS_PER_ARM, (
            f'burst {b}: n_treatment = {s.n_treatment}, '
            f'expected {_N_SEEDS_PER_ARM} — seed pooling lost cells'
        )
        assert s.n_baseline == _N_SEEDS_PER_ARM, (
            f'burst {b}: n_baseline = {s.n_baseline}, '
            f'expected {_N_SEEDS_PER_ARM}'
        )
        rel_err = abs(s.cohen_d - expected_d) / abs(expected_d)
        assert rel_err < _PER_BURST_D_REL_ERR, (
            f'burst {b}: cohen_d = {s.cohen_d:.4f}, expected '
            f'{expected_d:.4f} (rel err {rel_err:.4f})'
        )


# ============ Phase-flipping test ============

def test_per_burst_d_unmasks_phase_flip() -> None:
    """Bursts 0,1 have `Δβ_xz = +0.5` (treatment helps); bursts
    2,3 have `Δβ_xz = −0.5` (treatment hurts). The independent-
    samples Cohen's d at each (env, burst) recovers the
    alternating signs — the per-burst primitive does NOT silently
    average bursts.

    Closed-form d on positive-Δ bursts is +ve, on negative-Δ
    bursts is −ve, with EQUAL MAGNITUDE because the pooled SD
    is symmetric in (β_xz_t, β_xz_b) — only the numerator's
    sign flips. A per-burst primitive that mis-handles burst
    indexing would smear the alternation; one that mis-uses
    the paired SD would mask the magnitude."""
    treatments = (_scm(0.8), _scm(0.8), _scm(0.3), _scm(0.3))
    baselines = (_scm(0.3), _scm(0.3), _scm(0.8), _scm(0.8))
    n_bursts = len(treatments)
    cells = run_paired_phased_arms(
        treatments_per_burst=treatments,
        baselines_per_burst=baselines,
        seeds=range(_N_SEEDS_PER_ARM),
    )

    result = stratum_effect_panel_per_burst.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        source=_PER_BURST_Y_MEAN,
    )
    assert result.n_strata == n_bursts
    by_b = _by_burst(result.strata)
    expected_d_pos = _expected_per_burst_d(
        beta_xz_t=0.8, beta_xz_b=0.3,
    )
    expected_d_neg = _expected_per_burst_d(
        beta_xz_t=0.3, beta_xz_b=0.8,
    )
    # Symmetry sanity: the two values differ only in sign.
    assert math.isclose(expected_d_pos, -expected_d_neg, rel_tol=1e-9), (
        f'closed-form symmetry broken: pos={expected_d_pos}, '
        f'neg={expected_d_neg}'
    )

    for b in (0, 1):
        s = by_b[b]
        assert s.cohen_d > 0, (
            f'burst {b} (positive Δ): cohen_d = {s.cohen_d:.4f} '
            f'should be > 0'
        )
        rel_err = abs(s.cohen_d - expected_d_pos) / abs(expected_d_pos)
        assert rel_err < _PER_BURST_D_REL_ERR, (
            f'burst {b} (positive Δ): cohen_d = {s.cohen_d:.4f}, '
            f'expected {expected_d_pos:.4f} (rel err {rel_err:.4f})'
        )
    for b in (2, 3):
        s = by_b[b]
        assert s.cohen_d < 0, (
            f'burst {b} (negative Δ): cohen_d = {s.cohen_d:.4f} '
            f'should be < 0'
        )
        rel_err = abs(s.cohen_d - expected_d_neg) / abs(expected_d_neg)
        assert rel_err < _PER_BURST_D_REL_ERR, (
            f'burst {b} (negative Δ): cohen_d = {s.cohen_d:.4f}, '
            f'expected {expected_d_neg:.4f} (rel err {rel_err:.4f})'
        )


# ============ Power-handling: arm-count contract ============

def test_per_burst_d_returns_nan_when_arm_too_small() -> None:
    """With only 1 seed per arm, per-arm SD is undefined and
    Cohen's d is NaN. The framework's `_cohen_d_indep_samples`
    helper returns (NaN, NaN) for n_t < 2 OR n_b < 2 — this test
    pins that contract.

    Distinguishes "no data" (POWER_INSUFFICIENT downstream) from
    a numeric d value the bridge would treat as a real
    measurement."""
    treatments = (_scm(0.8),)
    baselines = (_scm(0.3),)
    cells = run_paired_phased_arms(
        treatments_per_burst=treatments,
        baselines_per_burst=baselines,
        seeds=range(1),  # 1 seed per arm — below the n>=2 floor
    )

    result = stratum_effect_panel_per_burst.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        source=_PER_BURST_Y_MEAN,
    )
    assert result.n_strata == 1
    s = result.strata[0]
    assert s.n_treatment == 1 and s.n_baseline == 1
    assert math.isnan(s.cohen_d), (
        f'cohen_d should be NaN at n_t=n_b=1 (SD undefined); '
        f'got {s.cohen_d}'
    )
    assert math.isnan(s.cohen_se), (
        f'cohen_se should be NaN at n_t=n_b=1; got {s.cohen_se}'
    )
