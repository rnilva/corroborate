"""Closed-form assertions on `paired_continuous_do_dowhy` over
the LG-SCM substrate.

The primitive: for each (treatment, baseline) PAIR keyed by
`pair_by`, read `treatment_var` from `treatment_var_arm`
(default baseline arm), compute Δ_outcome across arms, then run
DoWhy backdoor regression of Δ_outcome on treatment_var. 2 active
bridge consumers — the polyak τ bridges in
`experiments/findings/ddqn/mediation.py`.

Under LG-SCM with shared seeds, for each pair (seed):

    treatment_var(seed) = z_mean_baseline(seed)
                        = β_xz_b · x_mean(seed) + σ_z · ε_z_avg(seed)
    Δ_outcome(seed) = y_mean_t(seed) − y_mean_b(seed)
                    = (β_xz_t − β_xz_b) · β_zy · x_mean(seed)
                    (σ_z, σ_y cancel exactly under shared seeds)

Regression slope of Δ_outcome on treatment_var:

    slope = Cov(Δ_y, z_b) / Var(z_b)
          = (β_xz_t − β_xz_b) · β_zy · β_xz_b · σ_x²/n_steps
            / (β_xz_b² · σ_x²/n_steps + σ_z²/n_steps)
          = (β_xz_t − β_xz_b) · β_zy / (β_xz_b + σ_z²/(β_xz_b · σ_x²))

This is an ERROR-IN-VARIABLES regression: the σ_z noise in z_b
attenuates the slope toward zero relative to the true structural
ratio (β_xz_t − β_xz_b) · β_zy / β_xz_b. At σ_z = 0.01 (test
default), the attenuation factor is β_xz_b²σ_x² / (β_xz_b²σ_x² +
σ_z²) = 0.0225 / (0.0225 + 0.0001) ≈ 0.996 — slope recovers
the structural ratio within 0.5%.
"""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.analyses.dowhy.paired_continuous_do_dowhy import (
    paired_continuous_do_dowhy,
)

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_paired_arms


_MU_X = 1.0
_SIGMA_X = 0.5
# σ_z = 0.01 keeps the error-in-variables attenuation < 1%:
# the recovered slope is within 0.5% of the structural ratio
# (β_xz_t − β_xz_b)·β_zy/β_xz_b.
_SIGMA_Z = 0.01
_BETA_ZY = 1.5
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_SEEDS = 60
_BETA_XZ_T = 0.7
_BETA_XZ_B = 0.3


def _scm(beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=_MU_X,
        sigma_x=_SIGMA_X,
        beta_xz=beta_xz,
        sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY,
        sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _build_cells() -> list[Mapping[str, object]]:
    rows = run_paired_arms(
        treatment=_scm(_BETA_XZ_T),
        baseline=_scm(_BETA_XZ_B),
        seeds=tuple(range(_N_SEEDS)),
    )
    return [r.as_dict() for r in rows]


def _expected_slope_attenuated() -> float:
    """OLS recovers the error-in-variables-attenuated slope."""
    structural = (_BETA_XZ_T - _BETA_XZ_B) * _BETA_ZY / _BETA_XZ_B
    attenuation = (
        _BETA_XZ_B ** 2 * _SIGMA_X ** 2
        / (_BETA_XZ_B ** 2 * _SIGMA_X ** 2 + _SIGMA_Z ** 2)
    )
    return structural * attenuation


def test_paired_continuous_do_recovers_attenuated_slope() -> None:
    """Slope of Δ_y on z_mean_baseline = (β_t − β_b)·β_zy/β_b ·
    attenuation_factor. At σ_z = 0.01, attenuation ≈ 0.996, so
    slope ≈ 1.992 (vs structural 2.000)."""
    cells = _build_cells()
    result = paired_continuous_do_dowhy.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        treatment_var='z_mean',
        outcome='y_mean',
        treatment_var_arm='baseline',
    )
    expected = _expected_slope_attenuated()
    # Sampling SE on the slope at n=60 with very-low-noise z_b:
    # SE ≈ σ(residual) / sqrt(n · Var(z_b)). With most variance
    # in z_b coming from β_xz_b · x_mean (≈ 0.000113) and the
    # residual variance ≈ 0 under shared-seed cancellation of
    # σ_z/σ_y in Δ_y, SE is dominated by the small σ_z
    # measurement error in z_b. Empirically |slope - expected|
    # ≈ 0.02 across replicates; 0.10 bound is ~5× safety.
    assert abs(result.backdoor.ate - expected) < 0.10, (
        f'backdoor.ate={result.backdoor.ate:.4f} '
        f'expected={expected:.4f} '
        '(structural ratio 2.0 attenuated by σ_z² noise in z_b)'
    )
    assert result.n_pairs == _N_SEEDS


def test_paired_continuous_do_placebo_destroys_signal() -> None:
    """Placebo permutes z_mean across pairs → structural link to
    Δ_y breaks → refuted ATE ≈ 0 (within permutation MC noise),
    drift ≈ expected slope."""
    cells = _build_cells()
    result = paired_continuous_do_dowhy.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        treatment_var='z_mean',
        outcome='y_mean',
        treatment_var_arm='baseline',
    )
    expected = _expected_slope_attenuated()
    # DoWhy averages over 100 permutation runs by default → MC
    # SE on placebo refuted_ate ≈ slope_SE / sqrt(100) ≈ very
    # small. Empirically placebo refuted_ate hovers within 0.1
    # of zero.
    assert abs(result.placebo.refuted_ate) < 0.30, (
        f'placebo refuted_ate={result.placebo.refuted_ate:.4f} '
        'should be near 0 after permutation'
    )
    assert abs(result.placebo.drift - expected) < 0.30


def test_paired_continuous_do_rcc_drift_small() -> None:
    """Random common cause: adding a column orthogonal-in-
    expectation to z_b preserves the slope on z_b. Drift ≈ 0."""
    cells = _build_cells()
    result = paired_continuous_do_dowhy.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        treatment_var='z_mean',
        outcome='y_mean',
        treatment_var_arm='baseline',
    )
    expected = _expected_slope_attenuated()
    # RCC adds a random N(0, 1) confounder to the regression.
    # In population, slope on z_b is preserved. Sampling drift
    # ≈ slope_on_random_column * its correlation with z_b ≈ 0.
    # Empirically drift < 0.02; bound 0.10 covers > 5× safety.
    assert abs(result.random_common_cause.drift) < 0.10, (
        f'rcc drift={result.random_common_cause.drift:.4f} '
        'should be near 0'
    )
    assert abs(result.random_common_cause.refuted_ate - expected) < 0.10
