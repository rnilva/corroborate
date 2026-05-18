"""Closed-form assertions on `arm_mean_diff` over the LG-SCM
substrate.

Independent-samples counterpart to `test_paired_g.py` —
17 of 18 dqn_bridges in `experiments/findings/dqn_bridges.py`
migrated to this primitive (CLAUDE.md §"Methodology debt"), so a
bug in the Welch's t-test formula or the per-arm mean / SD
reduction would silently pass the analytic suite without this
test.

Under the LG-SCM (X → Z → Y, shared seeds across arms):

    y_mean(seed, arm) = β_xz(arm) · β_zy · μ_x
                       + (β_xz(arm) · β_zy) · (X_avg(seed) − μ_x)
                       + small ε_z / ε_y noise

Two arms differing only in β_xz give:

    E[mean_diff] = (β_xz_t − β_xz_b) · β_zy · μ_x      (closed-form)

    Var[y_mean_per_seed | arm]
        = (β_xz(arm) · β_zy)² · σ_x² / n_steps + O(σ_z², σ_y²)

    SE[mean_diff_independent_samples]
        = sqrt(Var_t / n_t + Var_b / n_b)              (Welch)

The test asserts `arm_mean_diff.fn(...).mean_diff` is within
`4 · SE_expected` of the closed-form expectation and the
framework's reported `.mean_diff_se` is within ±20% of the
closed-form SE.

Pairing-rho diagnostic: since the arms SHARE seeds (the
substrate's whole-point cancellation), `pairing_rho` should be
≈ 1.0 — exercising the "arms-share-seed-noise" diagnostic
branch. The null-contrast scenario (identical arms) corroborates
that the framework doesn't manufacture a mean_diff out of paired
i.i.d. noise.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from corroborate.analyses.paired.arm_mean_diff import arm_mean_diff
from corroborate.corpus.schema import RunRow

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_paired_arms


_MU_X = 1.0
_SIGMA_X = 0.5
_BETA_ZY = 1.5
_SIGMA_Z = 0.1
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_SEEDS_PER_ARM = 60


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


def _expected_mean_diff(*, beta_xz_t: float, beta_xz_b: float) -> float:
    return (beta_xz_t - beta_xz_b) * _BETA_ZY * _MU_X


def _expected_arm_var(*, beta_xz: float) -> float:
    """Closed-form Var[y_mean_per_seed | arm] including ALL
    structural noise terms.

    Per-step y_t = β_zy · (β_xz · x_t + σ_z · ε_z_t) + σ_y · ε_y_t.
    Mean over n_steps: y_mean = β_xz·β_zy·x_mean + β_zy·σ_z·ε_z_avg
                       + σ_y·ε_y_avg. The three terms are
    uncorrelated, so:

        Var[y_mean] = (β_xz · β_zy)² · σ_x² / n_steps
                    + (β_zy · σ_z)² / n_steps
                    + σ_y² / n_steps

    At β_xz_b = 0.2 the σ_z + σ_y noise terms account for ~59%
    of the variance (the X_avg term scales as β_xz²); omitting
    them would understate the per-arm variance by ~2.4×."""
    return (
        (beta_xz * _BETA_ZY) ** 2 * (_SIGMA_X ** 2) / _N_STEPS
        + (_BETA_ZY * _SIGMA_Z) ** 2 / _N_STEPS
        + (_SIGMA_Y ** 2) / _N_STEPS
    )


def _expected_mean_diff_se(
    *, beta_xz_t: float, beta_xz_b: float, n_per_arm: int,
) -> float:
    """Independent-samples Welch SE under equal n_t = n_b."""
    var_t = _expected_arm_var(beta_xz=beta_xz_t)
    var_b = _expected_arm_var(beta_xz=beta_xz_b)
    return math.sqrt(var_t / n_per_arm + var_b / n_per_arm)


def _expected_pairing_rho(*, beta_xz_t: float, beta_xz_b: float) -> float:
    """Closed-form pairing-rho under shared-seed cancellation.

    Under shared seeds the σ_z and σ_y noise components are
    IDENTICAL across arms (the runner uses
    `numpy.random.default_rng(seed)` to pre-draw all epsilons),
    so:
        y_t = β_xz_t·β_zy·X_avg + β_zy·σ_z·ε_z_avg + σ_y·ε_y_avg
        y_b = β_xz_b·β_zy·X_avg + β_zy·σ_z·ε_z_avg + σ_y·ε_y_avg

    The X_avg, ε_z_avg, ε_y_avg components are mutually
    uncorrelated. Cov(y_t, y_b) over seeds picks up the products
    of MATCHED component variances:

        Cov(y_t, y_b) = β_xz_t·β_xz_b · β_zy² · σ_x²/n_steps
                      + (β_zy · σ_z)² / n_steps
                      + σ_y² / n_steps

    And pairing_rho = Cov / sqrt(Var_t · Var_b).

    Not 1.0 because the X_avg component has DIFFERENT
    coefficients in the two arms (β_xz_t vs β_xz_b); the σ_z
    and σ_y components are perfectly correlated, but the
    structural X_avg component contributes less than perfectly.
    """
    var_t = _expected_arm_var(beta_xz=beta_xz_t)
    var_b = _expected_arm_var(beta_xz=beta_xz_b)
    cov = (
        beta_xz_t * beta_xz_b * (_BETA_ZY ** 2) * (_SIGMA_X ** 2) / _N_STEPS
        + (_BETA_ZY * _SIGMA_Z) ** 2 / _N_STEPS
        + (_SIGMA_Y ** 2) / _N_STEPS
    )
    return cov / math.sqrt(var_t * var_b)


def _as_dicts(rows: Sequence[RunRow]) -> list[Mapping[str, object]]:
    return [r.as_dict() for r in rows]


def test_arm_mean_diff_recovers_structural_contrast() -> None:
    beta_xz_t = 0.8
    beta_xz_b = 0.2
    rows = run_paired_arms(
        treatment=_scm(beta_xz_t),
        baseline=_scm(beta_xz_b),
        seeds=tuple(range(_N_SEEDS_PER_ARM)),
    )

    result = arm_mean_diff.fn(
        _as_dicts(rows),
        source='y_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
    )

    expected = _expected_mean_diff(beta_xz_t=beta_xz_t, beta_xz_b=beta_xz_b)
    se_expected = _expected_mean_diff_se(
        beta_xz_t=beta_xz_t,
        beta_xz_b=beta_xz_b,
        n_per_arm=_N_SEEDS_PER_ARM,
    )
    # 4-σ analytical window. The closed-form SE now includes all
    # three variance components (X_avg, σ_z, σ_y propagation), so
    # the bound is calibrated against actual sampling SD rather
    # than absorbing a structural omission.
    assert abs(result.mean_diff - expected) < 4.0 * se_expected, (
        f'mean_diff={result.mean_diff:.4f} expected={expected:.4f} '
        f'4*SE={4.0 * se_expected:.4f}'
    )
    # Framework Welch SE matches the closed-form full-variance SE
    # within ±10% — sample-SD CV at n=60 is ≈ 1/sqrt(2(n-1)) ≈
    # 9%, so 10% absorbs that one-σ slack on each per-arm SD.
    assert 0.9 * se_expected <= result.mean_diff_se <= 1.1 * se_expected, (
        f'mean_diff_se={result.mean_diff_se:.4f} '
        f'expected_se={se_expected:.4f} '
        f'(ratio={result.mean_diff_se / se_expected:.3f})'
    )
    assert result.n_treatment == _N_SEEDS_PER_ARM
    assert result.n_baseline == _N_SEEDS_PER_ARM
    # Closed-form pairing-rho derived from the variance
    # decomposition: shared σ_z/σ_y noise + arm-asymmetric X_avg
    # coefficient. At β=(0.8, 0.2) the population value is ≈ 0.83.
    expected_pairing = _expected_pairing_rho(
        beta_xz_t=beta_xz_t, beta_xz_b=beta_xz_b,
    )
    # Fisher-z SE on pairing rho at n=60 ≈ 1/sqrt(n-3) ≈ 0.131.
    # At ρ ≈ 0.83, the back-transformed bound is (1-ρ²) · z_se ≈
    # 0.041. 0.08 is a 2× safety on z_se to cover Pearson-vs-
    # population-Pearson sampling at n=60.
    assert abs(result.pairing_rho - expected_pairing) < 0.08, (
        f'pairing_rho={result.pairing_rho:.4f} '
        f'expected={expected_pairing:.4f}'
    )


def test_arm_mean_diff_null_contrast_indistinguishable_from_zero() -> None:
    rows = run_paired_arms(
        treatment=_scm(0.5),
        baseline=_scm(0.5),
        seeds=tuple(range(_N_SEEDS_PER_ARM)),
    )

    result = arm_mean_diff.fn(
        _as_dicts(rows),
        source='y_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
    )

    se_expected = _expected_mean_diff_se(
        beta_xz_t=0.5, beta_xz_b=0.5, n_per_arm=_N_SEEDS_PER_ARM,
    )
    # Under shared seeds + identical SCM, every paired Δ is
    # zero exactly. mean_diff IS zero up to floating-point. The
    # 4-sigma bound here is generous slack for any drift.
    assert abs(result.mean_diff) < 4.0 * se_expected
    # Sign: when arms are identical the mean_diff should be
    # exactly zero (modulo numerical noise) — independent-samples
    # arithmetic on shared-seed cells gives identical per-seed
    # contributions to both arms.
    assert abs(result.mean_diff) < 1e-9, (
        f'mean_diff={result.mean_diff} — expected exact 0.0 under '
        'identical SCM + shared seeds'
    )


def test_arm_mean_diff_sign_matches_contrast_direction() -> None:
    # Negative-contrast scenario: treatment β_xz < baseline β_xz.
    # E[mean_diff] = (β_t − β_b) · β_zy · μ_x is negative.
    beta_xz_t = 0.2
    beta_xz_b = 0.8
    rows = run_paired_arms(
        treatment=_scm(beta_xz_t),
        baseline=_scm(beta_xz_b),
        seeds=tuple(range(_N_SEEDS_PER_ARM)),
    )
    result = arm_mean_diff.fn(
        _as_dicts(rows),
        source='y_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
    )
    expected = _expected_mean_diff(beta_xz_t=beta_xz_t, beta_xz_b=beta_xz_b)
    assert expected < 0.0
    assert result.mean_diff < 0.0, (
        f'mean_diff={result.mean_diff} — expected negative under '
        f'beta_xz_t={beta_xz_t} < beta_xz_b={beta_xz_b}'
    )
    se_expected = _expected_mean_diff_se(
        beta_xz_t=beta_xz_t,
        beta_xz_b=beta_xz_b,
        n_per_arm=_N_SEEDS_PER_ARM,
    )
    assert abs(result.mean_diff - expected) < 4.0 * se_expected
