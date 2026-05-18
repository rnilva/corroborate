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
    """Var[y_mean_per_seed | arm]. Dominant term is the X_avg
    propagation; σ_z² / σ_y² contributions are smaller by
    σ_z²/(β_xz²·σ_x²) and σ_y²/((β_xz·β_zy)²·σ_x²) and ignored
    in the closed form (they're absorbed into the 4-sigma bound)."""
    return (beta_xz * _BETA_ZY) ** 2 * (_SIGMA_X ** 2) / _N_STEPS


def _expected_mean_diff_se(
    *, beta_xz_t: float, beta_xz_b: float, n_per_arm: int,
) -> float:
    """Independent-samples Welch SE under equal n_t = n_b."""
    var_t = _expected_arm_var(beta_xz=beta_xz_t)
    var_b = _expected_arm_var(beta_xz=beta_xz_b)
    return math.sqrt(var_t / n_per_arm + var_b / n_per_arm)


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
    # 4-sigma analytical window — Var carries only the dominant
    # X_avg-propagation term, so σ_z² / σ_y² contributions inflate
    # the empirical SD by a few %; 4σ accommodates that residual.
    assert abs(result.mean_diff - expected) < 4.0 * se_expected, (
        f'mean_diff={result.mean_diff:.4f} expected={expected:.4f} '
        f'4*SE={4.0 * se_expected:.4f}'
    )
    # Framework SE should match closed-form Welch SE within 20%
    # (the dominant-term approximation, sample-SD CV at n=60).
    assert 0.8 * se_expected <= result.mean_diff_se <= 1.25 * se_expected, (
        f'mean_diff_se={result.mean_diff_se:.4f} '
        f'expected_se={se_expected:.4f}'
    )
    assert result.n_treatment == _N_SEEDS_PER_ARM
    assert result.n_baseline == _N_SEEDS_PER_ARM
    # Shared seeds → paired noise cancels → high pairing rho. The
    # diagnostic should fire "would benefit from paired_g". The
    # population pairing-rho under shared X_avg noise is the ratio
    # of shared-variance to total per-arm variance, which is 1.0
    # in the closed form (X_avg is the dominant variance source);
    # the empirical value drops slightly from the σ_z²/σ_y² terms.
    assert result.pairing_rho >= 0.85, (
        f'pairing_rho={result.pairing_rho:.4f} — expected ≈ 1.0 '
        'under shared-seed noise cancellation'
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
