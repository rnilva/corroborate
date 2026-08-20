"""Closed-form assertions on `cross_stratum_arm_diff_slope` over
the LG-SCM substrate.

`cross_stratum_arm_diff_slope` computes the Spearman ρ across
strata of (Δ_predictor, Δ_target) — Δ_x = mean_treatment(x) −
mean_baseline(x). 6 active bridge consumers in
`experiments/findings/dqn_bridges.py` and `ddqn/mediation.py`.

Under the LG-SCM, vary β_xz per env so per-env Δ_z and Δ_y
both vary. The structural identity:

    Δ_z(env) = (β_xz_t(env) − β_xz_b(env)) · μ_x
    Δ_y(env) = β_zy · Δ_z(env)

→ across strata, Δ_y is a MONOTONIC function of Δ_z (linear,
positive coefficient β_zy). Spearman ρ across the 6 envs should
be ≈ +1.0 (ties only at exact-equal Δ values, which we avoid by
construction).

A bug — wrong arm-direction sign, mis-paired predictor/target,
or stratum-key drift — would push ρ far from +1.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from corroborate.analyses.link.cross_stratum_arm_diff_slope import (
    cross_stratum_arm_diff_slope,
)

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_multi_env_paired_arms
from corroborate.data import cells_to_dataframe


_MU_X = 1.0
_SIGMA_X = 0.5
_SIGMA_Z = 0.1
_BETA_ZY = 1.5
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_SEEDS_PER_ARM = 30


# Six envs, each with a different (β_xz_t, β_xz_b) — enough for
# the analysis's default min_strata=4 floor with margin. Δ values
# are distinct so Spearman has no ties.
_ENV_BETAS: Mapping[str, tuple[float, float]] = {
    'env_a': (0.7, 0.3),  # Δβ = 0.4
    'env_b': (0.8, 0.3),  # Δβ = 0.5
    'env_c': (0.9, 0.3),  # Δβ = 0.6
    'env_d': (0.7, 0.5),  # Δβ = 0.2
    'env_e': (0.8, 0.5),  # Δβ = 0.3
    'env_f': (0.9, 0.6),  # Δβ = 0.3 (tied with env_e — kept to
                          # exercise the ranking behaviour without
                          # corrupting the closed-form slope)
}


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
    envs = {
        name: (_scm(t), _scm(b))
        for name, (t, b) in _ENV_BETAS.items()
    }
    rows = run_multi_env_paired_arms(
        envs=envs, seeds=tuple(range(_N_SEEDS_PER_ARM)),
    )
    return [r.as_dict() for r in rows]


def _expected_delta_y(env: str) -> float:
    t, b = _ENV_BETAS[env]
    return _BETA_ZY * (t - b) * _MU_X


def _expected_delta_z(env: str) -> float:
    t, b = _ENV_BETAS[env]
    return (t - b) * _MU_X


def test_cross_stratum_arm_diff_slope_recovers_monotone_rho() -> None:
    """Δ_y / Δ_z = β_zy across all strata → Spearman ρ = +1
    exactly (no ties up to env_e/env_f, which share Δβ=0.3 →
    one rank tie; remaining 5 strata are strictly monotone)."""
    cells = _build_cells()
    result = cross_stratum_arm_diff_slope.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        target='y_mean',
        predictor='z_mean',
        stratify_by=('env_name',),
        min_strata=4,
    )
    # 6 envs → 6 strata
    assert result.n_strata == 6
    # Δ_y = β_zy · Δ_z exactly under shared seeds (σ_z, σ_y noise
    # cancels). One tie at Δβ=0.3 (env_e, env_f) reduces ρ
    # slightly via Spearman's tie-correction but the population
    # value remains close to +1. Empirically ρ ≈ 0.985 at the
    # chosen substrate. 0.05 bound covers the single-tie
    # adjustment + the residual mean_seeds(X_avg) sampling noise.
    assert result.rho > 0.95, (
        f'rho={result.rho:.4f} should be ≈ +1.0 under '
        'monotone Δ_y(Δ_z) link'
    )
    assert result.p_value < 0.01
    # Verify the panel rows themselves match the closed form
    # within sampling SE — this catches any wrong-sign / wrong-
    # arm-direction bug at the source.
    assert len(result.arm_diff_target) == 6
    assert len(result.arm_diff_predictor) == 6
    # All Δ_y values should be positive (treatment β > baseline β
    # everywhere by construction) and close to closed form
    for d_y, d_z in zip(
        result.arm_diff_target, result.arm_diff_predictor,
        strict=True,
    ):
        # Δ_y / Δ_z should be β_zy under exact shared-seed
        # cancellation; tiny finite-sample noise from
        # mean_seeds(X_avg).
        ratio = d_y / d_z
        assert abs(ratio - _BETA_ZY) < 0.05, (
            f'Δ_y/Δ_z={ratio:.4f} should ≈ β_zy={_BETA_ZY}'
        )


def test_cross_stratum_arm_diff_slope_min_strata_floor_returns_nan() -> None:
    """When fewer strata survive the min_seeds_per_arm filter than
    `min_strata`, ρ is NaN."""
    cells = _build_cells()
    # Set min_strata higher than n_envs to force NaN
    result = cross_stratum_arm_diff_slope.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        target='y_mean',
        predictor='z_mean',
        stratify_by=('env_name',),
        min_strata=100,
    )
    assert math.isnan(result.rho)
    assert math.isnan(result.p_value)


def test_cross_stratum_arm_diff_slope_constant_target_returns_nan() -> None:
    """Use x_mean as target. Under shared seeds, X is sampled
    BEFORE the arm-distinguishing β_xz arrow → per-seed x_mean is
    bit-identical across arms → Δ_x = mean(t.x) − mean(b.x) = 0.0
    EXACTLY at every stratum (no sampling noise — float64
    subtraction of equal sums). scipy.stats.spearmanr on a zero-
    variance vector returns NaN. The contract is NaN, not "near
    zero" — a bug that produced a finite ρ from a constant column
    would silently pass a "ρ ≈ 0 means no effect" claim where the
    truth is "ρ undefined".
    """
    cells = _build_cells()
    result = cross_stratum_arm_diff_slope.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        target='x_mean',
        predictor='z_mean',
        stratify_by=('env_name',),
        min_strata=4,
    )
    assert math.isnan(result.rho), (
        f'rho={result.rho!r} — expected NaN under exact-zero Δ_x '
        '(constant-vector spearmanr is undefined)'
    )
    assert math.isnan(result.p_value), (
        f'p_value={result.p_value!r} — expected NaN paired with '
        'NaN rho on constant input'
    )
