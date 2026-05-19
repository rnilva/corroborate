"""Closed-form assertions on `stratum_baseline_predictor_link_dowhy`
over the LG-SCM substrate.

The primitive tests `baseline_predictor → Δ_target` on stratum-
level rows via DoWhy backdoor, adjusting for env (one-hot). 3
active bridge consumers across `experiments/findings/`.

Setup needs multi-dim stratification: with stratify_by =
('env_name', 'mu_x'), 4 envs × 3 μ_x levels = 12 strata. Env one-
hot adds 3 dummies + intercept + v_pred = 5 predictors; 12 rows →
residual df 7, OLS slope identifiable.

Under the LG-SCM:

    v_pred(env, μ_x)        = β_xz_b · μ_x            (baseline mean z_mean)
    target_baseline(env, μ_x) = β_xz_b · β_zy · μ_x
    target_treatment(env, μ_x) = β_xz_t · β_zy · μ_x
    Δ_target(env, μ_x)        = (β_xz_t − β_xz_b) · β_zy · μ_x

→ Δ_target / v_pred = (β_xz_t − β_xz_b) · β_zy / β_xz_b
                    = (0.7 − 0.3) · 1.5 / 0.3 = 2.0

EXACT across all (env, μ_x) strata under shared-seed cancellation
of σ_z, σ_y noise streams. After env adjustment, the slope of
Δ_target on v_pred is 2.0 to machine precision.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from corroborate.analyses.dowhy.stratum_baseline_predictor_link_dowhy import (
    stratum_baseline_predictor_link_dowhy,
)

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_arm


_SIGMA_X = 0.5
_SIGMA_Z = 0.1
_BETA_ZY = 1.5
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_SEEDS_PER_ARM = 30
_BETA_XZ_T = 0.7
_BETA_XZ_B = 0.3


# 4 envs × 3 μ_x levels = 12 (env, μ_x) strata.
_ENVS: tuple[str, ...] = ('env_a', 'env_b', 'env_c', 'env_d')
_MU_X_LEVELS: tuple[float, ...] = (1.0, 2.0, 3.0)


def _scm(mu_x: float, beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=mu_x,
        sigma_x=_SIGMA_X,
        beta_xz=beta_xz,
        sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY,
        sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _build_cells() -> list[Mapping[str, object]]:
    """Build (env, μ_x) cells. Each combination contributes
    N_SEEDS_PER_ARM cells per arm. The runner stamps mu_x as a
    leaf field on each cell, so stratify_by=('env_name', 'mu_x')
    sees 12 strata. Seeds are offset per (env, μ_x) to ensure
    distinct cell IDs across the corpus."""
    rows = []
    for i, env in enumerate(_ENVS):
        for j, mu_x in enumerate(_MU_X_LEVELS):
            seed_offset = (i * len(_MU_X_LEVELS) + j) * _N_SEEDS_PER_ARM * 2
            seeds = tuple(range(
                seed_offset, seed_offset + _N_SEEDS_PER_ARM,
            ))
            rows.extend(run_arm(
                _scm(mu_x, _BETA_XZ_T),
                seeds=seeds,
                arm_key='treatment',
                env_name=env,
            ))
            rows.extend(run_arm(
                _scm(mu_x, _BETA_XZ_B),
                seeds=seeds,
                arm_key='baseline',
                env_name=env,
            ))
    return [r.as_dict() for r in rows]


def _expected_slope() -> float:
    """Δ_target / v_pred = (β_xz_t − β_xz_b) · β_zy / β_xz_b."""
    return (_BETA_XZ_T - _BETA_XZ_B) * _BETA_ZY / _BETA_XZ_B


def test_stratum_baseline_predictor_link_recovers_structural_slope() -> None:
    """Δ_target / v_pred = (β_xz_t − β_xz_b)·β_zy / β_xz_b = 2.0
    exactly at each stratum under shared-seed cancellation. After
    env one-hot adjustment, OLS slope recovers 2.0."""
    cells = _build_cells()
    result = stratum_baseline_predictor_link_dowhy.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        predictor_col='z_mean',
        target_col='y_mean',
        stratify_by=('env_name', 'mu_x'),
        min_seeds_per_arm=5,
        min_baseline_predictor=0.0,
    )
    expected = _expected_slope()
    # The OLS solve isn't quite to machine precision here because
    # mean_seeds(X_avg) introduces small (sub-1%) noise on v_pred
    # at n_seeds=30 — so it's not perfectly co-linear with the
    # structural form across (env, μ_x). The slope SE is
    # determined by that small residual variance. Empirically
    # |ate − 2.0| ≈ 0.003 to 0.01 across seed selections;
    # 0.05 bound is ~5× safety margin.
    assert abs(result.backdoor.ate - expected) < 0.05, (
        f'backdoor.ate={result.backdoor.ate:.4f} '
        f'expected={expected:.4f}'
    )
    assert result.n_strata == len(_ENVS) * len(_MU_X_LEVELS)


def test_stratum_baseline_predictor_link_placebo_drift() -> None:
    """Placebo refutation: permute v_pred → structural slope
    broken → refuted ATE ≈ 0, drift ≈ structural slope."""
    cells = _build_cells()
    result = stratum_baseline_predictor_link_dowhy.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        predictor_col='z_mean',
        target_col='y_mean',
        stratify_by=('env_name', 'mu_x'),
        min_seeds_per_arm=5,
        min_baseline_predictor=0.0,
    )
    expected = _expected_slope()
    # Placebo permutes the v_pred column → OLS coefficient on
    # the permuted column converges to 0 (no structural link to
    # Δ_target after env adjustment). The refuted ATE is
    # bounded by the permutation's MC noise on a finite (n=12)
    # panel — empirically ≤ 0.2 in magnitude.
    assert abs(result.placebo.refuted_ate) < 0.30, (
        f'placebo refuted_ate={result.placebo.refuted_ate:.4f} '
        'should be near 0 after permutation'
    )
    # Drift = original − refuted = expected slope − ~0 = expected.
    assert abs(result.placebo.drift - expected) < 0.30


def test_stratum_baseline_predictor_link_rcc_drift_small() -> None:
    """Random common cause adds a column orthogonal-in-expectation
    to v_pred. Since Δ_target is almost perfectly explained by
    (v_pred + env dummies), OLS allocates near-zero coefficient
    to the random column → drift ≈ 0."""
    cells = _build_cells()
    result = stratum_baseline_predictor_link_dowhy.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        predictor_col='z_mean',
        target_col='y_mean',
        stratify_by=('env_name', 'mu_x'),
        min_seeds_per_arm=5,
        min_baseline_predictor=0.0,
    )
    # Drift on RCC: with the structural slope nearly exact and a
    # random column added, OLS shrinks the random column's
    # coefficient toward 0; the v_pred coefficient absorbs at
    # most a small fluctuation. Empirically drift ≈ 0.001 at
    # this n_strata; 0.10 bound is huge safety.
    assert abs(result.random_common_cause.drift) < 0.10, (
        f'rcc drift={result.random_common_cause.drift:.4f} '
        'should be near 0 — random column carries no structure'
    )


def test_stratum_baseline_predictor_link_min_baseline_floor_drops_strata() -> None:
    """Setting min_baseline_predictor above the lowest stratum's
    v_pred drops those strata. Lowest v_pred = β_xz_b · min(μ_x)
    = 0.3 · 1.0 = 0.3. Setting floor at 0.6 should drop the μ_x=1
    strata (4 envs × 1 level = 4 strata)."""
    cells = _build_cells()
    result = stratum_baseline_predictor_link_dowhy.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        predictor_col='z_mean',
        target_col='y_mean',
        stratify_by=('env_name', 'mu_x'),
        min_seeds_per_arm=5,
        min_baseline_predictor=0.6,  # drops μ_x=1.0 strata
    )
    # Expected: 12 strata − 4 (μ_x=1) = 8 remaining
    assert result.n_strata == 8
    # Slope still recovers structurally (μ_x=2.0 and μ_x=3.0 still
    # give same ratio)
    expected = _expected_slope()
    assert not math.isnan(result.backdoor.ate)
    assert abs(result.backdoor.ate - expected) < 0.05
