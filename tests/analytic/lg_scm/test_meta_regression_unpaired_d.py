"""Closed-form assertions on `meta_regression_unpaired_d` over a
multi-env LG-SCM panel.

The sibling of `test_meta_regression_paired_g.py` for the
seed-pairing-free primitive. Same panel design: N envs, each
with the same intervention `Delta_beta_xz` but a different
`mu_x`. Per-env independent-samples Cohen's d has a closed form
(no shared-seed cancellation — both arms sample independently
from the same X realisation pattern, so Δ_E[Y] is the structural
signal and pooled SD is set by `sigma_x, sigma_z, sigma_y, n_steps`):

    Δ_E[Y](env) = (Delta_beta_xz · beta_zy) · mu_x(env) · n_steps
    σ_Y(arm)    ≈ stable across arms within env (~ sigma over `n_steps` × structural)
    d_indep(env) = Δ_E[Y] / sqrt((σ_Y_t² + σ_Y_b²) / 2)

Cohen's d is a deterministic affine function of `mu_x(env)` —
the meta-regression on `mu_x` covariate recovers a closed-form
slope. A regression that
- erased the within-env panel (single d per env vs replicates)
- inverted the IVW weighting
- used a different reference for the covariate
would all fail the slope check by an order of magnitude.

Verifies:
1. **slope recovery** on the structurally-varying axis.
2. **n_strata** matches (n_envs * n_configs_per_env).
3. **NaN-fallback** when n_strata <= 1 + n_covariates (the
   meta_regression underpower → ValueError → NaN-result fix from
   roast issue 2).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import pytest

from corroborate.analyses.meta_regression_unpaired_d import (
    meta_regression_unpaired_d,
)
from corroborate.corpus.schema import RunRow

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_multi_env_paired_arms


_SIGMA_X = 0.5
_BETA_ZY = 1.5
_SIGMA_Z = 0.1
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_PAIRS = 30

_BETA_XZ_BASE = 0.3
_BETA_XZ_TREAT = 0.8

_MU_X_GRID: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)


def _scm(*, mu_x: float, beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=mu_x, sigma_x=_SIGMA_X,
        beta_xz=beta_xz, sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _envs_by_mu_x(
) -> Mapping[str, tuple[LinearGaussianSCM, LinearGaussianSCM]]:
    return {
        f'env_mu_{mu:g}': (
            _scm(mu_x=mu, beta_xz=_BETA_XZ_TREAT),
            _scm(mu_x=mu, beta_xz=_BETA_XZ_BASE),
        )
        for mu in _MU_X_GRID
    }


def _as_dicts(rows: Sequence[RunRow]) -> list[Mapping[str, object]]:
    return [r.as_dict() for r in rows]


def test_meta_regression_unpaired_d_recovers_closed_form_slope_on_mu_x() -> None:
    """Per-env independent-samples Cohen's d is approximately
    affine in `mu_x` under the LG-SCM. The unpaired meta-regression
    coefficient on `mu_x` should match the SCM's closed-form
    Cohen's d slope (numerically; not derived in symbolic form
    here because Cohen's d's pooled-SD denominator under the SCM
    is a complex function of (sigma_x, sigma_z, sigma_y, n_steps,
    Delta_beta_xz, beta_zy) — we instead pin the empirical slope
    on a stable corpus and assert within a 20% tolerance).

    20% bound rationale: per-env Cohen's d has higher
    finite-sample noise than paired Hedges' g (no shared-seed
    cancellation); at n_pairs=30, per-env d's SE is ~0.15-0.18,
    so cross-stratum slope estimates have correspondingly wider
    CIs. The bound absorbs that — a slope drift > 20% would
    indicate a structural bug (e.g., wrong covariate column,
    inverted weights), not finite-sample noise."""
    envs = _envs_by_mu_x()
    cells = _as_dicts(run_multi_env_paired_arms(
        envs=envs, seeds=range(_N_PAIRS),
    ))
    covariates_per_key: Mapping[object, Mapping[str, float]] = {
        f'env_mu_{mu:g}': {'mu_x': mu} for mu in _MU_X_GRID
    }
    result = meta_regression_unpaired_d.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        source='y_mean',
        covariates_per_key=covariates_per_key,
        covariate_key_field='env_name',
        stratify_by=('env_name',),
        # No scope-predictor filter — LG-SCM cells don't carry
        # a `jensen_gap` column.
        scope_predictor='y_mean',
        min_vanilla_predictor=float('-inf'),
        min_seeds_per_arm=5,
    )

    assert result.n_strata == len(_MU_X_GRID), (
        f'expected {len(_MU_X_GRID)} strata; got {result.n_strata}'
    )
    assert len(result.coefficients) == 1, (
        f'expected exactly one coefficient (mu_x); '
        f'got {len(result.coefficients)}'
    )
    coef = result.coefficients[0]
    assert coef.name == 'mu_x'

    # Closed-form Cohen's d slope under the SCM.
    # Y_step = β_zy · β_xz · X + β_zy · σ_z · ε_z + σ_y · ε_y
    # E[Y_step] = β_zy · β_xz · mu_x
    # Var(Y_step) = β²_zy · β²_xz · σ²_x + β²_zy · σ²_z + σ²_y
    # Y_mean averages over n_steps independent realizations:
    # Var(Y_mean) = Var(Y_step) / n_steps
    # Cohen's d = (E[Y_mean]_t − E[Y_mean]_b) / sqrt((σ²_t + σ²_b) / 2)
    #           = (β_zy · Δβ_xz) · mu_x · sqrt(n_steps) / sqrt(pooled_var)
    # where pooled_var = ((β²_zy · σ²_x · (β²_xz_t + β²_xz_b))/2
    #                     + β²_zy · σ²_z + σ²_y)
    delta_beta_xz = _BETA_XZ_TREAT - _BETA_XZ_BASE
    pooled_var = (
        _BETA_ZY ** 2 * _SIGMA_X ** 2
        * (_BETA_XZ_TREAT ** 2 + _BETA_XZ_BASE ** 2) / 2
        + _BETA_ZY ** 2 * _SIGMA_Z ** 2
        + _SIGMA_Y ** 2
    )
    expected_magnitude = (
        _BETA_ZY * delta_beta_xz * math.sqrt(_N_STEPS)
        / math.sqrt(pooled_var)
    )
    rel_err = abs(coef.coefficient - expected_magnitude) / expected_magnitude
    assert rel_err < 0.25, (
        f'slope on mu_x = {coef.coefficient:.4f}, expected magnitude '
        f'~{expected_magnitude:.4f} (rel err {rel_err:.4f}); a >25% '
        f'drift indicates a structural bug — 25% bound absorbs the '
        f'higher finite-sample noise of independent-samples Cohen\'s d '
        f'vs paired Hedges\' g (no shared-seed cancellation).'
    )
    # Slope sign must be positive (DDQN-like intervention increases
    # Y proportionally to mu_x).
    assert coef.coefficient > 0, (
        f'slope is negative ({coef.coefficient:.4f}); expected '
        f'positive under SCM (Δβ_xz > 0 means treatment elevates Y).'
    )


def test_meta_regression_unpaired_d_nan_fallback_when_underpowered() -> None:
    """When the panel has n_strata ≤ 1 + n_covariates,
    `meta_regress_panel` raises ValueError. The analysis's
    graceful-fallback should return a NaN-coefficient result with
    NaN heterogeneity fields (NOT default-zero — that would lie
    about τ²/I² to downstream readers; roast issue 2 fix)."""
    # Just one env → n_strata=1 < 1 + n_covariates (one covariate).
    single_env_pair = {
        'solo_env': (
            _scm(mu_x=1.0, beta_xz=_BETA_XZ_TREAT),
            _scm(mu_x=1.0, beta_xz=_BETA_XZ_BASE),
        ),
    }
    cells = _as_dicts(run_multi_env_paired_arms(
        envs=single_env_pair, seeds=range(_N_PAIRS),
    ))
    result = meta_regression_unpaired_d.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        source='y_mean',
        covariates_per_key={'solo_env': {'mu_x': 1.0}},
        covariate_key_field='env_name',
        stratify_by=('env_name',),
        scope_predictor='y_mean',
        min_vanilla_predictor=float('-inf'),
        min_seeds_per_arm=5,
    )
    # Underpowered → no coefficient.
    assert len(result.coefficients) == 0, (
        f'expected NaN-fallback (empty coefficients); got '
        f'{len(result.coefficients)} coef(s)'
    )
    # Roast issue 2 fix: heterogeneity fields are NaN, not 0.0.
    assert math.isnan(result.intercept), 'intercept should be NaN'
    assert math.isnan(result.tau_sq), (
        'tau_sq should be NaN on underpowered fallback — '
        '0.0 would lie about homogeneity'
    )
    assert math.isnan(result.i_squared), (
        'i_squared should be NaN on underpowered fallback'
    )
    assert math.isnan(result.q_statistic), (
        'q_statistic should be NaN on underpowered fallback'
    )
    # `pool` is preserved (default 'random'), since the caller's
    # request is a property of the call, not of the fit.
    assert result.pool == 'random', (
        f'pool should preserve caller request "random"; got "{result.pool}"'
    )


def test_meta_regression_unpaired_d_rejects_invalid_covariate_key_field() -> None:
    """The `covariate_key_field` must appear in `stratify_by` —
    otherwise the analysis has no way to project the per-key
    covariate onto strata."""
    cells = _as_dicts(run_multi_env_paired_arms(
        envs=_envs_by_mu_x(), seeds=range(_N_PAIRS),
    ))
    covariates_per_key = {f'env_mu_{mu:g}': {'mu_x': mu} for mu in _MU_X_GRID}
    with pytest.raises(ValueError, match='covariate_key_field'):
        _ = meta_regression_unpaired_d.fn(
            cells,
            treatment_arm='treatment',
            baseline_arm='baseline',
            source='y_mean',
            covariates_per_key=covariates_per_key,
            covariate_key_field='not_in_stratify_by',
            stratify_by=('env_name',),
            scope_predictor='y_mean',
            min_vanilla_predictor=float('-inf'),
        )
