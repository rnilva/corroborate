"""Closed-form assertions on `meta_regression_per_burst` over a
multi-env phased LG-SCM corpus.

Where `meta_regression_paired_g` strata-grains on env only, this
analysis produces a per-(env, burst) panel: each env contributes
`n_bursts` rows to the meta-regression. Under shared-noise the
per-burst paired g within an env is approximately constant
(structurally, `g(env, burst) ≈ mu_x_env * sqrt(n_steps) /
sigma_x * c_4(n_pairs)`) but with finite-sample fluctuation across
bursts, so the panel has between-env signal + within-env residual.

The closed-form regression slope on `mu_x` is the same as the
env-only meta-regression case:

    slope(mu_x) = sqrt(n_steps_per_burst) / sigma_x * c_4(n_pairs)
    intercept   = 0

The added value vs `meta_regression_paired_g`: the panel
n_strata is `n_envs * n_bursts` (denser data → tighter inference)
and the analysis correctly *reuses* env-level covariates
broadcast across each env's bursts (testable via the rows-per-env
count assertion below)."""
from __future__ import annotations

import math
from collections.abc import Mapping

from corroborate.analyses.panel.meta_regression_per_burst import (
    meta_regression_per_burst,
)
from corroborate.measurables.reductions import from_key, reduce_axis
from corroborate.data import cells_to_dataframe

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import (
    PER_BURST_Y_KEY,
    run_multi_env_paired_phased_arms,
)


_SIGMA_X = 0.5
_BETA_ZY = 1.5
_SIGMA_Z = 0.1
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_PAIRS = 30
_N_BURSTS = 4

_BETA_XZ_BASE = 0.3
_BETA_XZ_TREAT = 0.8

_MU_X_GRID: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)


_PER_BURST_Y_MEAN = reduce_axis(
    from_key(PER_BURST_Y_KEY), axis=-1, op='mean',
)


def _scm(*, mu_x: float, beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=mu_x, sigma_x=_SIGMA_X,
        beta_xz=beta_xz, sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _build_phased_panel() -> list[Mapping[str, object]]:
    envs: dict[str, tuple[
        tuple[LinearGaussianSCM, ...], tuple[LinearGaussianSCM, ...],
    ]] = {}
    for mu in _MU_X_GRID:
        treatments = tuple(
            _scm(mu_x=mu, beta_xz=_BETA_XZ_TREAT) for _ in range(_N_BURSTS)
        )
        baselines = tuple(
            _scm(mu_x=mu, beta_xz=_BETA_XZ_BASE) for _ in range(_N_BURSTS)
        )
        envs[f'env_mu_{mu:g}'] = (treatments, baselines)
    return run_multi_env_paired_phased_arms(envs=envs, seeds=range(_N_PAIRS))


def _expected_slope(*, n_pairs: int) -> float:
    c4 = 1.0 - 3.0 / (4 * n_pairs - 5)
    return math.sqrt(_N_STEPS) / _SIGMA_X * c4


# ============ Slope recovery on the per-(env, burst) panel ============

def test_per_burst_meta_regression_recovers_closed_form_slope() -> None:
    """The (env, burst) panel has `n_envs * n_bursts = 5 * 4 = 20`
    strata. Per-burst g within an env is approximately constant
    (varies across envs as `g ∝ mu_x_env`), so the panel meta-
    regression's slope on `mu_x` matches the env-only closed form.

    A regression that mishandled the broadcast of env-level
    covariates across bursts (e.g., dropped them, or doubled them)
    would fail this test by changing the n_obs vs n_strata count
    or the slope value itself.
    """
    cells = _build_phased_panel()
    covariates_per_env: Mapping[str, Mapping[str, float]] = {
        f'env_mu_{mu:g}': {'mu_x': mu} for mu in _MU_X_GRID
    }
    result = meta_regression_per_burst.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source=_PER_BURST_Y_MEAN,
        covariates_per_env=covariates_per_env,
    )

    assert result.n_strata == len(_MU_X_GRID) * _N_BURSTS, (
        f'n_strata = {result.n_strata}, expected '
        f'{len(_MU_X_GRID) * _N_BURSTS} (n_envs * n_bursts)'
    )
    assert len(result.coefficients) == 1
    coef = result.coefficients[0]
    assert coef.name == 'mu_x'

    expected_slope = _expected_slope(n_pairs=_N_PAIRS)
    rel_err = abs(coef.coefficient - expected_slope) / expected_slope
    assert rel_err < 0.05, (
        f'slope on mu_x = {coef.coefficient:.4f}, expected '
        f'{expected_slope:.4f} (rel err {rel_err:.4f}); a >5% '
        f'drift indicates the per-burst panel is being mis-built '
        f'(env covariates not broadcast, or strata mis-grouped)'
    )
    # Intercept near zero — the closed form has g(mu_x=0) = 0.
    assert abs(result.intercept) < 0.1 * expected_slope, (
        f'intercept = {result.intercept:.4f}; expected near zero '
        f'(g(mu_x=0) = 0 by construction)'
    )


# ============ Fallback: covariates= path also works ============

def test_per_burst_meta_regression_resolves_covariates_from_cells() -> None:
    """`covariates: tuple[str, ...]` (without explicit
    `covariates_per_env`) tells the analysis to pull covariate
    values from the cells themselves (env-mean of the named columns).

    Each cell carries `mu_x` in its measurements (the implementation
    stamps it). The analysis should resolve the env-level covariate
    by averaging mu_x across each env's cells — yielding the same
    panel as supplying `covariates_per_env={mu_x: ...}` explicitly.
    """
    cells = _build_phased_panel()
    result = meta_regression_per_burst.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source=_PER_BURST_Y_MEAN,
        covariates=('mu_x',),
    )
    assert result.n_strata == len(_MU_X_GRID) * _N_BURSTS
    assert len(result.coefficients) == 1
    coef = result.coefficients[0]
    assert coef.name == 'mu_x'
    expected_slope = _expected_slope(n_pairs=_N_PAIRS)
    rel_err = abs(coef.coefficient - expected_slope) / expected_slope
    assert rel_err < 0.05, (
        f'slope on mu_x = {coef.coefficient:.4f}, expected '
        f'{expected_slope:.4f} (rel err {rel_err:.4f}); the '
        f'covariates= cell-level resolution path produced a '
        f'different fit than the explicit covariates_per_env one'
    )


def test_per_burst_meta_regression_honours_custom_arm_field() -> None:
    cells: list[Mapping[str, object]] = []
    for cell in _build_phased_panel():
        row = dict(cell)
        row['contrast_arm'] = row.pop('arm_key')
        cells.append(row)
    result = meta_regression_per_burst.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        arm_field='contrast_arm',
        pair_by=('seed',),
        source=_PER_BURST_Y_MEAN,
        covariates=('mu_x',),
    )
    assert result.n_strata == len(_MU_X_GRID) * _N_BURSTS
    assert [coef.name for coef in result.coefficients] == ['mu_x']
