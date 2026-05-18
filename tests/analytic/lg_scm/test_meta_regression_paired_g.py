"""Closed-form assertions on `meta_regression_paired_g` over a
multi-env LG-SCM panel.

The panel: N envs, each with the same intervention `Delta_beta_xz`
but a *different* `mu_x`. Per-env paired-g `g(env)` has the
closed form (under shared-seed cancellation):

    d(env)  = mu_x(env) * sqrt(n_steps) / sigma_x   (signed by Delta_beta)
    g(env)  = d(env) * c_4(n_pairs)
              c_4 = 1 - 3 / (4*n_pairs - 5)

So `g(env)` is a deterministic affine function of `mu_x(env)` —
the meta-regression of `g` on the `mu_x` covariate must recover:

    slope(mu_x)  = sqrt(n_steps) / sigma_x * c_4
    intercept    = 0

This pins both quantities to closed-form values; a regression
that mishandled inverse-variance weighting, normal-equation
solve, or covariate centering would fail the slope check by an
order of magnitude.

Two tests:

1. **slope recovery on the structurally-varying axis**
   (mu_x varies, mu_x is the regressor → slope ≈ closed-form)

2. **null covariate** — a per-env covariate uncorrelated with
   mu_x. The regression should report a coefficient
   indistinguishable from zero AND r_squared mostly attributable
   to the (unmodeled) mu_x axis.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from corroborate.analyses.panel.meta_regression_paired_g import (
    meta_regression_paired_g,
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

# Per-env mu_x grid. The points are spread to give the regressor
# enough variance for a stable WLS fit; the closed-form slope is
# the same regardless of grid spacing.
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
    """Build the env panel: one env per mu_x level. Within an env,
    treatment and baseline differ only in `beta_xz`."""
    return {
        f'env_mu_{mu:g}': (
            _scm(mu_x=mu, beta_xz=_BETA_XZ_TREAT),
            _scm(mu_x=mu, beta_xz=_BETA_XZ_BASE),
        )
        for mu in _MU_X_GRID
    }


def _expected_slope_on_mu_x(*, n_pairs: int) -> float:
    """Closed-form slope of g(env) on mu_x(env).

    g = mu_x * sqrt(n_steps) / sigma_x * c_4(n_pairs).
    Slope on mu_x = sqrt(n_steps) / sigma_x * c_4.
    """
    c4 = 1.0 - 3.0 / (4 * n_pairs - 5)
    return math.sqrt(_N_STEPS) / _SIGMA_X * c4


def _as_dicts(rows: Sequence[RunRow]) -> list[Mapping[str, object]]:
    return [r.as_dict() for r in rows]


# ============ Test 1: slope recovery ============

def test_meta_regression_recovers_closed_form_slope_on_mu_x() -> None:
    """Per-env g is a known affine function of mu_x. The
    regression's `mu_x` coefficient must match
    `sqrt(n_steps) / sigma_x * c_4(n_pairs)` within a tight bound.

    The fitted intercept, similarly, must be near zero (g(mu_x=0) = 0
    by construction).

    A regression that
        (a) failed to invert weights
        (b) confused covariate columns
        (c) included the env_name string as a column
    would all fail this test by 10x or more on the slope.
    """
    envs = _envs_by_mu_x()
    cells = _as_dicts(run_multi_env_paired_arms(
        envs=envs, seeds=range(_N_PAIRS),
    ))
    covariates_per_env: Mapping[str, Mapping[str, float]] = {
        f'env_mu_{mu:g}': {'mu_x': mu} for mu in _MU_X_GRID
    }
    result = meta_regression_paired_g.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source='y_mean',
        covariates_per_env=covariates_per_env,
    )

    assert result.n_strata == len(_MU_X_GRID), (
        f'expected {len(_MU_X_GRID)} strata in panel; '
        f'got {result.n_strata}'
    )
    assert len(result.coefficients) == 1, (
        f'expected exactly one coefficient (mu_x); '
        f'got {len(result.coefficients)}'
    )
    coef = result.coefficients[0]
    assert coef.name == 'mu_x'

    expected_slope = _expected_slope_on_mu_x(n_pairs=_N_PAIRS)
    # 5% tolerance — meta_regression's WLS uses Hedges' g se
    # (which itself depends on g via the Borenstein variance), so
    # the implied weighting introduces a small drift from the
    # uniform-weight closed form. 5% absorbs it; an order-of-
    # magnitude regression bug breaks orders of magnitude past it.
    rel_err = abs(coef.coefficient - expected_slope) / expected_slope
    assert rel_err < 0.05, (
        f'slope on mu_x = {coef.coefficient:.4f}, expected '
        f'{expected_slope:.4f} (rel err {rel_err:.4f}); a >5% '
        f'drift indicates the WLS solve is off — check weights, '
        f'covariate centering, or the c_4 correction.'
    )
    # Note on CI: under shared-seed cancellation the per-env g's
    # are nearly deterministic in mu_x, so the WLS residual
    # variance collapses and the CI shrinks to ~zero width
    # around the point estimate. We assert the fitted CI is
    # *narrow* relative to the slope (sanity-check the WLS solve
    # didn't blow up), not that it brackets the closed-form value.
    ci_width = coef.ci_hi - coef.ci_lo
    assert ci_width < 0.05 * expected_slope, (
        f'CI width {ci_width:.4f} > 5% of slope {expected_slope:.4f}; '
        f'shared-seed cells should give a tightly-determined fit'
    )
    # Intercept near zero. The closed form `g_env = slope · mu_x`
    # passes through the origin (g(mu_x=0) = 0), and the
    # framework does NOT center covariates (`meta_regression.py:251`
    # uses raw values), so the fit recovers intercept ≈ 0 directly.
    # The bound is `0.1 · expected_slope` in absolute units —
    # generous because the absolute magnitude is small (~2.7
    # vs slope ~27.5) and finite-sample noise on g_env bleeds in.
    assert abs(result.intercept) < 0.1 * expected_slope, (
        f'intercept = {result.intercept:.4f}; closed-form is 0 '
        f'(g_env = slope·mu_x passes through origin)'
    )


# ============ Test 2: null covariate ============

def test_meta_regression_returns_near_zero_slope_on_irrelevant_covariate() -> None:
    """If the panel's per-env covariate is *unrelated* to the
    structural axis driving g, the meta-regression should report
    a slope indistinguishable from zero.

    To engineer this: use the same multi-mu_x panel as Test 1, but
    pass `unrelated_signal` as the covariate (constant per env, no
    correlation with mu_x). The fit on this irrelevant column
    should give:

        - coefficient ≈ 0 (or non-significant)
        - intercept absorbs the average effect (since the column
          carries no information about g)

    Decoy column: each env gets `unrelated_signal = (-1)^index`
    — a centered alternating ±1, fully orthogonal to the strictly
    increasing `mu_x` grid.
    """
    envs = _envs_by_mu_x()
    cells = _as_dicts(run_multi_env_paired_arms(
        envs=envs, seeds=range(_N_PAIRS),
    ))
    covariates_per_env: Mapping[str, Mapping[str, float]] = {
        f'env_mu_{mu:g}': {'unrelated_signal': (-1.0) ** i}
        for i, mu in enumerate(_MU_X_GRID)
    }
    result = meta_regression_paired_g.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('seed',),
        source='y_mean',
        covariates_per_env=covariates_per_env,
    )
    assert len(result.coefficients) == 1
    coef = result.coefficients[0]
    assert coef.name == 'unrelated_signal'
    # Z-score bound on the structurally-null coefficient.
    # |coef / SE| < 2.5 catches both directions of regression:
    #   - inflated point estimate without proportional SE
    #   - SE collapse producing false-significance
    # Replaces the prior "not significant + CI covers zero" pair
    # which would pass on a framework returning garbage near-zero
    # estimates with overconfident CIs.
    #
    # CovariateCoefficient does not expose `.se` directly, so we
    # invert it from the CI: framework uses t-critical at df = n - p
    # (`meta_regression.py:253, 276`), where n = n_strata = 5 and
    # p = intercept + n_covariates = 2 → df = 3. Using Z=1.96 here
    # would inflate the implied SE by ~62%, weakening the bound.
    from scipy.stats import t as _t  # noqa: PLC0415  # local: scipy is heavy
    df = len(_MU_X_GRID) - 2  # n_strata - (intercept + 1 covariate)
    t_crit = float(_t.ppf(0.975, df=df))
    se = (coef.ci_hi - coef.ci_lo) / (2.0 * t_crit)
    assert se > 0.0, (
        f'CI [{coef.ci_lo:.4f}, {coef.ci_hi:.4f}] is degenerate; '
        f'framework should report finite uncertainty on a {len(_MU_X_GRID)}-'
        f'stratum fit'
    )
    z_score = abs(coef.coefficient) / se
    assert z_score < 2.5, (
        f'|coef / SE| = {z_score:.4f} (coef = {coef.coefficient:.4f}, '
        f'implied SE = {se:.4f}, df = {df}, t_crit = {t_crit:.4f}). '
        f'The decoy column is structurally orthogonal to mu_x '
        f'(alternating ±1 vs strictly-increasing mu_x); a Z-score '
        f'above 2.5 indicates either a spurious slope or under-'
        f'reported SE.'
    )
