"""Framework-as-instrument: `meta_regression_paired_g` recovers
the σ-scaling of Hasselt's |A|=2 closed-form bias.

Hasselt 2010, 2016: under iid N(0, σ²) noise across |A|=2
actions, vanilla DQN's `max_a Q̂` is biased upward by EXACTLY
σ/√π (Owen 1980). DDQN's double-greedify with independent
estimators is unbiased.

For fixed |A|=2 across a grid of σ values:

    E[Δ_jensen(σ)] = -σ / √π                                (exact)
    sd(Δ_jensen(σ)) = σ · √(2 - 1/π)                        (independent noise)
    Hedges' g(σ) ≈ -σ/√π / (σ · √(2 - 1/π)) · c_4(n)
                ≈ -1 / (√π · √(2 - 1/π)) · c_4              (independent of σ)

So per-env Hedges' g is APPROXIMATELY CONSTANT across σ envs!
Meta-regression on σ should report a slope ≈ 0 — the standardized
effect is structural (signal scales with noise, so the standardized
ratio is constant).

This makes the σ-scaling a clean negative-control test for the
framework's pooling: it must NOT report a non-zero slope on a
covariate that doesn't actually carry per-env variance.

A better closed-form test for a NON-trivial slope: regress
the RAW Δ_jensen mean per env on σ, where E[Δ] = -σ/√π gives
slope = -1/√π exactly. This requires a `source` per cell that's
the raw bias, not the standardized g — not what
`meta_regression_paired_g` does.

So this file pins TWO complementary properties:

1. **R² near 0 on σ-covariate at fixed |A|=2.** g(σ) is
   ~constant across σ, so σ as a covariate explains essentially
   no between-env variance. Pin the framework's R² calculation
   against returning a spurious high R² (which would indicate a
   pooling or weighting bug).

2. **Slope on σ near 0.** The standardized-effect invariance
   under σ scaling.

Both pin the framework's correctness on inputs whose closed-form
structural law is "no slope". A regression that introduced spurious
covariate-effect inflation (e.g., wrong inverse-variance weighting,
or a transform that confuses raw-mean with g) would breach.

For a TRUE non-zero slope test, the next file in this series
exercises `paired_g_per_burst` on the Banach γ-contraction rate.
"""
from __future__ import annotations

import math
import zlib

import numpy as np


def _det_seed(*parts: object) -> int:
    """Deterministic-across-processes seed via zlib.adler32 —
    Python's `hash()` randomizes per process under PYTHONHASHSEED=random."""
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF

from corroborate.analyses.meta_regression_paired_g import (
    meta_regression_paired_g,
)

from corroborate_rl.tabular import (
    double_greedify_tabular,
    max_greedify_tabular,
)


_N_PAIRS_PER_ENV = 200    # tight SE per-env on g

# σ grid: per-env σ value drives the raw bias σ/√π linearly.
# Hedges' g standardizes that bias by sd(Δ) ∝ σ, so g is
# approximately CONSTANT across the σ grid.
_SIGMA_VALUES: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)


def _generate_sigma_grid_cells() -> list[dict[str, object]]:
    """Per-σ paired vanilla/ddqn cells at fixed |A|=2. Per
    (env=σ, seed) pair: independent noise streams sized σ for
    vanilla, ddqn-online, ddqn-target."""
    cells: list[dict[str, object]] = []
    for sigma in _SIGMA_VALUES:
        for s in range(_N_PAIRS_PER_ENV):
            rng = np.random.default_rng(seed=_det_seed('sigma_grid', sigma, s))
            eps_v = (rng.standard_normal(2) * sigma).astype(np.float64)
            eps_online = (rng.standard_normal(2) * sigma).astype(np.float64)
            eps_target = (rng.standard_normal(2) * sigma).astype(np.float64)
            env_name = f'sigma_{sigma:g}'
            cells.append({
                'arm_key': 'vanilla',
                'seed': s,
                'env_name': env_name,
                'jensen_gap': max_greedify_tabular(eps_v),
            })
            cells.append({
                'arm_key': 'ddqn',
                'seed': s,
                'env_name': env_name,
                'jensen_gap': double_greedify_tabular(eps_online, eps_target),
            })
    return cells


def _covariates_per_env() -> dict[str, dict[str, float]]:
    return {
        f'sigma_{sigma:g}': {'sigma': sigma}
        for sigma in _SIGMA_VALUES
    }


def test_meta_regression_intercept_recovers_constant_hedges_g() -> None:
    """At fixed |A|=2, Hedges' g per env is approximately a
    structural constant:

        g ≈ -1 / (√π · √(2 - 1/π)) · c_4(n_pairs)
          ≈ -0.4622 · 0.9962  ≈  -0.4604        (n_pairs=200)

    Meta-regression on σ should recover an intercept matching
    this closed-form constant within sampling SE. The slope on
    σ is the negative-control assertion (next test).

    Per-env g SE ≈ √(1/n_pairs + g²/(2·n_pairs)) ≈ √(1/200 +
    0.21/400) ≈ 0.075. Across 5 envs the pooled intercept SE
    is √(0.075²/5) ≈ 0.034. 4·SE bound = 0.14, easily separates
    the structural -0.46 from -0 or -σ/√π.
    """
    cells = _generate_sigma_grid_cells()
    covariates = _covariates_per_env()
    result = meta_regression_paired_g.fn(
        cells,
        treatment_arm='ddqn',
        baseline_arm='vanilla',
        pair_by=('seed',),
        source='jensen_gap',
        covariates_per_env=covariates,
    )
    # Closed-form Hedges' g constant.
    n = _N_PAIRS_PER_ENV
    c4 = 1.0 - 3.0 / (4 * n - 5)
    expected_g = -1.0 / (math.sqrt(math.pi) * math.sqrt(2.0 - 1.0 / math.pi)) * c4

    bound = 0.14
    assert abs(result.intercept - expected_g) < bound, (
        f'intercept = {result.intercept:.4f}, closed-form '
        f'-1/(√π·√(2-1/π))·c_4 = {expected_g:.4f} '
        f'(4·SE bound = {bound:.4f}). At fixed |A|=2, g is a '
        f'structural constant — recovering it via meta-regression '
        f'verifies the per-env panel build + intercept fit.'
    )


def test_meta_regression_no_spurious_sigma_slope() -> None:
    """Negative control: σ does NOT predict Hedges' g at fixed
    |A|=2 (signal and noise scale together). The slope on σ
    should be near 0; a regression that introduced spurious
    covariate-effect inflation (wrong weighting, or a transform
    that confuses raw bias with g) would breach.

    Slope SE on a 5-env panel scales as
    sd(g) / √(n_envs · var(σ)). var(σ) ≈ 0.85 for the grid.
    sd(g) per env ≈ 0.075. Slope SE ≈ 0.075 / √(5·0.85) ≈ 0.036.
    4·SE bound on the slope = 0.15.
    """
    cells = _generate_sigma_grid_cells()
    covariates = _covariates_per_env()
    result = meta_regression_paired_g.fn(
        cells,
        treatment_arm='ddqn',
        baseline_arm='vanilla',
        pair_by=('seed',),
        source='jensen_gap',
        covariates_per_env=covariates,
    )
    by_name = {c.name: c for c in result.coefficients}
    sigma_coef = by_name['sigma']
    bound = 0.15
    assert abs(sigma_coef.coefficient) < bound, (
        f'σ-slope = {sigma_coef.coefficient:.4f}, expected ≈ 0 '
        f'(4·SE bound = {bound:.4f}). σ does NOT predict g at '
        f'fixed |A|=2 — Hedges g standardizes signal by noise so '
        f'the ratio is constant across σ. A non-zero slope here '
        f'means the framework introduced spurious covariate '
        f'inflation.'
    )
    # And the slope's CI must straddle zero (significance check).
    assert sigma_coef.ci_lo < 0 < sigma_coef.ci_hi, (
        f'σ-slope CI = [{sigma_coef.ci_lo:.4f}, '
        f'{sigma_coef.ci_hi:.4f}]; the structural-null slope '
        f'should not reject zero.'
    )


def test_meta_regression_low_r_squared_on_null_covariate() -> None:
    """When the covariate is structurally orthogonal to the
    response (σ at fixed |A|=2), R² should be near zero.

    Sampling-distribution-derived bound: under H_0 (no covariate
    effect) with 5 strata and 1 covariate (df_resid=3), the null
    distribution of R² is `Beta(p−1, n−p)/2 = Beta(1, 3)`. The
    75th percentile of Beta(1, 3) is ≈ 0.16; the 99th is ≈ 0.79.
    Use `R² < 0.20` as a tight bound that catches the typical
    spurious-inflation case (any framework bug that doubles the
    null R² breaches it). A naive `R² < 0.5` accepts ~95% of
    the null distribution and would miss meaningful inflation.
    """
    cells = _generate_sigma_grid_cells()
    covariates = _covariates_per_env()
    result = meta_regression_paired_g.fn(
        cells,
        treatment_arm='ddqn',
        baseline_arm='vanilla',
        pair_by=('seed',),
        source='jensen_gap',
        covariates_per_env=covariates,
    )
    assert result.r_squared < 0.20, (
        f'R² = {result.r_squared:.4f}, expected < 0.20 (Beta(1, 3) '
        f'null sampling distribution, 75th percentile ≈ 0.16). '
        f'σ is structurally orthogonal to g at |A|=2; high R² '
        f'would indicate the regression invented signal.'
    )
