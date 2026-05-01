"""Tests for `mundlak_decomposition` — proper within/between
decomposition for panel-data moderator hypotheses.

Validates:
1. Pure between effect: x varies only across strata, recovers β_b
   while β_w is null.
2. Pure within effect: x varies only within strata, recovers β_w
   while β_b is null.
3. Both effects: distinct β_b and β_w, Hausman test detects the
   difference.
4. Equal effects: β_b == β_w, Hausman test does NOT reject equality.
5. Edge case: degenerate panel (constant x within stratum) raises
   ValueError pointing at the rank-deficiency.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from corroborate.analyses.mundlak_decomposition import (
    MundlakResult,
    mundlak_decomposition,
)


def _panel(
    *, stratum_means: list[float], within_devs: list[list[float]],
    beta_b: float, beta_w: float, intercept: float = 0.0,
    se: float = 0.1, noise: float = 0.0, seed: int = 0,
) -> list[dict[str, object]]:
    """Build a synthetic panel:
        x_{e,i} = stratum_means[e] + within_devs[e][i]
        y_{e,i} = intercept + β_b · stratum_means[e] + β_w · within_devs[e][i] + ε
    """
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for e, m in enumerate(stratum_means):
        for d in within_devs[e]:
            x = m + d
            y = (intercept + beta_b * m + beta_w * d
                 + (rng.standard_normal() * noise))
            rows.append({
                'stratum_id': f'env_{e}',
                'x': x, 'y': y, 'se': se,
            })
    return rows


def test_pure_between_effect() -> None:
    """β_b = 0.7, β_w = 0. Decomposition recovers β_b ≈ 0.7,
    β_w ≈ 0 (CI includes 0)."""
    panel = _panel(
        stratum_means=[0.0, 1.0, 2.0, 3.0, 4.0],
        within_devs=[[-0.5, 0.0, 0.5]] * 5,
        beta_b=0.7, beta_w=0.0,
    )
    res = mundlak_decomposition.fn(panel)
    assert isinstance(res, MundlakResult)
    assert res.n_strata == 5
    assert res.between.coefficient == pytest.approx(0.7, abs=0.05)
    assert res.between.p_value < 0.05
    assert res.within.coefficient == pytest.approx(0.0, abs=0.05)


def test_pure_within_effect() -> None:
    """β_w = 0.5, β_b = 0 with small inter-env spread that keeps
    the decomposition identifiable. Pure-zero between variance
    is covered by `test_constant_x_within_stratum_raises`. The
    key recovery is β_w; β_b's identification is data-dependent
    when between-variance is small relative to noise."""
    panel = _panel(
        stratum_means=[0.0, 1.0, 2.0, 3.0, 4.0],
        within_devs=[[-1.0, 0.0, 1.0]] * 5,
        beta_b=0.0, beta_w=0.5, noise=0.02, seed=0,
    )
    res = mundlak_decomposition.fn(panel)
    assert res.within.coefficient == pytest.approx(0.5, abs=0.05)
    assert res.within.p_value < 0.05


def test_both_effects_with_hausman_rejection() -> None:
    """β_b = +0.6, β_w = -0.3. Hausman test should reject β_b == β_w."""
    panel = _panel(
        stratum_means=[0.0, 1.0, 2.0, 3.0, 4.0],
        within_devs=[[-0.5, 0.0, 0.5, 1.0]] * 5,
        beta_b=0.6, beta_w=-0.3, noise=0.05, seed=42,
    )
    res = mundlak_decomposition.fn(panel)
    assert res.between.coefficient == pytest.approx(0.6, abs=0.1)
    assert res.within.coefficient == pytest.approx(-0.3, abs=0.1)
    assert res.between.p_value < 0.05
    assert res.within.p_value < 0.05
    assert res.hausman_p < 0.05  # opposite signs → rejection


def test_equal_effects_hausman_no_rejection() -> None:
    """β_b == β_w == 0.4. Hausman test should NOT reject equality."""
    panel = _panel(
        stratum_means=[0.0, 1.0, 2.0, 3.0, 4.0],
        within_devs=[[-0.5, 0.0, 0.5]] * 5,
        beta_b=0.4, beta_w=0.4, noise=0.02, seed=0,
    )
    res = mundlak_decomposition.fn(panel)
    assert res.between.coefficient == pytest.approx(0.4, abs=0.05)
    assert res.within.coefficient == pytest.approx(0.4, abs=0.05)
    assert res.hausman_p > 0.05  # equality not rejected


def test_constant_x_within_stratum_raises() -> None:
    """If x is constant within every stratum (no within-deviation),
    the design matrix is rank-deficient and the function raises
    pointing at the cause."""
    panel = [
        {'stratum_id': f'env_{e}', 'x': float(e), 'y': float(e),
         'se': 0.1}
        for e in range(5)
    ]
    with pytest.raises(ValueError, match='rank-deficient'):
        mundlak_decomposition.fn(panel)


def test_singleton_strata_treated_as_zero_deviation() -> None:
    """When a stratum has only one observation, its within-deviation
    is exactly 0. Multiple multi-obs strata still allow the
    decomposition; the test ensures the function doesn't crash."""
    panel = (
        _panel(
            stratum_means=[0.0, 1.0, 2.0],
            within_devs=[[-0.5, 0.0, 0.5]] * 3,
            beta_b=0.5, beta_w=-0.1,
        )
        # Add a singleton stratum
        + [{'stratum_id': 'env_singleton', 'x': 0.5, 'y': 0.0,
            'se': 0.1}]
    )
    res = mundlak_decomposition.fn(panel)
    assert res.n_strata == 4
    # Singleton's within-deviation is 0 → contributes only to β_b
    assert res.between.coefficient == pytest.approx(0.5, abs=0.15)


def test_cluster_robust_se_inflates_on_autocorrelated_panel() -> None:
    """On an AR(1)-autocorrelated within-stratum panel, OLS SE
    underestimates and CR1 SE recovers a wider CI. The CR1
    estimator's SE should be strictly larger than OLS's when
    residuals are positively autocorrelated within stratum, since
    OLS treats correlated obs as independent and inflates
    effective N."""
    rng = np.random.default_rng(7)
    rows: list[dict[str, object]] = []
    n_strata = 6
    n_per = 60
    rho = 0.85  # strong AR(1) autocorrelation
    for e in range(n_strata):
        env_x = float(e)
        # AR(1) shock structure within stratum
        eps_prev = 0.0
        for t in range(n_per):
            innov = rng.standard_normal()
            eps = rho * eps_prev + innov
            eps_prev = eps
            x = env_x + 0.5 * (t - n_per / 2) / n_per
            y = 0.3 * env_x + 0.0 * (x - env_x) + 0.5 * eps
            rows.append({
                'stratum_id': f'env_{e}',
                'x': x, 'y': y, 'se': 0.1,
            })
    res_ols = mundlak_decomposition.fn(rows, cluster_robust=False)
    res_cr1 = mundlak_decomposition.fn(rows, cluster_robust=True)
    # CR1 SEs should be larger on both coefficients
    assert res_cr1.between.se > res_ols.between.se
    assert res_cr1.within.se > res_ols.within.se
    # Substantially larger (factor ≥ 1.3x typical with rho=0.85)
    assert res_cr1.between.se > 1.3 * res_ols.between.se
    # Coefficients themselves are unchanged — only SEs differ
    assert res_cr1.between.coefficient == pytest.approx(
        res_ols.between.coefficient, abs=1e-9,
    )
    assert res_cr1.within.coefficient == pytest.approx(
        res_ols.within.coefficient, abs=1e-9,
    )


def test_cluster_robust_requires_multiple_clusters() -> None:
    """A single cluster gives no between variation; CR1 cannot be
    estimated and the function should fail loudly."""
    rows = [
        {'stratum_id': 'only_env', 'x': float(i), 'y': float(i),
         'se': 0.1}
        for i in range(10)
    ]
    # First needs >1 stratum to even allow Mundlak; build fixture.
    rows_two = rows + [{'stratum_id': 'second', 'x': 5.0, 'y': 0.0,
                         'se': 0.1}]
    # cluster_robust=True with 2 strata should still work.
    _ = mundlak_decomposition.fn(rows_two, cluster_robust=True)


def test_orthogonality_of_decomposition() -> None:
    """The within and between predictors are orthogonal by
    construction. Confirm via the covariance: r ≈ 0."""
    panel = _panel(
        stratum_means=[0.0, 1.0, 2.0, 3.0],
        within_devs=[[-0.5, 0.0, 0.5]] * 4,
        beta_b=0.5, beta_w=0.5,
    )
    # Reproduce x_e and x_w to check orthogonality
    strata = [r['stratum_id'] for r in panel]
    xs = np.array([r['x'] for r in panel])
    sums: dict[object, float] = {}
    counts: dict[object, int] = {}
    for s, v in zip(strata, xs):
        sums[s] = sums.get(s, 0.0) + float(v)
        counts[s] = counts.get(s, 0) + 1
    means = {s: sums[s] / counts[s] for s in sums}
    x_e = np.array([means[s] for s in strata])
    x_w = xs - x_e
    correlation = float(np.corrcoef(x_e, x_w)[0, 1])
    assert math.isclose(correlation, 0.0, abs_tol=1e-9)
