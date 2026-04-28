"""Tests for `corroborate.causal_discovery` — PC + JCI primitives.

This file covers commit 1 (CI tests). PC algorithm + orientation
land in commit 2 with their own tests."""
from __future__ import annotations

import math

import numpy as np
from scipy.stats import spearmanr  # type: ignore[reportMissingTypeStubs]

from corroborate.causal_discovery import (
    partial_spearman_rho,
    partial_spearman_rho_multi,
    stratified_partial_spearman_rho,
    stratified_spearman_rho,
)


# ============ partial_spearman_rho — single Z closed form ============

def test_partial_reduces_to_marginal_when_z_is_orthogonal() -> None:
    """When Z is orthogonal to both X and Y (independent noise),
    partial(X, Y | Z) ≈ marginal Spearman(X, Y)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    # Y carries a real correlation with X plus noise.
    y = 0.7 * x + rng.standard_normal(200) * 0.7
    z = rng.standard_normal(200)  # independent of X, Y

    marginal_r, _ = spearmanr(x, y)  # type: ignore[reportUnknownMemberType]
    partial_r, _ = partial_spearman_rho(x, y, z)
    assert abs(float(marginal_r) - partial_r) < 0.05


def test_partial_zero_when_y_is_function_of_z_alone() -> None:
    """If Y is a deterministic function of Z and X is independent
    of Z, partial(X, Y | Z) ≈ 0 — the X→Y path doesn't survive
    conditioning on Z."""
    rng = np.random.default_rng(0)
    n = 300
    z = rng.standard_normal(n)
    y = z + rng.standard_normal(n) * 0.01
    x = rng.standard_normal(n)
    # Make X correlated with Z marginally (so X⫫Y becomes a
    # confounding test).
    x = x + 0.5 * z

    rho, p = partial_spearman_rho(x, y, z)
    assert abs(rho) < 0.15
    assert p > 0.05


def test_partial_returns_nan_on_constant_z() -> None:
    """Constant Z → singular partial-correlation denominator →
    NaN (no information)."""
    n = 100
    rng = np.random.default_rng(0)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    z = np.zeros(n)
    rho, p = partial_spearman_rho(x, y, z)
    assert math.isnan(rho)
    assert math.isnan(p)


# ============ partial_spearman_rho_multi — residual regression ============

def test_partial_multi_matches_single_z_to_within_tolerance() -> None:
    """With one Z, multi-Z partial via OLS residuals should agree
    with the closed-form single-Z partial."""
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal(n)
    y = 0.5 * x + rng.standard_normal(n) * 0.7
    z = rng.standard_normal(n)
    rho_closed, _ = partial_spearman_rho(x, y, z)
    rho_multi, _ = partial_spearman_rho_multi(x, y, z.reshape(-1, 1))
    # Residual-regression has higher variance; allow 0.1 slack.
    assert abs(rho_closed - rho_multi) < 0.1


def test_partial_multi_with_two_z_columns() -> None:
    """Y = Z1 + Z2 + noise; X independent. Partial(X, Y | Z1, Z2) ≈
    0."""
    rng = np.random.default_rng(0)
    n = 300
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    y = z1 + z2 + rng.standard_normal(n) * 0.01
    x = rng.standard_normal(n)  # independent
    rho, p = partial_spearman_rho_multi(x, y, np.column_stack([z1, z2]))
    assert abs(rho) < 0.15
    assert p > 0.05


# ============ stratified_spearman_rho — JCI / Simpson's paradox ============

def test_stratified_finds_within_stratum_correlation_masked_by_pooling() -> None:
    """Simpson's-paradox fixture: within-stratum positive
    correlation, but mean shifts across strata flip the sign of
    the pooled-marginal correlation. Stratified Spearman recovers
    the within-stratum sign."""
    rng = np.random.default_rng(0)
    n_per = 100
    # Stratum A: positive correlation, low mean.
    xa = rng.standard_normal(n_per)
    ya = 0.7 * xa + rng.standard_normal(n_per) * 0.3
    # Stratum B: positive correlation, high mean of x AND high
    # mean of y opposite to xa direction.
    xb = rng.standard_normal(n_per) + 5
    yb = 0.7 * xb + rng.standard_normal(n_per) * 0.3 - 8
    x = np.concatenate([xa, xb])
    y = np.concatenate([ya, yb])
    strata = ['a'] * n_per + ['b'] * n_per

    pooled_rho, _ = spearmanr(x, y)  # type: ignore[reportUnknownMemberType]
    strat_rho, strat_p = stratified_spearman_rho(x, y, strata)

    # Pooled marginal is misled (could be either sign depending on
    # exact shifts); stratified within is positive (~0.7).
    assert strat_rho > 0.5
    assert strat_p < 0.05
    # The pooled-vs-stratified separation is the load-bearing point
    # of JCI: they disagree.
    assert abs(float(pooled_rho) - strat_rho) > 0.05


def test_stratified_skips_too_small_strata() -> None:
    """Strata with n < min_stratum_size are dropped — they don't
    contribute to the Fisher-z pool."""
    rng = np.random.default_rng(0)
    # 50 samples in stratum 'a', 2 in stratum 'b'.
    xa = rng.standard_normal(50)
    ya = 0.7 * xa + rng.standard_normal(50) * 0.3
    xb = np.array([0.0, 1.0])
    yb = np.array([0.0, 1.0])
    x = np.concatenate([xa, xb])
    y = np.concatenate([ya, yb])
    strata = ['a'] * 50 + ['b'] * 2

    rho, _ = stratified_spearman_rho(x, y, strata, min_stratum_size=4)
    # 'b' is too small → dropped → result depends on 'a' alone.
    assert rho > 0.5


def test_stratified_returns_nan_when_no_eligible_strata() -> None:
    """All strata too small → NaN."""
    x = np.arange(6, dtype=np.float64)
    y = np.arange(6, dtype=np.float64)
    strata = list('abcdef')  # 1 obs per stratum
    rho, p = stratified_spearman_rho(x, y, strata, min_stratum_size=4)
    assert math.isnan(rho)
    assert math.isnan(p)


# ============ stratified_partial_spearman_rho ============

def test_stratified_partial_recovers_within_stratum_partial() -> None:
    """Each stratum has Y = f(Z) + noise, X independent of Z.
    Within-stratum partial(X, Y | Z) ≈ 0; stratified pool agrees."""
    rng = np.random.default_rng(0)
    n_per = 100

    def _stratum(_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z = rng.standard_normal(n_per)
        y = z + rng.standard_normal(n_per) * 0.5
        x = rng.standard_normal(n_per)
        return x, y, z

    xa, ya, za = _stratum(0)
    xb, yb, zb = _stratum(1)
    x = np.concatenate([xa, xb])
    y = np.concatenate([ya, yb])
    z = np.concatenate([za, zb])
    strata = ['a'] * n_per + ['b'] * n_per

    rho, p = stratified_partial_spearman_rho(x, y, z, strata)
    assert abs(rho) < 0.15
    assert p > 0.05
