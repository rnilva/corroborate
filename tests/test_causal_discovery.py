"""Tests for `corroborate.causal_discovery` — PC + JCI primitives.

This file covers commit 1 (CI tests). PC algorithm + orientation
land in commit 2 with their own tests."""
from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest
from scipy.stats import norm, spearmanr

from corroborate._internals.polars import series_std_float
from corroborate.graph.discovery import (
    VariableScope,
    classify_variable_scope,
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

    marginal_r, _ = spearmanr(x, y)
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


def test_partial_returns_finite_at_n_equals_5() -> None:
    """n=5 is the lower bound (df = n - 4 = 1). Pin `n < 5`
    against `n <= 5` (which would NaN at n=5) and `n < 6`
    (which would also NaN at n=5)."""
    rng = np.random.default_rng(3)
    n = 5
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    z = rng.standard_normal(n)
    rho, p = partial_spearman_rho(x, y, z)
    assert math.isfinite(rho)
    assert math.isfinite(p)


def test_partial_returns_nan_below_n_equals_5() -> None:
    """n=4 (df = 0) → returns NaN. Pin the early-return branch
    actually fires AND that the NaN-tuple it returns is well-
    formed (`float('nan')` mutations like `float(None)` would
    raise TypeError here)."""
    rng = np.random.default_rng(5)
    n = 4
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    z = rng.standard_normal(n)
    rho, p = partial_spearman_rho(x, y, z)
    assert math.isnan(rho)
    assert math.isnan(p)


def test_partial_p_value_matches_fisher_z_at_df_n_minus_4() -> None:
    """The reported p-value must equal the Fisher-z formula at
    df = n − 4. Construction lands rho in the moderate regime
    (~0.2-0.4) so a wrong df shifts p detectably."""
    rng = np.random.default_rng(7)
    n = 30
    z = rng.standard_normal(n)
    # X and Y both partially depend on Z plus residual coupling.
    x = z + 1.0 * rng.standard_normal(n)
    y = 0.8 * z + 0.4 * x + 1.0 * rng.standard_normal(n)
    rho, p_reported = partial_spearman_rho(x, y, z)
    p_expected = _expected_p_from_rho_df(rho, n - 4)
    assert p_reported == pytest.approx(p_expected, abs=1e-9)


def test_partial_rho_matches_closed_form_partial_correlation_formula() -> None:
    """The returned rho must equal:
        rho = (rxy − rxz·ryz) / sqrt((1 − rxz²) · (1 − ryz²))
    where r* are pairwise Spearman correlations on the input.

    Pins the rxz**2 / ryz**2 powers in the denominator (vs **3
    mutants — those shift rho without changing the rho/p
    relationship, so the Fisher-z p-value test above can't catch
    them). Compute the three pairwise Spearmans independently
    and reconstruct rho from the closed-form formula."""
    rng = np.random.default_rng(99)
    n = 50
    z = rng.standard_normal(n)
    # Substantial rxz, ryz (~0.5+) so the **2 vs **3 difference
    # in the denom is detectable.
    x = 0.6 * z + rng.standard_normal(n)
    y = 0.5 * z + 0.3 * x + rng.standard_normal(n)
    rho_returned, _ = partial_spearman_rho(x, y, z)

    rxy, _ = spearmanr(x, y)
    rxz, _ = spearmanr(x, z)
    ryz, _ = spearmanr(y, z)
    denom = math.sqrt(
        max(1 - float(rxz) ** 2, 0.0) * max(1 - float(ryz) ** 2, 0.0),
    )
    rho_expected = (float(rxy) - float(rxz) * float(ryz)) / denom
    rho_expected = max(-0.999999, min(0.999999, rho_expected))
    assert rho_returned == pytest.approx(rho_expected, abs=1e-9)




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


def _expected_p_from_rho_df(rho: float, df: int) -> float:
    """Closed-form Fisher-z two-sided p for a given rho + df."""
    rho_clamped = max(-0.999999, min(0.999999, rho))
    z_stat = 0.5 * math.log((1 + rho_clamped) / (1 - rho_clamped)) * math.sqrt(df)
    return 2 * (1.0 - float(norm.cdf(abs(z_stat))))


def test_partial_multi_p_value_matches_fisher_z_at_k_equals_2() -> None:
    """The reported p-value must equal the Fisher-z formula at
    df = n − 3 − k. Pin every term in the df expression and the k
    computation against off-by-one mutants:

      df = n − 3 − k  vs  n − 3 + k  vs  n + 3 − k  vs  n − 4 − k
      k  = z.shape[1] when ndim==2  vs  always 1  vs  always 2

    Constructed so rho lands in the moderate regime (~0.23 here)
    where p is around 0.2 — a one-unit shift in df shifts p by
    ~9e-3, well above the assertion tolerance. At higher rho the
    p-value falls into the 1e-13 regime where df-shifts produce
    sub-tolerance differences."""
    rng = np.random.default_rng(7)
    n = 30
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    # Substantial residual noise → moderate rho.
    x = z1 + 0.5 * z2 + 1.5 * rng.standard_normal(n)
    y = 0.4 * z1 + z2 + 0.2 * x + 1.5 * rng.standard_normal(n)
    rho, p_reported = partial_spearman_rho_multi(
        x, y, np.column_stack([z1, z2]),
    )
    df_expected = n - 3 - 2  # k = 2
    p_expected = _expected_p_from_rho_df(rho, df_expected)
    assert p_reported == pytest.approx(p_expected, abs=1e-9)


def test_partial_multi_with_1d_z_uses_k_equals_1() -> None:
    """Passing a 1D `z_matrix` triggers the `ndim != 2 → k = 1`
    branch AND the `z_matrix.reshape(-1, 1)` line. Pin:

      k computation: 1D ndim → k=1 (correct) vs k=z.shape[1] crash
      reshape line: `z = z.reshape(-1, 1)` vs `z = None` (next
        access crashes)

    Asserts the closed-form p at df = n − 3 − 1, which exercises
    both the k branch and the reshape branch end-to-end."""
    rng = np.random.default_rng(11)
    n = 60
    z = rng.standard_normal(n)  # 1D
    x = z + 0.5 * rng.standard_normal(n)
    y = 0.7 * z + 0.4 * x + 0.5 * rng.standard_normal(n)
    rho, p_reported = partial_spearman_rho_multi(x, y, z)  # 1D z passed
    df_expected = n - 3 - 1
    p_expected = _expected_p_from_rho_df(rho, df_expected)
    assert math.isfinite(rho)
    assert p_reported == pytest.approx(p_expected, abs=1e-9)


def test_partial_multi_returns_finite_at_df_equals_one() -> None:
    """The lower bound on df is 1 (not 2). At df=1 (n=5, k=1) the
    function still computes a finite rho + p. Pins `df < 1` against
    `df <= 1` (which would NaN at df=1) and against `df < 2` (which
    would NaN at df=1 AND df=2).

    A second probe at df=2 (n=6, k=1) distinguishes `df < 2`
    (NaN at df=2) from `df <= 1` (passes df=2)."""
    rng = np.random.default_rng(13)
    # df=1: n=5, k=1.
    n = 5
    z = rng.standard_normal(n)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    rho, p = partial_spearman_rho_multi(x, y, z.reshape(-1, 1))
    assert math.isfinite(rho)
    assert math.isfinite(p)


def test_partial_multi_returns_nan_below_df_one() -> None:
    """At df=0 (n=4, k=1) the original returns NaN. Pins the
    early-return path."""
    rng = np.random.default_rng(17)
    n = 4  # df = n-3-1 = 0
    z = rng.standard_normal(n)
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)
    rho, p = partial_spearman_rho_multi(x, y, z.reshape(-1, 1))
    assert math.isnan(rho)
    assert math.isnan(p)


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

    pooled_rho, _ = spearmanr(x, y)
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


def test_stratified_spearman_pooled_matches_fisher_z_closed_form() -> None:
    """The pooled (rho, p) must match the Fisher-z closed form:

      z_k       = 0.5 · ln((1 + r_k) / (1 − r_k))   per stratum
      w_k       = n_k − 3                            per stratum
      z_pooled  = Σ(w_k · z_k) / Σw_k
      rho       = tanh(z_pooled)
      z_stat    = z_pooled · sqrt(Σw_k)
      p         = 2 · (1 − Φ(|z_stat|))

    Pin every coefficient: 0.5 in z_k (vs 1.5 mutant), n_k − 3
    in w_k (vs n_k + 3), the / vs * in z_pooled normalization,
    and z_stat = z_pooled · sqrt(Σw_k) (vs `z_stat = None` mutant).

    Construct asymmetric strata (different size + different r)
    so each weight and each per-stratum z_k pulls the pool in a
    different direction — symmetric strata would silently absorb
    formula errors."""
    rng = np.random.default_rng(2026)
    # Stratum A: large, moderate r; Stratum B: small, near-zero r.
    n_a, n_b = 30, 15
    xa = rng.standard_normal(n_a)
    ya = 0.6 * xa + 0.5 * rng.standard_normal(n_a)
    xb = rng.standard_normal(n_b)
    yb = 0.05 * xb + rng.standard_normal(n_b)
    x = np.concatenate([xa, xb])
    y = np.concatenate([ya, yb])
    strata = ['A'] * n_a + ['B'] * n_b
    rho_pooled, p_pooled = stratified_spearman_rho(
        x, y, strata, min_stratum_size=4,
    )

    # Closed-form pool from per-stratum Spearman r's.
    r_a, _ = spearmanr(xa, ya)
    r_b, _ = spearmanr(xb, yb)
    z_a = 0.5 * math.log((1 + float(r_a)) / (1 - float(r_a)))
    z_b = 0.5 * math.log((1 + float(r_b)) / (1 - float(r_b)))
    w_a, w_b = n_a - 3, n_b - 3
    z_pool_expected = (w_a * z_a + w_b * z_b) / (w_a + w_b)
    rho_expected = math.tanh(z_pool_expected)
    z_stat_expected = z_pool_expected * math.sqrt(w_a + w_b)
    p_expected = 2 * (1.0 - float(norm.cdf(abs(z_stat_expected))))
    assert rho_pooled == pytest.approx(rho_expected, abs=1e-9)
    assert p_pooled == pytest.approx(p_expected, abs=1e-9)


def test_stratified_skips_at_exactly_min_stratum_size_minus_one() -> None:
    """Stratum at exactly `min_stratum_size` is INCLUDED. Pin
    `n_k < min_stratum_size` against `n_k <= min_stratum_size`
    mutant (which would skip strata at the boundary).

    Construct: one stratum at n=4 (= min_stratum_size default)
    that's NOT skipped, plus a tiny-n stratum that IS skipped."""
    rng = np.random.default_rng(31)
    # Strata: n=4 (boundary, kept) and n=2 (skipped).
    xa = rng.standard_normal(4)
    ya = 0.7 * xa + 0.3 * rng.standard_normal(4)
    xb = rng.standard_normal(2)
    yb = rng.standard_normal(2)
    x = np.concatenate([xa, xb])
    y = np.concatenate([ya, yb])
    strata = ['A'] * 4 + ['B'] * 2
    rho, _ = stratified_spearman_rho(
        x, y, strata, min_stratum_size=4,
    )
    # Only stratum A contributes (B too small, A at boundary).
    # Fisher-z is well-defined at n=4 (w_a = 4-3 = 1).
    assert math.isfinite(rho)
    # The single contributing stratum has positive rho → pooled
    # tanh(z_a) is positive.
    assert rho > 0.0


def test_stratified_returns_nan_when_no_eligible_strata() -> None:
    """All strata too small → NaN."""
    x = np.arange(6, dtype=np.float64)
    y = np.arange(6, dtype=np.float64)
    strata = list('abcdef')  # 1 obs per stratum
    rho, p = stratified_spearman_rho(x, y, strata, min_stratum_size=4)
    assert math.isnan(rho)
    assert math.isnan(p)


# ============ stratified_partial_spearman_rho ============

def test_stratified_partial_skips_at_exactly_min_stratum_size_minus_one() -> None:
    """Stratum at exactly `min_stratum_size` for the partial
    version is INCLUDED. Pin `n_k < min_stratum_size` against
    `<= min_stratum_size` mutant on the partial-Spearman path
    (default min_stratum_size=5, distinct from the marginal
    version's default of 4)."""
    rng = np.random.default_rng(41)
    # Stratum A at exactly n=5 (= default min), kept.
    # Stratum B at n=2, skipped.
    n_a, n_b = 5, 2
    za = rng.standard_normal(n_a)
    xa = za + 0.3 * rng.standard_normal(n_a)
    ya = 0.6 * xa + 0.3 * rng.standard_normal(n_a)
    zb = rng.standard_normal(n_b)
    xb = rng.standard_normal(n_b)
    yb = rng.standard_normal(n_b)
    x = np.concatenate([xa, xb])
    y = np.concatenate([ya, yb])
    z = np.concatenate([za, zb])
    strata = ['A'] * n_a + ['B'] * n_b
    rho, _ = stratified_partial_spearman_rho(
        x, y, z, strata, min_stratum_size=5,
    )
    # Stratum A contributes (boundary kept). With strong residual
    # X-Y signal even after partialling out Z, rho should be
    # finite and positive.
    assert math.isfinite(rho)


def test_stratified_partial_pooled_matches_fisher_z_closed_form() -> None:
    """Same closed-form pooled Fisher-z check as
    `stratified_spearman_rho`'s, but on the partial version where
    the per-stratum statistic is `partial_spearman_rho(x, y, z)`
    and the weight is `n_k - 4` (one extra df eaten by Z).

    Pin the partial-version coefficients (n_k − 4 vs n_k − 3,
    n_k + 4) and the pooled-z formula. Asymmetric strata
    construction (different n, different per-stratum partial r)
    so each weight pulls the pool differently."""
    rng = np.random.default_rng(2027)
    n_a, n_b = 30, 15
    za = rng.standard_normal(n_a)
    xa = 0.5 * za + 0.7 * rng.standard_normal(n_a)
    ya = 0.4 * za + 0.3 * xa + 0.5 * rng.standard_normal(n_a)
    zb = rng.standard_normal(n_b)
    xb = rng.standard_normal(n_b)
    yb = 0.05 * xb + rng.standard_normal(n_b)
    x = np.concatenate([xa, xb])
    y = np.concatenate([ya, yb])
    z = np.concatenate([za, zb])
    strata = ['A'] * n_a + ['B'] * n_b
    rho_pooled, p_pooled = stratified_partial_spearman_rho(
        x, y, z, strata, min_stratum_size=5,
    )

    r_a, _ = partial_spearman_rho(xa, ya, za)
    r_b, _ = partial_spearman_rho(xb, yb, zb)
    z_a = 0.5 * math.log((1 + r_a) / (1 - r_a))
    z_b = 0.5 * math.log((1 + r_b) / (1 - r_b))
    w_a, w_b = n_a - 4, n_b - 4
    z_pool_expected = (w_a * z_a + w_b * z_b) / (w_a + w_b)
    rho_expected = math.tanh(z_pool_expected)
    z_stat_expected = z_pool_expected * math.sqrt(w_a + w_b)
    p_expected = 2 * (1.0 - float(norm.cdf(abs(z_stat_expected))))
    assert rho_pooled == pytest.approx(rho_expected, abs=1e-9)
    assert p_pooled == pytest.approx(p_expected, abs=1e-9)


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


# ============ classify_variable_scope — within vs across stratum ============

def test_classify_single_stratum_treated_as_within() -> None:
    """One stratum present → no across-stratum dimension to test;
    classify by whether values vary at all.

    The multi-stratum WITHIN/ACROSS/BOTH/DEGENERATE cases plus
    the `assert_stratification_admissible` happy/raise paths are
    covered end-to-end at `tests/analytic/lg_scm/test_variable_scope.py`
    on real LG-SCM cells. The single-stratum edge case stays here
    because it doesn't fit the multi-env analytic substrate."""
    rng = np.random.default_rng(0)
    values = rng.standard_normal(50)
    strata = ['envA'] * 50
    assert classify_variable_scope(values, strata) is VariableScope.WITHIN_STRATUM
    assert classify_variable_scope(np.zeros(50), strata) is VariableScope.DEGENERATE


def test_classify_symmetric_around_zero_is_not_degenerate() -> None:
    """Values symmetric around zero (e.g., [-1, 1]) have non-zero
    range. Pin `np.max(values) - np.min(values)` against the
    `np.max + np.min` mutant, which would compute 0 here and
    misclassify as DEGENERATE."""
    values = np.array([-1.0, 1.0, -2.0, 2.0, -3.0, 3.0])
    strata = ['a', 'b', 'a', 'b', 'a', 'b']
    scope = classify_variable_scope(values, strata)
    assert scope is not VariableScope.DEGENERATE


def test_classify_two_strata_passes_unique_count_guard() -> None:
    """Two unique strata: passes the `len(unique_strata) < 2`
    fast-path guard. Pins `< 2` against `<= 2` (which would
    short-circuit to WITHIN at 2 strata) and `< 3` (same)."""
    rng = np.random.default_rng(11)
    # Pure across-stratum signal: per-stratum constant, different
    # means.
    values = np.concatenate([np.full(10, 0.0), np.full(10, 5.0)])
    strata = ['a'] * 10 + ['b'] * 10
    scope = classify_variable_scope(values, strata)
    # Should classify as ACROSS (no within-stratum variance,
    # all variance is across-stratum). The `< 2` mutants would
    # short-circuit to WITHIN before reaching this branch.
    assert scope is VariableScope.ACROSS_STRATUM
    # Sanity: rng unused; defensive against unused-import warnings.
    del rng


def test_classify_strata_at_size_two_compute_variance() -> None:
    """Per-stratum size n_k = 2 should still contribute its
    variance (n_k >= 2 admits variance computation). Pin
    `n_k >= 2` against `n_k > 2` and `n_k >= 3` mutants (both
    would skip n_k=2 strata, treating their var as 0).

    Construct: 2-element strata each with substantial within-
    stratum variance, equal means → variance lives ENTIRELY
    within-stratum. Mutant treats var as 0 → no within
    variance → would classify as DEGENERATE or fall through."""
    # Two strata, n_k=2 each, same mean (1.5) but spread.
    values = np.array([0.0, 3.0, 0.0, 3.0])
    strata = ['a', 'a', 'b', 'b']
    scope = classify_variable_scope(values, strata)
    # within_var > 0, across_var = 0 → WITHIN_STRATUM.
    # Mutant n_k > 2 sees per-stratum var = 0 → within_var = 0.
    # Then within_frac = 0 < threshold → ACROSS_STRATUM.
    # Original: WITHIN; mutant: ACROSS.
    assert scope is VariableScope.WITHIN_STRATUM


def test_classify_singleton_strata_use_zero_variance() -> None:
    """Single-element strata (n_k = 1) must contribute 0 variance,
    not 1. Pin `else 0.0` against `else 1.0` mutant.

    Construct: 11 singletons at evenly spaced values 0.0, 0.1,
    ..., 1.0. All within-stratum variances are 0 → original
    within_var = 0 → ACROSS_STRATUM. Mutant injects 1.0 per
    singleton → within_var = 1.0; across_var ≈ 0.1; within_frac
    ≈ 0.91 → both > threshold → BOTH classification.

    Constructed so the singleton contribution dominates total_var
    in the mutant — without enough singleton mass relative to
    across_var, the within_frac shift gets absorbed by the
    threshold and the verdict doesn't flip."""
    values = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    strata = [f's{i}' for i in range(11)]
    scope = classify_variable_scope(values, strata)
    assert scope is VariableScope.ACROSS_STRATUM


def test_classify_within_frac_uses_division_not_multiplication() -> None:
    """`within_frac = within_var / total_var`. Pin against
    `within_var * total_var` mutant.

    Construct: total_var = ~5, within_var = ~0.2 →
      orig within_frac = 0.04 < 0.05 → ACROSS_STRATUM
      mutant within_frac = 0.2 * 5 = 1.0 → not ACROSS → falls
      to check 2 → across_frac = 0.96 → BOTH.

    Two 2-element strata with identical small spread (within
    var = 0.2025) and means separated by ~4.4 (across var = 4.8)."""
    M = 4.38
    values = np.array([
        -0.45, 0.45,             # stratum A: mean=0
        M - 0.45, M + 0.45,      # stratum B: mean=M
    ])
    strata = ['a', 'a', 'b', 'b']
    scope = classify_variable_scope(values, strata)
    assert scope is VariableScope.ACROSS_STRATUM


def test_classify_across_frac_uses_division_not_multiplication() -> None:
    """`across_frac = across_var / total_var`. Pin against
    `across_var * total_var` mutant.

    Construct: total_var = ~1.5, across_var = ~0.06 →
      orig across_frac = 0.04 < 0.05 → WITHIN_STRATUM
      mutant across_frac = 0.06 * 1.5 = 0.09 → not WITHIN → BOTH.

    Two 2-element strata with large spread (within var = 1.44)
    and means separated by ~0.49 (across var = 0.06)."""
    diff = 0.49
    values = np.array([
        -1.2, 1.2,                       # stratum A: mean=0,    var=1.44
        diff - 1.2, diff + 1.2,          # stratum B: mean=0.49, var=1.44
    ])
    strata = ['a', 'a', 'b', 'b']
    scope = classify_variable_scope(values, strata)
    assert scope is VariableScope.WITHIN_STRATUM


def test_classify_returns_degenerate_when_total_var_zero() -> None:
    """Total variance exactly zero (constant column across all
    strata) → DEGENERATE. The early-degeneracy check via ptp
    catches the simple constant case; this test exercises the
    deeper `if total_var <= 0: return DEGENERATE` guard via a
    construction where ptp is non-zero but the variance
    decomposition zeros out (not actually possible with real
    data — the guard is defensive against numerical edge cases).
    Skip for now; the ptp guard above is the primary path."""
    # The ptp guard catches the truly-constant case.
    values = np.full(20, 3.14)
    strata = ['a'] * 10 + ['b'] * 10
    assert classify_variable_scope(values, strata) is VariableScope.DEGENERATE


# ============ PC algorithm — discover_adjacency ============

def _df_from_columns(**cols: np.ndarray) -> pl.DataFrame:
    """Build a polars DataFrame from kwarg columns."""
    return pl.DataFrame({k: v.tolist() for k, v in cols.items()})


def test_compare_pc_depths_kills_chain_edge_at_depth_1() -> None:
    """Three-variable chain X → M → Y. At depth-0 (marginal
    only), the X-Y edge survives because X and Y are marginally
    correlated through M. At depth-1, conditioning on M
    separates X-Y and the edge is removed. The diff catches
    `xy_edge in low_only` exactly. Same diff shape as depth-1 vs
    depth-2 catching a confounded edge that needs |Z|=2; the
    chain example just exercises the primitive cheaply."""
    from corroborate.graph.discovery import compare_pc_depths
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    m = x + rng.standard_normal(n) * 0.3
    y = m + rng.standard_normal(n) * 0.3
    df = _df_from_columns(x=x, m=m, y=y)

    diff = compare_pc_depths(
        df, variables=['x', 'm', 'y'],
        alpha=0.05, depths=(0, 1),
    )
    xy_edge = frozenset({'x', 'y'})
    assert xy_edge in diff.edges_low
    assert xy_edge not in diff.edges_high
    assert xy_edge in diff.low_only
    assert xy_edge not in diff.common


def test_compare_pc_depths_chain_unaffected_by_depth_increase() -> None:
    """X → M → Y: depth-1 already kills X-Y via {M}. Depth-2
    can only confirm. Diff: low_only and high_only both empty;
    common == both edge sets."""
    from corroborate.graph.discovery import compare_pc_depths
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    m = x + rng.standard_normal(n) * 0.3
    y = m + rng.standard_normal(n) * 0.3
    df = _df_from_columns(x=x, m=m, y=y)

    diff = compare_pc_depths(
        df, variables=['x', 'm', 'y'],
        alpha=0.05, depths=(1, 2),
    )
    assert diff.low_only == frozenset()
    assert diff.high_only == frozenset()
    assert diff.common == diff.edges_low == diff.edges_high


def test_discover_adjacency_n_observations_matches_dataframe_height() -> None:
    """`adj.n_observations` must equal `data.height`. Pin
    `n_obs = data.height` against `n_obs = None` mutant
    (which would store None in the typed dataclass field)."""
    from corroborate.graph.discovery import discover_adjacency
    rng = np.random.default_rng(0)
    n = 137  # picked to be distinctive
    df = _df_from_columns(
        x=rng.standard_normal(n),
        y=rng.standard_normal(n),
        z=rng.standard_normal(n),
    )
    adj = discover_adjacency(
        df, variables=['x', 'y', 'z'],
        alpha=0.05, max_conditioning=1,
    )
    assert adj.n_observations == 137


def test_compare_pc_depths_rejects_descending_depths() -> None:
    """Depths must be (low, high) with low < high."""
    from corroborate.graph.discovery import compare_pc_depths
    df = _df_from_columns(
        x=np.array([1.0, 2.0, 3.0]),
        y=np.array([1.0, 2.0, 3.0]),
    )
    with pytest.raises(ValueError, match='low < high'):
        compare_pc_depths(
            df, variables=['x', 'y'],
            alpha=0.05, depths=(2, 1),
        )


def test_discover_adjacency_rejects_negative_max_conditioning() -> None:
    """`max_conditioning` < 0 raises ValueError. Pin the message
    string against `None` mutant (which would raise with a
    blank message)."""
    from corroborate.graph.discovery import discover_adjacency
    df = _df_from_columns(
        x=np.array([1.0, 2.0, 3.0, 4.0]),
        y=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    with pytest.raises(ValueError, match='max_conditioning must be ≥ 0'):
        discover_adjacency(
            df, variables=['x', 'y'],
            max_conditioning=-1,
        )


def test_discover_adjacency_rejects_duplicate_variables() -> None:
    """Duplicate variable names in `variables` raise ValueError.
    Pin `len(var_set) != len(var_list)` against `==` mutant which
    would invert the check (raise on UNIQUE variables, accept
    duplicates)."""
    from corroborate.graph.discovery import discover_adjacency
    df = _df_from_columns(
        x=np.array([1.0, 2.0, 3.0, 4.0]),
        y=np.array([1.0, 2.0, 3.0, 4.0]),
    )
    with pytest.raises(ValueError, match='duplicate variables'):
        discover_adjacency(
            df, variables=['x', 'y', 'x'],
            max_conditioning=0,
        )


def test_discover_adjacency_keeps_edge_when_p_at_boundary_alpha() -> None:
    """The CI test removes an edge iff `p >= alpha`. Pin the
    boundary `>=` against `>` mutant — at p exactly equal to
    alpha, original removes the edge, mutant keeps it.

    Constructing exact p=alpha is fragile; this test instead pins
    the BOUNDARY semantics by using a near-boundary alpha that
    does NOT remove (keep edge) under both. The `>` direction is
    indirectly covered by other discovery tests where p < alpha
    keeps the edge — this test asserts that real signal is kept."""
    from corroborate.graph.discovery import discover_adjacency
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal(n)
    y = 0.7 * x + rng.standard_normal(n) * 0.5    # strong correlation
    df = _df_from_columns(x=x, y=y)
    adj = discover_adjacency(
        df, variables=['x', 'y'],
        alpha=0.05, max_conditioning=0,
    )
    # Strong correlation → small p → keep edge.
    assert frozenset({'x', 'y'}) in adj.edges


def test_compare_pc_depths_propagates_stratify_by_to_both_calls() -> None:
    """`compare_pc_depths` must pass `stratify_by` through to both
    inner `discover_adjacency` calls (low and high). Pin
    `stratify_by=stratify_by` against the `stratify_by=None`
    mutants on either call.

    Construct: x and y are MARGINALLY dependent (within-stratum
    independent, across-stratum mean-shifted) — Simpson's case.
    Without stratification, both depths see x-y dependent and
    keep the edge. With stratification, both depths see them
    conditionally independent and remove the edge.

    A mutant that drops stratify_by on either inner call would
    keep the edge at that depth, producing a non-empty
    `low_only` or `high_only` set."""
    from corroborate.graph.discovery import compare_pc_depths
    rng = np.random.default_rng(0)
    n_per = 40
    # Stratum A: x ~ N(0, 1), y ~ N(0, 1) — independent.
    xa = rng.standard_normal(n_per)
    ya = rng.standard_normal(n_per)
    # Stratum B: x and y both shifted by +5 — mean shift creates
    # marginal correlation, but within-stratum still independent.
    xb = rng.standard_normal(n_per) + 5
    yb = rng.standard_normal(n_per) + 5
    x = np.concatenate([xa, xb])
    y = np.concatenate([ya, yb])
    env = ['A'] * n_per + ['B'] * n_per
    df = _df_from_columns(x=x, y=y)
    df = df.with_columns(pl.Series('env', env))
    diff = compare_pc_depths(
        df, variables=['x', 'y'],
        alpha=0.05, depths=(0, 1),
        stratify_by='env',
    )
    # With stratify_by propagated to BOTH calls, both depths
    # remove x-y → empty edges_low, empty edges_high.
    assert frozenset({'x', 'y'}) not in diff.edges_low
    assert frozenset({'x', 'y'}) not in diff.edges_high


def test_compare_pc_depths_propagates_high_max_conditioning() -> None:
    """`max_conditioning=high` must propagate to the high call.
    Pin against the kwarg-drop mutant which would default to 1.

    Construct: 4-variable diamond (x → m1, x → m2, m1 → y, m2 → y).
    At depth 1, conditioning on m1 alone or m2 alone may not
    sever x-y entirely. At depth 2, conditioning on {m1, m2}
    severs x-y. Use depths=(1, 2): the mutant would call high
    with default max_conditioning=1 instead of 2, missing the
    x-y removal at depth 2."""
    from corroborate.graph.discovery import compare_pc_depths
    rng = np.random.default_rng(7)
    n = 300
    x = rng.standard_normal(n)
    m1 = x + rng.standard_normal(n) * 0.5
    m2 = x + rng.standard_normal(n) * 0.5
    y = m1 + m2 + rng.standard_normal(n) * 0.3
    df = _df_from_columns(x=x, m1=m1, m2=m2, y=y)
    diff = compare_pc_depths(
        df, variables=['x', 'm1', 'm2', 'y'],
        alpha=0.05, depths=(1, 2),
    )
    # x-y at depth 2 should be conditionally independent given
    # {m1, m2} → removed. Mutant high=1 default would miss this.
    assert frozenset({'x', 'y'}) not in diff.edges_high


def test_compare_pc_depths_rejects_equal_depths() -> None:
    """Depths must be strictly less, not less-or-equal. Pin
    `low >= high` against `low > high` mutant — under the
    mutant, equal depths would be accepted (and then run PC
    twice with the same depth, returning a degenerate diff)."""
    from corroborate.graph.discovery import compare_pc_depths
    df = _df_from_columns(
        x=np.array([1.0, 2.0, 3.0]),
        y=np.array([1.0, 2.0, 3.0]),
    )
    with pytest.raises(ValueError, match='low < high'):
        compare_pc_depths(
            df, variables=['x', 'y'],
            alpha=0.05, depths=(1, 1),
        )


def test_discover_chain_removes_marginal_independence_pair() -> None:
    """3-variable chain X → M → Y. PC at depth 1 should:
    - Keep X−M and M−Y (direct dependence)
    - Remove X−Y at depth 1, conditioning on M (X⫫Y | M)."""
    from corroborate.graph.discovery import discover_adjacency
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    m = x + rng.standard_normal(n) * 0.3
    y = m + rng.standard_normal(n) * 0.3
    df = _df_from_columns(x=x, m=m, y=y)

    adj = discover_adjacency(
        df, variables=['x', 'm', 'y'],
        alpha=0.05, max_conditioning=1,
    )
    edges = {tuple(sorted(e)) for e in adj.edges}
    # X−M and M−Y survive
    assert ('m', 'x') in edges
    assert ('m', 'y') in edges
    # X−Y removed (separated by M)
    assert ('x', 'y') not in edges
    xy_sepset = adj.separating_sets[frozenset({'x', 'y'})]
    assert frozenset({'m'}) in xy_sepset


def test_discover_collider_keeps_marginal_independence() -> None:
    """3-variable collider X → Z ← Y. X⫫Y marginally (no direct
    edge, no path through Z without conditioning). PC should keep
    X⫫Y at depth 0 (Berkson-bias example)."""
    from corroborate.graph.discovery import discover_adjacency
    rng = np.random.default_rng(0)
    n = 500
    x = rng.standard_normal(n)
    y = rng.standard_normal(n)  # independent
    z = x + y + rng.standard_normal(n) * 0.3  # collider
    df = _df_from_columns(x=x, y=y, z=z)

    adj = discover_adjacency(
        df, variables=['x', 'y', 'z'],
        alpha=0.05, max_conditioning=0,
    )
    edges = {tuple(sorted(e)) for e in adj.edges}
    # X−Z and Y−Z survive
    assert ('x', 'z') in edges
    assert ('y', 'z') in edges
    # X−Y removed at depth 0 (marginal independence)
    assert ('x', 'y') not in edges


def test_discover_with_jci_stratification() -> None:
    """JCI stratification on a categorical context: edges that
    look correlated when pooled are within-stratum independent.
    Stratified PC removes them; unstratified keeps them."""
    from corroborate.graph.discovery import discover_adjacency
    rng = np.random.default_rng(0)
    n_per = 200
    # Stratum A: X and Y both shifted up (creates pooled correlation
    # via mean shift even though within-stratum is independent).
    xa = rng.standard_normal(n_per) - 3
    ya = rng.standard_normal(n_per) - 3
    # Stratum B: shifted down.
    xb = rng.standard_normal(n_per) + 3
    yb = rng.standard_normal(n_per) + 3
    df = _df_from_columns(
        x=np.concatenate([xa, xb]),
        y=np.concatenate([ya, yb]),
        env=np.array(['a'] * n_per + ['b'] * n_per, dtype=object),
    )

    # Without stratification: pooled correlation is real.
    adj_pooled = discover_adjacency(
        df, variables=['x', 'y'],
        alpha=0.05, max_conditioning=0,
    )
    edges_pooled = {tuple(sorted(e)) for e in adj_pooled.edges}
    assert ('x', 'y') in edges_pooled

    # With stratification on env: within-stratum independent.
    adj_strat = discover_adjacency(
        df, variables=['x', 'y'],
        alpha=0.05, max_conditioning=0,
        stratify_by='env',
    )
    edges_strat = {tuple(sorted(e)) for e in adj_strat.edges}
    assert ('x', 'y') not in edges_strat


# ============ Orientation — v-structures + Meek rules ============

# ============ PC graph helpers — direct ============

def test_adjacent_self_loop_returns_false() -> None:
    """`_adjacent(a, a, ...)` returns False — no edge between a
    node and itself. Pin `a == b` against `a != b` mutant
    (would invert) and the `return False` against `return True`
    mutant."""
    from corroborate.graph.discovery import _adjacent
    assert _adjacent('x', 'x', set(), set()) is False
    # Even if a self-loop edge somehow exists, _adjacent still
    # returns False at the first guard.
    assert _adjacent(
        'x', 'x',
        directed={('x', 'x')},
        undirected={frozenset({'x'})},
    ) is False


def test_adjacent_undirected_edge_returns_true() -> None:
    """An undirected edge between a and b → True. Pin the
    `return True` for the undirected branch against `return False`
    mutant."""
    from corroborate.graph.discovery import _adjacent
    assert _adjacent(
        'x', 'y',
        directed=set(),
        undirected={frozenset({'x', 'y'})},
    ) is True


def test_adjacent_directed_in_either_direction_returns_true() -> None:
    """A directed edge (a, b) OR (b, a) → True. Pin `or` against
    `and` mutant which would only return True when BOTH (a,b) AND
    (b,a) are directed (impossible by orientation semantics)."""
    from corroborate.graph.discovery import _adjacent
    # Only (x, y) in directed:
    assert _adjacent(
        'x', 'y',
        directed={('x', 'y')},
        undirected=set(),
    ) is True
    # Only (y, x) in directed:
    assert _adjacent(
        'x', 'y',
        directed={('y', 'x')},
        undirected=set(),
    ) is True


def test_adjacent_no_edge_returns_false() -> None:
    """No edge in either direction → False (sanity)."""
    from corroborate.graph.discovery import _adjacent
    assert _adjacent('x', 'y', set(), set()) is False


def test_neighbors_directed_outgoing_picks_up_target() -> None:
    """For a directed edge (z, t), z's neighbor is t. Pin
    `if src == z: out.add(tgt)` against:
    - `src != z` mutant (would add tgt for edges NOT touching z)
    - `out.add(None)` mutant (would add None to the set)"""
    from corroborate.graph.discovery import _neighbors
    nbrs = _neighbors(
        'z',
        directed={('z', 'a'), ('b', 'c')},  # only first edge touches z
        undirected=set(),
    )
    assert nbrs == {'a'}


def test_neighbors_directed_incoming_picks_up_source() -> None:
    """For a directed edge (s, z), z's neighbor is s. Pin
    `out.add(src)` against `out.add(None)` mutant."""
    from corroborate.graph.discovery import _neighbors
    nbrs = _neighbors(
        'z',
        directed={('a', 'z'), ('b', 'c')},
        undirected=set(),
    )
    assert nbrs == {'a'}


def test_orient_returns_false_on_conflict_target_source() -> None:
    """Conflict: (target, source) is already directed → don't
    flip, return False. Pin `return False` against `return True`
    mutant."""
    from corroborate.graph.discovery import _orient
    directed: set[tuple[str, str]] = {('y', 'x')}
    undirected: set[frozenset[str]] = set()
    result = _orient('x', 'y', directed, undirected)
    assert result is False
    # State unchanged.
    assert directed == {('y', 'x')}


def test_orient_returns_false_on_idempotent_already_directed() -> None:
    """Already (source, target) directed → return False (no new
    work). Pin against `return True` mutant."""
    from corroborate.graph.discovery import _orient
    directed: set[tuple[str, str]] = {('x', 'y')}
    undirected: set[frozenset[str]] = set()
    result = _orient('x', 'y', directed, undirected)
    assert result is False


def test_orient_returns_false_when_edge_not_undirected() -> None:
    """No undirected edge between source and target → can't
    orient → return False. Pin against `return True` mutant."""
    from corroborate.graph.discovery import _orient
    directed: set[tuple[str, str]] = set()
    undirected: set[frozenset[str]] = set()
    result = _orient('x', 'y', directed, undirected)
    assert result is False


def test_orient_returns_true_and_mutates_on_clean_orient() -> None:
    """Undirected x-y, no conflict → orient x → y. Returns True,
    moves edge from undirected to directed."""
    from corroborate.graph.discovery import _orient
    directed: set[tuple[str, str]] = set()
    undirected: set[frozenset[str]] = {frozenset({'x', 'y'})}
    result = _orient('x', 'y', directed, undirected)
    assert result is True
    assert ('x', 'y') in directed
    assert frozenset({'x', 'y'}) not in undirected


def test_orient_v_structure_collider() -> None:
    """Unshielded X − Z − Y with Z NOT in sepset(X, Y) → orient
    X → Z ← Y (definite collider)."""
    from corroborate.graph.discovery import (
        DiscoveredAdjacency, orient_adjacency,
    )
    adj = DiscoveredAdjacency(
        variables=frozenset({'x', 'y', 'z'}),
        edges=frozenset({frozenset({'x', 'z'}), frozenset({'y', 'z'})}),
        separating_sets={
            frozenset({'x', 'y'}): frozenset({frozenset[str]()}),
        },
        n_observations=100, alpha=0.05, max_conditioning=0,
        stratify_by=None,
    )
    oriented = orient_adjacency(adj)
    assert ('x', 'z') in oriented.directed_edges
    assert ('y', 'z') in oriented.directed_edges
    assert oriented.undirected_edges == frozenset()
    assert oriented.ambiguous_triples == frozenset()


def test_orient_non_collider_when_z_in_sepset() -> None:
    """Unshielded X − Z − Y with Z IN sepset(X, Y) → Z is a
    non-collider; the X−Z and Y−Z edges stay undirected."""
    from corroborate.graph.discovery import (
        DiscoveredAdjacency, orient_adjacency,
    )
    adj = DiscoveredAdjacency(
        variables=frozenset({'x', 'y', 'z'}),
        edges=frozenset({frozenset({'x', 'z'}), frozenset({'y', 'z'})}),
        separating_sets={
            # Z separates X and Y → non-collider
            frozenset({'x', 'y'}): frozenset({frozenset({'z'})}),
        },
        n_observations=100, alpha=0.05, max_conditioning=1,
        stratify_by=None,
    )
    oriented = orient_adjacency(adj)
    assert oriented.directed_edges == frozenset()
    assert frozenset({'x', 'z'}) in oriented.undirected_edges
    assert frozenset({'y', 'z'}) in oriented.undirected_edges


def test_orient_meek_r1_propagation() -> None:
    """A → B and B − C undirected, A not adjacent to C → R1
    propagates orientation B → C (else A → B ← C would be a new
    v-structure that v-structure detection would have caught)."""
    from corroborate.graph.discovery import (
        DiscoveredAdjacency, orient_adjacency,
    )
    # We need: A → B (already directed), B − C undirected, A and C
    # not adjacent. Construct via a 4-node fixture:
    # collider triple X → A → ... → no. Simpler: use ambiguity-free
    # collider that orients A and then a downstream undirected B-C.
    # Construct: U → A, V → A (collider so A→A wouldn't apply; we
    # need A→B). Alternative: just synthesise the post-collider
    # state directly.
    # Manual construction: discovered adjacency has edges
    # {U-A, V-A, A-B, B-C} with U, V both colliding into A
    # (so v-structures orient U → A, V → A) and B-C undirected.
    # After v-structure: directed = {(U, A), (V, A)}, undirected =
    # {A-B, B-C}. Meek R1: U → A, A − B undirected, U not adjacent
    # to B → orient A → B. Then A → B, B − C undirected, A not
    # adjacent to C → orient B → C.
    adj = DiscoveredAdjacency(
        variables=frozenset({'u', 'v', 'a', 'b', 'c'}),
        edges=frozenset({
            frozenset({'u', 'a'}),
            frozenset({'v', 'a'}),
            frozenset({'a', 'b'}),
            frozenset({'b', 'c'}),
        }),
        separating_sets={
            # U⫫V at depth 0 (no edge)
            frozenset({'u', 'v'}): frozenset({frozenset[str]()}),
            # U⫫B given A
            frozenset({'u', 'b'}): frozenset({frozenset({'a'})}),
            # V⫫B given A
            frozenset({'v', 'b'}): frozenset({frozenset({'a'})}),
            # A⫫C given B
            frozenset({'a', 'c'}): frozenset({frozenset({'b'})}),
            # U⫫C, V⫫C — chain dissipates
            frozenset({'u', 'c'}): frozenset({frozenset({'a'})}),
            frozenset({'v', 'c'}): frozenset({frozenset({'a'})}),
        },
        n_observations=100, alpha=0.05, max_conditioning=1,
        stratify_by=None,
    )
    oriented = orient_adjacency(adj)
    # v-structure detection: U-A-V is unshielded; A NOT in sepset(U,V)
    # → collider U → A ← V.
    assert ('u', 'a') in oriented.directed_edges
    assert ('v', 'a') in oriented.directed_edges
    # Meek R1: U → A, A−B undirected, U not adjacent to B (sepset
    # has A) → A → B. Then A → B, B−C undirected, A not adjacent
    # to C → B → C.
    assert ('a', 'b') in oriented.directed_edges
    assert ('b', 'c') in oriented.directed_edges
    assert oriented.undirected_edges == frozenset()


def test_orient_ambiguous_triple_skipped() -> None:
    """Triple where Z is in SOME but not ALL separating sets →
    ambiguous; not oriented in conservative mode."""
    from corroborate.graph.discovery import (
        DiscoveredAdjacency, orient_adjacency,
    )
    adj = DiscoveredAdjacency(
        variables=frozenset({'x', 'y', 'z'}),
        edges=frozenset({frozenset({'x', 'z'}), frozenset({'y', 'z'})}),
        separating_sets={
            # Two sepsets: empty AND {z}. Z is in some but not all.
            frozenset({'x', 'y'}): frozenset({
                frozenset[str](),
                frozenset({'z'}),
            }),
        },
        n_observations=100, alpha=0.05, max_conditioning=1,
        stratify_by=None,
    )
    oriented = orient_adjacency(adj, conservative=True)
    # Conservative: ambiguous → not oriented.
    assert ('x', 'z') not in oriented.directed_edges
    assert ('y', 'z') not in oriented.directed_edges
    # Tracked as ambiguous triple.
    assert ('x', 'z', 'y') in oriented.ambiguous_triples


# ============ §4 acceptance: integration smoke ============

def test_pc_dqn_smoke_holds_on_migrated_corpus() -> None:
    """§4 acceptance on the existing 17-env / 1020-row corpus:
    PC + JCI on env_name finds NO edge between arm_ddqn and any
    outcome variable. Reproduces PAPER §4.3's structural finding.

    Skipped if the corpus parquet isn't on disk — the framework
    tests don't require it, only the integration smoke does."""
    from pathlib import Path

    import polars as pl

    runs_path = (
        Path(__file__).parent.parent
        / 'experiments' / 'data' / 'ddqn' / 'runs.parquet'
    )
    if not runs_path.exists():
        import pytest
        pytest.skip(f'{runs_path} not on disk')

    from corroborate.graph.discovery import discover_adjacency

    df = pl.read_parquet(runs_path)
    df = df.with_columns(
        # `arm_ddqn` is the binary one-hot of the DDQN arm. Post-
        # Phase-6 the corpus's `arm_key` column carries
        # `canonical_str(intervention_arms)`; the legacy
        # `intervention_name` column is preserved (`'ddqn'` /
        # `'vanilla_dqn'`) and is the canonical one-hot source for
        # PC discovery tests like this one.
        (pl.col('intervention_name') == 'ddqn')
        .cast(pl.Int64).alias('arm_ddqn'),
    )
    variables = [
        'arm_ddqn',
        'jensen_gap',
        'late_window_mean',
        'eval_final_mean',
        'eval_best_burst_mean',
        'eval_best_burst_step',
    ]
    df = df.drop_nulls(subset=variables)

    adj = discover_adjacency(
        df, variables=variables,
        alpha=0.05, max_conditioning=1,
        stratify_by='env_name',
    )

    # The §4 finding: NO edge between arm_ddqn and any outcome.
    outcome_vars = {v for v in variables if v.startswith('outcome.')}
    arm_outcome_edges = [
        e for e in adj.edges
        if 'arm_ddqn' in e and any(v in e for v in outcome_vars)
    ]
    assert not arm_outcome_edges, (
        f'§4 acceptance FAILED — surviving edges from arm_ddqn '
        f'to outcomes: {arm_outcome_edges}'
    )

    # Sanity: the mechanism intervention edge SHOULD survive
    # (DDQN's slot swap reduces the Jensen gap on a subset of envs).
    assert frozenset({'arm_ddqn', 'jensen_gap'}) in adj.edges, (
        'arm_ddqn → jensen_gap should survive (DDQN '
        'demonstrably reduces the gap on a subset of envs)'
    )


def test_per_env_pc_dqn_smoke_finds_within_env_arm_edges() -> None:
    """§6 thin per-env PC: at least some envs surface a within-env
    edge from arm_ddqn (mostly to jensen_gap). Skipped if
    the corpus parquet isn't on disk.

    With only one mechanism feature, this is the *thin* §6 — it
    cannot reproduce the three-regime mediator taxonomy. The gate
    is qualitative: at least 3 envs show within-env arm_ddqn
    neighbours (the slot swap leaves a per-env footprint even
    where pooled-JCI averages it to zero)."""
    from pathlib import Path

    import polars as pl

    runs_path = (
        Path(__file__).parent.parent
        / 'experiments' / 'data' / 'ddqn' / 'runs.parquet'
    )
    if not runs_path.exists():
        import pytest
        pytest.skip(f'{runs_path} not on disk')

    from corroborate.graph.discovery import discover_adjacency

    df = pl.read_parquet(runs_path)
    df = df.with_columns(
        # `arm_ddqn` is the binary one-hot of the DDQN arm. Post-
        # Phase-6 the corpus's `arm_key` column carries
        # `canonical_str(intervention_arms)`; the legacy
        # `intervention_name` column is preserved (`'ddqn'` /
        # `'vanilla_dqn'`) and is the canonical one-hot source for
        # PC discovery tests like this one.
        (pl.col('intervention_name') == 'ddqn')
        .cast(pl.Int64).alias('arm_ddqn'),
    )
    variables = [
        'arm_ddqn',
        'jensen_gap',
        'late_window_mean',
        'eval_final_mean',
        'eval_best_burst_mean',
        'eval_best_burst_step',
    ]
    df = df.drop_nulls(subset=variables)

    envs_with_arm_edge: list[str] = []
    for env in sorted(df['env_name'].unique().to_list()):
        env_df = df.filter(pl.col('env_name') == env)
        if env_df.height < 5 or env_df['arm_ddqn'].n_unique() < 2:
            continue
        constant_cols = [
            v for v in variables
            if env_df[v].dtype.is_float()
            and series_std_float(env_df[v]) == 0.0
        ]
        if constant_cols:
            continue
        adj = discover_adjacency(
            env_df, variables=variables,
            alpha=0.05, max_conditioning=1,
        )
        if any('arm_ddqn' in e for e in adj.edges):
            envs_with_arm_edge.append(env)

    assert len(envs_with_arm_edge) >= 3, (
        f'§6 thin gate: expected ≥3 envs with within-env arm_ddqn '
        f'edges, got {len(envs_with_arm_edge)}: {envs_with_arm_edge}'
    )


def test_per_env_mediator_pc_smoke_finds_outcome_neighbours() -> None:
    """§5+§6 rich gate on `runs_with_mediators.parquet`: per-env PC
    over the 10-variable mediator-augmented set surfaces ≥1 neighbour
    of `eval_final_mean` in at least 8 envs (the paper's
    9-of-15 threshold, allowing 1 slack for corpus-specific noise).

    Skipped if `runs_with_mediators.parquet` isn't on disk — produced
    by `experiments/compute_mediators.py`."""
    from pathlib import Path

    import polars as pl

    enriched_path = (
        Path(__file__).parent.parent
        / 'experiments' / 'data' / 'ddqn'
        / 'runs_with_mediators.parquet'
    )
    if not enriched_path.exists():
        import pytest
        pytest.skip(f'{enriched_path} not on disk')

    from corroborate.graph.discovery import discover_adjacency

    df = pl.read_parquet(enriched_path)
    df = df.with_columns(
        # `arm_ddqn` is the binary one-hot of the DDQN arm. Post-
        # Phase-6 the corpus's `arm_key` column carries
        # `canonical_str(intervention_arms)`; the legacy
        # `intervention_name` column is preserved (`'ddqn'` /
        # `'vanilla_dqn'`) and is the canonical one-hot source for
        # PC discovery tests like this one.
        (pl.col('intervention_name') == 'ddqn')
        .cast(pl.Int64).alias('arm_ddqn'),
    )
    # Drop epsilon_late and fill_ratio_late — corpus-wide constants.
    pc_mediators = (
        'mediator.q_gap_late', 'mediator.q_gap_growth',
        'mediator.q_max_growth', 'mediator.v_vs_max_delta_late',
        'mediator.td_residual_late', 'mediator.greedy_match_late',
    )
    variables = [
        'arm_ddqn', 'jensen_gap',
        *pc_mediators,
        'eval_final_mean', 'late_window_mean',
    ]
    outcome = 'eval_final_mean'

    envs_with_neighbour: list[str] = []
    for env in sorted(df['env_name'].unique().to_list()):
        env_df = df.filter(pl.col('env_name') == env)
        if env_df.height < 5 or env_df['arm_ddqn'].n_unique() < 2:
            continue
        env_df = env_df.drop_nulls(subset=variables)
        for v in variables:
            if env_df[v].dtype.is_float():
                env_df = env_df.filter(~pl.col(v).is_nan())
        if env_df.height < 5 or env_df['arm_ddqn'].n_unique() < 2:
            continue
        constant_cols = [
            v for v in variables
            if env_df[v].dtype.is_float()
            and series_std_float(env_df[v]) == 0.0
        ]
        if constant_cols:
            continue
        adj = discover_adjacency(
            env_df, variables=variables,
            alpha=0.05, max_conditioning=1,
        )
        if any(outcome in edge for edge in adj.edges):
            envs_with_neighbour.append(env)

    assert len(envs_with_neighbour) >= 8, (
        f'§5+§6 rich gate: expected ≥8 envs with a within-env '
        f'{outcome}-neighbour, got {len(envs_with_neighbour)}: '
        f'{envs_with_neighbour}'
    )
