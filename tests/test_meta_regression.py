"""Tests for `meta_regression` — per-stratum effect-size
regression on covariates with inverse-variance weighting.

Validates:
1. One true cleaver: g = β·x + ε with known β recovers β within
   CI; covariate flagged significant.
2. No effect: random g with no relation to covariates → no
   covariate flagged.
3. Two cleavers: g = β1·x1 + β2·x2 → both flagged with sensible
   coefficient estimates.
4. Edge cases: empty observations, n ≤ p, invalid SE, collinear
   covariates raise ValueError loudly.
5. Cleavage axes accessor returns names of significant
   coefficients in declaration order.
"""
from __future__ import annotations

import pytest

from corroborate.stats import (
    StratumObservation,
    meta_regression,
)
from corroborate.stats.meta_regression import (
    cross_validate_meta_regression,
)


# ============ Recovery on synthetic-known-effect data ============

def test_intercept_only_pools_inverse_variance_weighted_mean() -> None:
    """Intercept-only fit (no covariates) is the textbook
    inverse-variance-weighted pooled mean. With pool='fixed',
    intercept = Σwᵢgᵢ / Σwᵢ where wᵢ = 1/seᵢ²."""
    observations = [
        StratumObservation(stratum_id='a', g=1.0, se=0.1, covariates={}),
        StratumObservation(stratum_id='b', g=2.0, se=0.2, covariates={}),
        StratumObservation(stratum_id='c', g=3.0, se=0.3, covariates={}),
    ]
    result = meta_regression(observations, alpha=0.05, pool='fixed')
    # w = (100, 25, 11.111); sum=136.111; weighted_sum=100*1+25*2+11.111*3
    #   = 100+50+33.333 = 183.333; mean = 1.347
    expected = (100.0*1.0 + 25.0*2.0 + (1/0.09)*3.0) / (
        100.0 + 25.0 + 1/0.09
    )
    assert result.intercept == pytest.approx(expected, rel=1e-3)
    assert result.intercept_se > 0.0
    assert result.intercept_ci_lo < result.intercept < result.intercept_ci_hi


def test_homogeneous_data_yields_tau_sq_zero() -> None:
    """When effects are essentially identical across strata
    (only sampling noise), DL τ² estimator returns 0 and RE
    weights collapse to FE weights — RE intercept equals FE
    intercept."""
    observations = [
        StratumObservation(
            stratum_id=i, g=1.0 + (0.005 if i % 2 else -0.005),
            se=0.1, covariates={},
        )
        for i in range(10)
    ]
    fe = meta_regression(observations, alpha=0.05, pool='fixed')
    re = meta_regression(observations, alpha=0.05, pool='random')
    assert re.tau_sq == pytest.approx(0.0, abs=1e-9)
    assert re.intercept == pytest.approx(fe.intercept, rel=1e-9)
    assert re.intercept_se == pytest.approx(fe.intercept_se, rel=1e-9)
    assert re.i_squared == pytest.approx(0.0, abs=0.01)


def test_heterogeneous_data_yields_positive_tau_sq() -> None:
    """When per-stratum effects vary far more than each
    stratum's within-variance can explain, DL τ² > 0 and the
    Q-statistic exceeds df. The intercept-only fit's pooled g
    converges on the across-stratum mean (≈2.25 for these
    effects)."""
    # 10 strata with effects spread {0.0, 0.5, 1.0, …, 4.5} and
    # tight within-stratum SE. Between-stratum SD ≈ 1.4; within
    # SE 0.05. So τ² should land near (1.4)² ≈ 2.0 if estimator
    # is unbiased under the model — DL recovers within ~30%.
    observations = [
        StratumObservation(
            stratum_id=i, g=0.5 * i, se=0.05, covariates={},
        )
        for i in range(10)
    ]
    re = meta_regression(observations, alpha=0.05, pool='random')
    assert re.tau_sq > 0.5  # clearly positive
    assert re.q_statistic > re.n_strata  # heterogeneity beyond df
    assert re.i_squared > 0.9  # nearly all variance is between-stratum
    # Intercept estimates the across-stratum mean — RE with τ²>0
    # weights observations by 1/(v+τ²), which here are ~uniform
    # since τ² ≫ v. So the pooled g is approximately the
    # arithmetic mean of the per-stratum effects.
    assert re.intercept == pytest.approx(2.25, abs=0.1)


def test_pool_fixed_vs_random_with_nonuniform_weights() -> None:
    """When within-stratum SEs vary across strata (so FE and RE
    weights aren't proportional), random-effects pooling shifts
    the pooled mean toward the noisier strata (their RE weight
    is closer to FE because τ² adds the same constant)."""
    # 4 precise strata at g=0; 4 noisy strata at g=4. Without τ²
    # (FE), the precise strata dominate → pooled near 0. With
    # τ²≫v_FE for the precise strata, the RE weights flatten and
    # the pooled mean rises toward the across-stratum mean (2.0).
    precise = [
        StratumObservation(
            stratum_id=f'p{i}', g=0.0, se=0.05, covariates={},
        )
        for i in range(4)
    ]
    noisy = [
        StratumObservation(
            stratum_id=f'n{i}', g=4.0, se=1.0, covariates={},
        )
        for i in range(4)
    ]
    fe = meta_regression([*precise, *noisy], alpha=0.05, pool='fixed')
    re = meta_regression([*precise, *noisy], alpha=0.05, pool='random')
    assert re.tau_sq > 0.0
    # FE: precise strata dominate → pooled near 0
    assert abs(fe.intercept) < 0.5
    # RE: weights flatten → pooled shifts toward the across-stratum mean
    assert re.intercept > fe.intercept


def test_recovers_single_cleaver() -> None:
    """g_i = 2.0 + 0.5 * x_i with tight noise → coefficient on
    `x` is significant, ~0.5, intercept ~2.0."""
    observations = [
        StratumObservation(
            stratum_id=i,
            g=2.0 + 0.5 * i + (0.01 if i % 2 else -0.01),
            se=0.1,
            covariates={'x': float(i)},
        )
        for i in range(10)
    ]
    result = meta_regression(observations, alpha=0.05)
    assert result.n_strata == 10
    assert result.intercept == pytest.approx(2.0, abs=0.05)
    assert len(result.coefficients) == 1
    coef = result.coefficients[0]
    assert coef.name == 'x'
    assert coef.coefficient == pytest.approx(0.5, abs=0.05)
    assert coef.is_significant
    assert coef.ci_lo > 0.0  # CI excludes zero
    assert 'x' in result.cleavage_axes


def test_no_effect_no_significant_covariates() -> None:
    """g_i ≈ constant with covariate uncorrelated → coefficient
    not significant; cleavage_axes empty."""
    # Constant g; x cycles through values uncorrelated with g.
    observations = [
        StratumObservation(
            stratum_id=i,
            g=1.0 + (0.005 if i % 2 else -0.005),
            se=0.1,
            covariates={'x': float(i % 3)},
        )
        for i in range(12)
    ]
    result = meta_regression(observations, alpha=0.05)
    coef = result.coefficients[0]
    assert not coef.is_significant
    assert coef.ci_lo < 0.0 < coef.ci_hi  # CI brackets zero
    assert result.cleavage_axes == ()


def test_two_cleavers_both_flagged() -> None:
    """g_i = 0.5 * x1_i + 0.7 * x2_i with tight noise → both
    coefficients significant."""
    observations = [
        StratumObservation(
            stratum_id=(i, j),
            g=0.5 * i + 0.7 * j + (0.01 if (i + j) % 2 else -0.01),
            se=0.1,
            covariates={'x1': float(i), 'x2': float(j)},
        )
        for i in range(5) for j in range(5)
    ]
    result = meta_regression(observations, alpha=0.05)
    assert result.n_strata == 25
    by_name = {c.name: c for c in result.coefficients}
    assert by_name['x1'].coefficient == pytest.approx(0.5, abs=0.05)
    assert by_name['x2'].coefficient == pytest.approx(0.7, abs=0.05)
    assert by_name['x1'].is_significant
    assert by_name['x2'].is_significant
    assert set(result.cleavage_axes) == {'x1', 'x2'}


def test_inverse_variance_weighting_downweights_noisy_strata() -> None:
    """A noisy stratum (large SE) should contribute less than a
    precise one; the regression should still recover the precise
    signal."""
    # 5 precise strata showing slope=0.5, plus 1 noisy outlier.
    precise = [
        StratumObservation(
            stratum_id=f'p{i}',
            g=0.5 * i,
            se=0.05,
            covariates={'x': float(i)},
        )
        for i in range(5)
    ]
    noisy_outlier = StratumObservation(
        stratum_id='noisy',
        g=10.0,  # huge outlier
        se=10.0,  # but huge SE → near-zero weight
        covariates={'x': 2.5},
    )
    result = meta_regression([*precise, noisy_outlier], alpha=0.05)
    coef = result.coefficients[0]
    assert coef.coefficient == pytest.approx(0.5, abs=0.05)


# ============ Edge cases ============

def test_empty_observations_raises() -> None:
    with pytest.raises(ValueError, match='empty'):
        meta_regression([], alpha=0.05)


def test_n_too_small_for_covariate_count_raises() -> None:
    """n ≤ 1 + n_covariates can't fit OLS."""
    observations = [
        StratumObservation(
            stratum_id=i, g=1.0, se=0.1,
            covariates={'a': 1.0, 'b': 2.0, 'c': 3.0},
        )
        for i in range(3)
    ]
    with pytest.raises(ValueError, match='not enough strata'):
        meta_regression(observations, alpha=0.05)


def test_invalid_se_raises() -> None:
    observations = [
        StratumObservation(
            stratum_id=0, g=1.0, se=0.1, covariates={'x': 1.0},
        ),
        StratumObservation(
            stratum_id=1, g=2.0, se=0.0, covariates={'x': 2.0},  # invalid
        ),
        StratumObservation(
            stratum_id=2, g=3.0, se=0.1, covariates={'x': 3.0},
        ),
    ]
    with pytest.raises(ValueError, match='invalid se'):
        meta_regression(observations, alpha=0.05)


def test_collinear_covariates_raises() -> None:
    """x2 = 2 * x1 → singular design matrix; pyright cannot solve."""
    observations = [
        StratumObservation(
            stratum_id=i,
            g=float(i),
            se=0.1,
            covariates={'x1': float(i), 'x2': 2.0 * i},
        )
        for i in range(10)
    ]
    with pytest.raises(ValueError, match='singular'):
        meta_regression(observations, alpha=0.05)


def test_missing_covariate_defaults_to_zero() -> None:
    """When some observations don't have a covariate key, missing
    values default to 0.0 (one-hot-friendly encoding)."""
    observations = [
        StratumObservation(
            stratum_id=0, g=0.0, se=0.1, covariates={},
        ),
        StratumObservation(
            stratum_id=1, g=1.0, se=0.1, covariates={'x': 1.0},
        ),
        StratumObservation(
            stratum_id=2, g=2.0, se=0.1, covariates={'x': 2.0},
        ),
        StratumObservation(
            stratum_id=3, g=3.0, se=0.1, covariates={'x': 3.0},
        ),
    ]
    result = meta_regression(observations, alpha=0.05)
    # Treats stratum 0 as x=0 → fits a line through (0,0), (1,1),
    # (2,2), (3,3) → slope ≈ 1.
    assert result.coefficients[0].coefficient == pytest.approx(1.0, abs=0.05)


# ============ cross_validate_meta_regression ============

def test_cv_recovers_consistent_sign_on_known_cleaver() -> None:
    """g_i = 0.5 * x + ε with tight noise across enough strata
    that every fold's training set has clear signal → sign is
    consistent across all folds."""
    observations = [
        StratumObservation(
            stratum_id=i,
            g=0.5 * i + (0.01 if i % 2 else -0.01),
            se=0.1,
            covariates={'x': float(i)},
        )
        for i in range(20)
    ]
    cv = cross_validate_meta_regression(
        observations, k_folds=5, alpha=0.05, seed=0,
    )
    assert cv.n_folds == 5
    assert len(cv.per_fold) == 5
    # All folds should agree on sign of `x` (positive).
    assert cv.sign_consistency['x'] == 1.0
    mean, std = cv.coefficient_stability['x']
    assert mean == pytest.approx(0.5, abs=0.05)
    assert std < 0.05  # tight stability


def test_cv_unstable_signs_on_no_effect_data() -> None:
    """Truly random g (no relation to covariate) → coefficient
    sign flips across folds; sign_consistency typically below 1.0
    (we test < 1.0; the random seed makes the exact value
    deterministic but not 1.0)."""
    import random as rnd
    rng = rnd.Random(42)
    observations = [
        StratumObservation(
            stratum_id=i,
            g=rng.gauss(0, 0.5),
            se=0.5,
            covariates={'noise': rng.gauss(0, 1)},
        )
        for i in range(30)
    ]
    cv = cross_validate_meta_regression(
        observations, k_folds=5, alpha=0.05, seed=1,
    )
    # The coefficient on a noise covariate should flip sign in
    # at least one fold; sign_consistency strictly below 1.
    assert cv.sign_consistency['noise'] < 1.0


def test_cv_rejects_too_few_folds() -> None:
    observations = [
        StratumObservation(
            stratum_id=i, g=0.0, se=0.1, covariates={'x': float(i)},
        )
        for i in range(10)
    ]
    with pytest.raises(ValueError, match='k_folds must be ≥ 2'):
        cross_validate_meta_regression(
            observations, k_folds=1, alpha=0.05,
        )


def test_cv_rejects_too_many_folds() -> None:
    observations = [
        StratumObservation(
            stratum_id=i, g=0.0, se=0.1, covariates={'x': float(i)},
        )
        for i in range(5)
    ]
    with pytest.raises(ValueError, match='cannot exceed'):
        cross_validate_meta_regression(
            observations, k_folds=10, alpha=0.05,
        )


def test_cv_seed_reproducibility() -> None:
    """Same seed → same fold assignments → identical
    sign_consistency and coefficient_stability."""
    observations = [
        StratumObservation(
            stratum_id=i,
            g=0.5 * i + (0.05 if i % 3 == 0 else -0.05),
            se=0.1,
            covariates={'x': float(i)},
        )
        for i in range(15)
    ]
    cv1 = cross_validate_meta_regression(
        observations, k_folds=3, alpha=0.05, seed=7,
    )
    cv2 = cross_validate_meta_regression(
        observations, k_folds=3, alpha=0.05, seed=7,
    )
    assert dict(cv1.sign_consistency) == dict(cv2.sign_consistency)
    assert dict(cv1.coefficient_stability) == dict(cv2.coefficient_stability)
