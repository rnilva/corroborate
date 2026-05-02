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

from collections.abc import Mapping

import pytest

from corroborate.meta_regression import (
    StratumObservation,
    cross_validate_meta_regression,
    meta_regress_comparison,
    meta_regression,
)
from corroborate.schema import GroupStats, HypothesisComparisonRow
from corroborate.verdict import Verdict


# ============ Recovery on synthetic-known-effect data ============

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


# ============ meta_regress_comparison ============

def _gs(group_value: object, g: float, se: float) -> GroupStats:
    """Build a minimal GroupStats with only the fields B3 reads."""
    return GroupStats(
        group_value=group_value,
        n_pairs=10,
        arm_a_mean=0.0, arm_a_sd=None,
        arm_b_mean=0.0, arm_b_sd=None,
        effect_size_g=g,
        se=se,
        derived_q=None,
        delta_i=0.0,
        verdict=Verdict.HELD,
        refutation_class=None,
        adequately_powered=True,
    )


def _hcr(per_group: tuple[GroupStats, ...]) -> HypothesisComparisonRow:
    """Minimal HypothesisComparisonRow carrying just `per_group` —
    enough for `meta_regress_comparison` to do its job."""
    return HypothesisComparisonRow(
        id='test-row', parent_id=None, cycle_id=None,
        timestamp='2026-04-29T00:00:00Z',
        intervention_name='test',
        treatment_arm_key='treatment',
        baseline_arm_key='baseline',
        treatment_run_ids=(), baseline_run_ids=(),
        predicted_direction=None,
        pair_by=('seed',),
        group_by='env_name',
        arm_a_n=0, arm_a_mean=None, arm_a_sd=None,
        arm_b_n=0, arm_b_mean=None, arm_b_sd=None,
        effect_size_g=None, se=None, derived_q=None,
        delta_i_population=0.0,
        adequately_powered=False,
        verdict=Verdict.HELD,
        refutation_class=None,
        per_group=per_group,
        pooled=None,
        n_dropped_unpaired=0,
    )


def test_meta_regress_comparison_recovers_known_cleaver() -> None:
    """Stratified comparison row with per-stratum effects driven
    by one covariate → meta_regress_comparison flags it
    significant."""
    per_group = tuple(
        _gs(group_value=f'env{i}', g=2.0 + 0.5 * i, se=0.1)
        for i in range(10)
    )
    row = _hcr(per_group)

    def covariate_for(gv: object) -> Mapping[str, float]:
        # group_value is the env name like 'env3'; encode as int.
        assert isinstance(gv, str)
        return {'env_index': float(gv.removeprefix('env'))}

    result = meta_regress_comparison(row, covariate_for, alpha=0.05)
    assert result.n_strata == 10
    assert result.cleavage_axes == ('env_index',)


def test_meta_regress_comparison_skips_none_strata() -> None:
    """Strata with `effect_size_g=None` or `se=None` are silently
    dropped — those are degenerate per-strata flagged earlier in
    aggregation."""
    per_group = (
        _gs(group_value='env0', g=0.0, se=0.1),
        GroupStats(  # null stratum (no pairs)
            group_value='env_null', n_pairs=0,
            arm_a_mean=None, arm_a_sd=None,
            arm_b_mean=None, arm_b_sd=None,
            effect_size_g=None, se=None,
            derived_q=None, delta_i=0.0,
            verdict=Verdict.POWER_INSUFFICIENT,
            refutation_class=None,
            adequately_powered=False,
        ),
        _gs(group_value='env1', g=0.5, se=0.1),
        _gs(group_value='env2', g=1.0, se=0.1),
        _gs(group_value='env3', g=1.5, se=0.1),
    )
    row = _hcr(per_group)

    def covariate_for(gv: object) -> Mapping[str, float]:
        assert isinstance(gv, str)
        return {'idx': float(gv.removeprefix('env'))}

    result = meta_regress_comparison(row, covariate_for, alpha=0.05)
    assert result.n_strata == 4  # null stratum dropped


def test_meta_regress_comparison_empty_per_group_raises() -> None:
    row = _hcr(per_group=())

    def covariate_for(gv: object) -> Mapping[str, float]:
        del gv
        return {}

    with pytest.raises(ValueError, match='per_group is empty'):
        meta_regress_comparison(row, covariate_for, alpha=0.05)


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
