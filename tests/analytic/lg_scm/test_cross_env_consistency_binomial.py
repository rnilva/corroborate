"""Closed-form assertions on `cross_env_consistency_binomial` over
the LG-SCM substrate.

The primitive counts per-stratum directional agreement against a
predicted direction and tests it via a binomial sign-test. Under
the LG-SCM, control β_xz_t > β_xz_b at every env → Δ_y = β_zy ·
(β_xz_t − β_xz_b) · μ_x > 0 at every env (no sign-flip across
envs by construction).

Tests:

1. **All-positive direction → SUPPORTED at p ≤ 1e-3**: 10 envs,
   all Δβ > 0 → all 10 strata have d_y > 0 → 10/10 against
   one-tailed binomial (p = 0.5^10 = 0.000977).

2. **Mixed-direction → power-insufficient at small alignment**:
   5/10 positive vs 5/10 negative → p = 1.0 (one-tail, predicting
   positive); two-tailed p = 1.0. Honestly null.

3. **Null-floor drops near-zero d**: with null_floor=0.5, strata
   whose |d| < 0.5 drop out; only |d| ≥ 0.5 strata count toward
   the binomial.

4. **min_strata floor**: when n_strata after the panel filter
   falls below the binomial-power threshold (e.g. n=2), p=NaN.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.analyses.panel.cross_env_consistency_binomial import (
    CrossEnvConsistencyBinomialResult,
    cross_env_consistency_binomial,
    cross_env_consistency_binomial_verdict,
)

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_multi_env_paired_arms
from corroborate.data import cells_to_dataframe


_MU_X = 1.0
_SIGMA_X = 0.5
_SIGMA_Z = 0.1
_BETA_ZY = 1.5
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_SEEDS_PER_ARM = 30


def _scm(beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=_MU_X, sigma_x=_SIGMA_X,
        beta_xz=beta_xz, sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _build_cells(
    env_betas: Mapping[str, tuple[float, float]],
) -> list[Mapping[str, object]]:
    envs = {n: (_scm(t), _scm(b)) for n, (t, b) in env_betas.items()}
    rows = run_multi_env_paired_arms(
        envs=envs, seeds=tuple(range(_N_SEEDS_PER_ARM)),
    )
    return [r.as_dict() for r in rows]


# Case 1: 10 envs all with treatment β > baseline β → all d_y > 0
_ALL_POSITIVE_ENVS: Mapping[str, tuple[float, float]] = {
    f'env_{i:02d}': (0.7 + 0.02 * i, 0.3 + 0.01 * i)
    for i in range(10)
}


def test_consistency_binomial_all_positive_supported() -> None:
    """All 10 envs constructed with treatment β > baseline β
    → Δ_y > 0 at every env → 10/10 binomial against p=0.5 →
    one-tailed p = 0.5**10 = 0.000977. SUPPORTED."""
    cells = _build_cells(_ALL_POSITIVE_ENVS)
    result = cross_env_consistency_binomial.fn(
        cells_to_dataframe(cells),
        source='y_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
        stratify_by=('env_name',),
        predicted_direction='a_gt_b',
        scope_predictor='y_mean',  # use the source itself as filter (always > -1e9)
        min_baseline_predictor=-1e9,
    )
    # All 10 envs contributed
    assert result.n_strata_total == 10, (
        f'expected 10 strata, got {result.n_strata_total}'
    )
    # All 10 d>0
    assert result.n_signed_predicted == 10
    # Closed-form binomial p = 0.5^10
    expected_p = 0.5 ** 10
    assert abs(result.p_value - expected_p) < 1e-9, (
        f'p={result.p_value}, expected {expected_p}'
    )


# Case 2: 10 envs, 5 with t > b (Δ>0), 5 with t < b (Δ<0)
_MIXED_DIRECTION_ENVS: Mapping[str, tuple[float, float]] = {
    **{f'pos_{i:02d}': (0.7, 0.3) for i in range(5)},  # Δβ = +0.4
    **{f'neg_{i:02d}': (0.3, 0.7) for i in range(5)},  # Δβ = -0.4
}


def test_consistency_binomial_mixed_direction_null() -> None:
    """5 envs predict t>b (Δ_y>0) and 5 predict t<b (Δ_y<0).
    Predicting 'a_gt_b' direction: 5/10 envs match → one-tailed
    binomial p = 0.6230 (lower-bounded by tied count). NOT
    distinguishable from chance."""
    cells = _build_cells(_MIXED_DIRECTION_ENVS)
    result = cross_env_consistency_binomial.fn(
        cells_to_dataframe(cells),
        source='y_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
        stratify_by=('env_name',),
        predicted_direction='a_gt_b',
        scope_predictor='y_mean',
        min_baseline_predictor=-1e9,
    )
    assert result.n_strata_total == 10
    # 5 positives out of 10 — chance
    assert result.n_signed_predicted == 5
    # P(X >= 5 | n=10, p=0.5) = 0.6230
    assert abs(result.p_value - 0.62304687) < 1e-4


def test_consistency_binomial_null_floor_drops_small_d() -> None:
    """With a truly large null_floor (1000.0 Cohen d units), every
    stratum's d falls below the floor → n_above_floor = 0 → p=NaN.

    Tests the gating logic: |d|<null_floor strata are dropped
    from the binomial count. Picked to be unambiguously larger
    than any plausible LG-SCM Cohen's d."""
    cells = _build_cells(_ALL_POSITIVE_ENVS)
    result = cross_env_consistency_binomial.fn(
        cells_to_dataframe(cells),
        source='y_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
        stratify_by=('env_name',),
        predicted_direction='a_gt_b',
        null_floor=1000.0,  # Unambiguously larger than any LG-SCM Cohen d
        scope_predictor='y_mean',
        min_baseline_predictor=-1e9,
    )
    # All 10 strata above the seeds floor
    assert result.n_strata_total == 10
    # |d| << 1000 at every env (LG-SCM d ≈ 1-10 at construction)
    assert result.n_strata_above_floor == 0, (
        f'expected 0 above null_floor=1000.0, got '
        f'{result.n_strata_above_floor}; sample d values: '
        f'{result.cohen_d_per_stratum[:5]}'
    )
    assert math.isnan(result.p_value)


def test_consistency_binomial_either_two_tailed() -> None:
    """`predicted_direction='either'` runs two-tailed binomial.
    For 10 envs all d>0, two-tailed p against majority count = 10
    is 2 · 0.5^10 = 0.00195 — still SUPPORTED but weaker by ×2
    than one-tailed."""
    cells = _build_cells(_ALL_POSITIVE_ENVS)
    result = cross_env_consistency_binomial.fn(
        cells_to_dataframe(cells),
        source='y_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
        stratify_by=('env_name',),
        predicted_direction='either',
        scope_predictor='y_mean',
        min_baseline_predictor=-1e9,
    )
    assert result.n_strata_total == 10
    # Majority direction (positive) wins 10/10
    assert result.n_signed_predicted == 10
    # Two-tailed p for k=10 out of n=10
    expected_p = 2 * (0.5 ** 10)
    assert abs(result.p_value - expected_p) < 1e-9


# ============ cross_env_consistency_binomial_verdict band map ============


def _result(
    *, p_value: float, n_above: int, n_signed: int,
) -> CrossEnvConsistencyBinomialResult:
    """Minimal result carrying only the fields the verdict map reads."""
    return CrossEnvConsistencyBinomialResult(
        n_strata_total=n_above,
        n_strata_above_floor=n_above,
        n_signed_predicted=n_signed,
        p_value=p_value,
        measurable='x',
        predicted_direction='a_gt_b',
        null_floor=0.0,
        cohen_d_per_stratum=(),
        stratum_ids=(),
    )


def test_verdict_held_when_significant() -> None:
    """p ≤ 0.05 above the strata floor → HELD (10/10, p≈1e-3)."""
    v, refut = cross_env_consistency_binomial_verdict(
        _result(p_value=0.5 ** 10, n_above=10, n_signed=10),
    )
    assert v is Verdict.HELD and refut is None


def test_verdict_power_insufficient_band() -> None:
    """0.05 < p ≤ 0.15 → POWER_INSUFFICIENT (not yet HELD, not null)."""
    v, refut = cross_env_consistency_binomial_verdict(
        _result(p_value=0.11, n_above=8, n_signed=7),
    )
    assert v is Verdict.POWER_INSUFFICIENT and refut is None


def test_verdict_min_strata_floor_dominates() -> None:
    """Below min_strata the floor returns POWER_INSUFFICIENT regardless
    of alignment — the n=4 regime of ddqn2010_xenv: a perfect 4/4 is
    p=0.0625 (would-be near-HELD) yet 4 < 5 → POWER_INSUFFICIENT."""
    v, refut = cross_env_consistency_binomial_verdict(
        _result(p_value=0.0625, n_above=4, n_signed=4), min_strata=5,
    )
    assert v is Verdict.POWER_INSUFFICIENT and refut is None
    # Lower the floor to 4 and the same data clears the p≤0.15 band.
    v2, _ = cross_env_consistency_binomial_verdict(
        _result(p_value=0.0625, n_above=4, n_signed=4), min_strata=4,
    )
    assert v2 is Verdict.POWER_INSUFFICIENT  # 0.0625 in (0.05, 0.15]


def test_verdict_sign_flip_when_mostly_wrong_direction() -> None:
    """≥70% of strata in the WRONG direction → NO_EFFECT / SIGN_FLIP."""
    v, refut = cross_env_consistency_binomial_verdict(
        _result(p_value=0.99, n_above=10, n_signed=1),
    )
    assert v is Verdict.NO_EFFECT and refut is RefutationClass.SIGN_FLIP


def test_verdict_null_effect_when_no_alignment() -> None:
    """≤60% predicted-direction (and not a sign-flip) → NO_EFFECT/NULL."""
    v, refut = cross_env_consistency_binomial_verdict(
        _result(p_value=0.62, n_above=10, n_signed=5),
    )
    assert v is Verdict.NO_EFFECT and refut is RefutationClass.NULL_EFFECT
