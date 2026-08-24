"""Closed-form assertions on `cross_env_probability_of_improvement`
over the LG-SCM implementation + a hand-built saturation case.

Aggregates per-stratum Mann-Whitney `P(X > Y)` via two
complementary inference modes (exact sign-permutation + stratified
bootstrap CI). Under the LG-SCM, the SCM controls per-env
`β_xz_treatment` vs `β_xz_baseline`; positive Δβ → Δ_y > 0 →
per-stratum `P(X > Y) → 1` at large n_seeds.

Tests:

1. **All-positive direction**: 10 envs constructed with Δβ > 0 →
   per-stratum P_xy ≈ 1 at every env. Sign-permutation:
   `2 ** 10 = 1024` enumerations; only the all-positive flip yields
   `perm_dev ≥ observed_dev`. p_permutation ≈ 1/1024 ≈ 0.001.
   Bootstrap CI lower bound near 1.0.

2. **Mixed-direction null**: 5 positive Δβ, 5 negative Δβ →
   5 strata P_xy ≈ 1, 5 strata P_xy ≈ 0. observed_dev = 0;
   permutation gives p ≈ 0.5 by symmetry.

3. **Saturated stratum contributes neutrally**: hand-built
   synthetic cells where one stratum has both arms drawing from
   the same distribution. P_stratum ≈ 0.5; the (P_stratum − 0.5)
   contribution is zero → primitive doesn't inflate p_permutation.

4. **Per-stratum exact-enumeration with hand-checked p**: 3
   strata, hand-built P_xy values; 2^3 = 8 enumerations; expected
   p_permutation matches the exact count of sign-flips at-or-
   above the observed deviation.

5. **MC permutation fires above the cap**: with n_strata = 16
   (2^16 = 65536 > default permutation_cap=16384), primitive
   falls back to MC sampling; `permutation_exact = False`.
"""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.analyses.panel.cross_env_probability_of_improvement import (
    cross_env_probability_of_improvement,
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


_ALL_POSITIVE_ENVS: Mapping[str, tuple[float, float]] = {
    f'env_{i:02d}': (0.7 + 0.02 * i, 0.3 + 0.01 * i)
    for i in range(10)
}


def test_p_xy_all_positive_supported_by_permutation() -> None:
    """All 10 envs constructed with treatment β > baseline β →
    every per-stratum P_xy ≈ 1.0; observed_dev ≈ +5.0
    (= sum of 10 × +0.5). Sign-permutation p:
    `2 ** 10 = 1024` enumerations, only the all-flip-positive
    flip yields perm_dev ≥ +5.0, so p ≈ 1/1024."""
    cells = _build_cells(_ALL_POSITIVE_ENVS)
    result = cross_env_probability_of_improvement.fn(
        cells_to_dataframe(cells),
        source='y_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
        stratify_by=('env_name',),
    )
    assert result.n_strata == 10
    assert result.permutation_exact is True
    assert result.n_permutation_effective == 1024
    assert result.p_xy_mean > 0.95, (
        f'expected mean P_xy near 1.0, got {result.p_xy_mean}'
    )
    expected_p = 1.0 / 1024.0
    assert abs(result.p_permutation - expected_p) < 1e-9
    assert result.ci_bootstrap_lo > 0.90, (
        f'expected tight CI near 1.0, lo={result.ci_bootstrap_lo}'
    )
    assert result.ci_bootstrap_hi <= 1.0 + 1e-9


_MIXED_ENVS: Mapping[str, tuple[float, float]] = {
    **{f'pos_{i:02d}': (0.7, 0.3) for i in range(5)},
    **{f'neg_{i:02d}': (0.3, 0.7) for i in range(5)},
}


def test_p_xy_mixed_direction_null() -> None:
    """5 envs with Δβ > 0 (P_xy ≈ 1) and 5 with Δβ < 0
    (P_xy ≈ 0). Per-env (P_xy − 0.5) deviations: +0.5, +0.5,
    +0.5, +0.5, +0.5, −0.5, −0.5, −0.5, −0.5, −0.5.
    Observed sum = 0. By symmetry, half of the 2^10 sign-flips
    produce perm_dev ≥ 0 → p ≈ 0.5 (plus boundary contribution
    from the perm_dev = 0 cases since we use ≥)."""
    cells = _build_cells(_MIXED_ENVS)
    result = cross_env_probability_of_improvement.fn(
        cells_to_dataframe(cells),
        source='y_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
        stratify_by=('env_name',),
    )
    assert result.n_strata == 10
    assert result.permutation_exact is True
    # Mean P_xy near 0.5 with all-equal magnitudes
    assert abs(result.p_xy_mean - 0.5) < 0.10
    # By symmetry, p > 0.4 (boundary inflates p with the ≥ rule)
    assert result.p_permutation > 0.4
    assert result.p_permutation < 0.8


def test_p_xy_saturated_stratum_contributes_neutrally() -> None:
    """A stratum where both arms share identical values exactly
    (saturation: same distribution sampled per arm) has Mann-
    Whitney P(X > Y) = 0.5 + 0.5 × tie_fraction. With *identical*
    samples (every tie), U = 0.5 × n_t × n_b → P_xy = 0.5
    exactly. observed_dev contribution is zero → primitive
    doesn't inflate aggregate."""
    # Hand-build cells: 3 strata, 5 seeds each arm. Two strata
    # decisively positive (Δ ≈ +5); one stratum saturated (both
    # arms identical samples).
    cells: list[Mapping[str, object]] = []
    for env in ('env_pos1', 'env_pos2'):
        for seed in range(5):
            cells.append({
                'env_name': env, 'arm_key': 'baseline',
                'seed': seed, 'y': 1.0 + 0.1 * seed,
            })
            cells.append({
                'env_name': env, 'arm_key': 'treatment',
                'seed': seed, 'y': 6.0 + 0.1 * seed,
            })
    # Saturated stratum: both arms get identical samples
    saturated_vals = [10.0, 10.5, 11.0, 11.5, 12.0]
    for seed, v in enumerate(saturated_vals):
        cells.append({
            'env_name': 'env_sat', 'arm_key': 'baseline',
            'seed': seed, 'y': v,
        })
        cells.append({
            'env_name': 'env_sat', 'arm_key': 'treatment',
            'seed': seed, 'y': v,
        })
    result = cross_env_probability_of_improvement.fn(
        cells_to_dataframe(cells),
        source='y',
        treatment_arm='treatment',
        baseline_arm='baseline',
        stratify_by=('env_name',),
    )
    assert result.n_strata == 3
    # Find the saturated stratum
    by_id = {s.stratum_id: s for s in result.per_stratum}
    saturated = by_id[('env_sat',)]
    assert abs(saturated.p_xy - 0.5) < 1e-9, (
        f'saturated p_xy expected 0.5, got {saturated.p_xy}'
    )
    # The two decisive strata are P_xy = 1.0
    for env in ('env_pos1', 'env_pos2'):
        assert by_id[(env,)].p_xy == 1.0
    # Aggregate: (1.0 + 1.0 + 0.5) / 3 = 0.8333
    assert abs(result.p_xy_mean - 5.0 / 6.0) < 1e-9


def test_p_xy_exact_enumeration_hand_checked() -> None:
    """3-stratum hand-built case where the permutation distribution
    is enumerable in writing. P_xy values [1.0, 1.0, 0.5];
    deviations [+0.5, +0.5, 0.0]; observed_dev = +1.0.

    The 8 sign-flips and their perm_devs (sum of signs × |dev_i|):
      (+,+,+): +0.5 +0.5 +0.0 = +1.0   ≥ +1.0  ✓
      (+,+,-): +0.5 +0.5 -0.0 = +1.0   ≥ +1.0  ✓
      (+,-,+): +0.5 -0.5 +0.0 =  0.0
      (+,-,-): +0.5 -0.5 -0.0 =  0.0
      (-,+,+): -0.5 +0.5 +0.0 =  0.0
      (-,+,-): -0.5 +0.5 -0.0 =  0.0
      (-,-,+): -0.5 -0.5 +0.0 = -1.0
      (-,-,-): -0.5 -0.5 -0.0 = -1.0
    n_ge = 2 (the +1.0 cases); p_permutation = 2/8 = 0.25."""
    cells: list[Mapping[str, object]] = []
    # Two strata where treatment dominates entirely:
    for env in ('env_a', 'env_b'):
        for seed in range(5):
            cells.append({
                'env_name': env, 'arm_key': 'baseline',
                'seed': seed, 'y': float(seed),
            })
            cells.append({
                'env_name': env, 'arm_key': 'treatment',
                'seed': seed, 'y': 100.0 + float(seed),
            })
    # One saturated stratum (identical samples)
    for seed in range(5):
        cells.append({
            'env_name': 'env_sat', 'arm_key': 'baseline',
            'seed': seed, 'y': float(seed),
        })
        cells.append({
            'env_name': 'env_sat', 'arm_key': 'treatment',
            'seed': seed, 'y': float(seed),
        })
    result = cross_env_probability_of_improvement.fn(
        cells_to_dataframe(cells),
        source='y',
        treatment_arm='treatment',
        baseline_arm='baseline',
        stratify_by=('env_name',),
    )
    assert result.n_strata == 3
    assert result.permutation_exact is True
    assert result.n_permutation_effective == 8
    # Hand-counted: 2 of 8 sign-flips have perm_dev ≥ observed_dev = +1.0
    expected_p = 2.0 / 8.0
    assert abs(result.p_permutation - expected_p) < 1e-9, (
        f'hand-counted p=0.25, got {result.p_permutation}'
    )


def test_p_xy_min_seeds_filter_drops_stratum() -> None:
    """When a stratum has fewer than `min_seeds_per_arm` cells in
    either arm, it's dropped from the panel (not included in
    n_strata)."""
    cells: list[Mapping[str, object]] = []
    # Valid stratum: 5 cells per arm
    for seed in range(5):
        cells.append({
            'env_name': 'env_ok', 'arm_key': 'baseline',
            'seed': seed, 'y': 0.0,
        })
        cells.append({
            'env_name': 'env_ok', 'arm_key': 'treatment',
            'seed': seed, 'y': 1.0,
        })
    # Invalid stratum: only 2 cells per arm — below floor
    for seed in range(2):
        cells.append({
            'env_name': 'env_small', 'arm_key': 'baseline',
            'seed': seed, 'y': 0.0,
        })
        cells.append({
            'env_name': 'env_small', 'arm_key': 'treatment',
            'seed': seed, 'y': 1.0,
        })
    result = cross_env_probability_of_improvement.fn(
        cells_to_dataframe(cells),
        source='y',
        treatment_arm='treatment',
        baseline_arm='baseline',
        stratify_by=('env_name',),
        min_seeds_per_arm=5,
    )
    assert result.n_strata == 1
    assert result.per_stratum[0].stratum_id == ('env_ok',)


def test_p_xy_mc_fallback_above_cap() -> None:
    """At `n_strata` such that `2 ** n_strata > permutation_cap`,
    the primitive switches from exact enumeration to MC
    sampling. `permutation_exact` becomes False;
    `n_permutation_effective` = `n_permutation` (not 2^n)."""
    cells: list[Mapping[str, object]] = []
    for i in range(16):
        for seed in range(5):
            cells.append({
                'env_name': f'env_{i:02d}', 'arm_key': 'baseline',
                'seed': seed, 'y': float(seed),
            })
            cells.append({
                'env_name': f'env_{i:02d}', 'arm_key': 'treatment',
                'seed': seed, 'y': 100.0 + float(seed),
            })
    result = cross_env_probability_of_improvement.fn(
        cells_to_dataframe(cells),
        source='y',
        treatment_arm='treatment',
        baseline_arm='baseline',
        stratify_by=('env_name',),
        permutation_cap=16384,  # 2^14 < 2^16
        n_permutation=5000,
    )
    assert result.n_strata == 16
    assert result.permutation_exact is False
    assert result.n_permutation_effective == 5000
    # All 16 strata decisively positive → p should be small
    assert result.p_permutation < 0.001
    # Mean P_xy near 1.0
    assert result.p_xy_mean == 1.0
