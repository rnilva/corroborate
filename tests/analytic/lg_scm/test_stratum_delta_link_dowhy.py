"""Closed-form assertions on `stratum_delta_link_dowhy` over a
phased multi-env LG-SCM corpus.

The canonical RL-substrate mech→outcome link primitive
(CLAUDE.md §"Canonical analyses"): pool seeds within each arm
at each (env, burst), compute Δ_predictor and Δ_target at the
stratum level, then run DoWhy backdoor regression on the panel
adjusting for burst dummies.

Under the LG-SCM (X → Z → Y) with SHARED seeds across arms (the
substrate's design — `numpy.random.default_rng(seed)` pre-draws
all epsilons identically per seed), the σ_z and σ_y noise
streams are PERFECTLY MATCHED across arms. Seed-pooled means
at each (env, burst, arm) therefore differ ONLY by the
structural-coefficient channel:

    mean_z_arm(env, burst) = β_xz_arm · mean_seeds(X_avg)
                             + σ_z · mean_seeds(ε_z_avg)
    mean_y_arm(env, burst) = β_zy · mean_z_arm(env, burst)
                             + σ_y · mean_seeds(ε_y_avg)

Δ at each stratum is `mean_treatment − mean_baseline`:

    Δ_z(env, burst) = (β_xz_t − β_xz_b) · mean_seeds(X_avg)
    Δ_y(env, burst) = β_zy · Δ_z(env, burst)

The σ_z and σ_y components cancel EXACTLY in the Δ (identical
ε streams across arms). So Δ_y / Δ_z = β_zy holds NUMERICALLY
at every stratum, not just in expectation.

→ DoWhy backdoor regression of Δ_y on Δ_z (adjusting for burst
dummies) recovers β_zy = 1.5 to machine precision (empirically
1.5 ± 1e-15 — float64 OLS solve noise, not sampling noise).
The substrate's variation in μ_x across envs gives the panel
enough between-stratum spread for the OLS solve to converge
without singularity.

Refuters under shared-seed exactness:
- **Placebo**: random-permute the treatment column → structural
  link broken → refuted ATE = 0 exactly → drift = β_zy exactly.
- **Random common cause**: add an independent column to the
  regression. Since Δ_y is perfectly explained by Δ_z, OLS
  zeroes the new column's coefficient and preserves the Δ_z
  coefficient at β_zy → refuted ATE = β_zy → drift ≈ 0.

The DoWhy primitive itself is covered by `test_dowhy.py` (which
exercises finite-sample SE behavior with non-shared noise); this
test specifically verifies that the panel-construction +
stratum-Δ + burst-adjustment plumbing inside
`stratum_delta_link_dowhy.fn` doesn't corrupt the structural
recovery. The 1e-9 bound is well above float64 OLS noise
(~1e-15) and well below any plausible sign/scale/pooling bug.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.dowhy.stratum_delta_link_dowhy import (
    stratum_delta_link_dowhy,
)
from corroborate.measurables.reductions import from_key, reduce_axis

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import (
    PER_BURST_Y_KEY,
    PER_BURST_Z_KEY,
    run_paired_phased_arms,
)


_SIGMA_X = 0.5
_SIGMA_Z = 0.1
_BETA_ZY = 1.5
_SIGMA_Y = 0.1
_N_STEPS_PER_BURST = 100
_N_BURSTS = 4
_N_SEEDS_PER_ARM = 30


# Per-env μ_x — gives between-env variance in Δ_z and Δ_y.
_ENV_MU_X: Mapping[str, float] = {
    'env_a': 1.0,
    'env_b': 1.5,
    'env_c': 2.0,
}

# Shared treatment / baseline β_xz across envs. The contrast
# β_xz_t − β_xz_b is the intervention magnitude that propagates
# through the structural slope β_zy onto Δ_y.
_BETA_XZ_TREATMENT = 0.7
_BETA_XZ_BASELINE = 0.3


_PER_BURST_Y_MEAN = reduce_axis(
    from_key(PER_BURST_Y_KEY), axis=-1, op='mean',
)
_PER_BURST_Z_MEAN = reduce_axis(
    from_key(PER_BURST_Z_KEY), axis=-1, op='mean',
)


def _scm(beta_xz: float, mu_x: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=mu_x,
        sigma_x=_SIGMA_X,
        beta_xz=beta_xz,
        sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY,
        sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS_PER_BURST,
    )


def _build_phased_cells(
    *, beta_xz_t: float = _BETA_XZ_TREATMENT,
    beta_xz_b: float = _BETA_XZ_BASELINE,
) -> list[Mapping[str, object]]:
    cells: list[Mapping[str, object]] = []
    for env_name, mu_x in _ENV_MU_X.items():
        treatments = tuple(
            _scm(beta_xz_t, mu_x) for _ in range(_N_BURSTS)
        )
        baselines = tuple(
            _scm(beta_xz_b, mu_x) for _ in range(_N_BURSTS)
        )
        cells.extend(run_paired_phased_arms(
            treatments_per_burst=treatments,
            baselines_per_burst=baselines,
            seeds=tuple(range(_N_SEEDS_PER_ARM)),
            env_name=env_name,
        ))
    return cells


def _expected_link_slope() -> float:
    """Population slope of Δ_y on Δ_z under the LG-SCM. Since
    Δ_y = β_zy · Δ_z exactly at the population mean (X → Z → Y
    chain with linear arrows and no direct X → Y edge), the OLS
    slope recovers β_zy."""
    return _BETA_ZY


def test_stratum_delta_link_recovers_structural_slope() -> None:
    """Δ_y / Δ_z = β_zy exactly under shared-seed cancellation
    of σ_z and σ_y streams. The DoWhy backdoor on the
    (Δ_jens=Δ_z, Δ_out=Δ_y) panel adjusting for burst dummies
    recovers β_zy = 1.5 to machine precision."""
    cells = _build_phased_cells()

    result = stratum_delta_link_dowhy.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        link_predictor=_PER_BURST_Z_MEAN,
        link_target=_PER_BURST_Y_MEAN,
    )

    expected = _expected_link_slope()
    # Tight numerical bound — under shared seeds the σ_z + σ_y
    # noise streams cancel exactly between arms in seed-pooled
    # means, so Δ_y = β_zy · Δ_z holds numerically (not just in
    # expectation). 1e-9 is six orders of magnitude above float64
    # OLS noise (~1e-15) and detects any sign / scale / pooling
    # regression at far below "one part per million".
    assert abs(result.backdoor.ate - expected) < 1e-9, (
        f'backdoor.ate={result.backdoor.ate!r} '
        f'expected={expected}'
    )
    assert result.n_strata == 3 * _N_BURSTS, (
        f'n_strata={result.n_strata} expected '
        f'{3 * _N_BURSTS} — stratum-panel construction lost rows'
    )


def test_stratum_delta_link_placebo_destroys_signal() -> None:
    """Placebo refutation: random-permute Δ_jens. Under the
    deterministic structural recovery (Δ_y = β_zy · Δ_z exact),
    the permuted treatment carries no information about Δ_y →
    OLS slope on permuted column = 0 exactly. Drift = original
    − refuted = β_zy exactly."""
    cells = _build_phased_cells()
    result = stratum_delta_link_dowhy.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        link_predictor=_PER_BURST_Z_MEAN,
        link_target=_PER_BURST_Y_MEAN,
    )
    expected = _expected_link_slope()
    # Same numerical-precision rationale as the structural test:
    # 1e-9 covers float64 OLS noise on the permuted-column solve
    # and detects any "placebo doesn't fully break the link" bug.
    assert abs(result.placebo.refuted_ate) < 1e-9, (
        f'placebo refuted_ate={result.placebo.refuted_ate!r} '
        'should be 0 — permutation breaks Δ_jens → Δ_out signal'
    )
    drift = result.placebo.drift
    assert abs(drift - expected) < 1e-9, (
        f'placebo drift={drift!r} should match β_zy={expected}'
    )


def test_stratum_delta_link_random_common_cause_preserves_signal() -> None:
    """Random common cause: add an independent column to the
    regression. Since Δ_y is perfectly explained by Δ_z (no
    residual to allocate to the new column), OLS preserves the
    Δ_z coefficient exactly → refuted ATE = β_zy → drift = 0."""
    cells = _build_phased_cells()
    result = stratum_delta_link_dowhy.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        link_predictor=_PER_BURST_Z_MEAN,
        link_target=_PER_BURST_Y_MEAN,
    )
    expected = _expected_link_slope()
    assert abs(result.random_common_cause.refuted_ate - expected) < 1e-9, (
        f'rcc refuted_ate={result.random_common_cause.refuted_ate!r} '
        f'should match β_zy={expected}'
    )
    # Same precision rationale: under deterministic structural
    # recovery, OLS allocates zero coefficient to the random
    # column → drift is numerical OLS noise (~1e-15).
    assert abs(result.random_common_cause.drift) < 1e-9, (
        f'rcc drift={result.random_common_cause.drift!r} '
        'should be ≈ 0'
    )


def test_stratum_delta_link_null_contrast_yields_nan_or_zero() -> None:
    """Null intervention (β_xz_t == β_xz_b): under shared seeds,
    Δ_z = 0 EXACTLY at every (env, burst) stratum. The OLS solve
    has a singular treatment column — the framework should either
    return NaN (singular regression) or zero ATE, but never a
    spurious non-zero structural slope.

    The min_baseline_predictor=0.05 filter still admits all strata
    (mean_baseline_z = β_xz_b · μ_x ∈ {0.3, 0.45, 0.6} all > 0.05),
    so panel construction is valid; only the OLS solve is
    degenerate.
    """
    cells = _build_phased_cells(
        beta_xz_t=_BETA_XZ_BASELINE,
        beta_xz_b=_BETA_XZ_BASELINE,
    )
    result = stratum_delta_link_dowhy.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        link_predictor=_PER_BURST_Z_MEAN,
        link_target=_PER_BURST_Y_MEAN,
    )
    assert result.n_strata == 3 * _N_BURSTS
    # Under exact null treatment, OLS slope is either NaN
    # (singular column) or 0 (degenerate solve). NOT a non-zero
    # structural slope — that would mean the primitive is
    # manufacturing signal from machine precision.
    ate = result.backdoor.ate
    assert math.isnan(ate) or abs(ate) < 1e-9, (
        f'backdoor.ate={ate!r} under null contrast — expected NaN '
        '(singular OLS) or numerically zero'
    )
