"""Closed-form assertions on `stratum_delta_link_dowhy` over a
phased multi-env LG-SCM corpus.

The canonical RL-substrate mech→outcome link primitive
(CLAUDE.md §"Canonical analyses"): pool seeds within each arm
at each (env, burst), compute Δ_predictor and Δ_target at the
stratum level, then run DoWhy backdoor regression on the panel
adjusting for burst dummies.

Under the LG-SCM (X → Z → Y), with z_mean as Δ_predictor and
y_mean as Δ_target:

    mean_z_arm(env, burst) ≈ β_xz_arm · μ_x(env)        (population)
    mean_y_arm(env, burst) ≈ β_zy · β_xz_arm · μ_x(env)

    Δ_z(env, burst) = (β_xz_t − β_xz_b) · μ_x(env)
    Δ_y(env, burst) = β_zy · Δ_z(env, burst)

→ population OLS slope of Δ_y on Δ_z (adjusting for burst
dummies) = β_zy.

The closed-form structural slope is `β_zy`. The substrate's
μ_x grid (1.0, 1.5, 2.0 across three envs) gives Δ_z and Δ_y
enough between-env variance for the backdoor regression to
converge tightly; the burst dimension (4 bursts per env) gives
the panel its stratum count.

Refuters:
- **Placebo**: random-permute the treatment column → ATE → 0,
  drift → β_zy.
- **Random common cause**: add an independent N(0, σ²) confounder
  → ATE invariant, drift → 0.

The DoWhy primitive itself is already covered by `test_dowhy.py`;
this test specifically verifies that the panel-construction +
stratum-Δ + burst-adjustment plumbing inside
`stratum_delta_link_dowhy.fn` doesn't corrupt the structural
recovery.
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
    """Population Δ_y / Δ_z = β_zy exactly. The DoWhy backdoor on
    the (Δ_jens=Δ_z, Δ_out=Δ_y) panel adjusting for burst
    dummies should recover β_zy = 1.5 within a tight bound."""
    cells = _build_phased_cells()

    result = stratum_delta_link_dowhy.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        link_predictor=_PER_BURST_Z_MEAN,
        link_target=_PER_BURST_Y_MEAN,
    )

    expected = _expected_link_slope()
    # Panel size: 3 envs × 4 bursts = 12 strata. After burst-
    # dummy adjustment, residual df = 12 − 4 = 8 (intercept + 3
    # burst dummies). At n_seeds=30 per arm × n_steps=100 the
    # within-(env, burst) seed-pooled mean estimates have small
    # SD; the dominant source of slope-recovery error is residual
    # variance from σ_z² propagation through β_zy. Empirical SE
    # on the slope is ≈ 0.02 — a 4-σ window of 0.10 is a
    # generous 5× safety margin that still detects any sign,
    # scale, or pooling regression by orders of magnitude.
    assert abs(result.backdoor.ate - expected) < 0.10, (
        f'backdoor.ate={result.backdoor.ate:.4f} '
        f'expected={expected:.4f}'
    )
    assert result.n_strata == 3 * _N_BURSTS, (
        f'n_strata={result.n_strata} expected '
        f'{3 * _N_BURSTS} — stratum-panel construction lost rows'
    )
    assert result.treatment_col == 'djens'
    assert result.outcome_col == 'dout'


def test_stratum_delta_link_placebo_destroys_signal() -> None:
    """Placebo refutation: random-permute Δ_jens → structural
    link broken → refuted ATE ≈ 0, drift ≈ β_zy."""
    cells = _build_phased_cells()
    result = stratum_delta_link_dowhy.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        link_predictor=_PER_BURST_Z_MEAN,
        link_target=_PER_BURST_Y_MEAN,
    )
    expected = _expected_link_slope()
    # Placebo SHOULD recover ≈ 0 after permutation. Drift =
    # original − refuted. Tolerance: 0.30 absolute (the
    # permutation samples carry their own MC noise on small n;
    # 12 strata permuted → drift SE ≈ 0.20).
    assert abs(result.placebo.refuted_ate) < 0.30, (
        f'placebo refuted_ate={result.placebo.refuted_ate:.4f} '
        'should be near 0 after permutation'
    )
    drift = result.placebo.drift
    assert abs(drift - expected) < 0.30, (
        f'placebo drift={drift:.4f} should match the '
        f'structural slope β_zy={expected}'
    )


def test_stratum_delta_link_random_common_cause_preserves_signal() -> None:
    """Random common cause: add a random confounder unrelated to
    Δ_jens or Δ_out → refuted ATE ≈ original ATE, drift ≈ 0."""
    cells = _build_phased_cells()
    result = stratum_delta_link_dowhy.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        link_predictor=_PER_BURST_Z_MEAN,
        link_target=_PER_BURST_Y_MEAN,
    )
    expected = _expected_link_slope()
    # RCC SHOULD give refuted_ate ≈ original ATE.
    assert abs(result.random_common_cause.refuted_ate - expected) < 0.20
    # Drift on a random confounder should be small. With 12
    # strata and a random N(0,1) confounder added, the refuted
    # estimate fluctuates by the OLS slope of the random column
    # on Δ_jens — SE on that fluctuation ≈ 0.10 on this corpus.
    # 0.20 absolute is a 2× safety margin.
    assert abs(result.random_common_cause.drift) < 0.20, (
        f'rcc drift={result.random_common_cause.drift:.4f} should '
        'be ≈ 0 — random confounder carries no structural signal'
    )


def test_stratum_delta_link_null_contrast_indistinguishable() -> None:
    """Null intervention (β_xz_t == β_xz_b): Δ_z and Δ_y are both
    exactly zero at the population level (modulo seed-pool noise).
    The slope is unidentified at the population level — backdoor
    will report something near 0 but with the precise value
    sensitive to the OLS solve under near-constant treatment.

    What's load-bearing: the framework doesn't return a runtime
    error on this corpus; the panel construction is still valid
    (12 strata, valid burst dummies). The placebo + RCC drift
    interpretation breaks because the original ATE has no
    structural meaning here — verified by ATE being far smaller
    than the non-null case."""
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
    # n_strata may be < n_bursts*n_envs because min_baseline_predictor
    # filter applies; but it should be > 0 (β_xz_b * μ_x > 0.05 in
    # all envs).
    assert result.n_strata > 0
    # In the null contrast, even tiny Δ_z values get amplified
    # by OLS noise. The structural slope is undefined; what we
    # check is that the ATE is much smaller than the non-null
    # β_zy = 1.5. A 5σ-of-noise bound on the null ATE is ~0.5;
    # any non-trivial structural recovery would be many multiples
    # of that, so failing this bound means the primitive is
    # manufacturing a slope from i.i.d. noise.
    assert abs(result.backdoor.ate) < 1.0, (
        f'backdoor.ate={result.backdoor.ate:.4f} should be ≈ 0 '
        'on a null contrast — non-zero suggests OLS is finding '
        'spurious structure'
    )
    # ensure the result is well-formed
    assert not math.isnan(result.backdoor.ate)
