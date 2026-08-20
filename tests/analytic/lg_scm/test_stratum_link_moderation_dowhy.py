"""Closed-form assertions on `stratum_link_moderation_dowhy`
over the LG-SCM substrate.

The primitive tests whether `attenuator > binary_threshold`
MODERATES the Δ_predictor → Δ_target slope, via DoWhy backdoor
on an interaction term. The interaction coefficient on
`Δ_predictor × 1[env above threshold]` IS the moderation effect:

    Δ_target = β_main · Δ_predictor
             + β_inter · (Δ_predictor × 1[above])
             + env_dummies + intercept

For below-threshold envs (above=0): slope = β_main
For above-threshold envs (above=1): slope = β_main + β_inter

→ β_inter = slope_above − slope_below

CLAUDE.md notes this primitive is "currently UNCONSUMED — kept
provisionally for a future moderation-asking bridge"; this test
covers the contract so a future consumer can rely on the
recovery being correct.

Implementation setup: 4 envs split by μ_x threshold = 1.5.
- 2 below-threshold (μ_x ∈ {1.0, 1.4}, β_zy = 1.0)
- 2 above-threshold (μ_x ∈ {1.6, 2.0}, β_zy = 2.0)
- Under shared seeds, Δ_y = β_zy · Δ_z exactly per stratum
- Expected interaction coefficient = 2.0 − 1.0 = 1.0 EXACTLY
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from corroborate.analyses.link.stratum_link_moderation_dowhy import (
    stratum_link_moderation_dowhy,
)
from corroborate.measurables.reductions import from_key, reduce_axis
from corroborate.data import cells_to_dataframe

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import (
    PER_BURST_Y_KEY, PER_BURST_Z_KEY, run_paired_phased_arms,
)


_SIGMA_X = 0.5
_SIGMA_Z = 0.05
_SIGMA_Y = 0.05
_N_STEPS = 100
_N_SEEDS = 30
_N_BURSTS = 4
_BETA_XZ_T = 0.7
_BETA_XZ_B = 0.3
_BETA_ZY_BELOW = 1.0
_BETA_ZY_ABOVE = 2.0
_THRESHOLD = 1.5


_ENVS_BELOW: Mapping[str, float] = {'env_a': 1.0, 'env_b': 1.4}
_ENVS_ABOVE: Mapping[str, float] = {'env_c': 1.6, 'env_d': 2.0}


_PER_BURST_Z_MEAN = reduce_axis(from_key(PER_BURST_Z_KEY), axis=-1, op='mean')
_PER_BURST_Y_MEAN = reduce_axis(from_key(PER_BURST_Y_KEY), axis=-1, op='mean')


def _scm(mu_x: float, beta_xz: float, beta_zy: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=mu_x,
        sigma_x=_SIGMA_X,
        beta_xz=beta_xz,
        sigma_z=_SIGMA_Z,
        beta_zy=beta_zy,
        sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _build_cells(
    *,
    beta_zy_below: float = _BETA_ZY_BELOW,
    beta_zy_above: float = _BETA_ZY_ABOVE,
) -> list[Mapping[str, object]]:
    """Build per-(env, burst) cells. Per-env β_zy controlled
    independently for the above/below groups so the homogeneous-
    link scenario reuses the same builder with both args equal."""
    rows: list[Mapping[str, object]] = []
    for envs, beta_zy in (
        (_ENVS_BELOW, beta_zy_below), (_ENVS_ABOVE, beta_zy_above),
    ):
        for env, mu_x in envs.items():
            treatments = tuple(
                _scm(mu_x, _BETA_XZ_T, beta_zy)
                for _ in range(_N_BURSTS)
            )
            baselines = tuple(
                _scm(mu_x, _BETA_XZ_B, beta_zy)
                for _ in range(_N_BURSTS)
            )
            rows.extend(run_paired_phased_arms(
                treatments_per_burst=treatments,
                baselines_per_burst=baselines,
                seeds=tuple(range(_N_SEEDS)),
                env_name=env,
            ))
    return rows


def _expected_interaction() -> float:
    """β_inter = β_zy_above − β_zy_below, structurally."""
    return _BETA_ZY_ABOVE - _BETA_ZY_BELOW


def test_stratum_link_moderation_recovers_interaction_coefficient() -> None:
    """Interaction coefficient = β_zy_above − β_zy_below = 1.0
    exactly under shared-seed cancellation. The OLS solve
    isn't quite to machine precision (per-stratum Δ has
    mean_seeds(X_avg) noise, ~1e-3), but recovery is precise."""
    cells = _build_cells()
    result = stratum_link_moderation_dowhy.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        link_predictor=_PER_BURST_Z_MEAN,
        link_target=_PER_BURST_Y_MEAN,
        attenuator='mu_x',
        binary_threshold=_THRESHOLD,
        min_baseline_predictor=0.0,
    )
    expected = _expected_interaction()
    # All envs have mu_x_attenuator MEAN = mu_x (single SCM per
    # env, so env_mean(mu_x) = mu_x exactly). 2 below, 2 above
    # the 1.5 threshold.
    assert result.n_envs_below == 2
    assert result.n_envs_above == 2
    # 4 envs × 4 bursts = 16 strata after admission. Each stratum
    # passes min_baseline_predictor=0.0 (baseline z_mean per
    # stratum = β_xz_b · μ_x > 0.3 always).
    assert result.n_strata == 4 * _N_BURSTS
    # The interaction coefficient SE under OLS with the panel
    # carrying 16 rows + 5 predictors (intercept + d_pred + 3 env
    # dummies + interaction) leaves df = 10. Empirically the
    # recovered interaction lands within 0.01 of expected; 0.05
    # bound is ~5× safety margin and detects any sign / scale /
    # threshold-direction bug.
    assert abs(result.backdoor.ate - expected) < 0.05, (
        f'interaction ATE = {result.backdoor.ate:.4f}, '
        f'expected = {expected:.4f} (β_zy_above − β_zy_below)'
    )


def test_stratum_link_moderation_no_moderation_under_homogeneous_link() -> None:
    """When β_zy is the SAME in below and above envs (no true
    moderation), interaction coefficient should be ≈ 0. The
    OLS solve isn't exactly zero — it picks up tiny residual
    variance from per-stratum mean_seeds(X_avg) — but the
    interaction's contribution to outcome variance is null."""
    rows = _build_cells(beta_zy_below=1.5, beta_zy_above=1.5)
    result = stratum_link_moderation_dowhy.fn(
        cells_to_dataframe(rows),
        treatment_arm='treatment',
        baseline_arm='baseline',
        link_predictor=_PER_BURST_Z_MEAN,
        link_target=_PER_BURST_Y_MEAN,
        attenuator='mu_x',
        binary_threshold=_THRESHOLD,
        min_baseline_predictor=0.0,
    )
    # Under homogeneous link, structural moderation is exactly 0.
    # Numerical noise from finite-n mean_seeds(X_avg) keeps the
    # OLS interaction coefficient within ~0.02 of zero
    # empirically. 0.05 bound is ~2.5× the empirical noise.
    assert abs(result.backdoor.ate) < 0.05, (
        f'interaction ATE = {result.backdoor.ate:.4f} under '
        'homogeneous β_zy — expected ≈ 0 (no moderation)'
    )


def test_stratum_link_moderation_all_envs_below_threshold_returns_empty() -> None:
    """When no env exceeds the threshold, the panel filters
    everything (need at least 1 above + 1 below for the
    interaction to be identifiable). Result is NaN throughout."""
    # All envs have mu_x < 100 → all below the threshold.
    cells = _build_cells()
    result = stratum_link_moderation_dowhy.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        link_predictor=_PER_BURST_Z_MEAN,
        link_target=_PER_BURST_Y_MEAN,
        attenuator='mu_x',
        binary_threshold=100.0,
        min_baseline_predictor=0.0,
    )
    assert result.n_strata == 0
    assert result.n_envs_above == 0
    assert math.isnan(result.backdoor.ate)


def test_stratum_link_moderation_placebo_destroys_signal() -> None:
    """Placebo permutes the interaction column → structural
    interaction → 0 → refuted ATE near 0, drift ≈ structural."""
    cells = _build_cells()
    result = stratum_link_moderation_dowhy.fn(
        cells_to_dataframe(cells),
        treatment_arm='treatment',
        baseline_arm='baseline',
        link_predictor=_PER_BURST_Z_MEAN,
        link_target=_PER_BURST_Y_MEAN,
        attenuator='mu_x',
        binary_threshold=_THRESHOLD,
        min_baseline_predictor=0.0,
    )
    expected = _expected_interaction()
    # Placebo permutation breaks structural link to outcome —
    # refuted ATE collapses toward 0 modulo small-n permutation
    # MC noise on 16 strata.
    assert abs(result.placebo.refuted_ate) < 0.30, (
        f'placebo refuted_ate = {result.placebo.refuted_ate:.4f} '
        'should be near 0 after permutation of interaction column'
    )
    assert abs(result.placebo.drift - expected) < 0.30
