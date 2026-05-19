"""Closed-form assertions on `stratum_panel_jci_spearman` over
the LG-SCM substrate.

`stratum_panel_jci_spearman` returns three Spearman ρs on the
per-(env, config) stratum panel:

- `rho_marginal`: Spearman across all strata (no env adjustment)
- `rho_stratified`: JCI Spearman, per-env Fisher-z-pooled
- `rho_partial_stratified`: same as stratified but partialling
  out the baseline-arm target mean (partial_z)

2 active bridge consumers across `experiments/findings/`. Used
as a mediation falsification surface: if the predictor's
information about Δ_target is fully explained by the baseline
arm's target mean (config-quality confound), then
`rho_partial_stratified` drops to ≈ 0 while `rho_marginal` /
`rho_stratified` stay large.

Substrate setup: 4 envs × 3 μ_x levels = 12 (env, config) strata.
Within each env, μ_x varies → predictor, target, and partial_z
all vary linearly with μ_x. Predictor and target are perfectly
rank-monotone in μ_x → per-env Spearman ρ = +1 → JCI Fisher-z
pool = +1. Partial_z (baseline y_mean) is ALSO perfectly rank-
monotone in μ_x → partialling it out leaves no residual signal
→ per-env partial ρ = 0 (numerical noise only).
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from corroborate.analyses.spearman.stratum_panel_jci_spearman import (
    stratum_panel_jci_spearman,
)

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_arm


_SIGMA_X = 0.5
_SIGMA_Z = 0.05
_BETA_ZY = 1.5
_SIGMA_Y = 0.05
_N_STEPS = 200
_N_SEEDS_PER_ARM = 30
_BETA_XZ_T = 0.7
_BETA_XZ_B = 0.3

_ENVS: tuple[str, ...] = ('env_a', 'env_b', 'env_c', 'env_d')
# `stratified_spearman_rho` requires min_stratum_size=4 per env.
# 4 μ_x levels per env → 4 strata per env → 16 total strata.
_MU_X_LEVELS: tuple[float, ...] = (1.0, 2.0, 3.0, 4.0)


def _scm(mu_x: float, beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=mu_x,
        sigma_x=_SIGMA_X,
        beta_xz=beta_xz,
        sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY,
        sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _build_cells() -> list[Mapping[str, object]]:
    rows = []
    for i, env in enumerate(_ENVS):
        for j, mu_x in enumerate(_MU_X_LEVELS):
            seed_offset = (i * len(_MU_X_LEVELS) + j) * _N_SEEDS_PER_ARM * 2
            seeds = tuple(range(
                seed_offset, seed_offset + _N_SEEDS_PER_ARM,
            ))
            rows.extend(run_arm(
                _scm(mu_x, _BETA_XZ_T),
                seeds=seeds, arm_key='treatment', env_name=env,
            ))
            rows.extend(run_arm(
                _scm(mu_x, _BETA_XZ_B),
                seeds=seeds, arm_key='baseline', env_name=env,
            ))
    return [r.as_dict() for r in rows]


def test_stratum_panel_jci_marginal_rho_high() -> None:
    """predictor (baseline z_mean = β_xz_b · μ_x) and target Δ
    (Δ y_mean = (β_xz_t − β_xz_b)·β_zy·μ_x) are both ∝ μ_x with
    constant per-env multipliers. Marginal Spearman across all
    12 strata ≈ +1 (3 μ_x levels × 4 envs, perfectly monotone)."""
    cells = _build_cells()
    result = stratum_panel_jci_spearman.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        predictor_col='z_mean',
        target_col='y_mean',
        stratify_by=('env_name', 'mu_x'),
        min_seeds_per_arm=5,
        min_baseline_predictor=0.0,
    )
    assert result.n_strata == len(_ENVS) * len(_MU_X_LEVELS)
    # Marginal: predictor and target both ∝ μ_x AND the
    # multiplier on predictor (β_xz_b) is constant across envs;
    # the multiplier on target ((β_xz_t − β_xz_b)·β_zy) is also
    # constant. So the marginal relationship is strictly
    # monotone in μ_x with NO tie except across envs at same μ_x
    # (3 ties of 4 strata each). Empirically ρ ≈ 0.98.
    assert result.rho_marginal > 0.9, (
        f'rho_marginal={result.rho_marginal:.4f} should be ≈ +1.0 '
        'under strict-monotone predictor → target across strata'
    )
    assert result.p_marginal < 1e-3


def test_stratum_panel_jci_stratified_rho_high() -> None:
    """JCI per-env Fisher-z pool: within each env, 3 strata vary
    only in μ_x → per-env Spearman ρ = +1 exactly. Fisher-z pool
    of (1, 1, 1, 1) = +1.

    Note: at n_per_env=4, Spearman ρ is computed on 4 monotone
    points (μ_x=1, 2, 3, 4) — exactly +1 in the noiseless case.
    Empirically, baseline z_mean per stratum is
    β_xz_b · μ_x + σ_z noise, and Δ_y per stratum is
    (β_t − β_b)·β_zy·μ_x exactly under shared-seed cancellation;
    at σ_z = 0.05, predictor noise relative to signal is
    σ_z/sqrt(n_steps)/(β_xz_b · Δμ_x) ≈ 0.0035/(0.3) ≈ 1.2%, so
    ranks never flip → per-env ρ = +1.
    """
    cells = _build_cells()
    result = stratum_panel_jci_spearman.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        predictor_col='z_mean',
        target_col='y_mean',
        stratify_by=('env_name', 'mu_x'),
        min_seeds_per_arm=5,
        min_baseline_predictor=0.0,
    )
    # JCI stratified ≈ +1 (per-env all rank-monotone, Fisher-z
    # pool of (+1, +1, +1, +1) = +1). Empirically 0.98 ± 0.01.
    assert result.rho_stratified > 0.95, (
        f'rho_stratified={result.rho_stratified:.4f} should be ≈ +1 '
        'when per-env ranks are perfectly monotone'
    )


def test_stratum_panel_jci_partial_rho_collapses_under_full_mediation() -> None:
    """partial_z = baseline y_mean per stratum. Within each env,
    partial_z = β_xz_b·β_zy·μ_x is PERFECTLY rank-monotone in
    μ_x — same rank order as predictor (β_xz_b·μ_x) and target
    Δ ((β_t−β_b)·β_zy·μ_x). All three columns share the same
    within-env rank vector (1, 2, 3, 4).

    Partial Spearman ρ(x, dy | vy) at perfect collinearity is
    `(r_xy − r_xz · r_zy) / sqrt((1 − r_xz²)(1 − r_zy²))` with
    r_xz = r_zy = 1, giving 0/0 = NaN. The primitive's
    `stratified_partial_spearman_rho` propagates NaN out.

    This IS the correct mediation-falsification outcome under
    FULL mediation: the predictor's signal IS the partial_z's
    signal up to a constant, so there's nothing left to
    correlate. Reading NaN as "mediation supported" requires
    the primitive to NOT silently return 0 (which would be
    mistaken for "mediation falsified"). NaN forces the bridge
    to handle the singular case explicitly."""
    cells = _build_cells()
    result = stratum_panel_jci_spearman.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        predictor_col='z_mean',
        target_col='y_mean',
        stratify_by=('env_name', 'mu_x'),
        min_seeds_per_arm=5,
        min_baseline_predictor=0.0,
    )
    # Under perfect within-env collinearity of (x, dy, vy):
    # partial ρ is NaN. The framework must propagate NaN, NOT
    # return a garbage 0 (which would falsely signal "mediation
    # falsified").
    assert math.isnan(result.rho_partial_stratified), (
        f'rho_partial_stratified={result.rho_partial_stratified} — '
        'expected NaN under perfect within-env collinearity of '
        '(predictor, target, partial_z); a non-NaN return would '
        'silently misclassify the singular case'
    )


def test_stratum_panel_jci_returns_nan_when_no_strata() -> None:
    """Empty corpus → NaN throughout."""
    result = stratum_panel_jci_spearman.fn(
        [],
        treatment_arm='treatment',
        baseline_arm='baseline',
        predictor_col='z_mean',
        target_col='y_mean',
        stratify_by=('env_name',),
    )
    assert math.isnan(result.rho_marginal)
    assert math.isnan(result.rho_stratified)
    assert math.isnan(result.rho_partial_stratified)
    assert result.n_strata == 0
