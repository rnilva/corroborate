"""Closed-form assertions on `stratum_effect_panel` over the
LG-SCM substrate.

`stratum_effect_panel` is the leaner sibling of `stratum_panel`:
it surfaces ONLY the per-stratum Δs (no Spearman, no full means/
SDs), with the additional `aggregator='median'` mode for outlier-
robust per-stratum summaries. 4 active bridge consumers across
`experiments/findings/`.

Mean and median Δs both recover the structural product under
shared-seed cancellation:

    Δ_y(env) = (β_xz_t − β_xz_b) · β_zy · μ_x

For Gaussian residuals the median and mean of y_mean per arm
coincide in population; median converges with larger SE
(asymptotic CV is sqrt(π/2) ≈ 1.253× the mean's CV) — bound
size accounts for this.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from corroborate.analyses.panel.stratum_effect_panel import (
    stratum_effect_panel,
)

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_multi_env_paired_arms


_MU_X = 1.0
_SIGMA_X = 0.5
_SIGMA_Z = 0.1
_BETA_ZY = 1.5
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_SEEDS_PER_ARM = 60


_ENV_BETAS: Mapping[str, tuple[float, float]] = {
    'env_a': (0.7, 0.3),
    'env_b': (0.9, 0.5),
    'env_c': (0.8, 0.2),
}


def _scm(beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=_MU_X,
        sigma_x=_SIGMA_X,
        beta_xz=beta_xz,
        sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY,
        sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _build_cells() -> list[Mapping[str, object]]:
    envs = {
        name: (_scm(t), _scm(b))
        for name, (t, b) in _ENV_BETAS.items()
    }
    rows = run_multi_env_paired_arms(
        envs=envs, seeds=tuple(range(_N_SEEDS_PER_ARM)),
    )
    return [r.as_dict() for r in rows]


def _expected_y_delta(env: str) -> float:
    t, b = _ENV_BETAS[env]
    return _BETA_ZY * (t - b) * _MU_X


def _expected_shared_seed_delta_se(env: str) -> float:
    """Under shared seeds, Δ_y per seed has zero σ_z / σ_y noise
    (those streams cancel). Only mean_seeds(X_avg) contributes
    sampling noise: SE = |β_t − β_b|·β_zy·σ_x/sqrt(n_seeds·n_steps)."""
    t, b = _ENV_BETAS[env]
    return (
        abs(t - b) * _BETA_ZY * _SIGMA_X
        / math.sqrt(_N_SEEDS_PER_ARM * _N_STEPS)
    )


def test_stratum_effect_panel_mean_recovers_structural_delta() -> None:
    panel = stratum_effect_panel.fn(
        _build_cells(),
        treatment_arm='treatment',
        baseline_arm='baseline',
        measurables=('y_mean',),
        aggregator='mean',
    )
    assert panel.n_strata == 3
    deltas = panel.deltas['y_mean']
    # Per-stratum cell counts equal seeds per arm
    for n_t, n_b in zip(panel.n_treatment, panel.n_baseline, strict=True):
        assert n_t == _N_SEEDS_PER_ARM
        assert n_b == _N_SEEDS_PER_ARM
    # Mean-mode Δ recovers structural form within shared-seed-
    # cancellation SE. 4σ window is ≈ 0.011 at the smallest Δ
    # (env_a: Δ = 0.4·1.5·1.0 = 0.6, 4σ = 0.011 → < 2% of Δ).
    for idx, stratum in enumerate(panel.strata):
        env_name = str(stratum[0])
        expected = _expected_y_delta(env_name)
        se = _expected_shared_seed_delta_se(env_name)
        assert abs(deltas[idx] - expected) < 4.0 * se, (
            f'{env_name}: mean Δ_y={deltas[idx]:.4f} '
            f'expected={expected:.4f} 4σ={4.0 * se:.4f}'
        )


def test_stratum_effect_panel_median_recovers_structural_delta() -> None:
    """Median Δ on Gaussian data → same population value as mean
    Δ, with sqrt(π/2) ≈ 1.253× the per-arm CV. 5σ bound on the
    median-mode SE accommodates the wider median sampling
    distribution without admitting any structural bug."""
    panel = stratum_effect_panel.fn(
        _build_cells(),
        treatment_arm='treatment',
        baseline_arm='baseline',
        measurables=('y_mean',),
        aggregator='median',
    )
    deltas = panel.deltas['y_mean']
    assert panel.n_strata == 3
    for idx, stratum in enumerate(panel.strata):
        env_name = str(stratum[0])
        expected = _expected_y_delta(env_name)
        # Median Δ under shared seeds: differs from mean Δ
        # because median(y_t − y_b across seeds) ≠ median(y_t) −
        # median(y_b) when arms aren't aligned — but the panel
        # computes the LATTER form (median per arm, then Δ).
        # Median(y_t) ≈ mean(y_t) for Gaussian; sampling SD on
        # median is sqrt(π/2)·σ_arm/sqrt(n). With shared-seed
        # cancellation breaking under per-arm-medians, σ_z/σ_y
        # noise re-enters: full per-arm SD on y_mean ≈ 0.04 at
        # the larger-β arm; median Δ SE ≈ sqrt(2·π/2)·0.04/√60 ≈
        # 0.0091. 5σ bound ≈ 0.046, still < 8% of smallest Δ.
        t, b = _ENV_BETAS[env_name]
        var_arm = (
            (max(t, b) * _BETA_ZY) ** 2 * _SIGMA_X ** 2 / _N_STEPS
            + (_BETA_ZY * _SIGMA_Z) ** 2 / _N_STEPS
            + _SIGMA_Y ** 2 / _N_STEPS
        )
        sd_arm = math.sqrt(var_arm)
        median_delta_se = math.sqrt(math.pi) * sd_arm / math.sqrt(_N_SEEDS_PER_ARM)
        assert abs(deltas[idx] - expected) < 5.0 * median_delta_se, (
            f'{env_name}: median Δ_y={deltas[idx]:.4f} '
            f'expected={expected:.4f} 5σ={5.0 * median_delta_se:.4f}'
        )


def test_stratum_effect_panel_mean_vs_median_agree_under_gaussian() -> None:
    """Under Gaussian noise, mean Δ and median Δ should land
    close in population. The empirical spread between them is
    bounded by the sum of their sampling SDs."""
    cells = _build_cells()
    mean_panel = stratum_effect_panel.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        measurables=('y_mean',),
        aggregator='mean',
    )
    median_panel = stratum_effect_panel.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        measurables=('y_mean',),
        aggregator='median',
    )
    mean_deltas = mean_panel.deltas['y_mean']
    median_deltas = median_panel.deltas['y_mean']
    for idx, stratum in enumerate(mean_panel.strata):
        env_name = str(stratum[0])
        # Joint sampling SD on (mean Δ - median Δ) bounded by
        # sum of individual SDs ≈ 0.012 at env_a.
        # 4σ window: 0.048 — should comfortably hold under
        # Gaussian where the two aggregators agree in population.
        assert abs(mean_deltas[idx] - median_deltas[idx]) < 0.05, (
            f'{env_name}: mean Δ={mean_deltas[idx]:.4f} '
            f'vs median Δ={median_deltas[idx]:.4f} '
            'should agree under Gaussian noise'
        )
