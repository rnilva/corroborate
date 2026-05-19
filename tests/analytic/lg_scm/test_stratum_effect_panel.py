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

from tests.analytic.lg_scm._closed_form import y_mean_arm_sd
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
    Δ. The framework computes median(y_t) − median(y_b) on the
    per-arm seed populations (NOT median of per-seed Δs), so the
    shared-seed σ_z/σ_y cancellation that holds for the mean Δ
    only PARTIALLY holds for the median Δ — empirically the
    median picks similar-rank seeds in both arms, so the noise
    components are correlated but not identical.

    The closed-form `sqrt(π)·sd_arm/sqrt(n_seeds)` is the
    INDEPENDENT-ARMS asymptotic Δ-of-medians SE (sum of
    per-arm median SEs via independent-arm Pythagoras). It's a
    CONSERVATIVE UPPER BOUND: empirically the actual SE is
    ≈ 0.4-0.6× this value (shared-seed partial cancellation
    reduces the noise variance). The 5σ window with the
    conservative SE thus gives ~10σ-safety against the actual
    sampling distribution — bound passes by margin but still
    detects any sign / scale / pooling regression by orders of
    magnitude."""
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
        t, b = _ENV_BETAS[env_name]
        # Per-arm σ on y_mean (largest-β arm dominates the
        # conservative independent-arms bound). Δ-of-medians SE
        # under independent arms = sqrt(2 · π/2)·sd_arm/sqrt(n) =
        # sqrt(π)·sd_arm/sqrt(n). Shared-seed cancellation halves
        # this in practice.
        sd_arm = y_mean_arm_sd(
            beta_xz=max(t, b), beta_zy=_BETA_ZY,
            sigma_x=_SIGMA_X, sigma_z=_SIGMA_Z, sigma_y=_SIGMA_Y,
            n_steps=_N_STEPS,
        )
        median_delta_se_upper = (
            math.sqrt(math.pi) * sd_arm / math.sqrt(_N_SEEDS_PER_ARM)
        )
        assert abs(deltas[idx] - expected) < 5.0 * median_delta_se_upper, (
            f'{env_name}: median Δ_y={deltas[idx]:.4f} '
            f'expected={expected:.4f} '
            f'5σ(upper)={5.0 * median_delta_se_upper:.4f}'
        )


def test_stratum_effect_panel_mean_vs_median_agree_under_gaussian() -> None:
    """Under Gaussian noise, mean Δ and median Δ converge to the
    same population value. Joint sampling SD on (mean Δ −
    median Δ) is bounded by `mean_se + median_se_upper` per env
    (worst case under uncorrelated noise; both estimators on the
    same shared-seed cells have some correlation, so the actual
    SD is smaller)."""
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
        t, b = _ENV_BETAS[env_name]
        # Per-env bound = 4σ on (mean_se + median_se_upper) —
        # both estimators are unbiased for the same population Δ.
        mean_se = _expected_shared_seed_delta_se(env_name)
        sd_arm = y_mean_arm_sd(
            beta_xz=max(t, b), beta_zy=_BETA_ZY,
            sigma_x=_SIGMA_X, sigma_z=_SIGMA_Z, sigma_y=_SIGMA_Y,
            n_steps=_N_STEPS,
        )
        median_se_upper = (
            math.sqrt(math.pi) * sd_arm / math.sqrt(_N_SEEDS_PER_ARM)
        )
        bound = 4.0 * (mean_se + median_se_upper)
        assert abs(mean_deltas[idx] - median_deltas[idx]) < bound, (
            f'{env_name}: mean Δ={mean_deltas[idx]:.4f} '
            f'vs median Δ={median_deltas[idx]:.4f} '
            f'bound={bound:.4f} (4σ on sum of estimator SEs)'
        )
