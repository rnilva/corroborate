"""Closed-form assertions on `stratum_panel` over the LG-SCM
substrate.

`stratum_panel` is the unified per-stratum panel-construction
primitive — `stratified_arm_diff_pooled`, `stratum_effect_panel`,
the `stratum_*_dowhy` family, and `cross_stratum_*_slope` all
build on it. 7 active consumers across `experiments/findings/`.
A bug in the panel construction (wrong per-arm mean computation,
wrong stratification keying, wrong Spearman pair construction)
would silently propagate through every downstream analysis.

Under the LG-SCM with shared seeds across arms, varying β_xz
per env:

    mean_y_arm(env) = β_zy · β_xz_arm · μ_x      (population)
    Δ_y(env)        = β_zy · (β_xz_t − β_xz_b) · μ_x

The per-stratum measurements computed by the panel should
recover these closed forms within sampling SD.

Within-stratum Spearman r over the UNION of both arms' cells
takes a substantively different shape — the union has a
bimodal Y (treatment cluster + baseline cluster). Pearson r
over the union is suppressed (closed form below), but Spearman ρ
holds higher because the rank transform tolerates the bimodal
y distribution better. The closed form / empirical match is:

    Cov(X, Y_union)  = β_zy · (β_xz_t + β_xz_b)/2 · σ_x²/n_steps
    Var(X_union)     = σ_x²/n_steps
    Var(Y_union)     = (β_t − β_b)² · β_zy² · μ_x² / 4
                       + (β_t² + β_b²)/2 · β_zy² · σ_x²/n_steps
                       + β_zy² · σ_z²/n_steps + σ_y²/n_steps
    r_pearson_union  ≈ 0.09  (dominated by bimodal-y variance)
    ρ_spearman_union ≈ 0.40-0.46  (rank-robust to bimodality)

Empirical Spearman ρ across 15 replicates: mean 0.42, SD 0.025
per env at this substrate (σ_z = σ_y = 0.1, β_zy = 1.5).
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from corroborate.analyses.panel.stratum_panel import stratum_panel

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_multi_env_paired_arms


_MU_X = 1.0
_SIGMA_X = 0.5
_SIGMA_Z = 0.1
_BETA_ZY = 1.5
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_SEEDS_PER_ARM = 60


# Three envs with different β_xz_t / β_xz_b combinations. Same
# treatment-vs-baseline structure across envs (treatment β larger
# than baseline β); the magnitudes differ per env so per-stratum
# means + Δs are distinct and verifiable.
_ENV_BETAS: Mapping[str, tuple[float, float]] = {
    'env_a': (0.7, 0.3),  # Δβ = 0.4
    'env_b': (0.9, 0.5),  # Δβ = 0.4
    'env_c': (0.8, 0.2),  # Δβ = 0.6
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


def _expected_y_mean(beta_xz: float) -> float:
    return _BETA_ZY * beta_xz * _MU_X


def _expected_y_delta(env: str) -> float:
    t, b = _ENV_BETAS[env]
    return _BETA_ZY * (t - b) * _MU_X


def _expected_y_arm_sd(beta_xz: float) -> float:
    """Population per-arm SD of y_mean across seeds, including
    all three variance components (X_avg + σ_z + σ_y propagation
    through β_zy)."""
    var = (
        (beta_xz * _BETA_ZY) ** 2 * _SIGMA_X ** 2 / _N_STEPS
        + (_BETA_ZY * _SIGMA_Z) ** 2 / _N_STEPS
        + _SIGMA_Y ** 2 / _N_STEPS
    )
    return math.sqrt(var)


def test_stratum_panel_strata_indexing_and_counts() -> None:
    panel = stratum_panel.fn(
        _build_cells(),
        measurables=('x_mean', 'z_mean', 'y_mean'),
        treatment_arm='treatment',
        baseline_arm='baseline',
    )
    # 3 envs, all with enough seeds in both arms
    assert len(panel.strata) == 3
    assert panel.n_strata == 3
    # Strata are tuple-keyed by stratify_by values; verify the
    # panel surfaces each env_name as a distinct stratum.
    env_names = sorted(str(s[0]) for s in panel.strata)
    assert env_names == sorted(_ENV_BETAS.keys())
    # Per-arm cell counts at each stratum = _N_SEEDS_PER_ARM
    for n_t, n_b in zip(panel.n_treatment, panel.n_baseline, strict=True):
        assert n_t == _N_SEEDS_PER_ARM
        assert n_b == _N_SEEDS_PER_ARM


def test_stratum_panel_per_arm_means_recover_closed_form() -> None:
    panel = stratum_panel.fn(
        _build_cells(),
        measurables=('y_mean',),
        treatment_arm='treatment',
        baseline_arm='baseline',
    )
    # Per-arm mean y_mean per env = β_zy · β_xz_arm · μ_x.
    # Sample mean SE = σ_arm / sqrt(n_seeds). 4σ-of-sample-mean
    # bound: with n=60 the SE on per-arm mean is ~σ_arm/7.75.
    means_t = panel.means_treatment['y_mean']
    means_b = panel.means_baseline['y_mean']
    for idx, stratum in enumerate(panel.strata):
        env_name = str(stratum[0])
        beta_t, beta_b = _ENV_BETAS[env_name]
        expected_t = _expected_y_mean(beta_t)
        expected_b = _expected_y_mean(beta_b)
        se_t = _expected_y_arm_sd(beta_t) / math.sqrt(_N_SEEDS_PER_ARM)
        se_b = _expected_y_arm_sd(beta_b) / math.sqrt(_N_SEEDS_PER_ARM)
        assert abs(means_t[idx] - expected_t) < 4.0 * se_t, (
            f'{env_name}: mean_t={means_t[idx]:.4f} '
            f'expected={expected_t:.4f} 4σ={4.0 * se_t:.4f}'
        )
        assert abs(means_b[idx] - expected_b) < 4.0 * se_b, (
            f'{env_name}: mean_b={means_b[idx]:.4f} '
            f'expected={expected_b:.4f} 4σ={4.0 * se_b:.4f}'
        )


def test_stratum_panel_deltas_recover_structural_contrast() -> None:
    panel = stratum_panel.fn(
        _build_cells(),
        measurables=('y_mean',),
        treatment_arm='treatment',
        baseline_arm='baseline',
    )
    deltas = panel.deltas['y_mean']
    for idx, stratum in enumerate(panel.strata):
        env_name = str(stratum[0])
        expected = _expected_y_delta(env_name)
        beta_t, beta_b = _ENV_BETAS[env_name]
        # Under shared seeds the σ_z and σ_y noise streams cancel
        # in the per-seed Δ_y. The Δ across seed-pooled means is
        # then (β_xz_t − β_xz_b) · β_zy · mean_seeds(X_avg). The
        # only residual sampling source is mean_seeds(X_avg) →
        # μ_x with SE = σ_x / sqrt(n_seeds · n_steps).
        se = (
            abs(beta_t - beta_b) * _BETA_ZY * _SIGMA_X
            / math.sqrt(_N_SEEDS_PER_ARM * _N_STEPS)
        )
        # 4σ window — shared-seed cancellation gives a tight SE
        # (no σ_z/σ_y noise terms in the Δ); 4σ at n_seeds=60,
        # n_steps=200 is ≈ 0.007 → bound below 1% of the
        # smallest Δ in the test (env_a Δ_y = 0.4 → 1.75% of Δ).
        assert abs(deltas[idx] - expected) < 4.0 * se, (
            f'{env_name}: Δ_y={deltas[idx]:.4f} '
            f'expected={expected:.4f} 4σ={4.0 * se:.4f}'
        )


def test_stratum_panel_within_stratum_spearman_recovers_closed_form() -> None:
    panel = stratum_panel.fn(
        _build_cells(),
        measurables=('x_mean', 'y_mean'),
        treatment_arm='treatment',
        baseline_arm='baseline',
    )
    from corroborate.analyses.panel.stratum_panel import pair_key
    key = pair_key('x_mean', 'y_mean')
    rho_per_stratum = panel.spearman_within[key]
    assert len(rho_per_stratum) == 3
    # The within-stratum Spearman pools BOTH arms' cells. Y is
    # bimodal across arms (β_xz_t vs β_xz_b cluster); Spearman ρ
    # is rank-robust to that bimodality and lands at ≈ 0.40-0.46
    # at the higher-SNR substrate (σ_z = σ_y = 0.1, β_zy = 1.5).
    # Empirical SD across 15 replicates is ≈ 0.025 per env →
    # 3σ window of width ≈ 0.075. Lower 3σ bound at the weakest
    # env (env_c, mean ρ ≈ 0.39) is ≈ 0.32 — bound at 0.25
    # leaves a 5σ safety margin while still firmly catching any
    # sign / mixing bug (which would drop ρ below 0).
    for idx, stratum in enumerate(panel.strata):
        env_name = str(stratum[0])
        rho = rho_per_stratum[idx]
        assert rho > 0.25, (
            f'{env_name}: within-stratum Spearman ρ(x, y) = '
            f'{rho:.4f} should land at ≈ 0.40-0.46 under bimodal-y '
            'union with structural-positive within-arm link'
        )
        assert rho < 1.0  # well-formedness
        assert not math.isnan(rho)
