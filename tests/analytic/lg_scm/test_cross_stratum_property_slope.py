"""Closed-form assertions on `cross_stratum_property_slope` over
the LG-SCM substrate.

`cross_stratum_property_slope` computes Spearman ρ across strata
of (per-stratum scalar covariate, per-stratum Cohen's d on a
target measurable). 5 active bridge consumers across
`experiments/findings/`.

Under the LG-SCM with shared seeds and varying μ_x per env (β_xz
held constant across all envs):

    Δ_y(env)            = (β_xz_t − β_xz_b) · β_zy · μ_x(env)
    pooled_sd_y(env)    ≈ structural-noise-only (μ_x-independent
                           — the σ_x, σ_z, σ_y propagation gives
                           the SD a μ_x-free closed form)

→ Cohen's d(env) = Δ_y / pooled_sd ∝ μ_x

So a μ_x covariate must correlate +1 with Cohen's d across envs
(monotone increasing). Spearman ρ should be ≈ +1.0.
"""
from __future__ import annotations

import math
from collections.abc import Mapping

from corroborate.analyses.link.cross_stratum_property_slope import (
    DerivedCovariateSpec,
    cross_stratum_property_slope,
)

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_multi_env_paired_arms


_SIGMA_X = 0.5
_SIGMA_Z = 0.1
_BETA_ZY = 1.5
_SIGMA_Y = 0.1
_N_STEPS = 200
_N_SEEDS_PER_ARM = 30
_BETA_XZ_T = 0.7
_BETA_XZ_B = 0.3


# Eight envs with monotone-increasing μ_x. min_strata default is
# 8, so we need exactly 8 to test the floor + use Spearman's full
# discrimination power.
_ENV_MU_X: Mapping[str, float] = {
    f'env_{chr(ord("a") + i)}': 0.5 + 0.25 * i
    for i in range(8)
}


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
    envs = {
        env: (
            _scm(mu_x, _BETA_XZ_T), _scm(mu_x, _BETA_XZ_B),
        )
        for env, mu_x in _ENV_MU_X.items()
    }
    rows = run_multi_env_paired_arms(
        envs=envs, seeds=tuple(range(_N_SEEDS_PER_ARM)),
    )
    return [r.as_dict() for r in rows]


def test_cross_stratum_property_slope_monotone_in_mu_x() -> None:
    """Covariate = μ_x, target = y_mean. Cohen's d on y_mean
    scales linearly with μ_x (Δ_y ∝ μ_x; pooled SD μ_x-
    independent at fixed β_xz). Spearman ρ across strata should
    be ≈ +1.0 (no ties — μ_x values are strictly monotone)."""
    cells = _build_cells()
    # Key type `object`: the analysis accepts heterogeneous stratum
    # keys (`Mapping[object, ...]`), and Mapping is invariant in its
    # key parameter.
    covariates_per_key: dict[object, dict[str, float]] = {
        env: {'mu_x_cov': mu_x}
        for env, mu_x in _ENV_MU_X.items()
    }
    result = cross_stratum_property_slope.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        source='y_mean',
        covariate_name='mu_x_cov',
        covariates_per_key=covariates_per_key,
        covariate_key_field='env_name',
        # scope_predictor defaults to 'jensen_gap' which doesn't
        # exist in LG-SCM cells. Use z_mean (always positive in
        # LG-SCM at β_xz_b · μ_x ≥ 0.3·0.5 = 0.15 > 0). Set the
        # floor to 0 so all strata pass.
        scope_predictor='z_mean',
        min_baseline_predictor=0.0,
        min_strata=8,
    )
    assert result.n_strata == 8
    # Cohen's d ∝ μ_x with strictly monotone μ_x → Spearman ρ
    # exactly +1 in population. Empirical noise comes only from
    # the per-stratum Cohen's d sample-SD CV ≈ 1/sqrt(2(n-1)) ≈
    # 9% at n_per_arm=30; that doesn't break the rank order on
    # 8 well-separated covariates. ρ should be exactly +1 in
    # virtually every realisation. 0.05 below 1 absorbs the
    # rare case where two adjacent d values swap rank from
    # sampling noise.
    assert result.rho > 0.95, (
        f'rho={result.rho:.4f} — expected ≈ +1.0 under '
        'monotone Cohen\'s d in μ_x'
    )
    assert result.p_value < 0.01
    # Covariate values match the input map exactly (verifies the
    # env→covariate threading inside the primitive, NOT a pure
    # readback of an input arg).
    assert sorted(result.covariate_values) == sorted(_ENV_MU_X.values())
    # All Cohen's d should be positive (treatment β > baseline β
    # uniformly) and monotone increasing in μ_x.
    for d in result.cohen_d_per_stratum:
        assert d > 0, f'cohen_d={d} should be > 0 — treatment helps'


def test_cross_stratum_property_slope_below_min_strata_returns_nan() -> None:
    """Setting min_strata above n_envs forces NaN return."""
    cells = _build_cells()
    # Key type `object`: the analysis accepts heterogeneous stratum
    # keys (`Mapping[object, ...]`), and Mapping is invariant in its
    # key parameter.
    covariates_per_key: dict[object, dict[str, float]] = {
        env: {'mu_x_cov': mu_x}
        for env, mu_x in _ENV_MU_X.items()
    }
    result = cross_stratum_property_slope.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        source='y_mean',
        covariate_name='mu_x_cov',
        covariates_per_key=covariates_per_key,
        covariate_key_field='env_name',
        scope_predictor='z_mean',
        min_baseline_predictor=0.0,
        min_strata=100,
    )
    assert math.isnan(result.rho)
    assert math.isnan(result.p_value)
    assert result.n_strata == 8


def test_cross_stratum_property_slope_missing_covariates_dropped() -> None:
    """Strata with no covariate entry drop before the slope
    calculation; this catches a regression where the primitive
    might fill missing covariates with NaN or zero."""
    cells = _build_cells()
    # Drop 3 envs from the covariate map → only 5 strata survive
    # → < min_strata=8 → NaN.
    partial_covariates: dict[object, dict[str, float]] = {
        env: {'mu_x_cov': mu_x}
        for env, mu_x in list(_ENV_MU_X.items())[:5]
    }
    result = cross_stratum_property_slope.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        source='y_mean',
        covariate_name='mu_x_cov',
        covariates_per_key=partial_covariates,
        covariate_key_field='env_name',
        scope_predictor='z_mean',
        min_baseline_predictor=0.0,
        min_strata=8,
    )
    assert result.n_strata == 5
    assert math.isnan(result.rho)


def test_cross_stratum_property_slope_dataframe_input_identical_to_cells() -> None:
    """Canonical-input invariant: Iterable[Mapping] and
    pl.DataFrame inputs produce the same result. Guards against
    the DataFrame branch diverging from cells branch."""
    import polars as pl

    cells = _build_cells()
    # Key type `object`: the analysis accepts heterogeneous stratum
    # keys (`Mapping[object, ...]`), and Mapping is invariant in its
    # key parameter.
    covariates_per_key: dict[object, dict[str, float]] = {
        env: {'mu_x_cov': mu_x}
        for env, mu_x in _ENV_MU_X.items()
    }
    # A closure rather than a shared kwargs dict: the single
    # spelling of the arguments stays statically checked against
    # the analysis signature (a dict would erase them to
    # `dict[str, object]`).
    def run(
        cells_in: pl.DataFrame | list[Mapping[str, object]],
    ):
        return cross_stratum_property_slope.fn(
            cells_in,
            treatment_arm='treatment',
            baseline_arm='baseline',
            source='y_mean',
            covariate_name='mu_x_cov',
            covariates_per_key=covariates_per_key,
            covariate_key_field='env_name',
            scope_predictor='z_mean',
            min_baseline_predictor=0.0,
            min_strata=8,
        )

    result_cells = run(cells)
    result_panel = run(pl.DataFrame(cells))
    assert result_panel.n_strata == result_cells.n_strata
    assert result_panel.rho == result_cells.rho
    assert result_panel.p_value == result_cells.p_value
    assert (
        result_panel.cohen_d_per_stratum == result_cells.cohen_d_per_stratum
    )


def test_cross_stratum_property_slope_honours_custom_arm_field() -> None:
    """Both the effect panel and an arm-filtered derived covariate
    use the caller-selected arm column."""
    cells: list[Mapping[str, object]] = []
    for cell in _build_cells():
        row = dict(cell)
        row['contrast_arm'] = row.pop('arm_key')
        cells.append(row)
    result = cross_stratum_property_slope.fn(
        cells,
        treatment_arm='treatment',
        baseline_arm='baseline',
        arm_field='contrast_arm',
        source='y_mean',
        covariate_name='mu_x_cov',
        derived_covariate=DerivedCovariateSpec(
            column='mu_x', aggregator='mean', arm_filter='baseline',
        ),
        covariate_key_field='env_name',
        scope_predictor='z_mean',
        min_baseline_predictor=0.0,
        min_strata=8,
    )
    assert result.n_strata == len(_ENV_MU_X)
    assert result.rho > 0.95
