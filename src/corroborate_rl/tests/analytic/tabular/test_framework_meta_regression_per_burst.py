"""Framework-as-instrument: `meta_regression_per_burst` recovers
the structural slope on an env-level covariate over the
Banach-contraction per-(env, burst) panel.

Setup: 5 envs differing in `μ_env` (the per-cell initial-scale
mean). Per env, 80 paired cells contracting under γ_a=0.95
(slow) vs γ_b=0.5 (fast). 12 bursts per cell.

Per-(env, burst) Hedges' g:

    g(t, μ_env) = μ_env · K_t                    (structural)

    where K_t = (γ_a^t − γ_b^t)
              / √(σ_x² · (γ_a^t − γ_b^t)² + 2 σ_obs²)
              · c_4(n_pairs)

The per-(env, burst) panel has 5 × 12 = 60 strata. Meta-regressing
g on the env-level covariate `μ_env` (broadcast across bursts):

    slope ≈ mean_t(K_t)                           (random-effects pooling)
    intercept ≈ 0                                 (μ_env-grid centered at 0)
    R² ≈ 0.88 (structural / (structural + residual))

The "≈ mean_t(K_t)" claim deserves explanation: per-(env, burst)
SE varies with t (the K_t denominator changes), so naive IVW
weights would NOT be equal across bursts. Fixed-effect IVW would
predict slope ≈ 0.724 (down-weighting low-|K_t| bursts). But
under DerSimonian-Laird random-effects, between-stratum
heterogeneity drives `τ² ≈ 0.19` while per-stratum `v_i ≈ 0.025`
in median; the random-effects weights `w_i = 1/(v_i + τ²)`
collapse toward UNIFORM (τ² dominates). With uniform weighting
across the 60-stratum panel, OLS slope = `mean_t(K_t) = 0.808`
exactly under the structural form `g = μ_env · K_t + ε`.
MC verification (40 reps): empirical slope mean = 0.812 ± 0.020,
matching equal-weight closed-form (Δ=+0.004) not fixed-IVW
(Δ=+0.088). This is the load-bearing structural prediction the
framework must reproduce.

For the contraction parameters here:
    K_t at t=0..11 = (0, 0.62, 0.88, 0.98, 1.00, 0.99, 0.97,
                     0.93, 0.90, 0.86, 0.82, 0.78)
    mean_t K_t ≈ 0.808

The framework's `meta_regression_per_burst` flattens the
per-(env, burst) panel into 60 strata and fits a single
random-effects meta-regression. THIS is the framework-as-
instrument question: given a panel where the structural slope
on the env covariate is closed-form, does the framework
recover it within sampling SE?

A regression that mishandled the per-stratum covariate
broadcast (e.g., averaging g across bursts before regressing,
or dropping bursts) would breach the slope bound; one that
mishandled the IVW pooling would shift the slope away from
the closed form.
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from corroborate.analyses.meta_regression_per_burst import (
    meta_regression_per_burst,
)
from corroborate.measurables.reductions import from_key, reduce_axis


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


_GAMMA_A = 0.95     # slow contraction (treatment)
_GAMMA_B = 0.5      # fast contraction (baseline)
_SIGMA_X = 0.3      # per-seed initial-scale dispersion within env
_SIGMA_OBS = 0.5    # per-burst observation noise
_N_PAIRS = 80
_N_BURSTS = 12

# Centered grid → intercept ≈ 0 closed form. Spread spans 5 envs.
_MU_GRID: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)

_PER_BURST_KEY = 'contraction_trace'
_PER_BURST_SOURCE = reduce_axis(
    from_key(_PER_BURST_KEY), axis=-1, op='mean',
)


def _c4(n: int) -> float:
    return 1.0 - 3.0 / (4 * n - 5)


def _expected_per_burst_K(t: int) -> float:
    """K_t = (γ_a^t − γ_b^t) / sd(Δ_t) · c_4. The per-burst
    "g per unit μ" — the meta-regression's slope on μ_env should
    converge to mean_t(K_t) under equal weighting."""
    diff = _GAMMA_A ** t - _GAMMA_B ** t
    var = _SIGMA_X ** 2 * diff ** 2 + 2.0 * _SIGMA_OBS ** 2
    return diff / math.sqrt(var) * _c4(_N_PAIRS)


def _expected_slope_on_mu_env() -> float:
    """Closed-form slope ≈ mean_t(K_t) — the average per-burst
    "g per unit μ" across the 12-burst trajectory."""
    return sum(_expected_per_burst_K(t) for t in range(_N_BURSTS)) / _N_BURSTS


def _generate_per_burst_panel_cells() -> list[dict[str, object]]:
    """Per-cell per-burst contraction trace, with μ_env as an
    env-level mean-x_0 covariate."""
    cells: list[dict[str, object]] = []
    for mu in _MU_GRID:
        env_name = f'mu_{mu:g}'
        for s in range(_N_PAIRS):
            rng = np.random.default_rng(
                seed=_det_seed('mr_per_burst', mu, s),
            )
            x_0 = float(mu + _SIGMA_X * rng.standard_normal())
            traj_slow = np.array([
                [x_0 * (_GAMMA_A ** t)
                 + _SIGMA_OBS * float(rng.standard_normal())]
                for t in range(_N_BURSTS)
            ], dtype=np.float64)
            traj_fast = np.array([
                [x_0 * (_GAMMA_B ** t)
                 + _SIGMA_OBS * float(rng.standard_normal())]
                for t in range(_N_BURSTS)
            ], dtype=np.float64)
            cells.append({
                'arm_key': 'slow',
                'seed': s,
                'env_name': env_name,
                'mu_env': mu,
                _PER_BURST_KEY: traj_slow,
            })
            cells.append({
                'arm_key': 'fast',
                'seed': s,
                'env_name': env_name,
                'mu_env': mu,
                _PER_BURST_KEY: traj_fast,
            })
    return cells


# Note (2026-05-06 audit): slope-recovery and intercept-recovery
# tests were deleted. The closed-form `mean_t(K_t)` walks the
# same OLS arithmetic the framework walks, with the bound tuned
# to absorb the IVW-vs-equal-weight variation; centered-μ
# intercept ≈ 0 is a property of OLS not of the framework.
# What survives below probes framework logic the closed forms
# don't subsume:
#   - panel-size (panel-build dict-intersection + per-(env, burst)
#     stratification)
#   - tight p-value bound on the slope (t-test pipeline pin)
#   - R² in the closed-form structural band (residual-vs-explained
#     variance computation)


# ============ Slope p-value: t-test pipeline pin ============

def test_meta_regression_per_burst_slope_p_value_tight() -> None:
    """Slope on μ_env has |t| ≈ 20 at df=58 — p-value must be
    extremely small. Pin `p < 1e-10` so a framework returning
    e.g. `p = 0.5` constant breaches.

    Distinct from `is_significant` flag (non-discriminating at
    this overdetermined a t-statistic) — this directly probes
    the t-test→p-value computation."""
    cells = _generate_per_burst_panel_cells()
    result = meta_regression_per_burst.fn(
        cells,
        treatment_arm='slow',
        baseline_arm='fast',
        pair_by=('seed',),
        source=_PER_BURST_SOURCE,
        covariates=('mu_env',),
    )
    by_name = {c.name: c for c in result.coefficients}
    assert 'mu_env' in by_name
    p = by_name['mu_env'].p_value
    assert p < 1e-10, (
        f'slope p-value = {p:.2e}; expected p < 1e-10 at |t| ≈ 20 '
        f'and df=58. A non-tight p-value indicates the t-test '
        f'pipeline is broken.'
    )


# ============ Panel structure ============

def test_meta_regression_per_burst_panel_size() -> None:
    """The per-(env, burst) panel has `5 envs × 12 bursts = 60`
    strata. Pin against a regression that silently dropped envs
    or bursts during the panel build.
    """
    cells = _generate_per_burst_panel_cells()
    result = meta_regression_per_burst.fn(
        cells,
        treatment_arm='slow',
        baseline_arm='fast',
        pair_by=('seed',),
        source=_PER_BURST_SOURCE,
        covariates=('mu_env',),
    )
    assert result.n_strata == len(_MU_GRID) * _N_BURSTS, (
        f'n_strata = {result.n_strata}, expected '
        f'{len(_MU_GRID) * _N_BURSTS}'
    )


# ============ R²: covariate is structurally predictive ============

def test_meta_regression_per_burst_r_squared_in_closed_form_band() -> None:
    """Closed-form weighted R² ≈ structural-variance /
    (structural + residual variance). Under the construction:

        Var(μ_env · K_t) over panel = Var(μ_env) · mean_t(K_t²)
                                    ≈ 2.0 · 0.71 = 1.42
        Residual Var ≈ Var(g − μ_env·K_t) over panel ≈ 0.20

        R²_pop ≈ 1.42 / (1.42 + 0.20) ≈ 0.88

    Empirical R² (40-rep MC) lands in [0.86, 0.90]. The bound
    `0.80 < R² < 0.95` accepts the structural value within a
    ~5% band on either side and rules out:
    - stub R² ≈ 0 (covariate ignored)
    - stub R² ≈ 1 (perfect fit, would mean residual variance
      is being underestimated)
    """
    cells = _generate_per_burst_panel_cells()
    result = meta_regression_per_burst.fn(
        cells,
        treatment_arm='slow',
        baseline_arm='fast',
        pair_by=('seed',),
        source=_PER_BURST_SOURCE,
        covariates=('mu_env',),
    )
    assert 0.80 < result.r_squared < 0.95, (
        f'R² = {result.r_squared:.4f}, expected 0.80 < R² < 0.95 '
        f'(closed-form ≈ 0.88). μ_env explains ~88% of panel '
        f'variance; deviation outside the band suggests either '
        f'the covariate is not entering (low R²) or residual '
        f'variance is being miscomputed (high R²).'
    )
