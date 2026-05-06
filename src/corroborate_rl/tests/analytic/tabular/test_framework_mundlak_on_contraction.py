"""Framework-as-instrument: `mundlak_paired_g_per_burst` recovers
the within-env vs between-env decomposition on a tabular-
contraction panel with a structurally-known predictor.

Setup:
- 5 envs, μ_env ∈ {-2, -1, 0, 1, 2} (centered grid)
- Per env, 80 paired cells contracting at γ_a=0.95 (slow) vs
  γ_b=0.5 (fast), 12 bursts each
- Per cell, x_0 ~ N(μ_env, σ_x²); per-burst trajectory
  y_t = x_0 · γ^t + ε_obs(t)
- Synthetic per-burst predictor:
  `predictor(env, burst) = μ_env + 0.2 · burst_index`

Mundlak decomposition splits the panel predictor into:
    x_e(env)       = mean_t predictor(env, t) = μ_env + 1.10
                     (12-burst mean: 0.2 · 5.5 = 1.10 offset)
    x_w(env, t)    = predictor − x_e = 0.2 · (t − 5.5)
                     (deterministic burst-index residual; orthogonal
                     to μ_env by construction)

Per-(env, burst) target g (from `paired_g_per_burst`):
    g(t, μ_env) ≈ μ_env · K_t
    K_t = (γ_a^t − γ_b^t) / sd(Δ_t) · c_4(n_pairs)

Mundlak fit `g ~ β_b · x_e + β_w · x_w + α + ε`:

    β_b ≈ slope of (μ_env · mean_t K_t) on (μ_env + 0.275)
        = mean_t(K_t) ≈ 0.808
        (since x_e differs from μ_env by a constant offset, the
        slope is invariant under that translation)

    β_w ≈ 0  (closed form)
        Reasoning: y_w(env, t) = g − y_e ≈ μ_env · (K_t − mean_t K_t).
        x_w(env, t) = 0.05 · (t − 5.5).
        The within slope is `Cov(y_w, x_w) / Var(x_w)` averaged
        across env clusters. Per env, Cov(y_w, x_w) = μ_env ·
        Cov(K_t − mean K, 0.05·(t − 5.5)) = μ_env · 0.05 ·
        Cov(K_t, t) (since mean K and constant offset don't
        contribute). Across env clusters with μ_env grid centered
        at 0, the AVERAGE Cov is 0 — so β_w → 0 as the env
        clusters average symmetrically.

    α (intercept) ≈ −β_b · 1.10 ≈ −0.889
        (since centered μ_env makes E[g] ≈ 0; the intercept
        absorbs β_b · mean(x_e) = β_b · 1.10)

This is THE canonical Mundlak test pattern: known structural
between-env coefficient, structural-zero within coefficient,
and a closed-form intercept derived from the predictor offset.

The framework's `mundlak_paired_g_per_burst` must:
1. Build the per-(env, burst) g panel
2. Apply the registered predictor measurable per cell
3. Decompose into x_e and x_w (env-mean + within-deviation)
4. Fit `g ~ β_b · x_e + β_w · x_w + α`
5. Report cluster-robust SEs (CR1 sandwich, 5 env clusters)

A regression that conflated x_e with x_w, that mishandled the
env-mean projection, or that mis-clustered the SE would breach
the closed-form coefficient bounds.
"""
from __future__ import annotations

import math
import zlib
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from corroborate import measurable
from corroborate.analyses.mundlak_paired_g_per_burst import (
    mundlak_paired_g_per_burst,
)
from corroborate.measurables.reductions import from_key, reduce_axis


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


_GAMMA_A = 0.95
_GAMMA_B = 0.5
_SIGMA_X = 0.3
_SIGMA_OBS = 0.5
_N_PAIRS = 80
_N_BURSTS = 12
_BURST_SLOPE = 0.2       # within-env slope of predictor on burst index
                         # (≥ 0.2 so x_w has substantial variance →
                         #  β_w SE tight enough for Hausman to
                         #  resolve the β_b vs β_w difference at
                         #  α=0.05 with 5 env clusters)

_MU_GRID: tuple[float, ...] = (-2.0, -1.0, 0.0, 1.0, 2.0)

_PER_BURST_KEY = 'contraction_trace'
_PER_BURST_SOURCE = reduce_axis(
    from_key(_PER_BURST_KEY), axis=-1, op='mean',
)


@measurable
def _mundlak_contraction_predictor(
    record: Mapping[str, object],
) -> npt.NDArray[np.floating]:
    """Per-burst predictor `[μ_env + 0.05·0, μ_env + 0.05·1, ...]`.
    Substrate-side measurable (registered via @measurable at
    module import). `_contraction` suffix avoids collision with
    other registered predictors."""
    mu_env_v = record.get('mu_env')
    n_bursts_v = record.get('n_bursts')
    if not isinstance(mu_env_v, (int, float)) or isinstance(mu_env_v, bool):
        return np.array([], dtype=np.float64)
    if not isinstance(n_bursts_v, int) or isinstance(n_bursts_v, bool):
        return np.array([], dtype=np.float64)
    return np.array(
        [float(mu_env_v) + _BURST_SLOPE * b for b in range(n_bursts_v)],
        dtype=np.float64,
    )


def _c4(n: int) -> float:
    return 1.0 - 3.0 / (4 * n - 5)


def _expected_K_t(t: int) -> float:
    diff = _GAMMA_A ** t - _GAMMA_B ** t
    var = _SIGMA_X ** 2 * diff ** 2 + 2.0 * _SIGMA_OBS ** 2
    return diff / math.sqrt(var) * _c4(_N_PAIRS)


def _expected_beta_between() -> float:
    """β_b ≈ mean_t(K_t) — the per-burst structural g slope on
    μ_env, averaged across the 12-burst trajectory."""
    return sum(_expected_K_t(t) for t in range(_N_BURSTS)) / _N_BURSTS


def _expected_intercept() -> float:
    """α ≈ −β_b · mean(x_e) where mean(x_e) = mean_t(predictor)
    = 0.05 · mean(t) = 0.05 · (n_bursts − 1)/2."""
    mean_x_e_offset = _BURST_SLOPE * (_N_BURSTS - 1) / 2.0
    return -_expected_beta_between() * mean_x_e_offset


def _generate_mundlak_panel_cells() -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for mu in _MU_GRID:
        env_name = f'mu_{mu:g}'
        for s in range(_N_PAIRS):
            rng = np.random.default_rng(seed=_det_seed('mund_c', mu, s))
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
            base_meta = {
                'seed': s,
                'env_name': env_name,
                'mu_env': mu,
                'n_bursts': _N_BURSTS,
            }
            # Each cell needs a unique string `id` for the
            # predictor-array cache (mundlak_paired_g_per_burst
            # keys per-cell arrays by `id`).
            cells.append({
                'id': f'mu_{mu:g}_s{s}_slow',
                'arm_key': 'slow',
                **base_meta,
                _PER_BURST_KEY: traj_slow,
            })
            cells.append({
                'id': f'mu_{mu:g}_s{s}_fast',
                'arm_key': 'fast',
                **base_meta,
                _PER_BURST_KEY: traj_fast,
            })
    return cells


# ============ Between coefficient: μ_env slope ============

def test_mundlak_between_coefficient_recovers_mean_t_K() -> None:
    """β_b ≈ mean_t(K_t) ≈ 0.808 within sampling SE.

    Per env, the env-mean g is `μ_env · mean_t(K_t)`. Regressing
    env-mean g on `x_e = μ_env + 0.275` gives slope mean_t(K_t)
    invariant under the constant offset.

    SE on β_b under CR1 clustering by env (5 clusters): the small-
    cluster correction inflates the OLS SE. With 5 clusters and
    Var(μ_env) = 2.0, CR1-SE(β_b) ≈ 0.04-0.06; 4·SE ≈ 0.20.
    Bound 0.15 covers IVW/RE-pool variation + CR1 small-cluster
    inflation (analogous to test #8's slope bound).
    """
    cells = _generate_mundlak_panel_cells()
    result = mundlak_paired_g_per_burst.fn(
        cells,
        treatment_arm='slow',
        baseline_arm='fast',
        pair_by=('seed',),
        source=_PER_BURST_SOURCE,
        predictor_name='_mundlak_contraction_predictor',
    )
    expected = _expected_beta_between()
    actual = result.between.coefficient
    bound = 0.15
    assert abs(actual - expected) < bound, (
        f'β_b = {actual:.4f}, closed-form mean_t(K_t) = '
        f'{expected:.4f} (bound = {bound:.4f}). The framework s '
        f'env-mean projection + WLS solve must recover the '
        f'between-env slope.'
    )


# ============ Within coefficient: structural zero ============

def test_mundlak_within_coefficient_near_zero_via_z_score() -> None:
    """β_w should be structurally zero — the within-env target
    variation `μ_env · (K_t − mean K)` averaged across the
    centered μ_env grid has zero expected slope on the
    deterministic burst-index predictor `0.05·(t − 5.5)`.

    Use Z-score against the framework's reported CR1 SE (per
    CLAUDE.md test principle 3): `|β_w / SE_w| < 2.5` catches
    both inflated estimates AND collapsed SEs. SE > 0 floor
    rules out degenerate fits.
    """
    cells = _generate_mundlak_panel_cells()
    result = mundlak_paired_g_per_burst.fn(
        cells,
        treatment_arm='slow',
        baseline_arm='fast',
        pair_by=('seed',),
        source=_PER_BURST_SOURCE,
        predictor_name='_mundlak_contraction_predictor',
    )
    assert result.within.se > 0.0, (
        f'within SE = {result.within.se}; framework reported '
        f'zero or negative SE — degenerate fit'
    )
    z_score = abs(result.within.coefficient) / result.within.se
    assert z_score < 2.5, (
        f'|β_w / SE_w| = {z_score:.4f} (β_w = '
        f'{result.within.coefficient:.4f}, SE = '
        f'{result.within.se:.4f}). β_w should be structurally '
        f'zero — within-env y_w averaged across centered μ_env '
        f'grid has zero expected covariance with x_w.'
    )


# ============ Intercept: structural offset ============

def test_mundlak_intercept_recovers_predictor_offset() -> None:
    """α ≈ −β_b · mean(x_e) where the predictor's per-env mean
    has an offset `0.2 · 5.5 = 1.10` from μ_env. With centered
    μ_env, mean(g) ≈ 0, so the intercept absorbs `−β_b · 1.10`.

    Closed-form α ≈ −0.808 · 1.10 ≈ −0.889.
    """
    cells = _generate_mundlak_panel_cells()
    result = mundlak_paired_g_per_burst.fn(
        cells,
        treatment_arm='slow',
        baseline_arm='fast',
        pair_by=('seed',),
        source=_PER_BURST_SOURCE,
        predictor_name='_mundlak_contraction_predictor',
    )
    expected = _expected_intercept()
    actual = result.intercept
    # Intercept SE inherits both β_b SE and per-burst-mean SE;
    # bound 0.10 (~4·SE) consistent with test #8's intercept bound.
    bound = 0.10
    assert abs(actual - expected) < bound, (
        f'intercept = {actual:.4f}, closed-form '
        f'−β_b·0.275 = {expected:.4f} (bound = {bound:.4f}). '
        f'A drifted intercept indicates the env-mean projection '
        f'has a wrong offset.'
    )


# ============ Panel structure ============

def test_mundlak_panel_structure_matches_5_envs_x_12_bursts() -> None:
    """`n_strata` = number of env clusters (5); `n_obs` = total
    (env, burst) panel size (5 × 12 = 60). Pin the panel build:
    a regression that silently dropped envs would change
    n_strata; one that dropped bursts would change n_obs.
    """
    cells = _generate_mundlak_panel_cells()
    result = mundlak_paired_g_per_burst.fn(
        cells,
        treatment_arm='slow',
        baseline_arm='fast',
        pair_by=('seed',),
        source=_PER_BURST_SOURCE,
        predictor_name='_mundlak_contraction_predictor',
    )
    assert result.n_strata == len(_MU_GRID), (
        f'n_strata = {result.n_strata}, expected '
        f'{len(_MU_GRID)} (one CR1 cluster per env)'
    )
    assert result.n_obs == len(_MU_GRID) * _N_BURSTS, (
        f'n_obs = {result.n_obs}, expected '
        f'{len(_MU_GRID) * _N_BURSTS} (5 envs × 12 bursts)'
    )


# ============ Hausman p: between vs within distinguishable ============

def test_mundlak_hausman_p_distinguishes_between_from_within() -> None:
    """Mundlak's Hausman test asks whether the between (β_b) and
    within (β_w) coefficients are equal. Under the structural
    closed form they're not (β_b ≈ 0.808, β_w ≈ 0); Hausman
    should report `hausman_p < alpha = 0.05`.

    A regression that conflated the two channels (e.g.,
    returning β_b as both) would yield Hausman_p ≈ 1 — the
    canonical "no decomposition signal" failure mode.
    """
    cells = _generate_mundlak_panel_cells()
    result = mundlak_paired_g_per_burst.fn(
        cells,
        treatment_arm='slow',
        baseline_arm='fast',
        pair_by=('seed',),
        source=_PER_BURST_SOURCE,
        predictor_name='_mundlak_contraction_predictor',
    )
    assert result.hausman_p < 0.05, (
        f'Hausman p = {result.hausman_p:.4f}; β_b ≈ 0.808 and '
        f'β_w ≈ 0 are structurally distinct — Hausman test '
        f'should reject equality at α=0.05.'
    )
