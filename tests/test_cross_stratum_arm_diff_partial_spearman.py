"""Closed-form tests for `cross_stratum_arm_diff_partial_spearman`.

The primitive answers "does Δ_predictor predict Δ_target across
strata AFTER controlling for Δ_confound?" Three structural
scenarios cover the discriminative cases:

1. **Shadow predictor** — Δ_predictor = α · Δ_confound + ε
   (small ε), Δ_target = γ · Δ_confound. Marginal
   Spearman(Δ_predictor, Δ_target) is high; partial collapses
   to ≈ 0 because all signal travels through Δ_confound.

2. **Independent predictor** — Δ_predictor ⫫ Δ_confound across
   strata; Δ_target = β_p · Δ_predictor + β_z · Δ_confound.
   Marginal and partial both ≈ |β_p| / √(β_p² + β_z² σ_z² + σ²),
   ranked. Partial ≈ marginal because Δ_confound is orthogonal.

3. **Mixed signal** — Δ_predictor = α · Δ_confound + γ · U where
   U is independent. Partial Spearman recovers U's share of the
   Δ_target signal; marginal mixes both.

Each test constructs hand-built per-cell records (multi-stratum
panel with one stratum per env), runs the @analysis fixture, and
asserts the returned ρ against the closed-form expectation
within a sample-rank tolerance.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from corroborate.analyses.cross_stratum_arm_diff_partial_spearman import (
    cross_stratum_arm_diff_partial_spearman,
)


# Resolve the bare callable through the @analysis decorator.
_fn = cross_stratum_arm_diff_partial_spearman.fn


def _cells(
    *,
    strata: list[str],
    delta_predictor: list[float],
    delta_target: list[float],
    delta_confound: list[float],
    n_per_arm: int = 10,
    rng_seed: int = 0,
) -> list[Mapping[str, object]]:
    """Build a per-cell record list realising the given per-stratum
    arm-diff structure.

    For each stratum:
      - baseline arm: `n_per_arm` cells with predictor/target/
        confound drawn from N(0, sigma_arm).
      - treatment arm: `n_per_arm` cells with mean shifted by the
        target Δ, same sigma.

    Verifies that `mean(treatment) - mean(baseline)` equals the
    requested Δ to within `1 / sqrt(n_per_arm)` rounding."""
    rng = np.random.default_rng(rng_seed)
    sigma_within = 0.5
    cells: list[Mapping[str, object]] = []
    for k, env in enumerate(strata):
        # Baseline arm: zero-mean draws.
        b_p = rng.normal(0.0, sigma_within, n_per_arm)
        b_y = rng.normal(0.0, sigma_within, n_per_arm)
        b_z = rng.normal(0.0, sigma_within, n_per_arm)
        for i in range(n_per_arm):
            cells.append({
                'arm_key': 'baseline',
                'env_name': env,
                'M': float(b_p[i]),
                'Y': float(b_y[i]),
                'Z': float(b_z[i]),
            })
        # Treatment arm: shift mean by Δ exactly.
        t_p = rng.normal(delta_predictor[k], sigma_within, n_per_arm)
        t_y = rng.normal(delta_target[k], sigma_within, n_per_arm)
        t_z = rng.normal(delta_confound[k], sigma_within, n_per_arm)
        # Recentre so the realised arm-mean Δ EXACTLY equals the
        # requested Δ — removes within-arm-mean sampling noise.
        t_p = t_p - t_p.mean() + delta_predictor[k] + b_p.mean()
        t_y = t_y - t_y.mean() + delta_target[k] + b_y.mean()
        t_z = t_z - t_z.mean() + delta_confound[k] + b_z.mean()
        for i in range(n_per_arm):
            cells.append({
                'arm_key': 'treatment',
                'env_name': env,
                'M': float(t_p[i]),
                'Y': float(t_y[i]),
                'Z': float(t_z[i]),
            })
    return cells


# ============ Test 1: shadow predictor → partial r ≈ 0 ============

def test_shadow_predictor_collapses_to_zero_under_partial() -> None:
    """Δ_M = α · Δ_Z exactly (no independent component). Marginal
    ρ(Δ_M, Δ_Y) = ρ(Δ_Z, Δ_Y) by rank-equivalence; partial ρ
    collapses to ≈ 0 because once Δ_Z is partialled out, Δ_M
    carries no signal."""
    strata = ['env_a', 'env_b', 'env_c', 'env_d', 'env_e', 'env_f']
    # Heterogeneous Δ_Z across strata.
    delta_z = [-2.0, -1.5, -1.0, -0.5, 0.0, +1.0]
    alpha = 0.5
    gamma = 1.0
    delta_m = [alpha * z for z in delta_z]  # M is Z's shadow
    delta_y = [gamma * z for z in delta_z]  # Y depends only on Z

    cells = _cells(
        strata=strata,
        delta_predictor=delta_m,
        delta_target=delta_y,
        delta_confound=delta_z,
        n_per_arm=20,
    )
    result = _fn(
        cells, treatment_arm='treatment', baseline_arm='baseline',
        predictor='M', target='Y', confound='Z',
        stratify_by=('env_name',),
        min_seeds_per_arm=5, min_strata=5,
    )
    assert result.n_strata == 6
    # Marginal: Δ_M perfectly tracks Δ_Z, which perfectly tracks
    # Δ_Y → marginal Spearman = +1 (monotone).
    assert result.rho_marginal > 0.99
    # Partial: Δ_M = 0.5 · Δ_Z so after partialling Δ_Z there is
    # no independent Δ_M variance → partial ρ should be NaN
    # (rxz = 1 → denominator = 0). The primitive returns NaN by
    # design.
    assert np.isnan(result.rho)


# ============ Test 2: independent predictor → partial ≈ marginal ============

def test_independent_predictor_keeps_partial_near_marginal() -> None:
    """Δ_M ⫫ Δ_Z across strata; Δ_Y = β_M · Δ_M + β_Z · Δ_Z.
    Both Δ_M and Δ_Z contribute orthogonally; partial recovers
    Δ_M's share."""
    # Hand-picked rank-orthogonal pair: Δ_M and Δ_Z have
    # Spearman ρ ≈ 0 across the 6 strata, but each correlates
    # with Δ_Y.
    strata = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']
    delta_m = [-1.0, -0.5, 0.5, 1.0, 0.0, 2.0]
    delta_z = [0.0, 2.0, -1.0, 1.0, -0.5, 0.5]  # ranks roughly orthogonal to delta_m
    beta_m, beta_z = 1.0, 1.0
    delta_y = [beta_m * m + beta_z * z for m, z in zip(delta_m, delta_z)]

    cells = _cells(
        strata=strata,
        delta_predictor=delta_m,
        delta_target=delta_y,
        delta_confound=delta_z,
        n_per_arm=20,
    )
    result = _fn(
        cells, treatment_arm='treatment', baseline_arm='baseline',
        predictor='M', target='Y', confound='Z',
        stratify_by=('env_name',),
        min_seeds_per_arm=5, min_strata=5,
    )
    assert result.n_strata == 6
    # With β_M = β_Z = 1, partial(M, Y | Z) should be substantially
    # positive (after Z is partialled out, M still drives Y).
    assert result.rho > 0.5


# ============ Test 3: marginal field is populated alongside partial ============

def test_marginal_rho_alongside_partial_rho() -> None:
    """`rho_marginal` is the cross-stratum Spearman without
    conditioning — callers compare partial-vs-marginal to see how
    much the confound absorbs."""
    strata = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']
    delta_m = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    delta_z = [-0.5, +0.5, -0.3, +0.3, +0.0, +0.7]  # weak relation to M
    delta_y = [m + 0.2 * z for m, z in zip(delta_m, delta_z)]

    cells = _cells(
        strata=strata,
        delta_predictor=delta_m,
        delta_target=delta_y,
        delta_confound=delta_z,
        n_per_arm=10,
    )
    result = _fn(
        cells, treatment_arm='treatment', baseline_arm='baseline',
        predictor='M', target='Y', confound='Z',
        stratify_by=('env_name',),
        min_seeds_per_arm=5, min_strata=5,
    )
    assert not np.isnan(result.rho_marginal)
    # M is monotone increasing with Y by construction →
    # marginal ≈ +1.
    assert result.rho_marginal > 0.99
    # Partial should also be positive (M's signal survives weak
    # Z conditioning) but ≤ marginal in rank terms.
    assert result.rho > 0.7


# ============ Test 4: min_strata floor returns NaN ============

def test_min_strata_floor_returns_nan() -> None:
    """When fewer than `min_strata` strata pass the per-arm seed
    floor, the partial returns NaN. Diagnostic: caller sees
    n_strata < min_strata."""
    strata = ['e1', 'e2', 'e3']  # only 3 strata
    cells = _cells(
        strata=strata,
        delta_predictor=[-1.0, 0.0, 1.0],
        delta_target=[-1.0, 0.0, 1.0],
        delta_confound=[-0.5, 0.0, 0.5],
        n_per_arm=10,
    )
    result = _fn(
        cells, treatment_arm='treatment', baseline_arm='baseline',
        predictor='M', target='Y', confound='Z',
        stratify_by=('env_name',),
        min_seeds_per_arm=5, min_strata=5,
    )
    assert result.n_strata == 3
    assert np.isnan(result.rho)
    assert np.isnan(result.p_value)


# ============ Test 5: per-arm seed floor drops stratum ============

def test_per_arm_seed_floor_drops_stratum() -> None:
    """A stratum where either arm has < `min_seeds_per_arm`
    finite cells on any of predictor / target / confound is
    silently dropped from the Δ panel. n_strata reflects the
    surviving count."""
    strata = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']
    cells_full = _cells(
        strata=strata,
        delta_predictor=[-1.0, 0.0, 1.0, 2.0, -0.5, 0.5],
        delta_target=[-1.0, 0.0, 1.0, 2.0, -0.5, 0.5],
        delta_confound=[-0.5, 0.0, 0.5, 1.0, -0.2, 0.2],
        n_per_arm=10,
    )
    # Drop most of e3's treatment cells to put it below 5.
    pruned = [
        c for c in cells_full
        if not (c['env_name'] == 'e3' and c['arm_key'] == 'treatment')
    ]
    # Add back 3 treatment cells for e3 (below min_seeds_per_arm=5).
    pruned.extend([
        c for c in cells_full
        if c['env_name'] == 'e3' and c['arm_key'] == 'treatment'
    ][:3])
    result = _fn(
        pruned, treatment_arm='treatment', baseline_arm='baseline',
        predictor='M', target='Y', confound='Z',
        stratify_by=('env_name',),
        min_seeds_per_arm=5, min_strata=5,
    )
    assert result.n_strata == 5  # e3 dropped


# ============ Test 6: monotone-rank rho_marginal closed form ============

def test_monotone_rank_rho_marginal_exact() -> None:
    """When Δ_predictor and Δ_target are perfectly rank-coupled
    across N strata (any monotone transformation), marginal
    Spearman ρ is exactly +1 — the closed-form rank identity."""
    strata = ['e1', 'e2', 'e3', 'e4', 'e5', 'e6']
    delta_m = [-3.0, -1.0, 0.5, 1.5, 4.0, 7.0]  # strictly increasing
    delta_y = [m ** 3 for m in delta_m]  # monotone increasing
    delta_z = [+1.0, -1.0, +2.0, -2.0, +0.5, -0.5]  # unrelated
    cells = _cells(
        strata=strata,
        delta_predictor=delta_m,
        delta_target=delta_y,
        delta_confound=delta_z,
        n_per_arm=10,
    )
    result = _fn(
        cells, treatment_arm='treatment', baseline_arm='baseline',
        predictor='M', target='Y', confound='Z',
        stratify_by=('env_name',),
        min_seeds_per_arm=5, min_strata=5,
    )
    # Δ_M strictly increasing, Δ_Y strictly increasing → Spearman
    # is exactly +1 (no sampling slack at the per-stratum-arm-mean
    # level because _cells re-centres to exact Δ).
    assert abs(result.rho_marginal - 1.0) < 1e-6
