"""Closed-form analytic assertions on `dynamic_pc_adjacency`
over the LG-SCM substrate.

Builds a multi-burst, two-arm panel via `simulate_phased` (one
cell per (arm, seed); each cell carries an array of length
`n_bursts` for `y_mean_per_burst` and `z_mean_per_burst`). The
arm intervention is on `beta_xz` (treatment 0.6, baseline 0.4) so
the per-burst marginal ρ(arm, Ȳ_b) is non-zero with closed-form
prediction. In the LG-SCM chain `arm → Z → Y` the per-burst
mediator Z̄_b d-separates arm from Ȳ_b in population — so the
PC primitive should report `mediator_dseparates[b] == True` at
most bursts (within α-controlled type-I rate on the partial CI
test).

Substrate parameters at the test point:
  mu_x = 1.0, sigma_x = 0.5, sigma_z = 0.4, sigma_y = 0.4,
  beta_zy = 1.0, n_steps = 20, n_seeds_per_arm = 80
  beta_xz_T = 0.6, beta_xz_B = 0.4
  Δ_μ = (0.6 − 0.4) · 1.0 · 1.0 = 0.2 (per-burst-mean Ȳ shift)
  n per burst = 2 · 80 = 160

Closed-form expectations:

1. **Marginal edge always present**. Point-biserial Pearson r at
   the substrate (from
   `test_dynamic_partial_spearman.py`'s docstring derivation): r ≈
   0.585. At n=160 the Fisher-z CI test rejects null at α=0.05 with
   power ≈ 1.0 (z-stat ≈ 0.585·sqrt(157)·(1−0.585²)⁻¹ ≈ 11.1 →
   p ≈ 1e-28). So `n_bursts_marginal_edge` MUST equal n_bursts.

2. **Mediator d-separates at every burst**. Partial ρ(arm, Ȳ |
   Z̄) population value = 0 (Z d-separates arm from Y in the
   chain). Type-I rate of the partial CI test = α = 0.05. With
   n_bursts=3, expected false-positives on partial CI: ~0.15. We
   assert `n_bursts_mediator_dseparates >= 2` of 3 (allow 1
   false-positive per the type-I rate).

3. **Direct edge rarely fires**. Symmetric: `n_bursts_direct_edge
   <= 1` (the same 1-burst slack for partial-CI type-I).

The test ALSO sanity-checks aggregation_status == CONSISTENT_
DIRECTION (all bursts same sign at moderate ρ) — driven by the
shared `_classify_status` on the marginal-ρ trajectory.

Layer-B per the test design: anchors the primitive against
substrate-parameter-derived closed-form expectations rather than
just self-consistency on synthetic input.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import polars as pl

from corroborate.analyses.dynamic_mediation import (
    TimeAggregationStatus,
    dynamic_pc_adjacency,
)

from tests.analytic.lg_scm.composition import (
    LinearGaussianSCM,
    simulate_phased,
)


_MU_X = 1.0
_SIGMA_X = 0.5
_SIGMA_Z = 0.4
_BETA_ZY = 1.0
_SIGMA_Y = 0.4
_N_STEPS = 20
_N_BURSTS = 3
_N_SEEDS_PER_ARM = 80
_BETA_XZ_T = 0.6
_BETA_XZ_B = 0.4


def _scm(beta_xz: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=_MU_X, sigma_x=_SIGMA_X,
        beta_xz=beta_xz, sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _build_panel() -> pl.DataFrame:
    """Two-arm phased panel with `y_mean_per_burst` and
    `z_mean_per_burst` as `List(Float64)` columns of length
    `n_bursts`. Mirrors the partial-Spearman analytic test's
    panel construction so the two layer-B tests stay sibling-
    consistent."""
    scm_t = tuple(_scm(_BETA_XZ_T) for _ in range(_N_BURSTS))
    scm_b = tuple(_scm(_BETA_XZ_B) for _ in range(_N_BURSTS))
    cells: list[Mapping[str, object]] = []
    for s in range(_N_SEEDS_PER_ARM):
        obs_t = simulate_phased(scm_t, seed=s)
        cells.append({
            'env_name': 'lg_scm',
            'gamma': 0.99,
            'arm_key': 'treatment',
            'seed': s,
            'y_mean_per_burst': list(obs_t.y_mean_per_burst),
            'z_mean_per_burst': list(obs_t.z_mean_per_burst),
        })
        obs_b = simulate_phased(scm_b, seed=s)
        cells.append({
            'env_name': 'lg_scm',
            'gamma': 0.99,
            'arm_key': 'baseline',
            'seed': s,
            'y_mean_per_burst': list(obs_b.y_mean_per_burst),
            'z_mean_per_burst': list(obs_b.z_mean_per_burst),
        })
    return pl.DataFrame(cells)


def _stratum_key(
    keys: Sequence[object],
) -> tuple[object, ...]:
    return tuple(keys)


def test_lg_scm_marginal_edge_at_every_burst() -> None:
    """The marginal arm→outcome edge is present at every burst.
    At n=160 with planted r ≈ 0.585, the CI test's null rejection
    is overwhelming (z ≈ 11, p ≈ 1e-28) — `marginal_edge[b]` MUST
    fire at every burst regardless of seed variance."""
    df = _build_panel()
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='z_mean_per_burst',
        outcome_per_burst='y_mean_per_burst',
        stratify_by=('env_name', 'gamma'),
    )
    key = _stratum_key(('lg_scm', 0.99))
    assert key in results
    result = results[key]
    assert result.n_bursts == _N_BURSTS
    # Marginal p-values must all be tiny (CI test power ≈ 1.0).
    for b, p in enumerate(result.p_marginal):
        assert p < 1e-10, (
            f'burst {b}: marginal p={p!r} should be ≪ α at n=160 '
            f'planted r ≈ 0.585'
        )
    assert result.n_bursts_marginal_edge == _N_BURSTS


def test_lg_scm_mediator_d_separates_at_most_bursts() -> None:
    """Z̄_b d-separates arm from Ȳ_b at every burst in population
    → partial ρ ≈ 0 → conditional CI test rejects null at α=0.05
    with type-I rate exactly α. Across 3 bursts, expected false-
    positives: 0.15. Bound: `n_bursts_mediator_dseparates >= 2`
    (allow 1-burst slack for partial-CI type-I on the saturating-
    marginal end of the closed-form partial Spearman).

    The Fisher-z partial-CI's standard error at n=160 under
    d-separation is ≈ 1/sqrt(156) ≈ 0.080; the closed-form
    denominator inflation at r_xz ≈ 0.72, r_yz ≈ 0.82 (per the
    sibling test's derivation) is √(1/0.158) ≈ 2.5 → SE_partial
    ≈ 0.20. Under H0 (population partial = 0) the z-statistic
    has SD = 1, so the type-I rate IS exactly α=0.05 by
    construction. Per-burst false-positive prob = 0.05; binomial
    P(X ≥ 2 | n=3, p=0.05) = 0.0072. So our `>= 2` bound is met
    with probability > 0.99 across reasonable seeds."""
    df = _build_panel()
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='z_mean_per_burst',
        outcome_per_burst='y_mean_per_burst',
        stratify_by=('env_name', 'gamma'),
    )
    key = _stratum_key(('lg_scm', 0.99))
    result = results[key]
    # Each partial ρ should be small (close to 0). Bound 0.35
    # matches the partial-Spearman sibling's per-burst bound (2.5σ
    # on the SE-inflated partial Spearman).
    for b, rho in enumerate(result.rho_partial):
        assert abs(rho) < 0.35, (
            f'burst {b}: rho_partial={rho:.4f} should be ≈ 0 under '
            f'd-separation; SE ≈ 0.20 → 2.5σ = 0.50, we use 0.35 '
            f'as a tighter empirical bound'
        )
    # Edge classification: mediator d-separates at most bursts.
    assert result.n_bursts_mediator_dseparates >= 2, (
        f'expected mediator_dseparates >= 2/{_N_BURSTS}; got '
        f'{result.n_bursts_mediator_dseparates}. '
        f'p_conditional={result.p_conditional}'
    )
    assert result.n_bursts_direct_edge <= 1, (
        f'expected direct_edge <= 1/{_N_BURSTS} (partial-CI type-I '
        f'slack); got {result.n_bursts_direct_edge}. '
        f'p_conditional={result.p_conditional}'
    )


def test_lg_scm_aggregation_status_consistent_direction() -> None:
    """All bursts have the same sign on `rho_marginal` (positive,
    by the substrate construction). Magnitudes are close enough
    (planted point-biserial r ≈ 0.585 ± SE ≈ 0.05 per burst) to
    sit below the default `weak_time_varying_ratio=2.0`. Status
    MUST be CONSISTENT_DIRECTION."""
    df = _build_panel()
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='z_mean_per_burst',
        outcome_per_burst='y_mean_per_burst',
        stratify_by=('env_name', 'gamma'),
    )
    key = _stratum_key(('lg_scm', 0.99))
    result = results[key]
    # All marginal ρ positive (point-biserial direction matches
    # treatment > baseline encoding).
    assert all(r > 0 for r in result.rho_marginal), (
        f'all rho_marginal should be positive at this substrate; '
        f'got {result.rho_marginal}'
    )
    assert result.aggregation_status is (
        TimeAggregationStatus.CONSISTENT_DIRECTION
    ), (
        f'status={result.aggregation_status!r}; '
        f'rho_marginal={result.rho_marginal}'
    )


def test_lg_scm_n_per_burst_correct() -> None:
    """Sanity: every burst sees `2 * _N_SEEDS_PER_ARM` cells."""
    df = _build_panel()
    results = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='z_mean_per_burst',
        outcome_per_burst='y_mean_per_burst',
        stratify_by=('env_name', 'gamma'),
    )
    key = _stratum_key(('lg_scm', 0.99))
    result = results[key]
    assert result.n_bursts == _N_BURSTS
    assert all(n == 2 * _N_SEEDS_PER_ARM for n in result.n_per_burst), (
        f'n_per_burst={result.n_per_burst}'
    )


def test_lg_scm_dynamic_pc_consistent_with_partial_spearman() -> None:
    """The PC primitive and the partial-Spearman primitive share
    infrastructure (`_classify_status`, `_gather_burst_b`,
    `_encode_arm`) → on the SAME panel the per-burst `rho_marginal`
    arrays from both primitives MUST be bit-identical. This pins
    that the refactor's shared helpers actually return the same
    values when consumed from both primitives.

    The `rho_partial` arrays should also be bit-identical at every
    well-defined burst (both call the same `partial_spearman_rho`
    closed-form). Confirms infrastructure-sharing correctness."""
    from corroborate.analyses.dynamic_mediation import dynamic_partial_spearman
    df = _build_panel()
    res_pc = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='z_mean_per_burst',
        outcome_per_burst='y_mean_per_burst',
        stratify_by=('env_name', 'gamma'),
    )
    res_ps = dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst='z_mean_per_burst',
        outcome_per_burst='y_mean_per_burst',
        stratify_by=('env_name', 'gamma'),
    )
    key = _stratum_key(('lg_scm', 0.99))
    r_pc = res_pc[key]
    r_ps = res_ps[key]
    # rho_marginal bit-identical (same `_spearman_marginal` call).
    for b, (a, c) in enumerate(zip(r_pc.rho_marginal, r_ps.rho_marginal)):
        if math.isnan(a) and math.isnan(c):
            continue
        assert a == c, (
            f'burst {b}: PC.rho_marginal={a!r} differs from '
            f'partial_spearman.rho_marginal={c!r}; shared '
            f'infrastructure should return identical ρ'
        )
    # rho_partial bit-identical (both call `partial_spearman_rho`).
    for b, (a, c) in enumerate(zip(r_pc.rho_partial, r_ps.rho_partial)):
        if math.isnan(a) and math.isnan(c):
            continue
        assert a == c, (
            f'burst {b}: PC.rho_partial={a!r} differs from '
            f'partial_spearman.rho_partial={c!r}'
        )
    # And n_per_burst (ragged-tail alignment is the same primitive).
    assert r_pc.n_per_burst == r_ps.n_per_burst
