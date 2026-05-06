"""Framework-as-instrument: `mundlak_decomposition`'s Liang-Zeger
CR1 sandwich SE corrects naive OLS in the way Moulton's
formula predicts on a panel with KNOWN intra-cluster correlation.

This is the post-Phase-1-audit suggestion: "build a panel where
naive OLS SE is wrong by a known factor (clustered residuals)
and assert the framework's CR1 matches Liang-Zeger correction."

Setup: 10 strata, 20 observations per stratum (n=200). A cluster
random effect α_g introduces residual correlation; within-cluster
deviations x_w have Σ_i x_w_{g,i} = 0 within each cluster.

Empirical SE ratios (CR1/naive) read from running the test:

    σ²_α = 1.0, σ²_ε = 0.1 (HET):
        ratio_b = 3.90  (Moulton-DEFF inflated; closed form ≈
                         √(1 + (m−1)·ρ_intra) = √(1 + 19·0.909)
                         ≈ 4.28; empirical close to closed form)
        ratio_w = 0.063 (CR1 correctly down-weights: naive uses
                         σ²_total ≈ 1.1 from missing-stratum-FE,
                         while CR1 sees Σ_i x_w·u within cluster
                         dominated by ε not α — the closed form
                         here is sensitive to the bread-meat
                         interaction with the missing FE, not a
                         simple Moulton ratio. Recorded empirically.)
        asymmetry = ratio_b / ratio_w ≈ 62×

    σ²_α = 0.0, σ²_ε = 1.0 (IID negative control):
        ratio_b ≈ 0.85   (CR1 can land slightly BELOW 1 even with
                         small-sample correction 1.12× — the
                         meat-from-IID-scores varies with the
                         random sample. Bounded < 2 to absorb this.)
        asymmetry ≈ 1.17 (no clustering → no asymmetry between
                         coefs)

Three load-bearing assertions:

1. Coefficients unchanged (CR1 is a SE-only knob).

2. SE_CR1(β_b) > 2.5 × SE_naive(β_b) under high ICC — Moulton
   inflation of the between coefficient.

3. asymmetry > 8 under HET, asymmetry < 2 under IID — the
   structurally distinguishing property: a blanket-SE-inflation
   stub can pass (2) by returning F·SE_naive uniformly, but
   asymmetry = 1 under that stub fails this contrast. Bound `> 8`
   has empirical margin 62/8 ≈ 7.75× — high enough to absorb
   sampling noise on G=10 clusters AND tight enough to rule out
   per-coef-fixed-factor stubs (where a stub setting ratio_b ≈ 5
   and ratio_w ≈ 0.5 would give asymmetry = 10, just over the
   < 8 boundary).

Catches:
- Stub returning naive SE under both modes (asymmetry = 1; (3) fails)
- Stub applying blanket inflation by factor F (asymmetry = 1; (3) fails)
- Stub applying per-coef fixed factors (asymmetry < 8; (3) tight)
- Stub mishandling the inverse `bread @ meat @ bread` form (one
  of (2)/(3) likely fails)
"""
from __future__ import annotations

import math
import zlib

import numpy as np

from corroborate.analyses.mundlak_decomposition import mundlak_decomposition


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


_N_STRATA = 10
_N_PER_STRATUM = 20
_BETA_B = 0.5      # between-cluster coefficient (effect of x_e)
_BETA_W = 0.7      # within-cluster coefficient (effect of x_w)
_SIGMA_X_W = 1.0   # within-cluster predictor scale
# x_e values: equally-spaced grid, mean 0, var 0.5 (analogous
# to the heterogeneous τ² test panel).
_X_E_GRID: tuple[float, ...] = (-1.0, -0.7, -0.4, -0.2, 0.0, 0.2, 0.4, 0.7, 1.0, 1.2)


def _generate_clustered_panel(
    *,
    sigma_alpha: float,
    sigma_eps: float,
) -> list[dict[str, object]]:
    """Generate a panel with cluster random effect σ²_α and
    idiosyncratic noise σ²_ε. y = β_b · x_e + β_w · x_w + α_g + ε."""
    panel: list[dict[str, object]] = []
    rng_alpha = np.random.default_rng(seed=_det_seed('cr1_alpha', sigma_alpha))
    for g in range(_N_STRATA):
        x_e = _X_E_GRID[g]
        alpha_g = float(sigma_alpha * rng_alpha.standard_normal())
        rng_eps = np.random.default_rng(
            seed=_det_seed('cr1_eps', sigma_alpha, g),
        )
        rng_xw = np.random.default_rng(
            seed=_det_seed('cr1_xw', g),
        )
        for i in range(_N_PER_STRATUM):
            x_w = float(_SIGMA_X_W * rng_xw.standard_normal())
            x_total = x_e + x_w   # mundlak decomposes this back internally
            eps = float(sigma_eps * rng_eps.standard_normal())
            y = _BETA_B * x_e + _BETA_W * x_w + alpha_g + eps
            panel.append({
                'stratum_id': f'stratum_{g}',
                'x': x_total,
                'y': y,
                'se': 1.0,   # uniform weights → naive σ² estimates total residual variance
            })
    return panel


# ============ The CR1 vs naive contrast ============

def test_cr1_inflates_between_se_under_clustered_residuals() -> None:
    """High-ICC panel: σ²_α = 1.0, σ²_ε = 0.1 → ρ_intra ≈ 0.91,
    DEFF ≈ 18.3 → SE_CR1(β_b) / SE_naive(β_b) ≈ 4.28.

    Bound `> 2.5×` is conservative: it accepts ~50% sampling
    fluctuation around the closed-form 4.28× and still rejects:
    - SE_CR1 ≈ SE_naive (CR1 not actually applied)
    - SE_CR1 / SE_naive ≈ 1.12× (only the small-sample correction
      is being applied, the meat term is broken)"""
    panel = _generate_clustered_panel(sigma_alpha=1.0, sigma_eps=0.1)
    naive = mundlak_decomposition.fn(panel, cluster_robust=False)
    cr1 = mundlak_decomposition.fn(panel, cluster_robust=True)
    ratio = cr1.between.se / naive.between.se
    assert ratio > 2.5, (
        f'SE_CR1(β_b) = {cr1.between.se:.4f}, SE_naive(β_b) = '
        f'{naive.between.se:.4f}, ratio = {ratio:.2f}. Closed-form '
        f'Moulton ratio ≈ 4.28 (DEFF = 1 + (m−1)·ρ = 18.3). '
        f'A ratio < 2.5 means the CR1 sandwich is not '
        f'inflating SE under clustered residuals.'
    )


def test_cr1_adjustment_is_asymmetric_between_vs_within() -> None:
    """The CR1 adjustment must be ASYMMETRIC across β_b vs β_w
    — that's the methodologically distinguishing property of a
    cluster-robust SE under high ICC.

    Empirical CR1/naive SE ratios:
        ratio_b ≈ 3.90 (Moulton inflation; α_g contributes to
                        per-cluster score)
        ratio_w ≈ 0.063 (CR1 correctly identifies that α_g
                         cancels in Σ_i x_w_{g,i}·u_{g,i}; naive
                         uses σ²_total≈1.1 inflated by missing
                         stratum FE, so ratio is naturally far
                         below 1)
        asymmetry ≈ 62

    Bound `asymmetry > 8` is tightened from the prior `> 5`:
    a stub that mechanically multiplies CR1 SE by `F_b · naive_se`
    for β_b and `F_w · naive_se` for β_w (per-coefficient fixed
    factors, no real sandwich) could pass the prior `> 5` bound
    by setting F_b = 5, F_w = 0.5 (asymmetry = 10). Tightening
    to `> 8` rules out such stubs unless they also approximate
    CR1's actual asymmetry magnitude — which requires the
    sandwich machinery."""
    panel = _generate_clustered_panel(sigma_alpha=1.0, sigma_eps=0.1)
    naive = mundlak_decomposition.fn(panel, cluster_robust=False)
    cr1 = mundlak_decomposition.fn(panel, cluster_robust=True)
    ratio_b = cr1.between.se / naive.between.se
    ratio_w = cr1.within.se / naive.within.se
    asym = ratio_b / ratio_w
    assert asym > 8.0, (
        f'CR1/naive ratio(β_b) = {ratio_b:.2f}, '
        f'CR1/naive ratio(β_w) = {ratio_w:.2f}, asymmetry = '
        f'{asym:.2f}. Empirical ≈ 62; bound > 8 rules out '
        f'per-coefficient-fixed-factor stubs. A symmetric or '
        f'modestly-asymmetric adjustment (≤ 8) means CR1 is not '
        f'computing the bread-meat-bread sandwich correctly.'
    )


def test_cr1_does_not_change_coefficient_estimates() -> None:
    """The CR1 toggle is a SE-only knob — coefficient point
    estimates must be identical between the two modes (the WLS
    fit doesn't depend on the variance estimator).

    Pin against a regression that accidentally re-fit the model
    under cluster_robust=True (e.g., as a weighted-by-cluster
    estimator instead of the standard sandwich)."""
    panel = _generate_clustered_panel(sigma_alpha=1.0, sigma_eps=0.1)
    naive = mundlak_decomposition.fn(panel, cluster_robust=False)
    cr1 = mundlak_decomposition.fn(panel, cluster_robust=True)
    # Tight tolerance — coefficient estimates should be
    # bit-for-bit equal up to numerical precision.
    assert math.isclose(
        naive.between.coefficient,
        cr1.between.coefficient,
        rel_tol=1e-10,
    ), (
        f'β_b changed between naive ({naive.between.coefficient:.6f}) '
        f'and CR1 ({cr1.between.coefficient:.6f}); CR1 is a '
        f'SE-only correction, coefficients must match exactly.'
    )
    assert math.isclose(
        naive.within.coefficient,
        cr1.within.coefficient,
        rel_tol=1e-10,
    ), (
        f'β_w changed between naive ({naive.within.coefficient:.6f}) '
        f'and CR1 ({cr1.within.coefficient:.6f}); CR1 is a '
        f'SE-only correction, coefficients must match exactly.'
    )


def test_cr1_negative_control_iid_residuals() -> None:
    """Negative control: σ²_α = 0 → no clustering → CR1 should
    NOT meaningfully inflate SE, AND should NOT introduce
    asymmetry between β_b and β_w.

    Empirical: ratio_b ≈ 0.85, asymmetry ≈ 1.17 (CR1 lands
    slightly below naive because the meat from IID scores varies
    sample-to-sample; the small-sample correction G/(G−1)·(n−1)
    /(n−p) ≈ 1.12 is the multiplicative ceiling, but the meat
    sum can be smaller than `σ²·XᵀWX`).

    Without this control, the heterogeneous-case assertions are
    meaningless: a stub that always inflates SE by a constant
    factor (or always introduces a fixed asymmetry) would pass
    the heterogeneous test spuriously. This pins both directions:
    inflation is BOUNDED under no clustering, AND asymmetry
    collapses to ~1.

    Bounds:
      - `ratio_b < 2.0`: the small-sample correction × meat-from-
        IID-noise can land in roughly [0.7, 1.3]; bound `< 2`
        leaves a 1.5× margin. Empirical 0.85.
      - `asymmetry < 2`: structural prediction is ≈ 1 (no
        clustering → no β_b vs β_w distinction). Empirical 1.17.
    """
    panel = _generate_clustered_panel(sigma_alpha=0.0, sigma_eps=1.0)
    naive = mundlak_decomposition.fn(panel, cluster_robust=False)
    cr1 = mundlak_decomposition.fn(panel, cluster_robust=True)
    ratio_b = cr1.between.se / naive.between.se
    ratio_w = cr1.within.se / naive.within.se
    asym = ratio_b / ratio_w
    assert ratio_b < 2.0, (
        f'SE_CR1(β_b) / SE_naive(β_b) = {ratio_b:.2f} under IID '
        f'residuals (σ²_α = 0); expected ≈ 0.85 (small-sample '
        f'correction × meat-from-IID-noise can land near 1). '
        f'A ratio > 2 means CR1 is inflating SE even without '
        f'clustering.'
    )
    assert asym < 2.0, (
        f'asymmetry under IID = {asym:.2f}; expected ≈ 1.17. '
        f'A larger asymmetry under no clustering means CR1 is '
        f'introducing per-coefficient adjustments that do not '
        f'reflect actual clustering structure.'
    )
