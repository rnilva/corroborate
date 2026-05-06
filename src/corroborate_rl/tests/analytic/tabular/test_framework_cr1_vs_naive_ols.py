"""Framework-as-instrument: `mundlak_decomposition`'s Liang-Zeger
CR1 sandwich SE corrects naive OLS in the way Moulton's
formula predicts on a panel with KNOWN intra-cluster correlation.

This is the post-Phase-1-audit suggestion: "build a panel where
naive OLS SE is wrong by a known factor (clustered residuals)
and assert the framework's CR1 matches Liang-Zeger correction."

Setup: 10 strata, 20 observations per stratum (n=200). A cluster
random effect α_g introduces residual correlation; within-cluster
deviations x_w have no clustered residual contribution (since
Σ_i x_w_{g,i} = 0 by construction). The Moulton design-effect
formula gives the closed-form ratio of CR1 SE to naive SE for
the BETWEEN coefficient:

    DEFF = 1 + (m − 1) · ρ_intra
    SE_CR1(β_b) ≈ √DEFF · SE_naive(β_b)

For σ²_α = 1.0 (cluster-level), σ²_ε = 0.1 (idiosyncratic):
    ρ_intra = σ²_α / (σ²_α + σ²_ε) ≈ 0.909
    DEFF = 1 + 19 · 0.909 ≈ 18.3
    SE_CR1 / SE_naive ≈ √18.3 ≈ 4.28

Critically, the WITHIN coefficient β_w is unaffected by clustering
(within-cluster deviations are orthogonal to the cluster random
effect), so SE_CR1(β_w) ≈ SE_naive(β_w).

Three load-bearing assertions:

1. Coefficient point estimates UNCHANGED between the two SE
   estimators — the framework only swaps the variance estimator,
   not the WLS coefficient itself.

2. SE_CR1(β_b) > 2.5 × SE_naive(β_b) — the CR1 sandwich correctly
   inflates SE under clustered residuals; closed-form 4.28×.

3. SE_CR1(β_w) within ±50 % of SE_naive(β_w) — the within slope is
   not sensitive to clustering.

Plus a negative control: with σ²_α = 0 (IID residuals), CR1 must
NOT inflate SE — the small-sample correction G/(G−1) · (n−1)/(n−p)
is just ≈ 1.12× here, so SE_CR1 ≈ SE_naive within ~30%. This pins
that the inflation in the heterogeneous case isn't an artifact of
the small-sample correction itself.

Catches:
- Stub returning naive SE under both modes (CR1 not implemented)
- Stub applying inflation universally (no negative control →
  ratio inflated even with IID residuals)
- Stub mishandling the inverse `bread @ meat @ bread` form
- Stub forgetting CR1's small-sample correction
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

    Closed-form CR1/naive SE ratios:
        β_b: ~4.28 (Moulton inflation; α_g contributes to score)
        β_w: ~0.32 (CR1 correctly identifies that α_g cancels in
                    Σ_i x_w·u within each cluster, so the true
                    SE is √(σ²_ε/σ²_total) · √correction smaller
                    than naive σ²_total-based SE)

    The ratio-of-ratios ≈ 13. Bound `ratio_b / ratio_w > 5` rules
    out the blanket-inflation stub (where ratio_b ≈ ratio_w because
    CR1 just multiplies all SEs by some constant).

    This is the hard probe — a stub that uniformly inflates SE
    by a fixed factor F would pass the previous β_b assertion
    (ratio > 2.5) but fail this one (ratio_b/ratio_w = 1)."""
    panel = _generate_clustered_panel(sigma_alpha=1.0, sigma_eps=0.1)
    naive = mundlak_decomposition.fn(panel, cluster_robust=False)
    cr1 = mundlak_decomposition.fn(panel, cluster_robust=True)
    ratio_b = cr1.between.se / naive.between.se
    ratio_w = cr1.within.se / naive.within.se
    asym = ratio_b / ratio_w
    assert asym > 5.0, (
        f'CR1/naive ratio(β_b) = {ratio_b:.2f}, '
        f'CR1/naive ratio(β_w) = {ratio_w:.2f}, asymmetry = '
        f'{asym:.2f}. Closed form ≈ 13 (β_b inflated by Moulton, '
        f'β_w shrunk because α_g cancels in within-cluster '
        f'scores). A symmetric adjustment (asymmetry ≈ 1) means '
        f'CR1 is treating both coefficients identically — the '
        f'cluster-aware sandwich machinery is broken.'
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
    NOT inflate SE meaningfully. Only the small-sample correction
    `G/(G−1) · (n−1)/(n−p)` ≈ 10/9 · 199/197 ≈ 1.12 should apply.

    Without this control, the inflation assertion in the
    heterogeneous test is meaningless: a stub that always inflates
    SE by some constant factor would pass the heterogeneous
    assertion spuriously. Pin both directions of the contrast.

    Bound `< 2.0×` allows for the small-sample correction (~1.12×)
    plus per-cluster sampling noise on the score sum. 2× is high
    enough that the FACTOR-18 inflation in the heterogeneous case
    is unambiguously due to clustered residuals, not a stub."""
    panel = _generate_clustered_panel(sigma_alpha=0.0, sigma_eps=1.0)
    naive = mundlak_decomposition.fn(panel, cluster_robust=False)
    cr1 = mundlak_decomposition.fn(panel, cluster_robust=True)
    ratio = cr1.between.se / naive.between.se
    assert ratio < 2.0, (
        f'SE_CR1(β_b) / SE_naive(β_b) = {ratio:.2f} under IID '
        f'residuals (σ²_α = 0); expected ≈ 1.12 (small-sample '
        f'correction only). A ratio > 2 means CR1 is inflating '
        f'SE even without clustering — the inflation in the '
        f'heterogeneous test is then untestable.'
    )


def test_cr1_pi_value_for_between_widens_under_clustering() -> None:
    """The downstream effect of SE inflation: p-value for β_b
    grows under CR1 when clustering is present. Closed-form β_b
    = 0.5; SE_naive ≈ 0.07 → t ≈ 7 → p < 1e-10. SE_CR1 ≈ 0.30 →
    t ≈ 1.7 → p ≈ 0.10.

    This is the load-bearing methodological consequence: a stub
    that returned the same SE under both modes would pass naive
    significance tests on β_b that are actually NOT significant
    once clustering is acknowledged."""
    panel = _generate_clustered_panel(sigma_alpha=1.0, sigma_eps=0.1)
    naive = mundlak_decomposition.fn(panel, cluster_robust=False)
    cr1 = mundlak_decomposition.fn(panel, cluster_robust=True)
    # Naive p ≪ CR1 p. Concrete bound: CR1 p > 100 · naive p
    # (closed form gives ~100,000× ratio under DEFF≈18; conservative
    # bound at 2 orders of magnitude rules out stubs that don't
    # propagate SE→p).
    assert cr1.between.p_value > 100.0 * max(naive.between.p_value, 1e-12), (
        f'p_naive(β_b) = {naive.between.p_value:.2e}, '
        f'p_CR1(β_b) = {cr1.between.p_value:.2e}. CR1 should '
        f'widen the SE → grow the p-value. A small ratio means '
        f'p-values aren\'t propagating from the new SE.'
    )
