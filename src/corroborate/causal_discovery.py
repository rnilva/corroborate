"""Causal-discovery primitives — PC adjacency + JCI stratification.

Port of v9's `discover_adjacency` + supporting CI tests, sized to
fit corroborate's "don't reinvent the wheel" principle:
scipy.stats handles the statistics (Spearman ρ, normal CDF, OLS
residuals); this module owns only the algorithm-shape primitives
v9 invented (partial / stratified Spearman compositions, the PC
loop, v-structure + Meek-rule orientation).

The algorithm:

1. **Seed** a complete undirected graph over `variables`.
2. **Remove** edges (X, Y) where any conditioning set Z ⊆
   `variables \\ {X, Y}` with |Z| ≤ `max_conditioning` yields a
   conditional-independence test passing at `alpha` (p ≥ alpha
   means we cannot reject independence).
3. **Record** the separating set Z that removed the edge — used
   later for v-structure detection during orientation.
4. **Orient** with v-structures (X → Z ← Y when X⊥Y, Z ∉ sepset)
   then Meek rules R1–R4 to propagate.

JCI stratification (Mooij et al. 2020): when `stratify_by` names
a categorical column, CI tests run within-stratum and pool z-stats
across strata via Fisher's transform. This handles env-conditioned
heterogeneity without ordinal-encoding the env.

**§4 / §5 / §6 of PAPER_NOTES.md** all run on this primitive.
§4 uses `stratify_by='env_name'`; §6 iterates per-env subsets;
§5 composes specific conditioning sets to test mediator hypotheses.
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import numpy.typing as npt
import polars as pl
import statsmodels.api as sm  # type: ignore[reportMissingTypeStubs]
from scipy.stats import norm, spearmanr  # type: ignore[reportMissingTypeStubs]


# ============ CI-test primitives ============

def _spearman_marginal(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Marginal Spearman ρ + two-sided p. Returns (NaN, NaN) when
    either side is constant or `n < 4` (smallest n with a
    well-defined spearmanr p-value)."""
    if len(x) < 4 or float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return float('nan'), float('nan')
    r, p = spearmanr(x, y)  # type: ignore[reportUnknownMemberType]
    return float(r), float(p)  # type: ignore[reportUnknownArgumentType]


def partial_spearman_rho(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Partial Spearman ρ(X, Y | Z) via the closed-form first-
    order partial. Equivalent to: rank-transform X, Y, Z; pairwise
    Spearman; combine via the partial-correlation formula.

    Returns (rho, two-sided p). Fisher-z applied to rho with
    df = n − 4. Returns (NaN, NaN) when n < 5 or any variance is
    degenerate."""
    n = len(x)
    if n < 5:
        return float('nan'), float('nan')
    rxy, _ = _spearman_marginal(x, y)
    rxz, _ = _spearman_marginal(x, z)
    ryz, _ = _spearman_marginal(y, z)
    if any(math.isnan(r) for r in (rxy, rxz, ryz)):
        return float('nan'), float('nan')
    denom = math.sqrt(max(1.0 - rxz ** 2, 0.0)
                      * max(1.0 - ryz ** 2, 0.0))
    if denom <= 1e-12:
        return float('nan'), float('nan')
    rho = (rxy - rxz * ryz) / denom
    rho = max(-0.999999, min(0.999999, rho))
    z_stat = 0.5 * math.log((1 + rho) / (1 - rho)) * math.sqrt(n - 4)
    p = 2 * (1.0 - float(norm.cdf(abs(z_stat))))
    return rho, p


def partial_spearman_rho_multi(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z_matrix: npt.NDArray[np.float64],
) -> tuple[float, float]:
    """Multi-Z partial Spearman ρ(X, Y | Z₁, ..., Zₖ) via residual
    regression. Rank-transform X, Y, and each column of `z_matrix`;
    OLS-regress ranked X on ranked Z (with intercept) and ranked Y
    on ranked Z; Pearson the residuals.

    `z_matrix` shape: `(n, k)`. Returns (rho, two-sided p) with
    Fisher z df = n − 3 − k. Returns (NaN, NaN) when n is too small
    or OLS produces non-finite residuals (rank-deficient Z, etc.)."""
    n = len(x)
    k = z_matrix.shape[1] if z_matrix.ndim == 2 else 1
    df = n - 3 - k
    if df < 1:
        return float('nan'), float('nan')

    def _rank(a: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        order = np.argsort(a, kind='mergesort')
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(len(a))
        return ranks

    rx = _rank(x)
    ry = _rank(y)
    if z_matrix.ndim == 1:
        z_matrix = z_matrix.reshape(-1, 1)
    rz = np.column_stack([_rank(z_matrix[:, j]) for j in range(k)])
    rz_with_intercept = sm.add_constant(  # type: ignore[reportUnknownMemberType]
        rz,
    )
    try:
        x_resid: npt.NDArray[np.float64] = sm.OLS(  # type: ignore[reportUnknownMemberType]
            rx, rz_with_intercept,
        ).fit().resid
        y_resid: npt.NDArray[np.float64] = sm.OLS(  # type: ignore[reportUnknownMemberType]
            ry, rz_with_intercept,
        ).fit().resid
    except (ValueError, np.linalg.LinAlgError):
        return float('nan'), float('nan')
    if not np.all(np.isfinite(x_resid)) or not np.all(np.isfinite(y_resid)):
        return float('nan'), float('nan')
    sx = float(x_resid.std())
    sy = float(y_resid.std())
    if sx == 0.0 or sy == 0.0:
        return float('nan'), float('nan')
    rho = float(
        np.mean((x_resid - x_resid.mean()) * (y_resid - y_resid.mean()))
        / (sx * sy),
    )
    rho = max(-0.999999, min(0.999999, rho))
    z_stat = 0.5 * math.log((1 + rho) / (1 - rho)) * math.sqrt(df)
    p = 2 * (1.0 - float(norm.cdf(abs(z_stat))))
    return rho, p


def stratified_spearman_rho(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    strata: Sequence[object],
    *,
    min_stratum_size: int = 4,
) -> tuple[float, float]:
    """JCI-stratified marginal Spearman ρ(X, Y | strata).

    Per stratum k: compute Spearman ρ_k. Fisher z_k = 0.5 ln((1+ρ_k)
    / (1−ρ_k)). Pool across strata weighted by `(n_k − 3)`:
    z_pooled = Σ (n_k − 3) z_k / Σ (n_k − 3).

    Skips strata with `n_k < min_stratum_size` (Fisher z floor).
    Returns (rho_pooled, two-sided p) with z-stat normalised by
    pooled-weight √(Σ (n_k − 3))."""
    strata_arr = np.asarray(strata)
    unique = np.unique(strata_arr)
    z_vals: list[float] = []
    weights: list[float] = []
    for k in unique:
        mask = strata_arr == k
        n_k = int(mask.sum())
        if n_k < min_stratum_size:
            continue
        x_k = x[mask]
        y_k = y[mask]
        if float(x_k.std()) == 0.0 or float(y_k.std()) == 0.0:
            continue
        r, _ = _spearman_marginal(x_k, y_k)
        if math.isnan(r):
            continue
        r_clamped = max(-0.999999, min(0.999999, r))
        z_k = 0.5 * math.log((1 + r_clamped) / (1 - r_clamped))
        z_vals.append(z_k)
        weights.append(float(n_k - 3))
    if not z_vals:
        return float('nan'), float('nan')
    total_w = sum(weights)
    if total_w <= 0:
        return float('nan'), float('nan')
    z_pooled = sum(w * z for w, z in zip(weights, z_vals)) / total_w
    rho_pooled = float(math.tanh(z_pooled))
    z_stat = z_pooled * math.sqrt(total_w)
    p = 2 * (1.0 - float(norm.cdf(abs(z_stat))))
    return rho_pooled, p


def stratified_partial_spearman_rho(
    x: npt.NDArray[np.float64],
    y: npt.NDArray[np.float64],
    z_cond: npt.NDArray[np.float64],
    strata: Sequence[object],
    *,
    min_stratum_size: int = 5,
) -> tuple[float, float]:
    """JCI-stratified partial Spearman ρ(X, Y | Z, strata).

    Per stratum: closed-form `partial_spearman_rho(x_k, y_k, z_k)`.
    Fisher z pooled by `(n_k − 4)`. Returns (rho_pooled, p)."""
    strata_arr = np.asarray(strata)
    unique = np.unique(strata_arr)
    z_vals: list[float] = []
    weights: list[float] = []
    for k in unique:
        mask = strata_arr == k
        n_k = int(mask.sum())
        if n_k < min_stratum_size:
            continue
        rho_k, _ = partial_spearman_rho(
            x[mask], y[mask], z_cond[mask],
        )
        if math.isnan(rho_k):
            continue
        rho_clamped = max(-0.999999, min(0.999999, rho_k))
        z_k = 0.5 * math.log((1 + rho_clamped) / (1 - rho_clamped))
        z_vals.append(z_k)
        weights.append(float(n_k - 4))
    if not z_vals:
        return float('nan'), float('nan')
    total_w = sum(weights)
    if total_w <= 0:
        return float('nan'), float('nan')
    z_pooled = sum(w * z for w, z in zip(weights, z_vals)) / total_w
    rho_pooled = float(math.tanh(z_pooled))
    z_stat = z_pooled * math.sqrt(total_w)
    p = 2 * (1.0 - float(norm.cdf(abs(z_stat))))
    return rho_pooled, p


# Stub for commit 2 — PC algorithm.
@dataclass(frozen=True, slots=True)
class DiscoveredAdjacency:
    """PC adjacency-discovery result. Edges are unordered (frozenset
    pairs); separating sets record the conditioning Z that removed
    each tested edge (used by `orient_adjacency` for v-structure
    detection)."""
    edges: frozenset[frozenset[str]]
    separating_sets: dict[
        frozenset[str], frozenset[frozenset[str]]
    ]
    n_observations: int
    alpha: float
    max_conditioning: int
    stratify_by: str | None
