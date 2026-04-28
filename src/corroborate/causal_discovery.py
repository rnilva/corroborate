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


@dataclass(frozen=True, slots=True)
class DiscoveredAdjacency:
    """PC adjacency-discovery result. Edges are unordered (frozenset
    pairs); separating sets record the conditioning Z that removed
    each tested edge (used by `orient_adjacency` for v-structure
    detection)."""
    variables: frozenset[str]
    edges: frozenset[frozenset[str]]
    separating_sets: dict[
        frozenset[str], frozenset[frozenset[str]]
    ]
    n_observations: int
    alpha: float
    max_conditioning: int
    stratify_by: str | None


@dataclass(frozen=True, slots=True)
class OrientedAdjacency:
    """CPDAG: some edges are directed (from v-structure detection +
    Meek-rule orientation), the rest stay undirected (orientation
    not determinable from independence structure alone — every DAG
    in the equivalence class agrees on those edges' shape but not
    direction).

    `ambiguous_triples` carries unshielded triples `(X, Z, Y)` where
    Z's collider status is undetermined (Z is in SOME but not ALL
    separating sets of X⫫Y). Conservative PC skips these for
    orientation."""
    variables: frozenset[str]
    directed_edges: frozenset[tuple[str, str]]
    undirected_edges: frozenset[frozenset[str]]
    separating_sets: dict[
        frozenset[str], frozenset[frozenset[str]]
    ]
    ambiguous_triples: frozenset[tuple[str, str, str]]


# ============ PC algorithm ============

def discover_adjacency(
    data: pl.DataFrame,
    *,
    variables: Sequence[str],
    alpha: float = 0.05,
    max_conditioning: int = 1,
    stratify_by: str | None = None,
) -> DiscoveredAdjacency:
    """Conservative-PC adjacency over `variables` at depth ≤
    `max_conditioning`, using rank-based CI tests.

    `data`: polars DataFrame containing the named columns.
    `variables`: ordered sequence of column names. Order doesn't
    affect the surviving adjacency but affects which Z is checked
    first (irrelevant under conservative-PC since we collect ALL
    separating sets).
    `alpha`: significance level. Edge survives iff every tested Z
    yields `p < alpha`.
    `max_conditioning`: cap on |Z|. 0 = marginal only. 1 = single-
    Z partial via closed form. ≥2 = multi-Z residual regression.
    `stratify_by`: optional column for JCI categorical context.
    Within-stratum CI tests pooled via Fisher z. The stratifier
    is NOT a graph node.

    Conservative-PC (Ramsey-Zhang-Spirtes 2006): collects EVERY
    Z that separates the edge, not just the first. Lets the
    orientation pass distinguish definite-collider triples (Z in
    no sepsets) from ambiguous ones (Z in some but not all).

    Returns `DiscoveredAdjacency` with surviving edges + per-edge
    separating sets."""
    if max_conditioning < 0:
        raise ValueError(
            f'max_conditioning must be ≥ 0, got {max_conditioning}',
        )
    var_list = list(variables)
    var_set = frozenset(var_list)
    if len(var_set) != len(var_list):
        raise ValueError(f'duplicate variables in {var_list!r}')

    # Cache numpy column views once.
    columns: dict[str, npt.NDArray[np.float64]] = {
        v: np.asarray(
            data[v].to_list(),
            dtype=np.float64,
        ) for v in var_list
    }
    strata_arr: list[object] | None = None
    if stratify_by is not None:
        strata_arr = list(data[stratify_by].to_list())

    edges: set[frozenset[str]] = set()
    for x, y in combinations(var_list, 2):
        edges.add(frozenset({x, y}))
    separating_sets: dict[
        frozenset[str], frozenset[frozenset[str]]
    ] = {}

    # Depth 0 — marginal independence.
    for x, y in combinations(var_list, 2):
        edge = frozenset({x, y})
        if strata_arr is not None:
            _, p = stratified_spearman_rho(
                columns[x], columns[y], strata_arr,
            )
        else:
            _, p = _spearman_marginal(columns[x], columns[y])
        if math.isnan(p):
            continue  # CI test inconclusive; keep edge
        if p >= alpha:
            edges.discard(edge)
            separating_sets[edge] = frozenset(
                {frozenset[str]()},
            )

    # Depth k ≥ 1 — partial CI tests.
    n_obs = data.height
    for k in range(1, max_conditioning + 1):
        surviving = list(edges)
        for edge in surviving:
            if edge not in edges:
                continue  # removed in a prior iteration
            x, y = sorted(edge)  # deterministic order
            other_vars = [v for v in var_list if v != x and v != y]
            if len(other_vars) < k:
                continue
            found_sepsets: set[frozenset[str]] = set()
            for z_subset in combinations(other_vars, k):
                if k == 1 and strata_arr is not None:
                    _, p = stratified_partial_spearman_rho(
                        columns[x], columns[y],
                        columns[z_subset[0]], strata_arr,
                    )
                elif k == 1:
                    _, p = partial_spearman_rho(
                        columns[x], columns[y],
                        columns[z_subset[0]],
                    )
                else:
                    z_matrix = np.column_stack(
                        [columns[z] for z in z_subset],
                    )
                    _, p = partial_spearman_rho_multi(
                        columns[x], columns[y], z_matrix,
                    )
                if math.isnan(p):
                    continue
                if p >= alpha:
                    found_sepsets.add(frozenset(z_subset))
            if found_sepsets:
                edges.discard(edge)
                separating_sets[edge] = frozenset(found_sepsets)

    return DiscoveredAdjacency(
        variables=var_set,
        edges=frozenset(edges),
        separating_sets=separating_sets,
        n_observations=n_obs,
        alpha=alpha,
        max_conditioning=max_conditioning,
        stratify_by=stratify_by,
    )


# ============ Orientation ============

def _orient(
    source: str,
    target: str,
    directed: set[tuple[str, str]],
    undirected: set[frozenset[str]],
) -> bool:
    """Attempt to orient `source → target`. Returns True iff the
    edge was newly oriented. Skips conflicts (target → source
    already directed) and idempotent cases."""
    if (target, source) in directed:
        return False
    if (source, target) in directed:
        return False
    edge = frozenset({source, target})
    if edge not in undirected:
        return False
    undirected.discard(edge)
    directed.add((source, target))
    return True


def _adjacent(
    a: str, b: str,
    directed: set[tuple[str, str]],
    undirected: set[frozenset[str]],
) -> bool:
    """Any edge between a and b, directed or undirected."""
    if a == b:
        return False
    if frozenset({a, b}) in undirected:
        return True
    return (a, b) in directed or (b, a) in directed


def _neighbors(
    z: str,
    directed: set[tuple[str, str]],
    undirected: set[frozenset[str]],
) -> set[str]:
    out: set[str] = set()
    for edge in undirected:
        if z in edge:
            out.update(v for v in edge if v != z)
    for src, tgt in directed:
        if src == z:
            out.add(tgt)
        elif tgt == z:
            out.add(src)
    return out


def _detect_v_structures(
    adj: DiscoveredAdjacency,
    directed: set[tuple[str, str]],
    undirected: set[frozenset[str]],
    *,
    conservative: bool,
) -> frozenset[tuple[str, str, str]]:
    """For every unshielded triple X − Z − Y (X, Y both adjacent
    to Z; X, Y NOT adjacent in the discovered adjacency), decide
    whether Z is a collider:

    - Z in ALL sepsets(X, Y) → non-collider; skip.
    - Z in NO sepsets → collider; orient X → Z ← Y.
    - Z in SOME but not all → ambiguous.

    Conservative-PC tracks ambiguous triples and leaves them
    unoriented; standard-PC treats ambiguous as collider (more
    aggressive)."""
    ambiguous: set[tuple[str, str, str]] = set()
    var_list = sorted(adj.variables)
    for z in var_list:
        z_neighbors = sorted(_neighbors(z, directed, undirected))
        for i in range(len(z_neighbors)):
            for j in range(i + 1, len(z_neighbors)):
                x, y = z_neighbors[i], z_neighbors[j]
                xy = frozenset({x, y})
                # Unshielded: X, Y not adjacent in the discovered
                # adjacency.
                if xy in adj.edges:
                    continue
                sepsets = adj.separating_sets.get(xy)
                if not sepsets:
                    sepsets = frozenset([frozenset[str]()])
                z_in_all = all(z in s for s in sepsets)
                z_in_none = all(z not in s for s in sepsets)
                if z_in_all:
                    continue
                if z_in_none:
                    _ = _orient(x, z, directed, undirected)
                    _ = _orient(y, z, directed, undirected)
                    continue
                # Some but not all — ambiguous.
                a, b = (x, y) if x < y else (y, x)
                if conservative:
                    ambiguous.add((a, z, b))
                    continue
                # Standard PC: treat ambiguous as collider.
                _ = _orient(x, z, directed, undirected)
                _ = _orient(y, z, directed, undirected)
    return frozenset(ambiguous)


def _apply_meek_rules(
    directed: set[tuple[str, str]],
    undirected: set[frozenset[str]],
    variables: frozenset[str],
    *,
    ambiguous_triples: frozenset[tuple[str, str, str]],
) -> None:
    """Iterate Meek rules R1 and R2 until no more edges orient.

    R1: If A → B and B − C undirected and A not adjacent to C,
        orient B → C. (Else A → B ← C would be a new v-structure.)
    R2: If A → B → C and A − C undirected, orient A → C.
        (Else A → B → C → A would be a cycle.)

    Conservative gate on R1: skip when (A, B, C) is in
    `ambiguous_triples` — Z's collider status is undetermined."""
    changed = True
    while changed:
        changed = False
        # Rule 1
        for (a, b) in list(directed):
            for c in variables:
                if c == a or c == b:
                    continue
                if frozenset({b, c}) not in undirected:
                    continue
                if _adjacent(a, c, directed, undirected):
                    continue
                ac_lo, ac_hi = (a, c) if a < c else (c, a)
                if (ac_lo, b, ac_hi) in ambiguous_triples:
                    continue
                if _orient(b, c, directed, undirected):
                    changed = True
        # Rule 2
        for (a, b) in list(directed):
            for (b2, c) in list(directed):
                if b != b2:
                    continue
                if frozenset({a, c}) not in undirected:
                    continue
                if _orient(a, c, directed, undirected):
                    changed = True


def orient_adjacency(
    adj: DiscoveredAdjacency,
    *,
    conservative: bool = True,
) -> OrientedAdjacency:
    """Orient a `DiscoveredAdjacency` into a CPDAG. Two phases:
    v-structure detection (collider triples) then Meek-rule
    propagation. Returns `OrientedAdjacency` carrying directed
    edges, remaining undirected edges, and (in conservative mode)
    the ambiguous triples that were intentionally not oriented."""
    directed: set[tuple[str, str]] = set()
    undirected: set[frozenset[str]] = set(adj.edges)
    ambiguous = _detect_v_structures(
        adj, directed, undirected, conservative=conservative,
    )
    _apply_meek_rules(
        directed, undirected, adj.variables,
        ambiguous_triples=ambiguous,
    )
    return OrientedAdjacency(
        variables=adj.variables,
        directed_edges=frozenset(directed),
        undirected_edges=frozenset(undirected),
        separating_sets=adj.separating_sets,
        ambiguous_triples=ambiguous,
    )
