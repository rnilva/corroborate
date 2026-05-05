"""PC v-structure detection + Meek-rule orientation, against a
known collider DAG.

The chain-DAG test (`test_pc_discovery.py`) leaves PC unable to
orient X-Z-Y (Markov equivalence class includes both chains and
fork). To exercise v-structure detection AND Meek-rule
propagation, we need a DAG where some triple has a collider.

Smallest example: X → Z ← Y, with X ⫫ Y marginally.
- Marginal: ρ(X, Y) = 0 (independent), ρ(X, Z) > 0, ρ(Y, Z) > 0
- Conditional on Z: ρ(X, Y | Z) ≠ 0 (Berkson's bias —
  conditioning on a common effect induces dependence)

PC at depth 0 removes X-Y (marginally independent), keeps X-Z
and Z-Y. The X-Y separating set is the empty set {} — Z is NOT
in any separating set of X⫫Y, which is the textbook collider
signature. `orient_adjacency` then orients X→Z←Y (the only DAG
in the Markov equivalence class with this CI pattern).

This file targets the 251 survived + 304 no-tests mutants in
`corroborate.graph.discovery` that the chain test couldn't reach
— specifically `_detect_v_structures`, `_apply_meek_rules`,
`compare_pc_depths`, and the EdgeDiff machinery.
"""
from __future__ import annotations

import numpy as np
import polars as pl

from corroborate.graph.discovery import (
    DiscoveredAdjacency,
    OrientedAdjacency,
    compare_pc_depths,
    discover_adjacency,
    orient_adjacency,
)


_BETA_X_TO_Z = 1.0
_BETA_Y_TO_Z = 1.0
_SIGMA_NOISE = 1.0
_N_OBS = 300
_PC_ALPHA = 0.05  # default; collider's marginal X⫫Y is tight enough


def _collider_dataframe(seed: int = 0) -> pl.DataFrame:
    """Construct n samples from a collider DAG: X, Y independent;
    Z = β_x·X + β_y·Y + noise.

    Marginal independence X⫫Y is exact by construction (independent
    Gaussian draws). Conditioning on Z induces Berkson dependence,
    making PC's depth-1 test on (X, Y | Z) significant — but
    depth-0 already removed the X-Y edge before depth-1 runs."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(_N_OBS)
    y = rng.standard_normal(_N_OBS)  # independent of x
    z = _BETA_X_TO_Z * x + _BETA_Y_TO_Z * y + _SIGMA_NOISE * rng.standard_normal(_N_OBS)
    return pl.DataFrame({
        'X': x.tolist(), 'Y': y.tolist(), 'Z': z.tolist(),
    })


# ============ Adjacency: depth-0 already removes X-Y ============

def test_collider_adjacency_after_depth_zero_only() -> None:
    """At max_conditioning=0, only marginal CI tests run.
    X ⫫ Y by construction → X-Y edge removed at depth-0.
    X-Z and Z-Y survive (both marginally correlated through the
    structural arrows).

    Sepset for X-Y is the empty set {} — captures the textbook
    collider signature `Z ∉ sepset(X, Y)` that v-structure
    detection relies on."""
    df = _collider_dataframe()
    adj = discover_adjacency(
        df, variables=('X', 'Y', 'Z'),
        max_conditioning=0, alpha=_PC_ALPHA,
    )
    assert isinstance(adj, DiscoveredAdjacency)
    expected = frozenset({frozenset({'X', 'Z'}), frozenset({'Y', 'Z'})})
    assert adj.edges == expected, (
        f'depth-0 adjacency = {adj.edges!r}, expected {expected!r}. '
        f'X-Y must be removed (marginal independence); X-Z and '
        f'Y-Z survive.'
    )
    # Sepset of X-Y is empty (depth-0 separating set).
    xy_sepsets = adj.separating_sets[frozenset({'X', 'Y'})]
    assert frozenset[str]() in xy_sepsets, (
        f'X-Y sepset = {xy_sepsets!r}; expected to contain the '
        f'empty set (depth-0 separation), so Z is NOT in any '
        f'sepset → Z is a collider.'
    )


# ============ Orientation: v-structure detected ============

def test_orient_adjacency_detects_collider_at_z() -> None:
    """`orient_adjacency` on the collider adjacency must find
    the v-structure X→Z←Y. Test:
      - Both edges X-Z and Y-Z become DIRECTED (X→Z and Y→Z).
      - No undirected edges remain.
      - Zero ambiguous triples (Z is definitely NOT in any
        sepset → unambiguous collider).

    A regression in `_detect_v_structures` would either miss
    the collider (leaving X-Z, Y-Z undirected) or flip the
    direction (Z→X or Z→Y). Either case fails this test."""
    df = _collider_dataframe()
    adj = discover_adjacency(
        df, variables=('X', 'Y', 'Z'),
        max_conditioning=0, alpha=_PC_ALPHA,
    )
    oriented = orient_adjacency(adj, conservative=True)
    assert isinstance(oriented, OrientedAdjacency)
    # Both arrows go INTO Z.
    assert ('X', 'Z') in oriented.directed_edges, (
        f'X→Z not directed; got {oriented.directed_edges!r}'
    )
    assert ('Y', 'Z') in oriented.directed_edges, (
        f'Y→Z not directed; got {oriented.directed_edges!r}'
    )
    # No reverse arrows.
    assert ('Z', 'X') not in oriented.directed_edges, (
        f'Z→X (reverse) directed; collider must point INTO Z'
    )
    assert ('Z', 'Y') not in oriented.directed_edges, (
        f'Z→Y (reverse) directed; collider must point INTO Z'
    )
    # No undirected residue.
    assert oriented.undirected_edges == frozenset(), (
        f'undirected edges remain: {oriented.undirected_edges!r}; '
        f'collider should fully orient both edges'
    )
    # Conservative mode: no ambiguous triples on a clean collider.
    assert len(oriented.ambiguous_triples) == 0, (
        f'ambiguous triples = {oriented.ambiguous_triples!r}; '
        f'collider with empty sepset is unambiguous'
    )


# ============ compare_pc_depths: collider stable across depths ============

def test_compare_pc_depths_collider_invariant_across_depths() -> None:
    """`compare_pc_depths` runs PC at each requested depth and
    reports edge differences. For a collider DAG with 3 variables:
    - depth=0: X-Y removed → adjacency {X-Z, Y-Z}
    - depth=1: same adjacency (no new conditioning sets reach
      depth 1 with only 3 vars — the only Z-set of size 1 is
      {Z}, which would test X⫫Y|Z, but that edge is already gone).

    So the depth comparison should report zero changes between
    depth=0 and depth=1. Catches mutations on `compare_pc_depths`'s
    diff machinery."""
    df = _collider_dataframe()
    diff = compare_pc_depths(
        df, variables=('X', 'Y', 'Z'),
        depths=(0, 1), alpha=_PC_ALPHA,
    )
    # `compare_pc_depths` returns one EdgeDiff for the transition.
    assert diff.depth_low == 0 and diff.depth_high == 1
    assert diff.low_only == frozenset(), (
        f'edges at depth 0 but not 1: {diff.low_only!r}; '
        f'collider has no further edges to remove (X-Y already '
        f'gone at depth 0)'
    )
    assert diff.high_only == frozenset(), (
        f'edges at depth 1 but not 0: {diff.high_only!r}; PC only '
        f'removes edges across deepening — never adds'
    )
    # Common edges are the surviving collider arms.
    assert diff.common == frozenset({
        frozenset({'X', 'Z'}), frozenset({'Y', 'Z'}),
    })


# ============ Marginal independence is exact (sanity) ============

def test_marginal_x_y_independence_holds_in_collider_construction() -> None:
    """Sanity that the construction is what we say: X and Y are
    independent draws → marginal Spearman ρ ≈ 0, p > alpha. If
    this fails, the rest of the collider tests would silently
    leak X-Y dependence into the adjacency."""
    from scipy.stats import spearmanr

    df = _collider_dataframe()
    x = np.asarray(df['X'].to_list(), dtype=np.float64)
    y = np.asarray(df['Y'].to_list(), dtype=np.float64)
    rho, p = spearmanr(x, y)
    # SE(ρ) at n=300 is ~1/sqrt(297) = 0.058. 4·SE ≈ 0.23.
    assert abs(float(rho)) < 0.15, (
        f'marginal ρ(X, Y) = {rho:.4f}; construction draws X, Y '
        f'independently — should be near zero'
    )
    assert float(p) > 0.05, (
        f'marginal p = {p:.4f} significant; X⫫Y is the '
        f'construction guarantee'
    )
