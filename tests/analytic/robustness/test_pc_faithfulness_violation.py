"""Robustness probe: PC under faithfulness violations.

The PC algorithm assumes **faithfulness**: distribution-level
conditional independence ↔ graph-level d-separation. This
assumption fails when paths cancel exactly. The canonical case:

    True DAG:  X → Y (direct, coefficient +1)
               X → Z (coefficient +1)
               Z → Y (coefficient -1)

    Structural Y = X - Z + ε; expanding Z = X + ε_z:
        Y = X - (X + ε_z) + ε = -ε_z + ε
        Cov(X, Y) = Var(X) - Cov(X, Z) = 1 - 1 = 0

The marginal correlation between X and Y is exactly zero by
construction, even though the DAG has a direct X → Y edge.
PC's depth-0 marginal CI test removes the edge, producing
the WRONG skeleton.

Conditioning on Z reveals the truth: Cov(X, Y | Z) = Var(X) ≠ 0.
But PC's algorithm doesn't try depth-1 for edges already removed
at depth-0.

This probe pins:
  - That PC indeed produces the wrong adjacency on this fixture
  - That tightening alpha doesn't fix it (the marginal CI is
    asymptotically exact zero — alpha-thresholding only matters
    when the truth is borderline)
  - The structural diagnostic: partial cor(X, Y | Z) is large

Substrate-author guidance: PC's adjacency is robust to
distribution-level independence between non-adjacent nodes; it
is NOT robust to cancelling paths between adjacent nodes. The
framework cannot detect faithfulness violations from observational
data alone — they're a fundamental identifiability gap. Domain
knowledge is the only safeguard.

Empirical numbers anchored to deterministic seed=0; reproducible
across processes.
"""
from __future__ import annotations

import zlib

import numpy as np
import polars as pl

from corroborate.graph.discovery import (
    discover_adjacency,
    partial_spearman_rho,
)


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


def _faithfulness_violation_corpus(
    n: int = 500,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, pl.DataFrame,
]:
    """Construct (X, Z, Y) where X → Y direct + X → Z → Y indirect
    cancel marginally. Used by the probes below."""
    rng = np.random.default_rng(_det_seed('faithfulness', n))
    x = rng.normal(0, 1, n)
    z = x + rng.normal(0, 0.5, n)             # X → Z
    y = x - z + rng.normal(0, 0.3, n)         # X → Y (+1) + Z → Y (-1)
    df = pl.DataFrame({'x': x, 'z': z, 'y': y})
    return x, z, y, df


def test_marginal_correlation_is_near_zero_by_construction() -> None:
    """**Sanity**: the construction does what we claim.
    Marginal cor(X, Y) ≈ 0 within sampling SE.

    SE(r) at n=500 with population r=0: SE ≈ 1/√500 ≈ 0.045.
    Bound |cor| < 4·SE = 0.18 absorbs sampling drift while
    confirming the population correlation is structurally zero."""
    x, _, y, _ = _faithfulness_violation_corpus(n=500)
    r = float(np.corrcoef(x, y)[0, 1])
    assert abs(r) < 0.18, (
        f'marginal cor(X, Y) = {r:.4f}; construction targets '
        f'population r = 0. A larger r means the cancellation '
        f'didn\'t fire (probe is broken).'
    )


def test_partial_correlation_reveals_true_dependence() -> None:
    """**The hidden truth**: cor(X, Y | Z) is large positive
    despite cor(X, Y) ≈ 0. The X → Y direct edge IS there;
    conditioning on the cancelling mediator unmasks it.

    Closed form: cor(X, Y | Z) = Cov(X, Y | Z) / √(Var(X|Z)·Var(Y|Z)).
    With our construction, this is well above 0.6 — the bound
    `partial_r > 0.5` confirms the edge is structurally there
    and PC just can't see it via marginal CI alone."""
    x, z, y, _ = _faithfulness_violation_corpus(n=500)
    rho, p = partial_spearman_rho(x, y, z)
    assert rho > 0.5, (
        f'partial cor(X, Y | Z) = {rho:.4f}; expected > 0.5 '
        f'(the X → Y direct edge is structurally large; '
        f'conditioning on the cancelling mediator Z reveals it).'
    )
    assert p < 0.001, (
        f'partial p = {p:.4e}; expected < 0.001 (large partial r '
        f'at n=500 should be unambiguously significant).'
    )


def test_pc_removes_real_edge_under_faithfulness_violation() -> None:
    """**The bug PC cannot prevent**: PC's adjacency MISSES the
    X → Y direct edge because marginal CI removes it at depth 0.

    Pin: `frozenset({x, y}) NOT in adj.edges`. The framework
    silently produces a skeleton that disagrees with the true
    DAG; downstream consumers (DoWhy backdoor with this DAG,
    causal-chain claims) will compute on the wrong graph.

    This is a fundamental identifiability gap, not a fixable bug.
    Implementation authors must validate the DAG against domain
    knowledge — PC's output is the maximum-likelihood DAG given
    only the observational distribution; that's not the same as
    the structural truth when faithfulness fails."""
    _, _, _, df = _faithfulness_violation_corpus(n=500)
    adj = discover_adjacency(
        df, variables=('x', 'z', 'y'),
        alpha=0.05, max_conditioning=1,
    )
    assert frozenset({'x', 'y'}) not in adj.edges, (
        f'PC kept the X-Y edge despite faithfulness violation. '
        f'edges = {sorted(tuple(sorted(e)) for e in adj.edges)}. '
        f'If PC starts seeing this case correctly, the framework '
        f'or substrate author needs to update this test.'
    )
    # PC keeps the OTHER edges that are correctly identified.
    assert frozenset({'x', 'z'}) in adj.edges
    assert frozenset({'z', 'y'}) in adj.edges


def test_tightening_alpha_does_not_recover_real_edge() -> None:
    """**No simple fix**: lowering alpha (more conservative
    edge-removal) doesn't help. The marginal independence is
    structurally exact, not borderline.

    At α = 1e-6 (extremely conservative), the marginal CI at
    population r ≈ 0 still passes (p > 1e-6 with overwhelming
    probability), so PC still removes the edge.

    Pin that adjacency is identical at three orders of magnitude
    on alpha — confirming this isn't an alpha-tuning issue."""
    _, _, _, df = _faithfulness_violation_corpus(n=500)
    edges_alphas: dict[float, frozenset[frozenset[str]]] = {}
    for alpha in (0.05, 0.001, 1e-6):
        adj = discover_adjacency(
            df, variables=('x', 'z', 'y'),
            alpha=alpha, max_conditioning=1,
        )
        edges_alphas[alpha] = adj.edges
    assert all(
        frozenset({'x', 'y'}) not in e for e in edges_alphas.values()
    ), (
        f'X-Y edge unexpectedly recovered at some alpha; '
        f'edges per alpha: {edges_alphas}'
    )
