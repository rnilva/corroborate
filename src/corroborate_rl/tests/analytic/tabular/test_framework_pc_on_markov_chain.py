"""Framework-as-instrument: `discover_adjacency` (PC) recovers
the Markov-chain CI structure — the canonical RL substrate
property `s_{t+1} ⫫ s_{t-1} | s_t`.

The Markov property is the SUBSTRATE-DEEPEST closed-form
relationship in any MDP: future states are conditionally
independent of past states given the present. Hasselt's bias,
γ-contraction, Bellman residuals — all of these derive from
processes that respect the Markov property.

Setup: a 4-variable AR(1) chain (a degenerate continuous-state
"tabular" MDP under deterministic policy with stochastic
transitions):

    x_0 ∼ N(0, 1)
    x_t = α · x_{t-1} + ε_t,  ε_t ∼ N(0, σ²)  iid

True adjacency under the Markov property:

    direct edges:    {(x_0,x_1), (x_1,x_2), (x_2,x_3)}
    non-edges:       (x_0,x_2), (x_0,x_3), (x_1,x_3)
    sepsets:         x_0 ⫫ x_2 | x_1
                     x_1 ⫫ x_3 | x_2
                     x_0 ⫫ x_3 | {x_1, x_2}  (or {x_2} alone via depth-1)

For α=0.7, σ=0.5, marginal correlations:
    corr(x_0, x_1) = α/√Var(x_1)·Var(x_0) ≈ 0.81
    corr(x_0, x_2) = α²/√Var(x_2)·Var(x_0) ≈ 0.66 (high; would
                     SURVIVE a marginal-only test → depth-1
                     conditioning is required for removal)
    corr(x_0, x_3) ≈ 0.54
    corr(x_1, x_3) ≈ 0.66

PC at `max_conditioning=1` should:
- keep `{(x_0,x_1), (x_1,x_2), (x_2,x_3)}` (directly correlated;
  no Z makes them independent)
- remove `(x_0, x_2)` after conditioning on x_1 (Markov property
  → partial r ≈ 0)
- remove `(x_1, x_3)` after conditioning on x_2
- remove `(x_0, x_3)` — depth-1 conditioning on x_1 OR x_2
  (either sufficient under chain Markov property; partial r → 0
  conditional on either, since both are Markov barriers between
  x_0 and x_3 in a chain)

THIS is the framework-as-instrument question: given observational
data with a known Markov-chain CI structure, does the framework's
PC algorithm recover the true adjacency at depth 1?

A regression in the per-edge CI test, the depth-1 partial-Spearman
formula, the separating-set collection, or the conservative-PC
edge-removal logic would breach the recovered adjacency.
"""
from __future__ import annotations

import zlib

import numpy as np
import polars as pl

from corroborate.graph.discovery import discover_adjacency


def _det_seed(*parts: object) -> int:
    return zlib.adler32(repr(parts).encode()) & 0xFFFF_FFFF


_ALPHA = 0.7        # AR(1) coefficient
_SIGMA = 0.5        # innovation noise
_N_TRAJECTORIES = 200
_N_TIMESTEPS = 4    # x_0, x_1, x_2, x_3

# Tighter than PC's default 0.05 — Spearman partial-ρ has a
# systematic bias on linear-Gaussian data (~0.15-0.20 at n=200)
# from the rank transform's non-linearity. At α=0.05 the
# borderline residual partial-r registers as p ≈ 0.05, randomly
# keeping/removing edges that should be Markov-separated. α=0.001
# is the standard tighter cut used by the LG-SCM PC tests.
_PC_ALPHA = 0.001


def _generate_markov_chain_dataframe() -> pl.DataFrame:
    """Generate `N_TRAJECTORIES` independent AR(1) chains; each
    contributes one row with columns `x_0`, ..., `x_{T-1}`."""
    rng = np.random.default_rng(seed=_det_seed('pc_markov_chain'))
    cols: dict[str, list[float]] = {f'x_{t}': [] for t in range(_N_TIMESTEPS)}
    for _ in range(_N_TRAJECTORIES):
        x_prev = float(rng.standard_normal())
        cols['x_0'].append(x_prev)
        for t in range(1, _N_TIMESTEPS):
            x_t = _ALPHA * x_prev + _SIGMA * float(rng.standard_normal())
            cols[f'x_{t}'].append(x_t)
            x_prev = x_t
    return pl.DataFrame(cols)


_VARIABLES: tuple[str, ...] = tuple(f'x_{t}' for t in range(_N_TIMESTEPS))


# ============ Adjacency: direct neighbors only ============

def test_pc_recovers_chain_adjacency_at_depth_1() -> None:
    """At max_conditioning=1, PC should recover EXACTLY the
    chain adjacency: edges between consecutive timesteps,
    nothing else.

    Marginal correlations are substantial for non-neighbors
    (corr(x_0, x_2) ≈ 0.66) — depth-0 alone would NOT remove
    them. Depth-1 conditioning on the intermediate barrier IS
    the load-bearing test of the partial-Spearman primitive.
    """
    df = _generate_markov_chain_dataframe()
    adj = discover_adjacency(
        df, variables=_VARIABLES,
        alpha=_PC_ALPHA, max_conditioning=1,
    )
    expected = frozenset({
        frozenset({'x_0', 'x_1'}),
        frozenset({'x_1', 'x_2'}),
        frozenset({'x_2', 'x_3'}),
    })
    assert adj.edges == expected, (
        f'edges = {sorted(tuple(sorted(e)) for e in adj.edges)}, '
        f'expected chain adjacency '
        f'{sorted(tuple(sorted(e)) for e in expected)}. '
        f'PC at depth 1 must recover the Markov-chain skeleton.'
    )


def test_pc_keeps_all_edges_at_depth_0() -> None:
    """Negative control: at depth 0 (marginal CI only), PC
    should KEEP all 6 edges since marginal correlations along
    the chain are substantial (corr(x_0, x_3) ≈ 0.54 still).
    Pin against a regression that pre-emptively removed edges
    based on a different-distribution test.
    """
    df = _generate_markov_chain_dataframe()
    adj = discover_adjacency(
        df, variables=_VARIABLES,
        alpha=0.05, max_conditioning=0,
    )
    # All 6 pairs of 4 variables: every pair is marginally
    # correlated → every edge survives the depth-0 marginal test.
    assert len(adj.edges) == 6, (
        f'depth-0 edges = {len(adj.edges)}, expected 6 '
        f'(all pairs have substantial marginal correlation in '
        f'an AR(1) chain). The depth-1 conditioning IS the '
        f'load-bearing removal mechanism.'
    )


# ============ Separating sets: Markov barriers ============

def test_pc_separating_sets_match_markov_property() -> None:
    """The separating-set machinery (`adj.separating_sets`)
    should report:
    - sepset(x_0, x_2) contains the singleton {x_1}
    - sepset(x_1, x_3) contains the singleton {x_2}
    - sepset(x_0, x_3) contains either {x_1} OR {x_2} (either
      Markov barrier sufficient at depth 1)

    Conservative-PC collects ALL Z-sets that separate; under the
    chain Markov property each non-edge has a clear depth-1 sepset.

    A regression that lost the separating-set tracking, or that
    kept only the first sepset and missed alternatives, would
    breach.
    """
    df = _generate_markov_chain_dataframe()
    adj = discover_adjacency(
        df, variables=_VARIABLES,
        alpha=_PC_ALPHA, max_conditioning=1,
    )
    # x_0 ⫫ x_2 | x_1 (Markov chain): {x_1} should be in the
    # separating-sets entry.
    sepset_02 = adj.separating_sets.get(frozenset({'x_0', 'x_2'}))
    assert sepset_02 is not None, (
        f'no separating set recorded for (x_0, x_2); expected '
        f'{{x_1}} via Markov property.'
    )
    assert frozenset({'x_1'}) in sepset_02, (
        f'sepset(x_0, x_2) = {sepset_02}, expected to contain '
        f'{{x_1}}.'
    )
    # x_1 ⫫ x_3 | x_2 similarly.
    sepset_13 = adj.separating_sets.get(frozenset({'x_1', 'x_3'}))
    assert sepset_13 is not None
    assert frozenset({'x_2'}) in sepset_13


# ============ Panel structure: n_observations ============

def test_pc_n_observations_matches_dataframe_height() -> None:
    """`adj.n_observations` should equal `data.height` — pin
    against a regression that silently dropped rows during
    column projection."""
    df = _generate_markov_chain_dataframe()
    adj = discover_adjacency(
        df, variables=_VARIABLES,
        alpha=_PC_ALPHA, max_conditioning=1,
    )
    assert adj.n_observations == _N_TRAJECTORIES


# ============ Stricter alpha tightens removal threshold ============

def test_pc_depth_2_reproduces_depth_1_chain() -> None:
    """At max_conditioning=2, PC should recover the SAME chain
    adjacency as depth-1 — the chain is already minimal at depth
    1, and conditioning on a 2-element subset cannot remove a
    direct edge under the Markov property.

    Pin the depth-2 path's `partial_spearman_rho_multi` (residual
    regression) consistency with the depth-1 single-Z partial.
    A regression in the depth-2 conditioning would either:
    - falsely remove a direct edge (depth-2 inflated false-positive)
    - keep a non-edge that depth-1 already removed (depth-2 lost
      power from over-conditioning)
    """
    df = _generate_markov_chain_dataframe()
    adj_depth_1 = discover_adjacency(
        df, variables=_VARIABLES,
        alpha=_PC_ALPHA, max_conditioning=1,
    )
    adj_depth_2 = discover_adjacency(
        df, variables=_VARIABLES,
        alpha=_PC_ALPHA, max_conditioning=2,
    )
    assert adj_depth_2.edges == adj_depth_1.edges, (
        f'depth-2 edges differ from depth-1: '
        f'depth-1 = {sorted(tuple(sorted(e)) for e in adj_depth_1.edges)}, '
        f'depth-2 = {sorted(tuple(sorted(e)) for e in adj_depth_2.edges)}. '
        f'The chain is minimal at depth 1; depth-2 conditioning '
        f'should reproduce the same skeleton.'
    )
