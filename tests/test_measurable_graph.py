"""Tests for `corroborate.measurable_graph` — statistical graph
derived from per-step measurable series.

Covers:
1. `pairwise_correlations` builds a Graph with one edge per
   unordered pair, carrying Pearson r.
2. NaN handling: constant series, length-1 series, non-1-D fields.
3. Lexical edge ordering (deterministic regardless of dict
   insertion order).
4. `correlation_matrix_table` filters by |r| threshold + NaN.
5. `explained_by_claim_graph` returns True for claim-connected
   pairs (either direction), False for disjoint pairs."""
from __future__ import annotations

import math

import numpy as np

from corroborate.graph.computation import ComputationEdge
from corroborate.graph import Graph
from corroborate.graph.measurable import (
    Correlation,
    correlation_matrix_table,
    explained_by_claim_graph,
    pairwise_correlations,
)


# ============ pairwise_correlations ============

def test_pairwise_correlations_perfect_positive() -> None:
    """Identical series → r ≈ 1."""
    metrics = {
        'a': np.array([1.0, 2.0, 3.0, 4.0]),
        'b': np.array([1.0, 2.0, 3.0, 4.0]),
    }
    g = pairwise_correlations(metrics)
    assert g.nodes == frozenset({'a', 'b'})
    [edge] = g.edges
    assert math.isclose(edge.metadata.r, 1.0, abs_tol=1e-9)


def test_pairwise_correlations_perfect_negative() -> None:
    """Anti-correlated series → r ≈ -1."""
    metrics = {
        'a': np.array([1.0, 2.0, 3.0, 4.0]),
        'b': np.array([4.0, 3.0, 2.0, 1.0]),
    }
    g = pairwise_correlations(metrics)
    [edge] = g.edges
    assert math.isclose(edge.metadata.r, -1.0, abs_tol=1e-9)


def test_pairwise_correlations_constant_series_yields_nan() -> None:
    """A constant series has zero std; correlation is undefined."""
    metrics = {
        'varying': np.array([1.0, 2.0, 3.0]),
        'constant': np.array([5.0, 5.0, 5.0]),
    }
    g = pairwise_correlations(metrics)
    [edge] = g.edges
    assert math.isnan(edge.metadata.r)


def test_pairwise_correlations_skips_non_1d_fields() -> None:
    """Multi-D arrays (e.g., per-step (batch, n_actions) Q-tensors)
    don't form scalar series and are silently dropped."""
    metrics = {
        'reward': np.array([1.0, 0.5, 0.0]),
        'q_tensor': np.zeros((3, 2, 4)),  # 3-D, skipped
    }
    g = pairwise_correlations(metrics)
    assert g.nodes == frozenset({'reward'})  # q_tensor not registered
    assert g.edges == ()


def test_pairwise_correlations_n_choose_2_edges() -> None:
    """k scalar measurables → k*(k-1)/2 edges."""
    rng = np.random.default_rng(0)
    metrics = {f'm{i}': rng.standard_normal(20) for i in range(5)}
    g = pairwise_correlations(metrics)
    assert len(g.nodes) == 5
    assert len(g.edges) == 5 * 4 // 2  # 10


def test_pairwise_correlations_lexical_edge_order() -> None:
    """Edges use the lexically-first key as source for determinism
    across dict insertion orders."""
    metrics_a = {'b': np.arange(4, dtype=float), 'a': np.arange(4, dtype=float)}
    metrics_b = {'a': np.arange(4, dtype=float), 'b': np.arange(4, dtype=float)}
    g_a = pairwise_correlations(metrics_a)
    g_b = pairwise_correlations(metrics_b)
    assert g_a.edges == g_b.edges
    [e] = g_a.edges
    assert e.source == 'a' and e.target == 'b'


# ============ correlation_matrix_table ============

def test_correlation_matrix_table_sorts_by_abs_r() -> None:
    """Table output is sorted by |r| descending."""
    g = (
        Graph[str, Correlation]()
        .with_edge('a', 'b', Correlation(r=0.3))
        .with_edge('a', 'c', Correlation(r=-0.7))
        .with_edge('b', 'c', Correlation(r=0.1))
    )
    rows = correlation_matrix_table(g)
    assert len(rows) == 3
    # First row (largest |r|) should mention a <-> c.
    assert 'a' in rows[0] and 'c' in rows[0]


def test_correlation_matrix_table_threshold_filters() -> None:
    g = (
        Graph[str, Correlation]()
        .with_edge('a', 'b', Correlation(r=0.05))
        .with_edge('a', 'c', Correlation(r=0.5))
    )
    rows = correlation_matrix_table(g, threshold=0.1)
    assert len(rows) == 1
    assert 'c' in rows[0]


def test_correlation_matrix_table_drops_nan() -> None:
    g = (
        Graph[str, Correlation]()
        .with_edge('a', 'b', Correlation(r=float('nan')))
        .with_edge('a', 'c', Correlation(r=0.5))
    )
    rows = correlation_matrix_table(g)
    assert len(rows) == 1
    assert 'c' in rows[0]


# ============ explained_by_claim_graph ============

def test_explained_by_claim_graph_direct_edge() -> None:
    """A direct claim edge a→b explains corr(a, b)."""
    cg = Graph[str, ComputationEdge]().with_edge(
        'a', 'b', ComputationEdge(reader_arg='x'),
    )
    assert explained_by_claim_graph('a', 'b', cg)
    # And reverse direction (undirected reachability).
    assert explained_by_claim_graph('b', 'a', cg)


def test_explained_by_claim_graph_transitive_edge() -> None:
    """A path a→m→b also explains corr(a, b)."""
    cg = (
        Graph[str, ComputationEdge]()
        .with_edge('a', 'm', ComputationEdge(reader_arg='x'))
        .with_edge('m', 'b', ComputationEdge(reader_arg='y'))
    )
    assert explained_by_claim_graph('a', 'b', cg)
    assert explained_by_claim_graph('b', 'a', cg)


def test_explained_by_claim_graph_disconnected_components() -> None:
    """Two disjoint components — corr(a, c) is NOT explained."""
    cg = (
        Graph[str, ComputationEdge]()
        .with_edge('a', 'b', ComputationEdge(reader_arg='x'))
        .with_edge('c', 'd', ComputationEdge(reader_arg='y'))
    )
    assert not explained_by_claim_graph('a', 'c', cg)
    assert not explained_by_claim_graph('b', 'd', cg)


def test_explained_by_claim_graph_missing_node() -> None:
    """A measurable not in the claim graph at all → unexplained."""
    cg = Graph[str, ComputationEdge]().with_edge(
        'a', 'b', ComputationEdge(reader_arg='x'),
    )
    assert not explained_by_claim_graph('a', 'unknown', cg)
