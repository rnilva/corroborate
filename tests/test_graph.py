"""Tests for `corroborate.graph` — generic Graph[N, M] primitive.

Covers construction (with_node, with_edge), adjacency
(successors/predecessors/edges_between), reachability, paths,
multigraph semantics, subgraph projection, and structural diff."""
from __future__ import annotations

from corroborate.graph import Edge, Graph


# ============ Construction + identity ============

def test_empty_graph_has_no_nodes_or_edges() -> None:
    g: Graph[str, str] = Graph()
    assert g.nodes == frozenset()
    assert g.edges == ()


def test_with_node_idempotent() -> None:
    g: Graph[str, str] = Graph().with_node('a').with_node('a')
    assert g.nodes == frozenset({'a'})


def test_with_edge_adds_both_endpoints_as_nodes() -> None:
    g: Graph[str, str] = Graph().with_edge('a', 'b', 'meta')
    assert g.nodes == frozenset({'a', 'b'})
    assert len(g.edges) == 1
    assert g.edges[0] == Edge(source='a', target='b', metadata='meta')


def test_with_edge_multigraph_keeps_duplicates() -> None:
    """Two edges with the same (source, target) but distinct
    metadata are both kept."""
    g: Graph[str, str] = (
        Graph()
        .with_edge('a', 'b', 'kind1')
        .with_edge('a', 'b', 'kind2')
    )
    assert len(g.edges) == 2
    assert {e.metadata for e in g.edges} == {'kind1', 'kind2'}


def test_with_edge_same_metadata_kept_under_multigraph() -> None:
    """Identical edges (same source/target/metadata) are NOT
    deduplicated by the primitive — multigraph semantics. Caller
    handles dedup."""
    g: Graph[str, str] = (
        Graph()
        .with_edge('a', 'b', 'm')
        .with_edge('a', 'b', 'm')
    )
    assert len(g.edges) == 2


# ============ Adjacency ============

def test_successors_and_predecessors() -> None:
    g: Graph[str, str] = (
        Graph()
        .with_edge('a', 'b', 'm')
        .with_edge('b', 'c', 'm')
        .with_edge('a', 'c', 'm')
    )
    assert g.successors('a') == {'b', 'c'}
    assert g.successors('b') == {'c'}
    assert g.successors('c') == set()
    assert g.predecessors('c') == {'a', 'b'}


def test_edges_between_returns_all_multi_edges() -> None:
    g: Graph[str, str] = (
        Graph()
        .with_edge('a', 'b', 'm1')
        .with_edge('a', 'b', 'm2')
    )
    es = g.edges_between('a', 'b')
    assert len(es) == 2
    assert {e.metadata for e in es} == {'m1', 'm2'}


# ============ Reachability + paths ============

def test_reachable_walks_transitively() -> None:
    g: Graph[str, str] = (
        Graph()
        .with_edge('a', 'b', 'm')
        .with_edge('b', 'c', 'm')
        .with_edge('c', 'd', 'm')
    )
    assert g.reachable('a', 'd')
    assert not g.reachable('d', 'a')


def test_reachable_self_loop_requires_node_in_graph() -> None:
    g: Graph[str, str] = Graph().with_node('a')
    assert g.reachable('a', 'a')
    g_empty: Graph[str, str] = Graph()
    assert not g_empty.reachable('a', 'a')


def test_first_path_returns_edge_sequence() -> None:
    g: Graph[str, str] = (
        Graph()
        .with_edge('a', 'b', 'm1')
        .with_edge('b', 'c', 'm2')
    )
    path = g.first_path('a', 'c')
    assert path is not None
    assert len(path) == 2
    assert path[0].source == 'a' and path[0].target == 'b'
    assert path[1].source == 'b' and path[1].target == 'c'


def test_first_path_unreachable_returns_none() -> None:
    g: Graph[str, str] = Graph().with_edge('a', 'b', 'm')
    assert g.first_path('b', 'a') is None


# ============ Subgraph + diff ============

def test_subgraph_keeps_only_selected_nodes_and_their_edges() -> None:
    g: Graph[str, str] = (
        Graph()
        .with_edge('a', 'b', 'm')
        .with_edge('b', 'c', 'm')
        .with_edge('c', 'd', 'm')
    )
    sub = g.subgraph(['a', 'b', 'c'])
    assert sub.nodes == frozenset({'a', 'b', 'c'})
    assert {(e.source, e.target) for e in sub.edges} == {
        ('a', 'b'), ('b', 'c'),
    }


def test_diff_empty_when_graphs_identical() -> None:
    g1: Graph[str, str] = Graph().with_edge('a', 'b', 'm')
    g2: Graph[str, str] = Graph().with_edge('a', 'b', 'm')
    d = g1.diff(g2)
    assert d.is_empty()


def test_diff_reports_added_and_removed_edges() -> None:
    g1: Graph[str, str] = Graph().with_edge('a', 'b', 'm')
    g2: Graph[str, str] = (
        Graph().with_edge('a', 'b', 'm').with_edge('b', 'c', 'n')
    )
    d = g2.diff(g1)
    assert d.edges_only_in_self == (
        Edge(source='b', target='c', metadata='n'),
    )
    assert d.edges_only_in_other == ()
    assert d.nodes_only_in_self == frozenset({'c'})


def test_diff_multiset_semantics() -> None:
    """Two copies of the same edge in self vs one in other → one
    copy in `edges_only_in_self`. Multiset semantics preserved."""
    g1: Graph[str, str] = (
        Graph().with_edge('a', 'b', 'm').with_edge('a', 'b', 'm')
    )
    g2: Graph[str, str] = Graph().with_edge('a', 'b', 'm')
    d = g1.diff(g2)
    assert len(d.edges_only_in_self) == 1
    assert d.edges_only_in_self[0].metadata == 'm'


def test_diff_conflicting_pairs() -> None:
    """Same (source, target) with different metadata in each graph
    → reported as a conflicting pair."""
    g1: Graph[str, str] = Graph().with_edge('a', 'b', 'm1')
    g2: Graph[str, str] = Graph().with_edge('a', 'b', 'm2')
    d = g1.diff(g2)
    conflicts = d.conflicting_pairs()
    assert ('a', 'b') in conflicts
    self_edges, other_edges = conflicts[('a', 'b')]
    assert self_edges[0].metadata == 'm1'
    assert other_edges[0].metadata == 'm2'
