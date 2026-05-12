"""Tests for `corroborate.graph.causal` evidence(E) stamper +
cluster-identity queries.

Covers the framework primitives promoted from
`experiments/findings/ddqn/walks.py` per HYPOTHESIS_AS_GRAPH.md
(commit landing the principle wiring):

- `evaluated_graph(bridges, post_eval)` — stamps each edge's
  `evidentiary_level` (from verdict via `_stamp_level`) and
  `extent_hash` (from post_eval mapping).
- `clusters_by_extent(g)` — groups edges by `(source, target,
  extent_hash)`.
- `cluster_verdict(members)` → `ClusterVerdict` — composes
  per-member levels into a cluster-level label."""
from __future__ import annotations

import polars as pl

from corroborate.bridge.analysis import analysis
from corroborate.bridge.bridge import claim_bridge
from corroborate.bridge.verdict import Verdict
from corroborate.graph.causal import (
    BridgeEdge,
    ClusterVerdict,
    Direction,
    Tier,
    authored_graph,
    cluster_verdict,
    clusters_by_extent,
    evaluated_graph,
)


@analysis
def _stub_analysis(cells: list[object]) -> object:
    """No-op fixture so the bridges below can be evaluated."""
    return object()


# Test bridges — pure-string source/target keeps tier
# ASSOCIATIONAL by default. Used across all evaluated_graph tests.
@claim_bridge(
    source='m1', target='out',
    direction=Direction.DIRECT, tier=Tier.ASSOCIATIONAL,
    scope=pl.col('x') > 0, pair_by=(),
)
def _bridge_assoc_a(_stub_analysis: object) -> Verdict:
    return Verdict.HELD


@claim_bridge(
    source='m1', target='out',
    direction=Direction.DIRECT, tier=Tier.ASSOCIATIONAL,
    scope=pl.col('x') > 0, pair_by=(),
)
def _bridge_assoc_b(_stub_analysis: object) -> Verdict:
    return Verdict.HELD


@claim_bridge(
    source='m2', target='out',
    direction=Direction.DIRECT, tier=Tier.ASSOCIATIONAL,
    scope=pl.col('x') > 0, pair_by=(),
)
def _bridge_other(_stub_analysis: object) -> Verdict:
    return Verdict.NO_EFFECT


# Interventional-tier bridge: tier comes from explicit Tier.INTERVENTIONAL
# (DoEffect would auto-promote tier, but a plain measurable
# source is enough to lock the test invariant).
@claim_bridge(
    source='m1', target='out',
    direction=Direction.DIRECT, tier=Tier.INTERVENTIONAL,
    scope=pl.col('x') > 0, pair_by=(),
)
def _bridge_interventional(_stub_analysis: object) -> Verdict:
    return Verdict.HELD


# ============ evaluated_graph stamping ============

def test_evaluated_graph_held_associational_correlational() -> None:
    g = evaluated_graph(
        (_bridge_assoc_a,),
        {'_bridge_assoc_a': (Verdict.HELD, 12345)},
    )
    edges = tuple(g.edges)
    assert len(edges) == 1
    assert edges[0].metadata.evidentiary_level == 'correlational'
    assert edges[0].metadata.extent_hash == 12345


def test_evaluated_graph_held_interventional_causal_one_sided() -> None:
    g = evaluated_graph(
        (_bridge_interventional,),
        {'_bridge_interventional': (Verdict.HELD, 0)},
    )
    edges = tuple(g.edges)
    assert edges[0].metadata.evidentiary_level == 'causal_one_sided'


def test_evaluated_graph_held_with_scope_flag_stamps_like_held() -> None:
    """HELD_WITH_SCOPE_FLAG is a corroboration verdict (per
    `Verdict.is_corroboration()`) — must stamp the same way HELD
    does. The promoted `_stamp_level` dispatches via the enum
    predicate, so this case is handled correctly even though no
    live bridge emits HELD_WITH_SCOPE_FLAG in shipped snapshots."""
    g = evaluated_graph(
        (_bridge_assoc_a,),
        {'_bridge_assoc_a': (Verdict.HELD_WITH_SCOPE_FLAG, 99)},
    )
    edges = tuple(g.edges)
    assert edges[0].metadata.evidentiary_level == 'correlational'
    assert edges[0].metadata.extent_hash == 99


def test_evaluated_graph_no_effect_refuted() -> None:
    g = evaluated_graph(
        (_bridge_assoc_a,),
        {'_bridge_assoc_a': (Verdict.NO_EFFECT, 7)},
    )
    edges = tuple(g.edges)
    assert edges[0].metadata.evidentiary_level == 'refuted'


def test_evaluated_graph_invariant_violation_unevaluated() -> None:
    """Per verdict.py:71 — INVARIANT_VIOLATION means 'test was
    out of scope', NOT refuted. Stamps as 'unevaluated' (the
    framework's choice, documented in `_stamp_level`)."""
    g = evaluated_graph(
        (_bridge_assoc_a,),
        {'_bridge_assoc_a': (Verdict.INVARIANT_VIOLATION, 3)},
    )
    edges = tuple(g.edges)
    assert edges[0].metadata.evidentiary_level == 'unevaluated'


def test_evaluated_graph_power_insufficient_unevaluated() -> None:
    g = evaluated_graph(
        (_bridge_assoc_a,),
        {'_bridge_assoc_a': (Verdict.POWER_INSUFFICIENT, 5)},
    )
    edges = tuple(g.edges)
    assert edges[0].metadata.evidentiary_level == 'unevaluated'


def test_evaluated_graph_missing_from_post_eval_unevaluated() -> None:
    """Bridges absent from `post_eval` keep authored defaults:
    `evidentiary_level='unevaluated'`, `extent_hash=0`."""
    g = evaluated_graph((_bridge_assoc_a,), {})
    edges = tuple(g.edges)
    assert edges[0].metadata.evidentiary_level == 'unevaluated'
    assert edges[0].metadata.extent_hash == 0


def test_evaluated_graph_preserves_topology() -> None:
    """Stamping doesn't add/remove edges — only mutates metadata."""
    bridges = (_bridge_assoc_a, _bridge_other)
    pre = authored_graph(bridges)
    post = evaluated_graph(
        bridges,
        {
            '_bridge_assoc_a': (Verdict.HELD, 1),
            '_bridge_other': (Verdict.NO_EFFECT, 2),
        },
    )
    assert len(tuple(pre.edges)) == len(tuple(post.edges))
    assert set(pre.nodes) == set(post.nodes)


# ============ clusters_by_extent ============

def test_clusters_by_extent_groups_by_triple() -> None:
    """Two bridges sharing `(source, target, extent_hash)` cluster
    together; the framework derives identity from the data."""
    g = evaluated_graph(
        (_bridge_assoc_a, _bridge_assoc_b),
        {
            '_bridge_assoc_a': (Verdict.HELD, 42),
            '_bridge_assoc_b': (Verdict.HELD, 42),
        },
    )
    clusters = clusters_by_extent(g)
    assert ('m1', 'out', 42) in clusters
    assert len(clusters[('m1', 'out', 42)]) == 2


def test_clusters_by_extent_separates_distinct_extent_hashes() -> None:
    """Same `(source, target)` but different extents → two
    clusters, not one. Cluster identity is the triple."""
    g = evaluated_graph(
        (_bridge_assoc_a, _bridge_assoc_b),
        {
            '_bridge_assoc_a': (Verdict.HELD, 10),
            '_bridge_assoc_b': (Verdict.HELD, 20),
        },
    )
    clusters = clusters_by_extent(g)
    assert ('m1', 'out', 10) in clusters
    assert ('m1', 'out', 20) in clusters
    assert len(clusters[('m1', 'out', 10)]) == 1
    assert len(clusters[('m1', 'out', 20)]) == 1


def test_clusters_by_extent_singletons_at_distinct_sources() -> None:
    """Distinct source nodes → distinct cluster keys → singletons."""
    g = evaluated_graph(
        (_bridge_assoc_a, _bridge_other),
        {
            '_bridge_assoc_a': (Verdict.HELD, 1),
            '_bridge_other': (Verdict.NO_EFFECT, 1),
        },
    )
    clusters = clusters_by_extent(g)
    assert ('m1', 'out', 1) in clusters
    assert ('m2', 'out', 1) in clusters
    assert len(clusters[('m1', 'out', 1)]) == 1
    assert len(clusters[('m2', 'out', 1)]) == 1


# ============ cluster_verdict ============

def _edge(level: str, extent_hash: int = 1) -> BridgeEdge:
    """Synthetic BridgeEdge for cluster_verdict tests — only
    `evidentiary_level` and `extent_hash` matter."""
    return BridgeEdge(
        bridge_name='stub',
        direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL,
        evidentiary_level=level,  # pyright: ignore[reportArgumentType]
        extent_hash=extent_hash,
    )


def test_cluster_verdict_supported_all_correlational() -> None:
    members = (_edge('correlational'), _edge('correlational'))
    assert cluster_verdict(members) == ClusterVerdict.SUPPORTED


def test_cluster_verdict_supported_mixed_correlational_causal() -> None:
    """SUPPORTED admits members at either admit rung
    (correlational + causal_one_sided)."""
    members = (
        _edge('correlational'),
        _edge('causal_one_sided'),
    )
    assert cluster_verdict(members) == ClusterVerdict.SUPPORTED


def test_cluster_verdict_refuted_any_refuted() -> None:
    """One refutation is enough to flip the cluster to REFUTED."""
    members = (
        _edge('correlational'),
        _edge('refuted'),
        _edge('causal_one_sided'),
    )
    assert cluster_verdict(members) == ClusterVerdict.REFUTED


def test_cluster_verdict_underpowered_mixed_unevaluated() -> None:
    """Mix of admit + unevaluated → UNDERPOWERED."""
    members = (
        _edge('correlational'),
        _edge('unevaluated'),
    )
    assert cluster_verdict(members) == ClusterVerdict.UNDERPOWERED


def test_cluster_verdict_empty_extent_all_empty() -> None:
    """All members admit zero cells (shared empty-frozenset hash)
    → EMPTY_EXTENT, even if their levels suggest otherwise."""
    empty_hash = hash(frozenset[str]())
    members = (
        _edge('correlational', empty_hash),
        _edge('refuted', empty_hash),
    )
    assert cluster_verdict(members) == ClusterVerdict.EMPTY_EXTENT


def test_cluster_verdict_empty_members_underpowered() -> None:
    """Defensive: empty cluster is UNDERPOWERED, not crash."""
    assert cluster_verdict(()) == ClusterVerdict.UNDERPOWERED
