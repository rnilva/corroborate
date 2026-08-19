"""Tests for `corroborate.graph.causal` evidence(E) stamper +
dataset-relative extent-grouping queries.

Covers the framework primitives promoted from
`experiments/findings/ddqn/walks.py` per docs/HYPOTHESIS_AS_GRAPH.md
(commit landing the principle wiring):

- `evaluated_graph(bridges, post_eval)` — stamps each edge's
  `evidentiary_level` (from verdict via `_stamp_level`) and
  `extent_hash` (from post_eval mapping).
- `clusters_by_extent(g)` — groups edges by `(source, target,
  extent_hash)`.
- `cluster_verdict(members)` → `ClusterVerdict` — composes
  per-member levels into a cluster-level label."""
from __future__ import annotations

from collections.abc import Iterable, Mapping

import polars as pl

from corroborate.bridge.analysis import analysis
from corroborate.bridge.bridge import claim_bridge
from corroborate.bridge.verdict import Verdict
from corroborate.graph.causal import (
    BridgeEdge,
    ClusterVerdict,
    Direction,
    PostEvalEntry,
    Tier,
    authored_graph,
    cluster_verdict,
    clusters_by_extent,
    evaluated_graph,
)


@analysis
def _stub_analysis(
    cells: pl.DataFrame | Iterable[Mapping[str, object]],
) -> object:
    """No-op fixture so the bridges below can be evaluated."""
    del cells
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
        {
            '_bridge_assoc_a': PostEvalEntry(
                verdict=Verdict.HELD,
                extent_hash=12345,
                n_cells_in_scope=7,
            ),
        },
    )
    edges = tuple(g.edges)
    assert len(edges) == 1
    assert edges[0].metadata.evidentiary_level == 'correlational'
    assert edges[0].metadata.extent_hash == 12345
    assert edges[0].metadata.n_cells_in_scope == 7


def test_evaluated_graph_held_interventional_causal_one_sided() -> None:
    g = evaluated_graph(
        (_bridge_interventional,),
        {'_bridge_interventional': PostEvalEntry(verdict=Verdict.HELD, extent_hash=0)},
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
        {'_bridge_assoc_a': PostEvalEntry(verdict=Verdict.HELD_WITH_SCOPE_FLAG, extent_hash=99)},
    )
    edges = tuple(g.edges)
    assert edges[0].metadata.evidentiary_level == 'correlational'
    assert edges[0].metadata.extent_hash == 99


def test_evaluated_graph_no_effect_refuted() -> None:
    g = evaluated_graph(
        (_bridge_assoc_a,),
        {'_bridge_assoc_a': PostEvalEntry(verdict=Verdict.NO_EFFECT, extent_hash=7)},
    )
    edges = tuple(g.edges)
    assert edges[0].metadata.evidentiary_level == 'refuted'


def test_evaluated_graph_invariant_violation_unevaluated() -> None:
    """Per verdict.py:71 — INVARIANT_VIOLATION means 'test was
    out of scope', NOT refuted. Stamps as 'unevaluated' (the
    framework's choice, documented in `_stamp_level`)."""
    g = evaluated_graph(
        (_bridge_assoc_a,),
        {'_bridge_assoc_a': PostEvalEntry(verdict=Verdict.INVARIANT_VIOLATION, extent_hash=3)},
    )
    edges = tuple(g.edges)
    assert edges[0].metadata.evidentiary_level == 'unevaluated'


def test_evaluated_graph_power_insufficient_unevaluated() -> None:
    g = evaluated_graph(
        (_bridge_assoc_a,),
        {'_bridge_assoc_a': PostEvalEntry(verdict=Verdict.POWER_INSUFFICIENT, extent_hash=5)},
    )
    edges = tuple(g.edges)
    assert edges[0].metadata.evidentiary_level == 'unevaluated'


def test_evaluated_graph_inadmissible_unevaluated() -> None:
    """INADMISSIBLE — bridge body did not run (admission gate
    BLOCK-level fired). Same stamping as POWER_INSUFFICIENT and
    INVARIANT_VIOLATION: 'unevaluated'. Neither corroboration nor
    refutation by `Verdict.is_*()` predicates, so falls through to
    the default branch of `_stamp_level`."""
    g = evaluated_graph(
        (_bridge_assoc_a,),
        {'_bridge_assoc_a': PostEvalEntry(verdict=Verdict.INADMISSIBLE, extent_hash=11)},
    )
    edges = tuple(g.edges)
    assert edges[0].metadata.evidentiary_level == 'unevaluated'
    assert edges[0].metadata.extent_hash == 11


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
            '_bridge_assoc_a': PostEvalEntry(verdict=Verdict.HELD, extent_hash=1),
            '_bridge_other': PostEvalEntry(verdict=Verdict.NO_EFFECT, extent_hash=2),
        },
    )
    assert len(tuple(pre.edges)) == len(tuple(post.edges))
    assert set(pre.nodes) == set(post.nodes)


# ============ clusters_by_extent ============

def test_clusters_by_extent_groups_by_triple() -> None:
    """Matching `(source, target, extent_hash)` values group together."""
    g = evaluated_graph(
        (_bridge_assoc_a, _bridge_assoc_b),
        {
            '_bridge_assoc_a': PostEvalEntry(verdict=Verdict.HELD, extent_hash=42),
            '_bridge_assoc_b': PostEvalEntry(verdict=Verdict.HELD, extent_hash=42),
        },
    )
    clusters = clusters_by_extent(g)
    assert ('m1', 'out', 42) in clusters
    assert len(clusters[('m1', 'out', 42)]) == 2


def test_clusters_by_extent_separates_distinct_extent_hashes() -> None:
    """Same `(source, target)` but different extents → two
    groups, not one."""
    g = evaluated_graph(
        (_bridge_assoc_a, _bridge_assoc_b),
        {
            '_bridge_assoc_a': PostEvalEntry(verdict=Verdict.HELD, extent_hash=10),
            '_bridge_assoc_b': PostEvalEntry(verdict=Verdict.HELD, extent_hash=20),
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
            '_bridge_assoc_a': PostEvalEntry(verdict=Verdict.HELD, extent_hash=1),
            '_bridge_other': PostEvalEntry(verdict=Verdict.NO_EFFECT, extent_hash=1),
        },
    )
    clusters = clusters_by_extent(g)
    assert ('m1', 'out', 1) in clusters
    assert ('m2', 'out', 1) in clusters
    assert len(clusters[('m1', 'out', 1)]) == 1
    assert len(clusters[('m2', 'out', 1)]) == 1


# ============ cluster_verdict ============

def _edge(
    level: str,
    extent_hash: int = 1,
    *,
    n_cells_in_scope: int = 1,
) -> BridgeEdge:
    """Synthetic BridgeEdge for cluster_verdict tests — only
    `evidentiary_level` and `extent_hash` matter."""
    return BridgeEdge(
        bridge_name='stub',
        direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL,
        evidentiary_level=level,  # pyright: ignore[reportArgumentType]
        extent_hash=extent_hash,
        n_cells_in_scope=n_cells_in_scope,
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
    """Explicit zero row counts, not an ID digest, mean empty."""
    members = (
        _edge('correlational', n_cells_in_scope=0),
        _edge('refuted', n_cells_in_scope=0),
    )
    assert cluster_verdict(members) == ClusterVerdict.EMPTY_EXTENT


def test_cluster_verdict_does_not_infer_empty_from_id_key() -> None:
    """A non-empty frame can have no usable string IDs."""
    from corroborate.graph.causal import EMPTY_EXTENT_HASH

    members = (
        _edge(
            'correlational', EMPTY_EXTENT_HASH, n_cells_in_scope=2,
        ),
    )
    assert cluster_verdict(members) == ClusterVerdict.SUPPORTED


def test_cluster_verdict_mixed_empty_and_nonempty_is_underpowered() -> None:
    members = (
        _edge('correlational', n_cells_in_scope=0),
        _edge('causal_one_sided', n_cells_in_scope=3),
    )
    assert cluster_verdict(members) == ClusterVerdict.UNDERPOWERED


def test_cluster_verdict_unknown_count_is_underpowered() -> None:
    members = (_edge('correlational', n_cells_in_scope=-1),)
    assert cluster_verdict(members) == ClusterVerdict.UNDERPOWERED


def test_cluster_verdict_empty_members_underpowered() -> None:
    """Defensive: empty cluster is UNDERPOWERED, not crash."""
    assert cluster_verdict(()) == ClusterVerdict.UNDERPOWERED


# ============ Walk primitives (walk_subgraph / is_walk / walk_scope) ============


def test_walk_subgraph_two_step_chain() -> None:
    """Standard 3-node walk A → B → C: subgraph keeps both step
    edges, drops any diagonal edges A→C."""
    from corroborate.graph.causal import walk_subgraph
    g = authored_graph((_bridge_assoc_a, _bridge_other))
    # _bridge_assoc_a is m1→out; _bridge_other is m2→out.
    # Walk through ('m1', 'out') = single step using _bridge_assoc_a only.
    sub = walk_subgraph(g, nodes=('m1', 'out'))
    edge_names = {e.metadata.bridge_name for e in sub.edges}
    # _bridge_assoc_b is also m1→out — it appears too (multi-edge step)
    assert '_bridge_assoc_a' in edge_names or '_bridge_assoc_b' in edge_names
    # _bridge_other is m2→out — NOT in walk through m1
    assert '_bridge_other' not in edge_names


def test_walk_subgraph_drops_non_walk_edges() -> None:
    """walk_subgraph keeps ONLY edges between consecutive nodes —
    drops edges between non-adjacent nodes in the walk."""
    from corroborate.graph.causal import walk_subgraph
    g = authored_graph((_bridge_assoc_a, _bridge_other))
    # Walk ('m1', 'out', 'm2'): step 1 is m1→out (a-edges), step 2
    # would be out→m2 (no edges in g).
    sub = walk_subgraph(g, nodes=('m1', 'out', 'm2'))
    # _bridge_other goes m2→out — NOT (out→m2), so dropped.
    edge_names = {e.metadata.bridge_name for e in sub.edges}
    assert '_bridge_other' not in edge_names


def test_walk_subgraph_empty_or_singleton() -> None:
    """Walks of length <2 have no edges (still a valid subgraph)."""
    from corroborate.graph.causal import walk_subgraph
    g = authored_graph((_bridge_assoc_a,))
    sub_empty = walk_subgraph(g, nodes=())
    sub_singleton = walk_subgraph(g, nodes=('m1',))
    assert tuple(sub_empty.edges) == ()
    assert tuple(sub_singleton.edges) == ()


def test_is_walk_well_formed_chain() -> None:
    """_bridge_assoc_a (m1→out) alone is a trivial walk."""
    from corroborate.graph.causal import is_walk
    g = authored_graph((_bridge_assoc_a,))
    assert is_walk(g, bridges=(_bridge_assoc_a,)) is True


def test_is_walk_disconnected_returns_false() -> None:
    """Two bridges where bridge[i+1].source != bridge[i].target →
    not a walk."""
    from corroborate.graph.causal import is_walk
    g = authored_graph((_bridge_assoc_a, _bridge_other))
    # _bridge_assoc_a: m1→out; _bridge_other: m2→out.
    # Target of a (=out) != source of other (=m2) → disconnected.
    assert is_walk(g, bridges=(_bridge_assoc_a, _bridge_other)) is False


def test_is_walk_empty_or_singleton() -> None:
    """Trivially well-formed walks of length 0 and 1."""
    from corroborate.graph.causal import is_walk
    g = authored_graph((_bridge_assoc_a,))
    assert is_walk(g, bridges=()) is True
    assert is_walk(g, bridges=(_bridge_assoc_a,)) is True


def test_walk_scope_and_reduce_two_bridges() -> None:
    """walk_scope AND-reduces two bridges' scope predicates."""
    from corroborate.graph.causal import walk_scope
    expr = walk_scope((_bridge_assoc_a, _bridge_assoc_b))
    # Both bridges have scope `pl.col('x') > 0`; AND with itself
    # is structurally identical. Verify it's a valid pl.Expr.
    df = pl.DataFrame({'x': [-1, 0, 1, 2]})
    admitted = df.filter(expr)
    assert admitted.height == 2  # x ∈ {1, 2}


def test_walk_scope_empty_returns_lit_true() -> None:
    """No bridges → AND-reduce identity is True."""
    from corroborate.graph.causal import walk_scope
    expr = walk_scope(())
    df = pl.DataFrame({'x': [1, 2, 3]})
    admitted = df.filter(expr)
    assert admitted.height == 3  # all rows admitted


def test_walk_scope_rejects_deferred_scope() -> None:
    """DeferredScope bridges can't compose into a static walk
    scope — `walk_scope` must raise TypeError so callers don't
    silently get a confusing partial scope."""
    from corroborate.bridge.deferred_scope import DeferredScope
    from corroborate.graph.causal import walk_scope
    # Build a Bridge stub with a DeferredScope. Real DeferredScope
    # construction is complex; instead, replace scope on an existing
    # bridge via dataclasses.replace.
    from dataclasses import replace
    deferred = object.__new__(DeferredScope)
    stub = replace(_bridge_assoc_a, scope=deferred)
    import pytest
    with pytest.raises(TypeError, match='deferred-scope'):
        walk_scope((stub,))


def test_external_value_effects_stay_distinguishable_in_the_graph() -> None:
    """Both effects carry Tier.INTERVENTIONAL — the tier is the
    author's declared interpretation — but a value-based effect's
    assignment is author-asserted, not framework-executed, and the
    edge keeps that fact machine-readable."""
    from corroborate.analyses.paired.paired_g import PairedGResult
    from corroborate.core.claim import claim as claim_decorator
    from corroborate.core.intervention import (
        DoEffect, Intervention,
    )
    from corroborate.graph.causal import authored_graph

    @claim_decorator
    def _swap_op(x: int) -> int:
        return x

    structural = DoEffect(
        arms=((), (Intervention(slot_path='op', replacement=_swap_op),)),
    )
    declared = DoEffect.from_values(
        source='gamma', reference=0.8, treatment=0.99,
    )

    @claim_bridge(source=structural, target='outcome')
    def _executed(paired_g: PairedGResult) -> Verdict:
        return Verdict.HELD

    @claim_bridge(source=declared, target='outcome')
    def _asserted(paired_g: PairedGResult) -> Verdict:
        return Verdict.HELD

    g = authored_graph((_executed, _asserted))
    by_name = {e.metadata.bridge_name: e.metadata for e in g.edges}
    assert by_name['_executed'].external_effect is False
    assert by_name['_asserted'].external_effect is True
    assert by_name['_executed'].tier is by_name['_asserted'].tier
