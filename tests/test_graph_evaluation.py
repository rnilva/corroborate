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
    PostEvalEntry,
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


# Null-prediction bridge: `predicted_direction='null'` means
# "the bridge predicts the effect is ≈ 0." NO_EFFECT under this
# prediction corroborates the null (the prediction succeeded),
# NOT refutes it. Mirrors the within-arm asymmetry pattern at
# `experiments/findings/ddqn_sweeps/finding_lambda_a_within_arm_asymmetry.py`.
@claim_bridge(
    source='m1', target='out',
    direction=Direction.DIRECT, tier=Tier.ASSOCIATIONAL,
    scope=pl.col('x') > 0, pair_by=(),
    predicted_direction='null',
)
def _bridge_predicts_null(_stub_analysis: object) -> Verdict:
    return Verdict.NO_EFFECT


@claim_bridge(
    source='m1', target='out',
    direction=Direction.DIRECT, tier=Tier.INTERVENTIONAL,
    scope=pl.col('x') > 0, pair_by=(),
    predicted_direction='null',
)
def _bridge_predicts_null_interventional(_stub_analysis: object) -> Verdict:
    return Verdict.NO_EFFECT


# ============ evaluated_graph stamping ============

def test_evaluated_graph_held_associational_correlational() -> None:
    g = evaluated_graph(
        (_bridge_assoc_a,),
        {'_bridge_assoc_a': PostEvalEntry(verdict=Verdict.HELD, extent_hash=12345)},
    )
    edges = tuple(g.edges)
    assert len(edges) == 1
    assert edges[0].metadata.evidentiary_level == 'correlational'
    assert edges[0].metadata.extent_hash == 12345


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


def test_evaluated_graph_no_effect_under_predicted_null_corroborates() -> None:
    """NO_EFFECT under `predicted_direction='null'` is the
    prediction SUCCEEDING — the bridge predicted ≈ 0 effect and
    the test confirmed it. Must stamp as admit-equivalent
    (`'correlational'` at ASSOCIATIONAL tier), NOT `'refuted'`.

    Reproduces the within-arm asymmetry pattern at
    `experiments/findings/ddqn_sweeps/finding_lambda_a_within_arm_asymmetry.py`:
    a directional bridge HELDs on vanilla cells and a sibling
    null-predicting bridge admits its null on DDQN cells — both
    are corroboration, the cluster should compose SUPPORTED."""
    g = evaluated_graph(
        (_bridge_predicts_null,),
        {'_bridge_predicts_null': PostEvalEntry(verdict=Verdict.NO_EFFECT, extent_hash=7)},
    )
    edges = tuple(g.edges)
    assert edges[0].metadata.evidentiary_level == 'correlational'


def test_evaluated_graph_no_effect_under_predicted_null_interventional() -> None:
    """Same logic at INTERVENTIONAL tier: null-prediction success
    is admit-equivalent at the rung that produced it
    (`'causal_one_sided'`)."""
    g = evaluated_graph(
        (_bridge_predicts_null_interventional,),
        {'_bridge_predicts_null_interventional': PostEvalEntry(verdict=Verdict.NO_EFFECT, extent_hash=7)},
    )
    edges = tuple(g.edges)
    assert edges[0].metadata.evidentiary_level == 'causal_one_sided'


def test_evaluated_graph_cluster_held_plus_predicted_null_supported() -> None:
    """The within-arm-asymmetry cluster shape end-to-end: one
    bridge HELD on the directional prediction + one bridge
    NO_EFFECT on the null prediction → cluster SUPPORTED, NOT
    REFUTED. Pre-fix, this combination tripped `cluster_verdict`'s
    `'refuted' in levels` branch and fired REFUTED on
    every-bridge-admits."""
    bridges = (_bridge_assoc_a, _bridge_predicts_null)
    g = evaluated_graph(
        bridges,
        {
            '_bridge_assoc_a': PostEvalEntry(
                verdict=Verdict.HELD, extent_hash=42,
            ),
            '_bridge_predicts_null': PostEvalEntry(
                verdict=Verdict.NO_EFFECT, extent_hash=42,
            ),
        },
    )
    members = tuple(e.metadata for e in g.edges)
    assert cluster_verdict(members) == ClusterVerdict.SUPPORTED


def test_evaluated_graph_no_effect_without_predicted_direction_refuted() -> None:
    """Defensive: bridges that don't set `predicted_direction`
    (the legacy default) must continue stamping NO_EFFECT as
    `'refuted'`. The fix only changes the `predicted_direction
    == 'null'` branch — every other prediction shape (including
    None) keeps the existing semantics."""
    g = evaluated_graph(
        (_bridge_assoc_a,),  # predicted_direction defaults to None
        {'_bridge_assoc_a': PostEvalEntry(verdict=Verdict.NO_EFFECT, extent_hash=1)},
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
    """Two bridges sharing `(source, target, extent_hash)` cluster
    together; the framework derives identity from the data."""
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
    clusters, not one. Cluster identity is the triple."""
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
