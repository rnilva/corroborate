"""Tests for `corroborate.causal_graph` — Pearl-tier-typed
BridgeEdges + composition algebra + bridged-evidence promotion."""
from __future__ import annotations

import pytest

from corroborate.bridge import BridgeResult
from corroborate.causal_graph import (
    BridgeEdge,
    Direction,
    InvalidTierTransition,
    Tier,
    build_causal_graph,
    chain_tier,
    compose_direction,
    promote_bridged_evidence,
)
from corroborate.verdict import Verdict


# ============ Direction ============

def test_direction_multiplication_identity_direct() -> None:
    assert Direction.DIRECT * Direction.DIRECT == Direction.DIRECT


def test_direction_multiplication_inverse_inverse_yields_direct() -> None:
    """Two negatives cancel."""
    assert Direction.INVERSE * Direction.INVERSE == Direction.DIRECT


def test_direction_multiplication_mixed_yields_inverse() -> None:
    assert Direction.DIRECT * Direction.INVERSE == Direction.INVERSE
    assert Direction.INVERSE * Direction.DIRECT == Direction.INVERSE


# ============ Tier ============

def test_tier_promote_associational_to_interventional() -> None:
    assert Tier.ASSOCIATIONAL.promote() == Tier.INTERVENTIONAL


def test_tier_promote_at_top_raises() -> None:
    with pytest.raises(InvalidTierTransition):
        Tier.INTERVENTIONAL.promote()


def test_tier_demote_interventional_to_associational() -> None:
    assert Tier.INTERVENTIONAL.demote() == Tier.ASSOCIATIONAL


def test_tier_demote_at_bottom_raises() -> None:
    with pytest.raises(InvalidTierTransition):
        Tier.ASSOCIATIONAL.demote()


# ============ Chain composition ============

def test_compose_direction_empty_chain_is_direct_identity() -> None:
    assert compose_direction([]) == Direction.DIRECT


def test_compose_direction_two_inverse_edges_yield_direct() -> None:
    e1 = BridgeEdge(
        bridge_name='a', direction=Direction.INVERSE,
        tier=Tier.ASSOCIATIONAL, evidentiary_level='correlational',
    )
    e2 = BridgeEdge(
        bridge_name='b', direction=Direction.INVERSE,
        tier=Tier.ASSOCIATIONAL, evidentiary_level='correlational',
    )
    assert compose_direction([e1, e2]) == Direction.DIRECT


def test_compose_direction_one_inverse_one_direct_yields_inverse() -> None:
    e1 = BridgeEdge(
        bridge_name='a', direction=Direction.INVERSE,
        tier=Tier.ASSOCIATIONAL, evidentiary_level='correlational',
    )
    e2 = BridgeEdge(
        bridge_name='b', direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL, evidentiary_level='correlational',
    )
    assert compose_direction([e1, e2]) == Direction.INVERSE


def test_chain_tier_empty_is_associational() -> None:
    assert chain_tier([]) == Tier.ASSOCIATIONAL


def test_chain_tier_takes_minimum_along_chain() -> None:
    e_intv = BridgeEdge(
        bridge_name='a', direction=Direction.DIRECT,
        tier=Tier.INTERVENTIONAL, evidentiary_level='causal_one_sided',
    )
    e_assoc = BridgeEdge(
        bridge_name='b', direction=Direction.DIRECT,
        tier=Tier.ASSOCIATIONAL, evidentiary_level='correlational',
    )
    # Intervention + Association → chain is only Associational.
    assert chain_tier([e_intv, e_assoc]) == Tier.ASSOCIATIONAL
    # Pure Intervention chain.
    assert chain_tier([e_intv, e_intv]) == Tier.INTERVENTIONAL


# ============ build_causal_graph ============

def _br(
    name: str, targets: tuple[str, ...], verdict: Verdict,
    stats: dict[str, float | int | bool | str] | None = None,
) -> BridgeResult:
    return BridgeResult(
        verdict=verdict, reason='test',
        stats=stats or {}, name=name, targets=targets,
    )


def test_build_causal_graph_single_target_creates_node_only() -> None:
    g = build_causal_graph([
        _br('node_only', ('alpha',), Verdict.HELD),
    ])
    assert 'alpha' in g.nodes
    assert len(g.edges) == 0


def test_build_causal_graph_binary_held_correlational() -> None:
    """HELD without `tier=interventional` → ASSOCIATIONAL +
    correlational level."""
    g = build_causal_graph([
        _br('couples', ('a', 'b'), Verdict.HELD,
            stats={'rho': 0.7}),
    ])
    assert ('a', 'b') in {(e.source, e.target) for e in g.edges}
    edge = g.edges[0].metadata
    assert edge.bridge_name == 'couples'
    assert edge.direction == Direction.DIRECT
    assert edge.tier == Tier.ASSOCIATIONAL
    assert edge.evidentiary_level == 'correlational'
    assert edge.rho == 0.7


def test_build_causal_graph_negative_rho_yields_inverse_direction() -> None:
    g = build_causal_graph([
        _br('inv', ('a', 'b'), Verdict.HELD, stats={'rho': -0.5}),
    ])
    assert g.edges[0].metadata.direction == Direction.INVERSE
    assert g.edges[0].metadata.rho == -0.5


def test_build_causal_graph_ate_overrides_rho_for_direction() -> None:
    """When `ate` is in stats, it takes priority over `rho` for
    direction inference."""
    g = build_causal_graph([
        _br('mixed', ('a', 'b'), Verdict.HELD,
            stats={'rho': -0.5, 'ate': 0.3}),
    ])
    # ate is positive → DIRECT, even though rho is negative.
    assert g.edges[0].metadata.direction == Direction.DIRECT


def test_build_causal_graph_held_interventional_promoted() -> None:
    """HELD + `stats['tier']=='interventional'` → INTERVENTIONAL +
    causal_one_sided level."""
    g = build_causal_graph([
        _br('intv', ('a', 'b'), Verdict.HELD,
            stats={'rho': 0.5, 'tier': 'interventional'}),
    ])
    edge = g.edges[0].metadata
    assert edge.tier == Tier.INTERVENTIONAL
    assert edge.evidentiary_level == 'causal_one_sided'


def test_build_causal_graph_no_effect_marks_refuted() -> None:
    g = build_causal_graph([
        _br('null', ('a', 'b'), Verdict.NO_EFFECT, stats={'rho': 0.1}),
    ])
    edge = g.edges[0].metadata
    assert edge.tier == Tier.ASSOCIATIONAL
    assert edge.evidentiary_level == 'refuted'


def test_build_causal_graph_power_insufficient_marks_refuted() -> None:
    """Underpowered verdict → refuted (not enough evidence)."""
    g = build_causal_graph([
        _br('underpowered', ('a', 'b'), Verdict.POWER_INSUFFICIENT),
    ])
    assert g.edges[0].metadata.evidentiary_level == 'refuted'


def test_build_causal_graph_invariant_violation_marks_refuted() -> None:
    g = build_causal_graph([
        _br('violator', ('a', 'b'), Verdict.INVARIANT_VIOLATION),
    ])
    assert g.edges[0].metadata.evidentiary_level == 'refuted'


def test_build_causal_graph_joint_bridge_emits_one_edge_per_source() -> None:
    """≥3 targets: last is joint target, others are sources. Emits
    one edge per source with co_sources = the others."""
    g = build_causal_graph([
        _br('joint', ('s1', 's2', 's3', 'tgt'), Verdict.HELD,
            stats={'rho': 0.4}),
    ])
    assert len(g.edges) == 3
    edges_by_source = {e.source: e for e in g.edges}
    assert set(edges_by_source) == {'s1', 's2', 's3'}
    for e in g.edges:
        assert e.target == 'tgt'
        assert e.metadata.bridge_name == 'joint'
        assert len(e.metadata.co_sources) == 2
    # Source s1's co_sources should be (s2, s3).
    assert set(edges_by_source['s1'].metadata.co_sources) == {'s2', 's3'}


def test_build_causal_graph_feedback_flag_propagates() -> None:
    g = build_causal_graph([
        _br('cycle', ('a', 'b'), Verdict.HELD,
            stats={'rho': 0.5, 'feedback': True}),
    ])
    assert g.edges[0].metadata.feedback is True


# ============ promote_bridged_evidence ============

def test_promote_bridged_evidence_two_intv_admits_promote() -> None:
    """≥2 causal_one_sided edges on the same (source, target) →
    BOTH promote to causal_bridged."""
    g = build_causal_graph([
        _br('estimate', ('a', 'b'), Verdict.HELD,
            stats={'rho': 0.5, 'tier': 'interventional'}),
        _br('refuter', ('a', 'b'), Verdict.HELD,
            stats={'rho': 0.45, 'tier': 'interventional'}),
    ])
    promoted = promote_bridged_evidence(g)
    levels = {e.metadata.evidentiary_level for e in promoted.edges}
    assert levels == {'causal_bridged'}
    # Both edges promoted.
    assert all(
        e.metadata.evidentiary_level == 'causal_bridged'
        for e in promoted.edges
    )


def test_promote_bridged_evidence_correlational_does_not_count() -> None:
    """One INTERVENTIONAL admit + one correlational admit on same
    pair → no promotion. Correlational doesn't count toward
    bridging."""
    g = build_causal_graph([
        _br('intv', ('a', 'b'), Verdict.HELD,
            stats={'rho': 0.5, 'tier': 'interventional'}),
        _br('corr', ('a', 'b'), Verdict.HELD,
            stats={'rho': 0.45}),  # no tier marker → correlational
    ])
    promoted = promote_bridged_evidence(g)
    levels = {e.metadata.evidentiary_level for e in promoted.edges}
    assert levels == {'causal_one_sided', 'correlational'}


def test_promote_bridged_evidence_refuted_unaffected() -> None:
    """Refuted edge on same pair as interventional admits — refuted
    edge keeps its 'refuted' level; admits don't promote because
    only ONE causal_one_sided exists."""
    g = build_causal_graph([
        _br('intv', ('a', 'b'), Verdict.HELD,
            stats={'rho': 0.5, 'tier': 'interventional'}),
        _br('reject', ('a', 'b'), Verdict.NO_EFFECT,
            stats={'rho': 0.0}),
    ])
    promoted = promote_bridged_evidence(g)
    levels = {e.metadata.evidentiary_level for e in promoted.edges}
    assert levels == {'causal_one_sided', 'refuted'}


def test_promote_bridged_evidence_no_op_when_nothing_to_promote() -> None:
    """All edges already correlational → returned graph is the
    same object structure (no upgrades, no edits)."""
    g = build_causal_graph([
        _br('corr', ('a', 'b'), Verdict.HELD, stats={'rho': 0.5}),
    ])
    promoted = promote_bridged_evidence(g)
    assert promoted == g


def test_promote_bridged_evidence_independent_pairs_independent() -> None:
    """Pair (a, b) has 2 INTERVENTIONAL admits → promoted. Pair
    (c, d) has only 1 → unchanged."""
    g = build_causal_graph([
        _br('e1', ('a', 'b'), Verdict.HELD,
            stats={'rho': 0.5, 'tier': 'interventional'}),
        _br('e2', ('a', 'b'), Verdict.HELD,
            stats={'rho': 0.4, 'tier': 'interventional'}),
        _br('e3', ('c', 'd'), Verdict.HELD,
            stats={'rho': 0.6, 'tier': 'interventional'}),
    ])
    promoted = promote_bridged_evidence(g)
    by_target = {(e.source, e.target): e.metadata.evidentiary_level
                 for e in promoted.edges}
    # (a, b) edges promoted (2 entries; both should be 'causal_bridged').
    ab_levels = [v for (s, t), v in by_target.items()
                 if (s, t) == ('a', 'b')]
    # by_target stores last-write-wins for same key — gather edges directly.
    ab_levels = [
        e.metadata.evidentiary_level
        for e in promoted.edges if (e.source, e.target) == ('a', 'b')
    ]
    assert ab_levels == ['causal_bridged', 'causal_bridged']
    cd_levels = [
        e.metadata.evidentiary_level
        for e in promoted.edges if (e.source, e.target) == ('c', 'd')
    ]
    assert cd_levels == ['causal_one_sided']
