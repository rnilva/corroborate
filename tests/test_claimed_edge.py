"""Tests for typed `ClaimedEdge` + role factories.

Validates the structural shape of edges produced by each factory
and that conventional defaults (source, tier) match the §3
verdict pattern's role semantics."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.bridge import Bridge, BridgeResult, bridge
from corroborate.causal_graph import Tier
from corroborate.claimed_edge import (
    ClaimedEdge,
    link_edge,
    mechanism_edge,
    outcome_edge,
    refuter_edge,
)
from corroborate.verdict import Verdict


def _stub_bridge(target: str) -> Bridge[Mapping[str, object]]:
    """Minimal Bridge fixture for tests that only care about
    the edge metadata, not the verdict logic."""
    @bridge(targets=(target,), name=f'stub({target})')
    def _b(record: Mapping[str, object]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name=f'stub({target})', targets=(target,),
        )
    return _b


# ============ mechanism_edge ============

def test_mechanism_edge_defaults() -> None:
    """`mechanism_edge` defaults source to `'do(arm)'` and tier to
    INTERVENTIONAL. Role is 'mechanism'."""
    b = _stub_bridge('mechanism.jensen_gap')
    e = mechanism_edge(
        target='mechanism.jensen_gap',
        predicted_direction='a_lt_b',
        bridge=b,
    )
    assert e.role == 'mechanism'
    assert e.source == 'do(arm)'
    assert e.target == 'mechanism.jensen_gap'
    assert e.predicted_direction == 'a_lt_b'
    assert e.tier is Tier.INTERVENTIONAL
    assert e.bridge is b


def test_mechanism_edge_custom_source() -> None:
    """Source is overridable for non-conventional usages (e.g.
    a chained intervention with a non-default sentinel)."""
    b = _stub_bridge('quantity_x')
    e = mechanism_edge(
        target='quantity_x',
        predicted_direction='a_gt_b',
        bridge=b,
        source='do(replay)',
    )
    assert e.source == 'do(replay)'


# ============ outcome_edge ============

def test_outcome_edge_defaults() -> None:
    b = _stub_bridge('outcome.return')
    e = outcome_edge(
        target='outcome.return',
        predicted_direction='a_gt_b',
        bridge=b,
    )
    assert e.role == 'outcome'
    assert e.source == 'do(arm)'
    assert e.tier is Tier.INTERVENTIONAL


# ============ link_edge ============

def test_link_edge_associational_tier() -> None:
    """`link_edge` is between two measurements (no do() at the
    source) — tier defaults to ASSOCIATIONAL."""
    b = _stub_bridge('outcome.return')
    e = link_edge(
        source='mechanism.jensen_gap',
        target='outcome.return',
        predicted_direction='a_gt_b',
        bridge=b,
    )
    assert e.role == 'link'
    assert e.source == 'mechanism.jensen_gap'
    assert e.target == 'outcome.return'
    assert e.tier is Tier.ASSOCIATIONAL


# ============ refuter_edge ============

def test_refuter_edge_defaults() -> None:
    b = _stub_bridge('quantity_y')
    e = refuter_edge(
        target='quantity_y',
        predicted_direction='a_gt_b',
        bridge=b,
    )
    assert e.role == 'refuter'
    assert e.source == 'do(arm)'
    assert e.tier is Tier.INTERVENTIONAL


# ============ Equality ============

def test_equal_args_produce_equal_edges() -> None:
    """Frozen dataclass equality — same args produce equal
    ClaimedEdges. (Bridge identity is referenced; same Bridge
    instance gives equal edges.)"""
    b = _stub_bridge('m')
    e1 = mechanism_edge(
        target='m', predicted_direction='a_lt_b', bridge=b,
    )
    e2 = mechanism_edge(
        target='m', predicted_direction='a_lt_b', bridge=b,
    )
    assert e1 == e2


def test_different_roles_not_equal() -> None:
    """Same target/direction/bridge but different roles ⇒ not
    equal."""
    b = _stub_bridge('m')
    e_mech = mechanism_edge(
        target='m', predicted_direction='a_lt_b', bridge=b,
    )
    e_ref: ClaimedEdge[Mapping[str, object]] = refuter_edge(
        target='m', predicted_direction='a_lt_b', bridge=b,
    )
    assert e_mech != e_ref


# ============ Slot enforcement ============

def test_edge_is_frozen() -> None:
    """`@dataclass(frozen=True, slots=True)` — attribute writes
    raise."""
    b = _stub_bridge('m')
    e = mechanism_edge(
        target='m', predicted_direction='a_lt_b', bridge=b,
    )
    try:
        e.role = 'outcome'  # pyright: ignore[reportAttributeAccessIssue]
    except (AttributeError, TypeError):
        return
    raise AssertionError('expected frozen-dataclass attribute write to raise')
