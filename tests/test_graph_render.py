"""Contract tests for generic evidence/computation graph rendering."""
from __future__ import annotations

from types import MappingProxyType

from corroborate.bridge.bridge import BridgeEvaluation, claim_bridge
from corroborate.bridge.verdict import Verdict
from corroborate.core.claim import claim, trace_context
from corroborate.core.intervention import DoEffect, Intervention
from corroborate.graph.causal import ClusterVerdict, Direction, Tier
from corroborate.graph.computation import build_computation_graph
from corroborate.graph.render import (
    computation_graph_to_dot,
    computation_graph_to_svg,
    evidence_graph_to_dot,
    evidence_graph_to_svg,
)


@claim
def _alternative(value: object) -> object:
    return value


_CONTRAST = DoEffect(arms=(
    (),
    (Intervention(slot_path='policy', replacement=_alternative),),
))


@claim_bridge(
    source=_CONTRAST,
    target='state_visitation_breadth',
    direction=Direction.DIRECT,
    pair_by=('seed',),
    predicted_direction='a_gt_b',
)
def _visitation_edge() -> Verdict:
    return Verdict.POWER_INSUFFICIENT


@claim_bridge(
    source=_CONTRAST,
    target='state_visitation_breadth',
    direction=Direction.DIRECT,
    pair_by=('seed',),
    predicted_direction='a_gt_b',
)
def _parallel_edge_not_in_finding() -> Verdict:
    return Verdict.HELD


def _evaluation(bridge_name: str, verdict: Verdict) -> BridgeEvaluation:
    return BridgeEvaluation(
        bridge_name=bridge_name,
        verdict=verdict,
        analysis_results=MappingProxyType({}),
        n_cells_in_scope=12,
        source_name=_CONTRAST.node_key(),
        target_name='state_visitation_breadth',
        extent_hash=123,
    )


def test_evidence_renderer_filters_by_bridge_identity() -> None:
    evaluations = {
        _visitation_edge.name: _evaluation(
            _visitation_edge.name,
            Verdict.POWER_INSUFFICIENT,
        ),
        _parallel_edge_not_in_finding.name: _evaluation(
            _parallel_edge_not_in_finding.name,
            Verdict.HELD,
        ),
    }
    dot = evidence_graph_to_dot(
        (_visitation_edge, _parallel_edge_not_in_finding),
        evaluations,
        bridge_names=(_visitation_edge.name,),
        edge_labels={_visitation_edge.name: 'changes exploration breadth'},
        edge_summaries={_visitation_edge.name: 'g = 0.31; n = 6'},
    )
    assert 'changes exploration breadth' in dot
    assert 'label="changes exploration breadth' in dot
    assert 'g = 0.31; n = 6' in dot
    assert 'POWER INSUFFICIENT' in dot
    assert 'parallel edge not in finding' not in dot
    assert 'style=dashed' in dot


def test_evidence_svg_is_standalone_and_exact() -> None:
    svg = evidence_graph_to_svg(
        (_visitation_edge,),
        {
            _visitation_edge.name: _evaluation(
                _visitation_edge.name,
                Verdict.POWER_INSUFFICIENT,
            ),
        },
        node_labels={
            _CONTRAST.node_key(): 'do(entropy bonus)',
            'state_visitation_breadth': 'early visitation breadth',
        },
        title='Exploration claim',
        aggregate_verdict=ClusterVerdict.UNDERPOWERED,
    )
    assert svg.startswith('<?xml')
    assert '<svg ' in svg
    assert 'do(entropy bonus)' in svg
    assert 'early visitation breadth' in svg
    assert 'POWER INSUFFICIENT' in svg
    assert 'UNDERPOWERED' in svg
    assert 'stroke-dasharray="9 7"' in svg


@claim
def _produce(seed: object) -> dict[str, object]:
    return {'payload': [seed]}


@claim
def _consume(payload: object) -> object:
    return payload


def test_computation_renderers_label_identity_flow() -> None:
    with trace_context() as records:
        result = _produce('seed')
        _consume(result['payload'])
    graph = build_computation_graph(records)
    dot = computation_graph_to_dot(graph, title='Structural probe')
    svg = computation_graph_to_svg(graph, title='Structural probe')
    assert '_produce' in dot and '_consume' in dot
    assert '.payload ← .payload' in dot
    assert 'observed identity flow' in dot
    assert '_produce' in svg and '_consume' in svg


def test_renderers_are_deterministic() -> None:
    evaluation = {
        _visitation_edge.name: _evaluation(
            _visitation_edge.name,
            Verdict.POWER_INSUFFICIENT,
        ),
    }
    assert evidence_graph_to_dot((_visitation_edge,), evaluation) == (
        evidence_graph_to_dot((_visitation_edge,), evaluation)
    )
    assert evidence_graph_to_svg((_visitation_edge,), evaluation) == (
        evidence_graph_to_svg((_visitation_edge,), evaluation)
    )


def test_unknown_finding_bridge_fails_loudly() -> None:
    try:
        evidence_graph_to_dot(
            (_visitation_edge,),
            {},
            bridge_names=('missing-bridge',),
        )
    except KeyError as error:
        assert 'missing-bridge' in str(error)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError('unknown bridge name should fail')


@claim_bridge(
    source='state_visitation_diversity',
    target='checkpoint_return',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    pair_by=(),
    predicted_direction='a_gt_b',
)
def _associational_edge() -> Verdict:
    return Verdict.HELD_WITH_SCOPE_FLAG


def test_supported_association_remains_visually_dashed() -> None:
    evaluation = BridgeEvaluation(
        bridge_name=_associational_edge.name,
        verdict=Verdict.HELD_WITH_SCOPE_FLAG,
        analysis_results=MappingProxyType({}),
        n_cells_in_scope=96,
        source_name=_associational_edge.source_name,
        target_name=_associational_edge.target_name,
        extent_hash=456,
    )
    evaluations = {_associational_edge.name: evaluation}
    dot = evidence_graph_to_dot((_associational_edge,), evaluations)
    svg = evidence_graph_to_svg((_associational_edge,), evaluations)
    assert 'style=dashed' in dot
    assert 'stroke-dasharray="9 7"' in svg
