"""Tests for `hypothesis_subgraph_verdict` — typed verdict-walk
over a Hypothesis's typed claimed edges into `HypothesisVerdict`.

Validates:
1. 3-edge subgraph (two intervention edges + a coupling edge)
   produces per-edge BridgeResults, a typed CausalGraph, and the
   target-keyed verdict lookup.
2. Single-intervention-edge subgraphs work.
3. Coupling edges with missing-source raise loudly.
4. Empty edges raise (the typed surface is required).
5. The graph carries Tier-typed BridgeEdges; INTERVENTIONAL for
   intervention edges, ASSOCIATIONAL for coupling edges."""
from __future__ import annotations

from collections.abc import Mapping

import pytest

from corroborate.causal_graph import (
    CausalGraph,
    Direction,
    Tier,
    Tier as GraphTier,
    promote_bridged_evidence,
)
from corroborate.claim import claim
from corroborate.claim_bridge import Bridge as ClaimBridge
from corroborate.hypothesis import Hypothesis, PredictedDirection
from corroborate.hypothesis_verdict import (
    HypothesisVerdict,
    hypothesis_subgraph_verdict,
)
from corroborate.intervention import DoEffect, Intervention
from corroborate.schema import RunRow
from corroborate.verdict import Verdict


# Stub claim so the test Hypothesis has non-empty intervention_arms,
# avoiding the same-arm-key check in `from_cells`.
@claim
def _stub_arm(x: int) -> int:
    return x


_TREATMENT_ARMS = (Intervention(slot_path='stub', replacement=_stub_arm),)
_TEST_DO = DoEffect(treatment_arm='treat', baseline_arm='baseline')


def _intervention_edge(
    target: str, predicted_direction: PredictedDirection,
) -> ClaimBridge:
    return ClaimBridge(
        name=f'do->{target}',
        source=_TEST_DO.node_key(), target=target,
        intervention=_TEST_DO,
        tier=Tier.INTERVENTIONAL, direction=Direction.DIRECT,
        predicted_direction=predicted_direction,
    )


def _coupling_edge(
    source: str, target: str, predicted_direction: PredictedDirection,
) -> ClaimBridge:
    return ClaimBridge(
        name=f'{source}->{target}',
        source=source, target=target,
        tier=Tier.ASSOCIATIONAL, direction=Direction.DIRECT,
        predicted_direction=predicted_direction,
    )


def _run(
    cell_id: str, *, env: str, seed: int,
    arm_key: str, mech: float, outcome_v: float,
) -> RunRow:
    """Synthetic per-cell RunRow with both mechanism + outcome
    measurements (so a single corpus supports both edges)."""
    return RunRow(
        id=cell_id, parent_id=None, cycle_id=None,
        timestamp='2026-04-29T00:00:00Z',
        verdict=Verdict.HELD,
        arm_key=arm_key,
        measurements={
            'env_name': env,
            'seed': seed,
            'mechanism.q': mech,
            'outcome.r': outcome_v,
            'intervention_name': (
                'treat' if arm_key != 'baseline' else 'baseline'
            ),
        },
    )


def _three_edge_hypothesis() -> Hypothesis[Mapping[str, object]]:
    return Hypothesis(
        name='three_edge',
        intervention={},
        intervention_arms=_TREATMENT_ARMS,
        edges=(
            _intervention_edge('mechanism.q', 'a_lt_b'),
            _intervention_edge('outcome.r', 'a_gt_b'),
            _coupling_edge('mechanism.q', 'outcome.r', 'a_gt_b'),
        ),
    )


# ============ Three-edge subgraph ============

def test_three_edge_subgraph_produces_per_edge_bridge_results() -> None:
    """Build a 3-edge hypothesis with synthetic data where the
    treatment arm has a clear effect on both mechanism (smaller
    q is better) and outcome (larger r is better). The verdict
    carries:
    - one BridgeResult per edge keyed by (source, target).
    - one HypothesisComparisonRow per non-link edge keyed by target.
    - a typed CausalGraph with one BridgeEdge per (source, target).
    `pattern()` returns the canonical (mechanism, outcome, link)
    chain."""
    treatment: list[RunRow] = []
    baseline: list[RunRow] = []
    treatment_arm_key = (
        Hypothesis(
            name='_', intervention={}, intervention_arms=_TREATMENT_ARMS,
        ).arm_key()
    )
    envs = ('A', 'B', 'C', 'D')
    for i, env in enumerate(envs):
        for seed in range(8):
            # Bigger Δ_mech in env 'A', shrinking through 'D'.
            # Bigger Δ_outcome correlated with Δ_mech.
            base_mech = 1.0 + i * 0.3 + seed * 0.01
            base_outcome = 0.5 + seed * 0.01
            mech_reduction = 0.5 - i * 0.1
            outcome_lift = 0.4 - i * 0.08
            treatment.append(_run(
                f't{env}{seed}', env=env, seed=seed,
                arm_key=treatment_arm_key,
                mech=base_mech - mech_reduction,
                outcome_v=base_outcome + outcome_lift,
            ))
            baseline.append(_run(
                f'b{env}{seed}', env=env, seed=seed,
                arm_key='baseline',
                mech=base_mech,
                outcome_v=base_outcome,
            ))

    h = _three_edge_hypothesis()
    v = hypothesis_subgraph_verdict(
        h, treatment, baseline,
        pair_by=('seed',), group_by='env_name',
    )
    assert isinstance(v, HypothesisVerdict)
    assert len(v.edge_verdicts) == 3
    # Two intervention edges drive comparison rows; coupling does not.
    assert len(v.comparison_rows) == 2

    # Each edge has a verdict keyed by (source, target).
    do_node = _TEST_DO.node_key()
    assert (do_node, 'mechanism.q') in v.edge_verdicts
    assert (do_node, 'outcome.r') in v.edge_verdicts
    assert ('mechanism.q', 'outcome.r') in v.edge_verdicts

    # Comparison rows are keyed by target; only intervention edges.
    assert 'mechanism.q' in v.comparison_rows
    assert 'outcome.r' in v.comparison_rows

    # The typed graph carries one edge per (source, target) pair.
    edges_by_pair = {
        (e.source, e.target): e.metadata for e in v.graph.edges
    }
    assert (do_node, 'mechanism.q') in edges_by_pair
    assert (do_node, 'outcome.r') in edges_by_pair
    assert ('mechanism.q', 'outcome.r') in edges_by_pair

    # Target-path verdict lookup (replaces the role-based pattern()).
    _ = v.verdict_at('mechanism.q')
    _ = v.verdict_at('outcome.r')


def test_per_edge_predicted_direction_drives_sign_test() -> None:
    """A bridge declaring `predicted_direction='a_lt_b'` should
    HELD when treatment_minus_baseline is negative; flip the sign
    by declaring `'a_gt_b'` and the same data should refute. Per-
    edge override flows through `hypothesis_subgraph_verdict` →
    `hypothesis_comparison_from_cells` → `_per_group_stats` →
    `verdict_from_paired_stats`."""
    treatment: list[RunRow] = []
    baseline: list[RunRow] = []
    treatment_arm_key = (
        Hypothesis(
            name='_', intervention={}, intervention_arms=_TREATMENT_ARMS,
        ).arm_key()
    )
    # Treatment mech < baseline mech across all envs and seeds.
    for env in ('A', 'B', 'C', 'D'):
        for seed in range(8):
            treatment.append(_run(
                f't{env}{seed}', env=env, seed=seed,
                arm_key=treatment_arm_key,
                mech=0.0 + seed * 0.001,
                outcome_v=1.0,
            ))
            baseline.append(_run(
                f'b{env}{seed}', env=env, seed=seed,
                arm_key='baseline',
                mech=1.0 + seed * 0.001,
                outcome_v=1.0,
            ))

    # Predict a_lt_b ("treatment less than baseline") — matches the data.
    h_correct: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='correct', intervention={},
        intervention_arms=_TREATMENT_ARMS,
        edges=(_intervention_edge('mechanism.q', 'a_lt_b'),),
    )
    v_correct = hypothesis_subgraph_verdict(
        h_correct, treatment, baseline,
        pair_by=('seed',), group_by='env_name',
    )
    assert v_correct.verdict_at('mechanism.q') is Verdict.HELD

    # Predict a_gt_b ("treatment greater than baseline") — sign-
    # opposite of the data; the same evidence refutes.
    h_wrong: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='wrong', intervention={},
        intervention_arms=_TREATMENT_ARMS,
        edges=(_intervention_edge('mechanism.q', 'a_gt_b'),),
    )
    v_wrong = hypothesis_subgraph_verdict(
        h_wrong, treatment, baseline,
        pair_by=('seed',), group_by='env_name',
    )
    assert v_wrong.verdict_at('mechanism.q') is Verdict.NO_EFFECT


def test_verdict_at_returns_power_insufficient_for_missing_target() -> None:
    """A hypothesis with only one intervention edge produces a
    verdict for that target; absent targets fall back to
    POWER_INSUFFICIENT (paper-narrative reading is best-effort)."""
    treatment: list[RunRow] = []
    baseline: list[RunRow] = []
    treatment_arm_key = (
        Hypothesis(
            name='_', intervention={}, intervention_arms=_TREATMENT_ARMS,
        ).arm_key()
    )
    for env in ('A', 'B', 'C'):
        for seed in range(6):
            treatment.append(_run(
                f't{env}{seed}', env=env, seed=seed,
                arm_key=treatment_arm_key,
                mech=0.5, outcome_v=0.0,
            ))
            baseline.append(_run(
                f'b{env}{seed}', env=env, seed=seed,
                arm_key='baseline',
                mech=1.0, outcome_v=0.0,
            ))

    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='mech_only',
        intervention={},
        intervention_arms=_TREATMENT_ARMS,
        edges=(
            _intervention_edge('mechanism.q', 'a_lt_b'),
        ),
    )
    v = hypothesis_subgraph_verdict(
        h, treatment, baseline,
        pair_by=('seed',), group_by='env_name',
    )
    # Mechanism edge produces a real verdict for its target.
    _ = v.verdict_at('mechanism.q')
    # Targets that no edge claims are reported as POWER_INSUFFICIENT.
    assert v.verdict_at('outcome.r') is Verdict.POWER_INSUFFICIENT
    assert v.verdict_at('not_a_real_target') is Verdict.POWER_INSUFFICIENT


def test_edge_verdict_lookup() -> None:
    """`edge_verdict(edge)` returns the verdict for a specific
    claimed edge; missing edges raise KeyError."""
    treatment: list[RunRow] = []
    baseline: list[RunRow] = []
    treatment_arm_key = (
        Hypothesis(
            name='_', intervention={}, intervention_arms=_TREATMENT_ARMS,
        ).arm_key()
    )
    for env in ('A', 'B', 'C'):
        for seed in range(6):
            treatment.append(_run(
                f't{env}{seed}', env=env, seed=seed,
                arm_key=treatment_arm_key,
                mech=0.5, outcome_v=1.0,
            ))
            baseline.append(_run(
                f'b{env}{seed}', env=env, seed=seed,
                arm_key='baseline',
                mech=1.0, outcome_v=0.5,
            ))

    h = _three_edge_hypothesis()
    v = hypothesis_subgraph_verdict(
        h, treatment, baseline,
        pair_by=('seed',), group_by='env_name',
    )
    mech_edges = h.edges_by_target('mechanism.q')
    assert len(mech_edges) == 1
    # Reading the verdict back through the edge.
    _ = v.edge_verdict(mech_edges[0])


# ============ Error paths ============

def test_empty_edges_raises() -> None:
    """A Hypothesis with no typed edges can't be verdict-walked."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='no_edges', intervention={},
        intervention_arms=_TREATMENT_ARMS,
    )
    with pytest.raises(ValueError, match='no typed edges'):
        hypothesis_subgraph_verdict(
            h, [], [], pair_by=('seed',),
        )


def test_link_with_unknown_source_raises() -> None:
    """A coupling edge whose source isn't produced by any
    intervention edge is an authoring bug — raise loudly."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='dangling_link', intervention={},
        intervention_arms=_TREATMENT_ARMS,
        edges=(
            _intervention_edge('outcome.r', 'a_gt_b'),
            _coupling_edge(
                source='mechanism.q',  # not produced by any edge
                target='outcome.r',
                predicted_direction='a_gt_b',
            ),
        ),
    )
    treatment_arm_key = (
        Hypothesis(
            name='_', intervention={}, intervention_arms=_TREATMENT_ARMS,
        ).arm_key()
    )
    treatment = [
        _run(f't{i}', env='A', seed=i, arm_key=treatment_arm_key,
             mech=0.0, outcome_v=1.0)
        for i in range(6)
    ]
    baseline = [
        _run(f'b{i}', env='A', seed=i, arm_key='baseline',
             mech=0.0, outcome_v=0.0)
        for i in range(6)
    ]
    with pytest.raises(ValueError, match='source='):
        hypothesis_subgraph_verdict(
            h, treatment, baseline,
            pair_by=('seed',), group_by='env_name',
        )


# ============ Causal graph integration ============

def test_verdict_graph_has_typed_tiers() -> None:
    """End-to-end: 3-edge subgraph runs verdict-walk; the graph
    inside the verdict carries Tier-typed BridgeEdges. mechanism
    + outcome edges should have INTERVENTIONAL tier; link edge
    should have ASSOCIATIONAL."""
    treatment_arm_key = (
        Hypothesis(
            name='_', intervention={}, intervention_arms=_TREATMENT_ARMS,
        ).arm_key()
    )
    treatment: list[RunRow] = []
    baseline: list[RunRow] = []
    for i, env in enumerate(('A', 'B', 'C', 'D')):
        for seed in range(8):
            treatment.append(_run(
                f't{env}{seed}', env=env, seed=seed,
                arm_key=treatment_arm_key,
                mech=0.5 - i * 0.1, outcome_v=1.0 + i * 0.05,
            ))
            baseline.append(_run(
                f'b{env}{seed}', env=env, seed=seed,
                arm_key='baseline',
                mech=1.0, outcome_v=0.5,
            ))

    h = _three_edge_hypothesis()
    v = hypothesis_subgraph_verdict(
        h, treatment, baseline,
        pair_by=('seed',), group_by='env_name',
        promote_bridged=False,
    )
    g = v.graph
    do_node = _TEST_DO.node_key()

    edges_by_pair = {
        (e.source, e.target): e.metadata for e in g.edges
    }
    assert (do_node, 'mechanism.q') in edges_by_pair
    assert (do_node, 'outcome.r') in edges_by_pair
    assert ('mechanism.q', 'outcome.r') in edges_by_pair

    mech_edge = edges_by_pair[(do_node, 'mechanism.q')]
    out_edge = edges_by_pair[(do_node, 'outcome.r')]
    link_meta = edges_by_pair[('mechanism.q', 'outcome.r')]

    assert mech_edge.tier is GraphTier.INTERVENTIONAL or (
        # NO_EFFECT verdicts demote to ASSOCIATIONAL/refuted.
        mech_edge.evidentiary_level == 'refuted'
    )
    assert out_edge.tier is GraphTier.INTERVENTIONAL or (
        out_edge.evidentiary_level == 'refuted'
    )
    assert link_meta.tier is GraphTier.ASSOCIATIONAL


def test_promote_bridged_runs_when_two_interventional_admits() -> None:
    """When ≥2 INTERVENTIONAL HELD edges share a (source, target)
    pair, `promote_bridged_evidence` upgrades them to
    `causal_bridged`. The Hypothesis surface doesn't naturally
    produce duplicates, but the post-pass should hand off
    correctly when given them. Sanity-only test of the wiring."""
    from corroborate.causal_graph import BridgeEdge, Direction
    from corroborate.graph import Graph
    e1 = BridgeEdge(
        bridge_name='estimate',
        direction=Direction.DIRECT,
        tier=Tier.INTERVENTIONAL,
        evidentiary_level='causal_one_sided',
        ate=1.0,
    )
    e2 = BridgeEdge(
        bridge_name='placebo_refuter',
        direction=Direction.DIRECT,
        tier=Tier.INTERVENTIONAL,
        evidentiary_level='causal_one_sided',
        ate=1.0,
    )
    g: 'CausalGraph' = Graph()
    g = g.with_edge('a', 'b', e1)
    g = g.with_edge('a', 'b', e2)
    g = promote_bridged_evidence(g)
    edges = [
        e.metadata for e in g.edges
        if e.source == 'a' and e.target == 'b'
    ]
    assert any(
        e.evidentiary_level == 'causal_bridged'
        and e.tier is GraphTier.INTERVENTIONAL
        for e in edges
    )
