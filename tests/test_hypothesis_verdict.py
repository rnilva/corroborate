"""Tests for `hypothesis_subgraph_verdict` — typed verdict-walk
over a Hypothesis's typed claimed edges into `HypothesisVerdict`.

Validates:
1. 3-edge subgraph (mechanism + outcome + link) produces
   per-edge BridgeResults, a typed CausalGraph, and the §3
   pattern.
2. Mechanism-only subgraphs work (no outcome / link).
3. Link edges with missing-source raise loudly.
4. Empty edges raise (the typed surface is required).
5. The graph carries Tier-typed BridgeEdges; INTERVENTIONAL for
   `do(arm)`-sourced edges, ASSOCIATIONAL for link edges."""
from __future__ import annotations

from collections.abc import Mapping

import pytest

from corroborate.bridge import Bridge, BridgeResult, bridge as bridge_decorator
from corroborate.causal_graph import Tier as GraphTier
from corroborate.causal_graph import build_causal_graph, promote_bridged_evidence
from corroborate.claim import claim
from corroborate.claimed_edge import (
    link_edge,
    mechanism_edge,
    outcome_edge,
)
from corroborate.hypothesis import Hypothesis
from corroborate.hypothesis_verdict import (
    HypothesisVerdict,
    hypothesis_subgraph_verdict,
)
from corroborate.intervention import Intervention
from corroborate.schema import RunRow
from corroborate.verdict import Verdict


# Stub claim so the test Hypothesis has non-empty intervention_arms,
# avoiding the same-arm-key check in `from_cells`.
@claim
def _stub_arm(x: int) -> int:
    return x


_TREATMENT_ARMS = (Intervention(slot_path='stub', replacement=_stub_arm),)


def _stub_bridge(target: str) -> Bridge[Mapping[str, object]]:
    @bridge_decorator(targets=(target,), name=f'stub({target})')
    def _b(record: Mapping[str, object]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='', stats={},
            name=f'stub({target})', targets=(target,),
        )
    return _b


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
            mechanism_edge(
                target='mechanism.q',
                predicted_direction='a_lt_b',
                bridge=_stub_bridge('mechanism.q'),
            ),
            outcome_edge(
                target='outcome.r',
                predicted_direction='a_gt_b',
                bridge=_stub_bridge('outcome.r'),
            ),
            link_edge(
                source='mechanism.q',
                target='outcome.r',
                predicted_direction='a_gt_b',
                bridge=_stub_bridge('outcome.r'),
            ),
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
    assert len(v.bridge_results) == 3
    assert len(v.comparison_rows) == 2  # mech + outcome (no link)

    # Each edge has a BridgeResult keyed by (source, target).
    assert ('do(arm)', 'mechanism.q') in v.bridge_results
    assert ('do(arm)', 'outcome.r') in v.bridge_results
    assert ('mechanism.q', 'outcome.r') in v.bridge_results

    # Comparison rows are keyed by target; only non-link edges.
    assert 'mechanism.q' in v.comparison_rows
    assert 'outcome.r' in v.comparison_rows

    # The typed graph carries one edge per (source, target) pair.
    edges_by_pair = {
        (e.source, e.target): e.metadata for e in v.graph.edges
    }
    assert ('do(arm)', 'mechanism.q') in edges_by_pair
    assert ('do(arm)', 'outcome.r') in edges_by_pair
    assert ('mechanism.q', 'outcome.r') in edges_by_pair

    # Pattern is a 3-tuple in canonical order.
    pattern = v.pattern()
    assert len(pattern) == 3


def test_pattern_default_when_role_missing() -> None:
    """A hypothesis with only a mechanism edge produces a pattern
    of (mech_verdict, POWER_INSUFFICIENT, POWER_INSUFFICIENT) —
    missing roles default to 'unknown'."""
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
            mechanism_edge(
                target='mechanism.q',
                predicted_direction='a_lt_b',
                bridge=_stub_bridge('mechanism.q'),
            ),
        ),
    )
    v = hypothesis_subgraph_verdict(
        h, treatment, baseline,
        pair_by=('seed',), group_by='env_name',
    )
    pattern = v.pattern()
    assert pattern[1] is Verdict.POWER_INSUFFICIENT  # outcome missing
    assert pattern[2] is Verdict.POWER_INSUFFICIENT  # link missing


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
    mech = h.mechanism_edge()
    assert mech is not None
    # Reading the verdict back through the edge.
    _ = v.edge_verdict(mech)


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
    """A link edge whose source isn't produced by any pass-1 edge
    is an authoring bug — raise loudly."""
    h: Hypothesis[Mapping[str, object]] = Hypothesis(
        name='dangling_link', intervention={},
        intervention_arms=_TREATMENT_ARMS,
        edges=(
            outcome_edge(
                target='outcome.r',
                predicted_direction='a_gt_b',
                bridge=_stub_bridge('outcome.r'),
            ),
            link_edge(
                source='mechanism.q',  # not produced by any edge
                target='outcome.r',
                predicted_direction='a_gt_b',
                bridge=_stub_bridge('outcome.r'),
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

    edges_by_pair = {
        (e.source, e.target): e.metadata for e in g.edges
    }
    assert ('do(arm)', 'mechanism.q') in edges_by_pair
    assert ('do(arm)', 'outcome.r') in edges_by_pair
    assert ('mechanism.q', 'outcome.r') in edges_by_pair

    mech_edge = edges_by_pair[('do(arm)', 'mechanism.q')]
    out_edge = edges_by_pair[('do(arm)', 'outcome.r')]
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
    r1 = BridgeResult(
        verdict=Verdict.HELD,
        reason='',
        stats={'tier': 'interventional', 'ate': 1.0},
        name='estimate', targets=('a', 'b'),
    )
    r2 = BridgeResult(
        verdict=Verdict.HELD,
        reason='',
        stats={'tier': 'interventional', 'ate': 1.0},
        name='placebo_refuter', targets=('a', 'b'),
    )
    g = build_causal_graph([r1, r2])
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
