"""Hypothesis verdict — typed walk over a Hypothesis's claimed
edges into a `HypothesisVerdict[R]`.

A `HypothesisVerdict` carries:

- `hypothesis: Hypothesis[R]` — the original subgraph claim.
- `graph: CausalGraph` — the typed graph: one `BridgeEdge` per
  claimed edge, keyed by `(source, target)` with Pearl tier ×
  direction × evidentiary level + per-edge stats (ate / rho /
  pvalue / n_observations).
- `edge_verdicts: Mapping[(source, target), Verdict]` — raw
  4-bucket verdicts (HELD / NO_EFFECT / POWER_INSUFFICIENT /
  INVARIANT_VIOLATION) preserving full resolution that the
  graph's evidentiary_level field collapses ('refuted' /
  'correlational' / 'causal_one_sided' / 'causal_bridged').
- `comparison_rows: Mapping[str, HypothesisComparisonRow]` —
  rich per-edge detail (per_group, pooled) for *intervention*
  edges, keyed by target path. Coupling edges' rho/pvalue/n live
  on the BridgeEdge directly.

Verdict logic per edge category:

- Intervention edges (`bridge.intervention is not None`) —
  paired comparison via `hypothesis_comparison_from_cells`,
  stratified by `group_by`, reading the edge's `target` as the
  outcome path. Verdict comes from the random-effects PI test.
- Coupling edges (`bridge.intervention is None`) — Pearson r
  over the per-group effect sizes of the edge's `source` and
  `target` intervention comparisons. Requires that an
  intervention edge with the source path as its target has
  already been computed (coupling edges run paired-second).

The verdict-walk constructs `BridgeEdge`s inline as it processes
each claimed edge — no intermediate `BridgeResult` shape; the
edge metadata is the typed product the graph layer wants."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import scipy.stats as ss

from corroborate.aggregate import hypothesis_comparison_from_cells
from corroborate.causal_graph import (
    BridgeEdge,
    CausalGraph,
    Direction,
    EvidentiaryLevel,
    Tier,
    promote_bridged_evidence,
)
from corroborate.claim_bridge import Bridge as ClaimBridge
from corroborate.graph import Graph
from corroborate.hypothesis import Hypothesis
from corroborate.schema import HypothesisComparisonRow, RunRow
from corroborate.verdict import Verdict


@dataclass(frozen=True, slots=True)
class HypothesisVerdict[R: Mapping[str, object]]:
    """Corroboration verdict of a Hypothesis: typed CausalGraph
    plus per-edge raw verdicts plus rich per-edge detail for
    intervention edges.

    The `graph` is the canonical typed subgraph artifact (BridgeEdge
    per claimed edge, with ate/rho/pvalue/n_observations populated).
    `edge_verdicts` carries the raw 4-bucket Verdict per edge
    (HELD / NO_EFFECT / POWER_INSUFFICIENT / INVARIANT_VIOLATION),
    preserving resolution that the graph's evidentiary_level
    collapses. `comparison_rows` carries meta-regression inputs
    (per_group, pooled) for the paired-comparison edges."""
    hypothesis: Hypothesis[R]
    graph: CausalGraph
    edge_verdicts: Mapping[tuple[str, str], Verdict]
    comparison_rows: Mapping[str, HypothesisComparisonRow]

    def edge_verdict(self, edge: ClaimBridge) -> Verdict:
        """Verdict for a specific claimed edge — looked up via
        the edge's `(source, target)` key in `edge_verdicts`."""
        v = self.edge_verdicts.get((edge.source_name, edge.target_name))
        if v is None:
            raise KeyError(
                f'no verdict for ({edge.source_name!r}, '
                f'{edge.target_name!r}) — edge not in this hypothesis '
                f'verdict',
            )
        return v

    def verdict_at(self, target: str) -> Verdict:
        """Verdict for the rung-2 intervention edge whose target
        is `target`. Returns POWER_INSUFFICIENT if no intervention
        edge in this hypothesis claims that target — paper-narrative
        reading is best-effort, substrate code that asks for a
        path the hypothesis didn't claim shouldn't crash.

        Coupling edges (intervention is None) typically share the
        target with the matched outcome intervention edge; `verdict_at`
        always reports the intervention-edge verdict to keep the
        paper-narrative reading unambiguous. Use `edge_verdict(edge)`
        when you need the coupling edge's verdict explicitly."""
        from corroborate.intervention import DoEffect
        for edge in self.hypothesis.edges:
            if edge.target_name != target:
                continue
            if not isinstance(edge.source, DoEffect):
                continue
            return self.edge_verdicts.get(
                (edge.source_name, edge.target_name),
                Verdict.POWER_INSUFFICIENT,
            )
        return Verdict.POWER_INSUFFICIENT


# ============ Edge constructors ============

def _intervention_edge(
    edge: ClaimBridge, row: HypothesisComparisonRow,
) -> tuple[BridgeEdge, Verdict]:
    """Build a `BridgeEdge` + raw `Verdict` for an intervention
    edge from its paired-comparison row. Direction inferred from
    `ate` sign; tier/level inferred from corroboration status."""
    raw_verdict = row.verdict
    is_held = raw_verdict.is_corroboration()
    ate: float | None = (
        float(row.effect_size_g)
        if row.effect_size_g is not None
        and not math.isnan(row.effect_size_g)
        else None
    )
    promoted = is_held and edge.tier is Tier.INTERVENTIONAL
    tier = Tier.INTERVENTIONAL if promoted else Tier.ASSOCIATIONAL
    level: EvidentiaryLevel = (
        'causal_one_sided' if promoted
        else 'correlational' if is_held
        else 'refuted'
    )
    direction = (
        Direction.INVERSE if (ate is not None and ate < 0)
        else Direction.DIRECT
    )
    return BridgeEdge(
        bridge_name=edge.name,
        direction=direction,
        tier=tier,
        evidentiary_level=level,
        ate=ate,
        n_observations=row.arm_a_n if row.arm_a_n > 0 else None,
    ), raw_verdict


def _coupling_edge(
    edge: ClaimBridge,
    source_row: HypothesisComparisonRow,
    target_row: HypothesisComparisonRow,
    *,
    alpha: float,
) -> tuple[BridgeEdge, Verdict]:
    """Build a `BridgeEdge` + raw `Verdict` for a coupling edge
    by computing Pearson r over the source's and target's per-
    stratum effect sizes. Always associational tier; verdict
    follows sign-and-significance against the edge's predicted
    direction."""
    src_by_group = {
        gs.group_value: gs.effect_size_g
        for gs in source_row.per_group
        if gs.effect_size_g is not None
        and not math.isnan(gs.effect_size_g)
    }
    tgt_by_group = {
        gs.group_value: gs.effect_size_g
        for gs in target_row.per_group
        if gs.effect_size_g is not None
        and not math.isnan(gs.effect_size_g)
    }
    paired = sorted(
        src_by_group.keys() & tgt_by_group.keys(),
        key=repr,
    )
    src: list[float] = [src_by_group[k] for k in paired]
    tgt: list[float] = [tgt_by_group[k] for k in paired]
    n = len(src)

    if n < 3:
        return BridgeEdge(
            bridge_name=edge.name,
            direction=Direction.DIRECT,
            tier=Tier.ASSOCIATIONAL,
            evidentiary_level='refuted',
            n_observations=n,
        ), Verdict.POWER_INSUFFICIENT

    # scipy boundary — `pearsonr` returns a NamedTuple-like
    # PearsonRResult whose attribute / element types are
    # stub-typed loosely. Coerce both elements at the boundary.
    r_value, p_value = ss.pearsonr(src, tgt)
    r = float(r_value)  # pyright: ignore[reportArgumentType]
    p = float(p_value)  # pyright: ignore[reportArgumentType]

    pred = edge.predicted_direction
    if p >= alpha:
        verdict = Verdict.NO_EFFECT
    elif pred == 'two_sided':
        verdict = Verdict.HELD
    elif pred == 'a_gt_b' and r > 0.0:
        verdict = Verdict.HELD
    elif pred == 'a_lt_b' and r < 0.0:
        verdict = Verdict.HELD
    else:
        verdict = Verdict.NO_EFFECT

    is_held = verdict is Verdict.HELD
    direction = Direction.INVERSE if r < 0 else Direction.DIRECT
    level: EvidentiaryLevel = 'correlational' if is_held else 'refuted'
    return BridgeEdge(
        bridge_name=edge.name,
        direction=direction,
        tier=Tier.ASSOCIATIONAL,
        evidentiary_level=level,
        rho=r,
        pvalue=p,
        n_observations=n,
    ), verdict


# ============ Top-level verdict walk ============

def hypothesis_subgraph_verdict(
    h: Hypothesis[Mapping[str, object]],
    treatment_runs: Sequence[RunRow],
    baseline_runs: Sequence[RunRow],
    *,
    pair_by: tuple[str, ...],
    group_by: str = 'env_name',
    alpha: float = 0.05,
    power: float = 0.8,
    baseline_h: Hypothesis[Mapping[str, object]] | None = None,
    promote_bridged: bool = True,
) -> HypothesisVerdict[Mapping[str, object]]:
    """Verdict-walk a Hypothesis's typed claimed edges.

    Two passes:
    1. Intervention edges (`bridge.intervention is not None`) —
       paired comparison via `hypothesis_comparison_from_cells`,
       stratified by `group_by`, reading the edge's `target` as
       outcome_path. Per-edge `predicted_direction` (when set)
       overrides the hypothesis-level prior in the sign test.
    2. Coupling edges (`bridge.intervention is None`) — Pearson
       r over the source's and target's per-group effect sizes
       from pass 1.

    Pass 2 strictly requires that the source and target paths of
    every coupling edge match a comparison computed in pass 1. If
    a coupling edge references paths not produced by any pass-1
    edge, the framework raises ValueError — this is an authoring
    bug, not a runtime degenerate case.

    Each pass produces a `BridgeResult` keyed by `(source, target)`.
    The graph is built via `build_causal_graph` over those results,
    optionally followed by `promote_bridged_evidence` (default on)."""
    if not h.edges:
        raise ValueError(
            'hypothesis_subgraph_verdict: hypothesis has no '
            'typed edges; populate `Hypothesis.edges` with one or '
            'more `claim_bridge.Bridge` declarations (intervention '
            'edges set `intervention=DoEffect(...)`; coupling '
            'edges leave `intervention=None`).',
        )

    edges_built: dict[tuple[str, str], BridgeEdge] = {}
    edge_verdicts: dict[tuple[str, str], Verdict] = {}
    comparison_rows: dict[str, HypothesisComparisonRow] = {}

    from corroborate.intervention import DoEffect
    # Pass 1: intervention edges (rung-2 paired contrasts).
    for edge in h.edges:
        if not isinstance(edge.source, DoEffect):
            continue
        row = hypothesis_comparison_from_cells(
            h, treatment_runs, baseline_runs,
            outcome_path=edge.target_name,
            pair_by=pair_by,
            group_by=group_by,
            alpha=alpha,
            power=power,
            baseline_h=baseline_h,
            predicted_direction=edge.predicted_direction,
        )
        comparison_rows[edge.target_name] = row
        be, v = _intervention_edge(edge, row)
        edges_built[(edge.source_name, edge.target_name)] = be
        edge_verdicts[(edge.source_name, edge.target_name)] = v

    # Pass 2: coupling edges — Pearson r over per-stratum effects
    # produced in pass 1.
    for edge in h.edges:
        if isinstance(edge.source, DoEffect):
            continue
        if edge.source_name not in comparison_rows:
            raise ValueError(
                f'coupling edge references source={edge.source_name!r} '
                f'which is not the target of any intervention edge '
                f'in this hypothesis. Add an intervention edge that '
                f'produces the source path before the coupling edge '
                f'can be evaluated.',
            )
        if edge.target_name not in comparison_rows:
            raise ValueError(
                f'coupling edge references target={edge.target_name!r} '
                f'which is not the target of any intervention edge '
                f'in this hypothesis.',
            )
        if edge.predicted_direction is None:
            raise ValueError(
                f'coupling edge ({edge.source_name!r} -> '
                f'{edge.target_name!r}) must declare '
                f'`predicted_direction` — Pearson sign check needs '
                f'a prior.',
            )
        be, v = _coupling_edge(
            edge,
            comparison_rows[edge.source_name],
            comparison_rows[edge.target_name],
            alpha=alpha,
        )
        edges_built[(edge.source_name, edge.target_name)] = be
        edge_verdicts[(edge.source_name, edge.target_name)] = v

    graph: CausalGraph = Graph()
    for (s, t), be in edges_built.items():
        graph = graph.with_edge(s, t, be)
    if promote_bridged:
        graph = promote_bridged_evidence(graph)

    return HypothesisVerdict(
        hypothesis=h,
        graph=graph,
        edge_verdicts=edge_verdicts,
        comparison_rows=comparison_rows,
    )
