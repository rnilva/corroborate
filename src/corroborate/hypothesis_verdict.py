"""Hypothesis verdict — typed walk over a Hypothesis's claimed
edges into a `HypothesisVerdict[R]`.

A `HypothesisVerdict` carries:

- `hypothesis: Hypothesis[R]` — the original subgraph claim.
- `graph: CausalGraph` — the typed graph (BridgeEdges keyed by
  Pearl tier × direction × evidentiary level), eagerly built from
  the per-edge BridgeResults via `build_causal_graph` +
  `promote_bridged_evidence`.
- `bridge_results: Mapping[(source, target), BridgeResult]` —
  the per-edge BridgeResult keyed by (source, target). Single
  source of truth for the per-edge verdict.
- `comparison_rows: Mapping[str, HypothesisComparisonRow]` —
  rich per-edge detail (per_group, pooled, facts, reads_set) for
  the *paired-comparison* edges (mechanism / outcome / refuter),
  keyed by target path. Link edges are pure cross-stratum
  Pearson; their richer detail collapses cleanly into the
  BridgeResult and they don't contribute here.

Verdict logic per role:

- `mechanism` / `outcome` / `refuter` — paired comparison via
  `hypothesis_comparison_from_cells`, stratified by `group_by`,
  reading the edge's `target` as the outcome path. Verdict comes
  from the random-effects PI test (HELD / HELD_WITH_SCOPE_FLAG /
  NO_EFFECT / POWER_INSUFFICIENT).
- `link` — corpus-level Pearson r over the per-group effect
  sizes of the edge's `source` and `target` comparisons. Requires
  that the source's mechanism / outcome verdict has already been
  computed (link is paired-second)."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import scipy.stats as ss

from corroborate.aggregate import hypothesis_comparison_from_cells
from corroborate.bridge import BridgeResult
from corroborate.causal_graph import (
    CausalGraph,
    Tier,
    build_causal_graph,
    promote_bridged_evidence,
)
from corroborate.claimed_edge import BridgeRole, ClaimedEdge
from corroborate.hypothesis import Hypothesis, PredictedDirection
from corroborate.schema import HypothesisComparisonRow, MeasurementLeaf
from corroborate.schema import RunRow
from corroborate.verdict import RefutationClass, Verdict


@dataclass(frozen=True, slots=True)
class HypothesisVerdict[R: Mapping[str, object]]:
    """Corroboration verdict of a Hypothesis: typed CausalGraph
    plus the per-edge BridgeResults that built it, plus rich
    per-edge detail for paired-comparison edges.

    The `graph` is the canonical typed subgraph artifact (stage 8
    output). The `bridge_results` map is the single source of
    truth for per-edge verdicts (the graph's evidentiary_level is
    derived). The `comparison_rows` map carries the meta-regression
    inputs (per_group, pooled) for the paired-comparison edges."""
    hypothesis: Hypothesis[R]
    graph: CausalGraph
    bridge_results: Mapping[tuple[str, str], BridgeResult]
    comparison_rows: Mapping[str, HypothesisComparisonRow]

    def edge_verdict(self, edge: ClaimedEdge[R]) -> Verdict:
        """Verdict for a specific claimed edge — looked up via
        the edge's `(source, target)` key in `bridge_results`."""
        br = self.bridge_results.get((edge.source, edge.target))
        if br is None:
            raise KeyError(
                f'no bridge result for ({edge.source!r}, '
                f'{edge.target!r}) — edge not in this hypothesis '
                f'verdict',
            )
        return br.verdict

    def pattern(self) -> tuple[Verdict, ...]:
        """The §3 verdict chain in canonical order: mechanism →
        outcome → link. Roles absent from the Hypothesis produce
        POWER_INSUFFICIENT in their slot ('unknown')."""
        out: list[Verdict] = []
        roles: tuple[BridgeRole, ...] = ('mechanism', 'outcome', 'link')
        for role in roles:
            edges = self.hypothesis.edges_by_role(role)
            if not edges:
                out.append(Verdict.POWER_INSUFFICIENT)
                continue
            edge = edges[0]
            br = self.bridge_results.get((edge.source, edge.target))
            out.append(
                br.verdict if br is not None
                else Verdict.POWER_INSUFFICIENT
            )
        return tuple(out)


# ============ link-edge Pearson computation ============

def _pearson_link_to_bridge_result(
    source_row: HypothesisComparisonRow,
    target_row: HypothesisComparisonRow,
    *,
    source_path: str,
    target_path: str,
    predicted_direction: PredictedDirection,
    alpha: float,
) -> BridgeResult:
    """Compute Pearson r of per-group effect sizes and package as
    a `BridgeResult` for graph consumption.

    Pairs strata by `group_value`; only strata where BOTH
    comparisons have a finite `effect_size_g` contribute. The
    result is a BridgeResult with `targets=(source, target)` and
    stats carrying `rho`, `pvalue`, `n_groups`, plus refutation
    class string when the verdict isn't HELD."""
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

    name = f'link({source_path}->{target_path})'
    targets = (source_path, target_path)

    if n < 3:
        return BridgeResult(
            verdict=Verdict.POWER_INSUFFICIENT,
            reason=f'only {n} paired strata; need ≥3 for link Pearson',
            stats={
                'n_groups': n,
                'refutation_class': RefutationClass.UNDERPOWERED.value,
                'predicted_direction': predicted_direction,
            },
            name=name,
            targets=targets,
        )

    # scipy boundary — `pearsonr` returns a NamedTuple-like
    # PearsonRResult whose attribute / element types are
    # stub-typed loosely. Coerce both elements at the boundary.
    r_value, p_value = ss.pearsonr(src, tgt)
    r = float(r_value)  # pyright: ignore[reportArgumentType]
    p = float(p_value)  # pyright: ignore[reportArgumentType]

    base_stats: dict[str, MeasurementLeaf] = {
        'rho': r,
        'pvalue': p,
        'n_groups': n,
        'predicted_direction': predicted_direction,
    }

    if p >= alpha:
        return BridgeResult(
            verdict=Verdict.NO_EFFECT,
            reason=f'p={p:.3g} ≥ α={alpha}',
            stats=base_stats | {
                'refutation_class': RefutationClass.NULL_EFFECT.value,
            },
            name=name, targets=targets,
        )
    # p < alpha — significant; check direction.
    if predicted_direction == 'two_sided':
        return BridgeResult(
            verdict=Verdict.HELD,
            reason=f'r={r:+.3f}, p={p:.3g} < α',
            stats=base_stats, name=name, targets=targets,
        )
    if predicted_direction == 'a_gt_b' and r > 0.0:
        return BridgeResult(
            verdict=Verdict.HELD,
            reason=f'r={r:+.3f}, p={p:.3g} < α',
            stats=base_stats, name=name, targets=targets,
        )
    if predicted_direction == 'a_lt_b' and r < 0.0:
        return BridgeResult(
            verdict=Verdict.HELD,
            reason=f'r={r:+.3f}, p={p:.3g} < α',
            stats=base_stats, name=name, targets=targets,
        )
    return BridgeResult(
        verdict=Verdict.NO_EFFECT,
        reason=f'sign mismatch: r={r:+.3f}, predicted={predicted_direction}',
        stats=base_stats | {
            'refutation_class': RefutationClass.SIGN_FLIP.value,
        },
        name=name, targets=targets,
    )


# ============ comparison-edge → BridgeResult ============

def _comparison_to_bridge_result(
    edge: ClaimedEdge[Mapping[str, object]],
    row: HypothesisComparisonRow,
) -> BridgeResult:
    """Convert a paired-comparison `HypothesisComparisonRow` into
    a `BridgeResult` shaped for `build_causal_graph` consumption.

    `targets=(source, target)` — the directed pair this edge
    spans. `stats['tier']='interventional'` is set when the
    claimed edge declared INTERVENTIONAL tier so
    `build_causal_graph`'s promotion logic fires.

    `HELD_WITH_SCOPE_FLAG` is mapped down to `HELD` for the graph
    builder's binary `is_held` check — both attest to corroboration
    at population level. The heterogeneity flag remains on the
    HypothesisComparisonRow for meta-regression to consume; the
    graph layer just records the corroboration."""
    stats: dict[str, MeasurementLeaf] = {}
    if edge.tier is Tier.INTERVENTIONAL:
        stats['tier'] = 'interventional'
    if (
        row.effect_size_g is not None
        and not math.isnan(row.effect_size_g)
    ):
        stats['ate'] = float(row.effect_size_g)

    raw_verdict = row.verdict
    graph_verdict = (
        Verdict.HELD if raw_verdict.is_corroboration() else raw_verdict
    )

    return BridgeResult(
        verdict=graph_verdict,
        reason='',
        stats=stats,
        name=f'{edge.role}({edge.target})',
        targets=(edge.source, edge.target),
    )


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
    1. Mechanism / outcome / refuter edges — paired comparison
       via `hypothesis_comparison_from_cells`, stratified by
       `group_by`, reading the edge's `target` as outcome_path.
    2. Link edges — Pearson r over the source's and target's
       per-group effect sizes from pass 1.

    Pass 2 strictly requires that the source and target paths of
    every link edge match a comparison computed in pass 1. If a
    link edge references paths not produced by any pass-1 edge,
    the framework raises ValueError — this is an authoring bug,
    not a runtime degenerate case.

    Each pass produces a `BridgeResult` keyed by `(source, target)`.
    The graph is built via `build_causal_graph` over those results,
    optionally followed by `promote_bridged_evidence` (default on)."""
    if not h.edges:
        raise ValueError(
            'hypothesis_subgraph_verdict: hypothesis has no '
            'typed edges; populate `Hypothesis.edges` via the '
            'role factories (mechanism_edge, outcome_edge, '
            'link_edge, refuter_edge).',
        )

    bridge_results: dict[tuple[str, str], BridgeResult] = {}
    comparison_rows: dict[str, HypothesisComparisonRow] = {}

    for edge in h.edges:
        if edge.role == 'link':
            continue
        row = hypothesis_comparison_from_cells(
            h, treatment_runs, baseline_runs,
            outcome_path=edge.target,
            pair_by=pair_by,
            group_by=group_by,
            alpha=alpha,
            power=power,
            baseline_h=baseline_h,
        )
        comparison_rows[edge.target] = row
        bridge_results[(edge.source, edge.target)] = (
            _comparison_to_bridge_result(edge, row)
        )

    for edge in h.edges:
        if edge.role != 'link':
            continue
        if edge.source not in comparison_rows:
            raise ValueError(
                f'link edge references source={edge.source!r} '
                f'which is not the target of any mechanism / '
                f'outcome / refuter edge in this hypothesis. '
                f'Add an edge that produces the source path '
                f'before the link edge can be evaluated.',
            )
        if edge.target not in comparison_rows:
            raise ValueError(
                f'link edge references target={edge.target!r} '
                f'which is not the target of any mechanism / '
                f'outcome / refuter edge in this hypothesis.',
            )
        bridge_results[(edge.source, edge.target)] = (
            _pearson_link_to_bridge_result(
                comparison_rows[edge.source],
                comparison_rows[edge.target],
                source_path=edge.source,
                target_path=edge.target,
                predicted_direction=edge.predicted_direction,
                alpha=alpha,
            )
        )

    graph = build_causal_graph(bridge_results.values())
    if promote_bridged:
        graph = promote_bridged_evidence(graph)

    return HypothesisVerdict(
        hypothesis=h,
        graph=graph,
        bridge_results=bridge_results,
        comparison_rows=comparison_rows,
    )
