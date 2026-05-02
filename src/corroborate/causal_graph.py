"""Causal graph — Pearl-tier-typed BridgeEdges over a Graph[str, M].

Nodes are measurable / record-key names (strings). Edges are
`BridgeEdge`s carrying:

- `direction: Direction` — sign of the coupling. DIRECT = source↑
  ⇒ target↑; INVERSE = source↑ ⇒ target↓. Composes along chains
  via `__mul__`.
- `tier: Tier` — Pearl-ladder rung. ASSOCIATIONAL = observational
  coupling (PC adjacency / Spearman correlation); INTERVENTIONAL =
  confirmed do() effect (paired Hedges' g HELD). Promotion algebra
  is structurally constrained: only ASSOCIATIONAL → INTERVENTIONAL
  via `.promote()`, and only INTERVENTIONAL → ASSOCIATIONAL via
  `.demote()`.
- `evidentiary_level: str` — the verdict-derived label
  ('refuted' / 'correlational' / 'causal_one_sided' /
  'causal_bridged'). Lifecycle the bridge is in.
- `ate` / `rho` / `pvalue` / `n_observations` — typed evidence
  fields. Intervention edges populate `ate` + `n_observations`;
  coupling edges populate `rho` + `pvalue` + `n_observations`.

`compose_direction` and `chain_tier` walk an edge sequence to
produce path-level direction + tier — chain composition for
admissibility / promotion checks.

`hypothesis_subgraph_verdict` (in `hypothesis_verdict.py`) is the
sole producer of `BridgeEdge`s: it constructs them inline as it
walks a Hypothesis's typed claimed edges, then passes the
resulting graph through `promote_bridged_evidence` for the
`causal_one_sided` → `causal_bridged` post-pass.

`promote_bridged_evidence(g)` is the post-pass: for any (source,
target) pair with ≥2 `causal_one_sided` edges (≥2 INTERVENTIONAL
HELD bridges under the same DAG), those edges promote to
`causal_bridged`. Reading: do-calculus inference is corroborated
by an INDEPENDENT bridge — typically an estimate plus a refuter —
not just an estimate matched by a correlational coupling."""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Literal

from corroborate.graph import Edge, Graph
from corroborate.verdict import Verdict

if TYPE_CHECKING:
    # Forward import: `claim_bridge` depends on `causal_graph`
    # transitively via verdict; lazy-typed to avoid the cycle.
    from corroborate.claim_bridge import Bridge as ClaimBridge


# ============ Direction — sign of an edge ============

class Direction(Enum):
    """Sign of an edge: DIRECT = source↑ ⇒ target↑; INVERSE =
    source↑ ⇒ target↓. Multiplicative chain composition: DIRECT
    is identity, INVERSE squared is DIRECT (two negatives cancel)."""
    DIRECT = 'direct'
    INVERSE = 'inverse'

    def __mul__(self, other: 'Direction') -> 'Direction':
        if self is other:
            return Direction.DIRECT
        return Direction.INVERSE


# ============ Tier — Pearl-ladder rung ============

class InvalidTierTransition(ValueError):
    """A `Tier` transition violated the Pearl-ladder lifecycle
    algebra. Raised by `Tier.promote()` / `Tier.demote()` when
    called on a tier without further movement available."""


class Tier(IntEnum):
    """Pearl-ladder rung the evidence has reached.

    - `ASSOCIATIONAL` (rung 1) = observational coupling. PC
      adjacency, Spearman correlation, mediator-features within-
      env Pearson — all rung 1.
    - `INTERVENTIONAL` (rung 2) = confirmed do() effect. Paired
      Hedges' g HELD on a treatment-vs-baseline comparison.

    Promotion is structurally constrained: only ASSOCIATIONAL →
    INTERVENTIONAL via `.promote()`. The reverse via `.demote()`
    exists for downgrade scenarios (refuter contradicts an
    interventional admit), but the framework treats refutation
    primarily through `evidentiary_level='refuted'` on the
    BridgeEdge — `.demote()` is the rare case."""
    ASSOCIATIONAL = 1
    INTERVENTIONAL = 2

    def promote(self) -> 'Tier':
        if self is Tier.ASSOCIATIONAL:
            return Tier.INTERVENTIONAL
        raise InvalidTierTransition(
            f'{self.name} cannot promote — already top.',
        )

    def demote(self) -> 'Tier':
        if self is Tier.INTERVENTIONAL:
            return Tier.ASSOCIATIONAL
        raise InvalidTierTransition(
            f'{self.name} cannot demote — already bottom.',
        )


# ============ EvidentiaryLevel ============

EvidentiaryLevel = Literal[
    'unevaluated',
    'refuted', 'correlational', 'causal_one_sided', 'causal_bridged',
]


# ============ BridgeEdge — graph edge metadata ============

@dataclass(frozen=True, slots=True)
class BridgeEdge:
    """Metadata stored on each edge of a `CausalGraph`.

    `bridge_name` — the claim_bridge.Bridge.name that produced
    this edge.
    `direction` — DIRECT or INVERSE, inferred from `ate` sign
    (priority) or `rho` sign (fallback) at construction.
    `tier` — ASSOCIATIONAL by default; INTERVENTIONAL when the
    edge represents an intervention contrast AND the verdict is
    HELD.
    `evidentiary_level` — 'refuted' for non-HELD;
    'causal_one_sided' for INTERVENTIONAL admit; 'correlational'
    for ASSOCIATIONAL admit; 'causal_bridged' is set only by
    `promote_bridged_evidence` post-pass.

    `ate` — effect-size-g for intervention edges (paired Hedges'
    g across cells in the comparison row). None for coupling
    edges.
    `rho` — Pearson r for coupling edges (cross-stratum on
    per-stratum effect sizes). None for intervention edges.
    `pvalue` — significance test p-value where defined (coupling
    edges' Pearson p, etc.). None where not applicable.
    `n_observations` — sample size used for the verdict's
    statistical test (n_pairs for intervention, n_groups for
    coupling). None where not applicable.

    `feedback` — set on edges that intentionally participate in
    cycles. Graph walks use this to break cycle traversal.
    `condition_desc` — optional condition annotation (e.g.
    'when reward_scale > 0')."""
    bridge_name: str
    direction: Direction
    tier: Tier
    evidentiary_level: EvidentiaryLevel
    ate: float | None = None
    rho: float | None = None
    pvalue: float | None = None
    n_observations: int | None = None
    feedback: bool = False
    condition_desc: str | None = None

    def __str__(self) -> str:
        bits = [
            self.bridge_name,
            f'{self.direction.value}/{self.tier.name.lower()}/{self.evidentiary_level}',
        ]
        if self.ate is not None:
            bits.append(f'ate={self.ate:+.3f}')
        if self.rho is not None:
            bits.append(f'ρ={self.rho:+.2f}')
        if self.feedback:
            bits.append('feedback')
        return ' '.join(bits)


type CausalGraph = Graph[str, BridgeEdge]


# ============ Chain composition ============

def compose_direction(edges: Iterable[BridgeEdge]) -> Direction:
    """Chain composition of directions along an edge sequence.
    Empty chain → DIRECT (multiplicative identity). Two INVERSE
    edges compose to DIRECT (negatives cancel)."""
    result = Direction.DIRECT
    for e in edges:
        result = result * e.direction
    return result


def chain_tier(edges: Iterable[BridgeEdge]) -> Tier:
    """Minimum tier along a chain — the chain is no stronger than
    its weakest link. Empty chain → ASSOCIATIONAL (no evidence,
    coarsest tier as default)."""
    result = Tier.INTERVENTIONAL
    any_edge = False
    for e in edges:
        any_edge = True
        if e.tier < result:
            result = e.tier
    return result if any_edge else Tier.ASSOCIATIONAL


# ============ Pre-evaluation authored graph ============

def authored_graph(
    bridges: 'Iterable[ClaimBridge]',
) -> CausalGraph:
    """Build the unevaluated graph topology from a `Sequence[Bridge]`.

    Each bridge contributes one edge. When `bridge.intervention is
    None`, the edge is `bridge.source → bridge.target` (measurable-
    to-measurable). When set, the edge is
    `bridge.intervention.node_key() → bridge.target` — an
    *intervention → measurable* edge with the do-node as source.
    All edges get `evidentiary_level='unevaluated'`; tier is
    INTERVENTIONAL when an intervention is declared (the edge is
    by construction a Pearl-rung-2 contrast), else inherits the
    bridge's declared `tier`.

    Used by analyses that want to inspect the authored graph
    topology BEFORE running bridges — e.g. cache builders,
    Protocol-conforming module discoverers."""
    g: CausalGraph = Graph()
    for b in bridges:
        if b.intervention is not None:
            source_key = b.intervention.node_key()
            tier = Tier.INTERVENTIONAL
        else:
            source_key = b.source
            tier = b.tier
        edge = BridgeEdge(
            bridge_name=b.name,
            direction=b.direction,
            tier=tier,
            evidentiary_level='unevaluated',
        )
        g = g.with_edge(source_key, b.target, edge)
    return g


# ============ Bridged-evidence promotion ============

def promote_bridged_evidence(g: CausalGraph) -> CausalGraph:
    """Post-pass: for each `(source, target)` pair with ≥2
    `causal_one_sided` edges (≥2 INTERVENTIONAL HELD bridges
    under the same DAG), promote those edges to `causal_bridged`.

    Pearl-ladder reading: bridged evidence means do-calculus
    inference is corroborated by an INDEPENDENT bridge — typically
    an estimate (backdoor / IV) plus a refuter (placebo /
    random-common-cause) — not just an estimate matched by a
    correlational coupling.

    Correlational edges (`Tier.ASSOCIATIONAL`) on the same pair do
    NOT count toward bridging and stay `'correlational'`. They're
    real evidence at a lower rung; conflating them with
    interventional corroboration would over-promote.

    Conservative wrt refutation: a `'refuted'` edge on the same
    pair does not demote the others — refutation flows through
    the per-result pass already (the refuted edge carries
    `evidentiary_level='refuted'` on its own metadata)."""
    by_pair: dict[tuple[str, str], list[BridgeEdge]] = {}
    for e in g.edges:
        by_pair.setdefault((e.source, e.target), []).append(e.metadata)

    upgrades: set[tuple[str, str]] = set()
    for pair, edges in by_pair.items():
        interventional_admits = [
            m for m in edges
            if m.evidentiary_level == 'causal_one_sided'
        ]
        if len(interventional_admits) >= 2:
            upgrades.add(pair)

    if not upgrades:
        return g

    new_edges: list[Edge[str, BridgeEdge]] = []
    for e in g.edges:
        if (
            (e.source, e.target) in upgrades
            and e.metadata.evidentiary_level == 'causal_one_sided'
        ):
            promoted_md = replace(
                e.metadata, evidentiary_level='causal_bridged',
            )
            new_edges.append(Edge(
                source=e.source, target=e.target,
                metadata=promoted_md,
            ))
        else:
            new_edges.append(e)
    return replace(g, edges=tuple(new_edges))
