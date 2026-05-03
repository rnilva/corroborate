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

if TYPE_CHECKING:
    # Forward import: `claim_bridge` depends on `causal_graph`
    # transitively via verdict; lazy-typed to avoid the cycle.
    from corroborate.claim_bridge import Bridge as ClaimBridge


# ============ Direction — sign or predicate of an edge ============

class Direction(Enum):
    """Sign or predicate carried by an edge.

    Two axes coexist:

    - **Coupling sign**: DIRECT = source↑ ⇒ target↑; INVERSE =
      source↑ ⇒ target↓. Multiplicative chain composition: DIRECT
      is identity, INVERSE squared is DIRECT.
    - **Threshold predicate**: AT_MOST = `source ≤ threshold`;
      AT_LEAST = `source ≥ threshold`. Used by INVARIANT self-loop
      bridges with `Bridge.threshold` set; carried as direction so
      the predicate is one typed field instead of a separate enum.
      Predicates are NOT chain-composable — invariants are
      self-loops, never cross-node chain edges."""
    DIRECT = 'direct'
    INVERSE = 'inverse'
    AT_MOST = 'at_most'
    AT_LEAST = 'at_least'

    def __mul__(self, other: 'Direction') -> 'Direction':
        if (
            self in (Direction.AT_MOST, Direction.AT_LEAST)
            or other in (Direction.AT_MOST, Direction.AT_LEAST)
        ):
            raise TypeError(
                f'cannot compose threshold direction {self.name} with '
                f'{other.name}; AT_MOST/AT_LEAST are predicates on '
                f'self-loop invariants, not chain-edge signs.',
            )
        if self is other:
            return Direction.DIRECT
        return Direction.INVERSE


# ============ Tier — Pearl-ladder rung ============

class InvalidTierTransition(ValueError):
    """A `Tier` transition violated the Pearl-ladder lifecycle
    algebra. Raised by `Tier.promote()` / `Tier.demote()` when
    called on a tier without further movement available."""


class Tier(IntEnum):
    """Pearl-ladder rung the evidence has reached, plus INVARIANT
    for substrate-axiom claims that are NOT on the rung.

    - `INVARIANT` (numeric 0) = substrate-author axiom expressed
      as a self-loop bridge with a threshold predicate (e.g.
      `jensen_dormancy_gap ≤ 0` is the Hasselt-2010 premise
      check). Pre-statistical: a structural constraint the
      substrate declares as the precondition under which the
      mechanism's causal-chain bites. Never appears in cross-node
      chains — `chain_tier` skips it.
    - `ASSOCIATIONAL` (rung 1) = observational coupling. PC
      adjacency, Spearman correlation, mediator-features within-
      env Pearson — all rung 1.
    - `INTERVENTIONAL` (rung 2) = confirmed do() effect. Paired
      Hedges' g HELD on a treatment-vs-baseline comparison.

    Promotion is structurally constrained: only ASSOCIATIONAL →
    INTERVENTIONAL via `.promote()`. INVARIANT is orthogonal —
    `.promote()` / `.demote()` raise on it. The reverse via
    `.demote()` exists for downgrade scenarios (refuter
    contradicts an interventional admit), but the framework
    treats refutation primarily through
    `evidentiary_level='refuted'` on the BridgeEdge — `.demote()`
    is the rare case."""
    INVARIANT = 0
    ASSOCIATIONAL = 1
    INTERVENTIONAL = 2

    def promote(self) -> 'Tier':
        if self is Tier.ASSOCIATIONAL:
            return Tier.INTERVENTIONAL
        raise InvalidTierTransition(
            f'{self.name} cannot promote — already top or off-ladder.',
        )

    def demote(self) -> 'Tier':
        if self is Tier.INTERVENTIONAL:
            return Tier.ASSOCIATIONAL
        raise InvalidTierTransition(
            f'{self.name} cannot demote — already bottom or off-ladder.',
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
    coarsest tier as default).

    INVARIANT edges are SKIPPED: they're substrate-axiom self-loops,
    never compose into cross-node chains; pulling the chain min
    down to INVARIANT (numerically 0) would misrepresent the
    chain's evidence rung."""
    result = Tier.INTERVENTIONAL
    any_edge = False
    for e in edges:
        if e.tier is Tier.INVARIANT:
            continue
        any_edge = True
        if e.tier < result:
            result = e.tier
    return result if any_edge else Tier.ASSOCIATIONAL


# ============ Pre-evaluation authored graph ============

def authored_graph(
    bridges: 'Iterable[ClaimBridge]',
) -> CausalGraph:
    """Build the unevaluated graph topology from a `Sequence[Bridge]`.

    Each bridge contributes one edge. The source-node rendering:

    - `bridge.source` is a `DoEffect`:
      `do(treatment|vs=baseline) → bridge.target`, tier
      INTERVENTIONAL.
    - Otherwise: `bridge.source → bridge.target`
      (measurable-to-measurable), tier inherits `bridge.tier`.

    Used by analyses that want to inspect the authored graph
    topology BEFORE running bridges — e.g. cache builders,
    Protocol-conforming module discoverers."""
    from corroborate.intervention import DoEffect
    g: CausalGraph = Graph()
    for b in bridges:
        if isinstance(b.source, DoEffect):
            source_key = b.source.node_key()
            tier = Tier.INTERVENTIONAL
        else:
            source_key = b.source_name
            tier = b.tier
        edge = BridgeEdge(
            bridge_name=b.name,
            direction=b.direction,
            tier=tier,
            evidentiary_level='unevaluated',
        )
        g = g.with_edge(source_key, b.target_name, edge)
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
