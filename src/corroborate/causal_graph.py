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

`compose_direction` and `chain_tier` walk an edge sequence to
produce path-level direction + tier — chain composition for
admissibility / promotion checks.

`build_causal_graph(bridge_results)` consumes a list of corroborate
`BridgeResult`s and produces a `CausalGraph`. Convention on
targets-arity (mirrors v10):

- `len(targets) == 1` → node-only annotation (no outgoing edge).
- `len(targets) == 2` → binary edge `(source → target)`.
- `len(targets) >= 3` → JOINT bridge. Last target is the joint
  target; preceding targets are sources. Emits one `BridgeEdge`
  per `(source_i → target)` with `co_sources` = the OTHER sources
  of the same joint bridge. Graph stays binary; multi-source
  nature is preserved on each edge's metadata, and the joint
  bridge can be reconstructed by grouping edges with the same
  `bridge_name`.

`promote_bridged_evidence(g)` is the post-pass: for any (source,
target) pair with ≥2 `causal_one_sided` edges (≥2 INTERVENTIONAL
HELD bridges under the same DAG), those edges promote to
`causal_bridged`. Reading: do-calculus inference is corroborated
by an INDEPENDENT bridge — typically an estimate plus a refuter —
not just an estimate matched by a correlational coupling."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from typing import Literal

from corroborate.bridge import BridgeResult
from corroborate.graph import Edge, Graph
from corroborate.verdict import Verdict


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
    'refuted', 'correlational', 'causal_one_sided', 'causal_bridged',
]


# ============ BridgeEdge — graph edge metadata ============

@dataclass(frozen=True, slots=True)
class BridgeEdge:
    """Metadata stored on each edge of a `CausalGraph`.

    `bridge_name` — the source bridge / invariant name (matches
    `BridgeResult.name`).
    `direction` — DIRECT or INVERSE, inferred from `stats`'s `ate`
    sign (priority) or `rho` sign (fallback) at construction.
    `tier` — ASSOCIATIONAL by default; INTERVENTIONAL when the
    bridge result carries `stats['tier'] == 'interventional'` AND
    the verdict is HELD.
    `evidentiary_level` — 'refuted' for non-HELD; 'causal_one_sided'
    for INTERVENTIONAL admit; 'correlational' for ASSOCIATIONAL
    admit; 'causal_bridged' is set only by
    `promote_bridged_evidence` post-pass.

    `co_sources` — for joint bridges (len(targets) ≥ 3 emit one
    edge per source), this lists the OTHER sources of the same
    bridge so the joint claim is reconstructable by grouping
    edges with the same `bridge_name`.

    `feedback` — set when `stats['feedback'] == True`; signals
    intentional cycle participation. Graph walks use this to
    break cycle traversal."""
    bridge_name: str
    direction: Direction
    tier: Tier
    evidentiary_level: EvidentiaryLevel
    rho: float | None = None
    co_sources: tuple[str, ...] = ()
    feedback: bool = False
    condition_desc: str | None = None

    def __str__(self) -> str:
        bits = [
            self.bridge_name,
            f'{self.direction.value}/{self.tier.name.lower()}/{self.evidentiary_level}',
        ]
        if self.rho is not None:
            bits.append(f'ρ={self.rho:+.2f}')
        if self.co_sources:
            bits.append(f'joint:co_sources={list(self.co_sources)}')
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


# ============ build_causal_graph ============

def _direction_from_stats(
    stats: 'Mapping[str, float | int | bool | str]',
) -> tuple[Direction, float | None]:
    """Infer direction from stats. Priority: ate sign >
    rho sign > DIRECT default. Returns (direction, rho_or_None)."""
    rho_raw = stats.get('rho')
    ate_raw = stats.get('ate')
    rho: float | None = (
        float(rho_raw) if isinstance(rho_raw, (int, float))
        and not isinstance(rho_raw, bool) else None
    )
    ate: float | None = (
        float(ate_raw) if isinstance(ate_raw, (int, float))
        and not isinstance(ate_raw, bool) else None
    )

    if ate is not None and ate != 0:
        direction = Direction.INVERSE if ate < 0 else Direction.DIRECT
    elif rho is not None and rho < 0:
        direction = Direction.INVERSE
    else:
        direction = Direction.DIRECT
    return direction, rho


def build_causal_graph(
    bridge_results: Iterable[BridgeResult],
) -> CausalGraph:
    """Construct a `CausalGraph` from a list of `BridgeResult`s.

    Verdict mapping (corroborate's typology):
    - `Verdict.HELD` + `stats['tier'] == 'interventional'` →
      tier=INTERVENTIONAL, evidentiary_level='causal_one_sided'.
    - `Verdict.HELD` (no interventional tier marker) →
      tier=ASSOCIATIONAL, evidentiary_level='correlational'.
    - Any other verdict (`NO_EFFECT`, `POWER_INSUFFICIENT`,
      `INVARIANT_VIOLATION`) → tier=ASSOCIATIONAL,
      evidentiary_level='refuted'.

    `'causal_bridged'` is NOT derivable from a single BridgeResult;
    see `promote_bridged_evidence` for the graph-level post-pass."""
    g: CausalGraph = Graph()
    for r in bridge_results:
        # Single-target → node-only annotation.
        if len(r.targets) == 1:
            g = g.with_node(r.targets[0])
            continue
        if len(r.targets) < 2:
            continue

        direction, rho = _direction_from_stats(r.stats)
        tier_marker = r.stats.get('tier')
        is_held = r.verdict is Verdict.HELD
        promoted = is_held and tier_marker == 'interventional'
        tier = Tier.INTERVENTIONAL if promoted else Tier.ASSOCIATIONAL

        level: EvidentiaryLevel
        if not is_held:
            level = 'refuted'
        elif promoted:
            level = 'causal_one_sided'
        else:
            level = 'correlational'

        feedback_v = r.stats.get('feedback', False)
        feedback_flag = bool(feedback_v) if isinstance(
            feedback_v, (bool, int)
        ) else False

        if len(r.targets) == 2:
            a, b = r.targets[0], r.targets[1]
            edge = BridgeEdge(
                bridge_name=r.name,
                direction=direction,
                tier=tier,
                evidentiary_level=level,
                rho=rho,
                feedback=feedback_flag,
            )
            g = g.with_edge(a, b, edge)
        else:
            # Joint bridge: last is target, others are sources.
            sources = r.targets[:-1]
            target = r.targets[-1]
            for source in sources:
                co_sources = tuple(s for s in sources if s != source)
                edge = BridgeEdge(
                    bridge_name=r.name,
                    direction=direction,
                    tier=tier,
                    evidentiary_level=level,
                    rho=rho,
                    co_sources=co_sources,
                    feedback=feedback_flag,
                )
                g = g.with_edge(source, target, edge)
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
