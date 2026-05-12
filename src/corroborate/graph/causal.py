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
  ('refuted' / 'correlational' / 'causal_one_sided'). Lifecycle
  the bridge is in.
- `extent_hash` — frozenset-of-admitted-cell-ids hash carried
  from `BridgeEvaluation.extent_hash`. Two edges with the same
  `(source, target, extent_hash)` were evaluated against identical
  cell-sets — the cluster-identity primitive. Authors group their
  refutation clusters by sharing scope predicates (extracted as
  module-level constants); the framework derives cluster identity
  empirically rather than from author labels.

`compose_direction` and `chain_tier` walk an edge sequence to
produce path-level direction + tier — chain composition for
admissibility checks along paths.

`evaluated_graph(bridges, post_eval)` realizes the principle's
`evidence(E)` component: take an authored topology and stamp
each edge with its verdict-derived `evidentiary_level` and the
per-bridge `extent_hash`. The result IS the principle's
"hypothesis = (V, E, evidence(E))". `clusters_by_extent` +
`cluster_verdict` + `ClusterVerdict` are the cluster-identity
query primitives that operate on the stamped graph.

**No auto-promotion.** Refutation-cluster identity is queryable
post-evaluation via `(source, target, extent_hash)` grouping.
Per-bridge `evidentiary_level` carries the Pearl-rung admit
fact; cluster-level "this edge has multiple INTERVENTIONAL HELD
bridges sharing an extent" is a structural query authors
compose, not a central aggregator on the graph.

**Restoration path for effect-size on edges**: this module
intentionally does NOT carry numeric effect summaries (ate /
rho / pvalue / n_observations) on `BridgeEdge`. When a
consumer materializes (DOT renderer, chain effect-size
product, pooled cluster effect summary), restore an
`EffectStats` dataclass + the per-edge numeric fields
together with the consumer in the same PR — typed surface
follows the caller, per CLAUDE.md primitive discipline."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum, IntEnum
from typing import TYPE_CHECKING, Literal, override

from corroborate.bridge.verdict import Verdict
from corroborate.graph.graph import Edge, Graph

if TYPE_CHECKING:
    # Forward import: `claim_bridge` depends on `causal_graph`
    # transitively via verdict; lazy-typed to avoid the cycle.
    from corroborate.bridge.bridge import Bridge as ClaimBridge


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
      as a self-loop bridge with a threshold predicate (e.g. the
      substrate's invariance-gap measurable falling on the side
      of its theorem's precondition). Pre-statistical: a
      structural constraint the substrate declares as the
      precondition under which the mechanism's causal-chain
      bites. Never appears in cross-node chains — `chain_tier`
      skips it.
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
    'refuted', 'correlational', 'causal_one_sided',
]


# ============ BridgeEdge — graph edge metadata ============

@dataclass(frozen=True, slots=True)
class BridgeEdge:
    """Metadata stored on each edge of a `CausalGraph`.

    `bridge_name` — the claim_bridge.Bridge.name that produced
    this edge.
    `direction` — DIRECT or INVERSE, declared on the bridge.
    `tier` — ASSOCIATIONAL by default; INTERVENTIONAL when the
    edge's source is a DoEffect.
    `evidentiary_level` — 'refuted' for NO_EFFECT;
    'causal_one_sided' for INTERVENTIONAL admit; 'correlational'
    for ASSOCIATIONAL admit; 'unevaluated' otherwise. No
    auto-promotion: cluster-level corroboration is a structural
    query over the post-evaluated graph, not a baked-in level.

    `feedback` — set on edges that intentionally participate in
    cycles. Graph walks use this to break cycle traversal.

    `extent_hash` — `hash(frozenset(admitted_cell_ids))` carried
    from `BridgeEvaluation.extent_hash`. Two edges with the same
    `(source, target, extent_hash)` admit identical cell-sets on
    the cache that produced them — the cluster identity primitive.
    Empty extent → all empties share `hash(frozenset())`, honestly
    reflecting "framework cannot distinguish these on this cache."
    Cluster identity is corpus-dependent by design (bridge verdicts
    already are; cluster identity inherits the dependency)."""
    bridge_name: str
    direction: Direction
    tier: Tier
    evidentiary_level: EvidentiaryLevel
    feedback: bool = False
    extent_hash: int = 0

    @override
    def __str__(self) -> str:
        bits = [
            self.bridge_name,
            f'{self.direction.value}/{self.tier.name.lower()}/{self.evidentiary_level}',
        ]
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
    from corroborate.core.intervention import DoEffect
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


# ============ Post-evaluation evidence(E) stamper ============


@dataclass(frozen=True, slots=True)
class PostEvalEntry:
    """Per-bridge post-evaluation pair: the verdict the bridge's
    `holds_when` body returned and the `extent_hash` of the
    admitted cell-set on the cache that produced it.

    Callers construct this from either a `BridgeEvaluation`
    (`PostEvalEntry(ev.verdict, ev.extent_hash)`) or from a
    persisted `*.run.json` snapshot. Tightening the
    `evaluated_graph` signature with a typed record keeps the
    shape explicit and survives the addition of a third field
    (e.g. `refutation_class`, `assumption_violations`) without
    breaking callers."""
    verdict: Verdict
    extent_hash: int


def _stamp_level(tier: Tier, verdict: Verdict) -> EvidentiaryLevel:
    """Map `(Tier, Verdict)` → `EvidentiaryLevel`.

    Dispatch via `Verdict.is_corroboration()` /
    `Verdict.is_refutation()` — the enum's own predicates —
    so HELD and HELD_WITH_SCOPE_FLAG both stamp as admit
    rungs (per verdict-enum semantics: both are positive
    evidence the claim holds at population level; the
    scope-flag refines uniformity but not corroboration).
    NO_EFFECT stamps as 'refuted'. POWER_INSUFFICIENT,
    INVARIANT_VIOLATION, INADMISSIBLE all stamp as
    'unevaluated' — per verdict.py:71, INVARIANT_VIOLATION
    means the test was out of scope, NOT a refutation."""
    if verdict.is_corroboration():
        return (
            'causal_one_sided' if tier is Tier.INTERVENTIONAL
            else 'correlational'
        )
    if verdict.is_refutation():
        return 'refuted'
    return 'unevaluated'


def evaluated_graph(
    bridges: 'Iterable[ClaimBridge]',
    post_eval: Mapping[str, PostEvalEntry],
) -> CausalGraph:
    """Realize the principle's `evidence(E)` component.

    Build the authored graph topology then stamp each edge's
    `evidentiary_level` (from verdict via `_stamp_level`) and
    `extent_hash` from the `post_eval` mapping
    `{bridge_name: PostEvalEntry(verdict, extent_hash)}`.

    Bridges absent from `post_eval` keep authored defaults
    (`evidentiary_level='unevaluated'`, `extent_hash=0`).

    Per HYPOTHESIS_AS_GRAPH.md: the resulting graph IS the
    hypothesis under the principle's definition
    `Hypothesis = (V, E, evidence(E))`. Cluster-shaped queries
    on this graph use `clusters_by_extent` + `cluster_verdict`."""
    g = authored_graph(bridges)
    new_edges: list[Edge[str, BridgeEdge]] = []
    for e in g.edges:
        pe = post_eval.get(e.metadata.bridge_name)
        if pe is None:
            new_edges.append(e)
            continue
        new_meta = replace(
            e.metadata,
            evidentiary_level=_stamp_level(e.metadata.tier, pe.verdict),
            extent_hash=pe.extent_hash,
        )
        new_edges.append(replace(e, metadata=new_meta))
    return replace(g, edges=tuple(new_edges))


# ============ Cluster-identity queries ============


class ClusterVerdict(Enum):
    """Verdict on a refutation cluster (multi-bridge edge group).

    Mirrors the framework's three-verdict-not-binary discipline at
    the cluster level: SUPPORTED / REFUTED / UNDERPOWERED, plus
    EMPTY_EXTENT for clusters whose member bridges all admit zero
    cells on the current cache (the framework cannot empirically
    distinguish them)."""
    SUPPORTED = 'supported'
    REFUTED = 'refuted'
    UNDERPOWERED = 'underpowered'
    EMPTY_EXTENT = 'empty_extent'


_EMPTY_EXTENT_HASH = hash(frozenset[str]())

_ADMIT_LEVELS: frozenset[EvidentiaryLevel] = frozenset(
    {'correlational', 'causal_one_sided'},
)


def clusters_by_extent(
    g: CausalGraph,
) -> dict[tuple[str, str, int], tuple[BridgeEdge, ...]]:
    """Group every edge in `g` by `(source, target, extent_hash)`.

    Multi-edge groups are refutation clusters (≥2 bridges
    corroborating the same edge identity); singletons are
    standalone bridges. Empty-extent edges all share
    `hash(frozenset())` per the cluster-identity invariant."""
    buckets: dict[tuple[str, str, int], list[BridgeEdge]] = {}
    for e in g.edges:
        key = (e.source, e.target, e.metadata.extent_hash)
        buckets.setdefault(key, []).append(e.metadata)
    return {k: tuple(v) for k, v in buckets.items()}


def cluster_verdict(
    members: tuple[BridgeEdge, ...],
) -> ClusterVerdict:
    """Compose member edges' evidentiary_level into a cluster
    verdict.

    Empty-extent (all members admit zero cells) is its own
    bucket — the corpus can't test the cluster. REFUTED if any
    member refutes. SUPPORTED if every member admits with a
    non-empty extent. Otherwise UNDERPOWERED (mix of admits and
    unevaluateds)."""
    if not members:
        return ClusterVerdict.UNDERPOWERED
    if all(m.extent_hash == _EMPTY_EXTENT_HASH for m in members):
        return ClusterVerdict.EMPTY_EXTENT
    levels = {m.evidentiary_level for m in members}
    if 'refuted' in levels:
        return ClusterVerdict.REFUTED
    if levels <= _ADMIT_LEVELS:
        return ClusterVerdict.SUPPORTED
    return ClusterVerdict.UNDERPOWERED


def composed_verdict(
    g: CausalGraph,
    *,
    bridges: 'Iterable[ClaimBridge]',
) -> ClusterVerdict:
    """Compose verdicts over a hand-listed set of bridges.

    The Finding-walk primitive: a finding declares the `Bridge`
    instances that support its claim (imported by Python name so
    rename → ImportError, no stringly-typed references); this
    helper looks them up in the post-eval graph and composes
    their evidentiary_levels into a single cluster verdict.

    Single helper for cluster- and envelope-shaped Findings —
    cluster integrity (extent uniformity) is a separate concern
    surfaced by the `run_hypothesis.py` cluster rollup, not
    re-checked here. UNDERPOWERED on len-mismatch (one or more
    declared bridges absent from graph — possible when a bridge
    was authored but its hypothesis hasn't been re-evaluated)."""
    expected_names = {b.name for b in bridges}
    found = tuple(
        e.metadata for e in g.edges
        if e.metadata.bridge_name in expected_names
    )
    if len(found) != len(expected_names):
        return ClusterVerdict.UNDERPOWERED
    return cluster_verdict(found)
