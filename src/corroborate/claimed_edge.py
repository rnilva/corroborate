"""ClaimedEdge — typed edge in a Hypothesis's causal subgraph
claim.

A `Hypothesis` is structurally a *subgraph claim*: a set of typed
edges, each declaring a role in the §3 verdict pattern (mechanism
/ outcome / link / refuter). v10's `causal_graph.py` provides the
lower-level `BridgeEdge` (typed by Direction × Tier ×
EvidentiaryLevel — observed sign + Pearl rung + admission state);
this module provides the *Hypothesis-side* edge, which composes a
prospective edge with:

- the bridge's `role` in the §3 pattern (Hypothesis-specific
  taxonomy, not graph-side),
- the predicted direction in comparison-space
  (`PredictedDirection` literal: `'a_gt_b'` | `'a_lt_b'` |
  `'two_sided'`) — author's *prior* sign claim, distinct from
  the graph-side `Direction` enum which records the *observed*
  sign post-verdict,
- the `Bridge[R]` test function that produces the per-edge
  verdict at run-time.

The mechanism edge is the *load-bearing* claim: it carries the
theoretical content the rest of the subgraph hangs from. Outcome
and link edges test its implications. The factories below
(`mechanism_edge`, `outcome_edge`, `link_edge`, `refuter_edge`)
make declaring a §3-shaped subgraph ergonomic — each fills in
conventional defaults (`source='do(arm)'`, tier per Pearl-tier
convention) so authors only specify what's distinctive about
their edge."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from corroborate.bridge import Bridge
from corroborate.causal_graph import Tier
from corroborate.hypothesis import PredictedDirection


type BridgeRole = Literal['mechanism', 'outcome', 'link', 'refuter']
"""Hypothesis-side taxonomy for typed claimed edges. The §3
verdict pattern (`mechanism HELD ↛ outcome HELD ↛ link HELD`)
ranges over these roles. A refuter is a mechanism-shaped edge
whose verdict, if HELD, would *contradict* the claim — used to
test alternative explanations. Roles are NOT graph-side: the
same `BridgeEdge` could play different roles in different
hypotheses; role attaches at this layer."""


@dataclass(frozen=True, slots=True)
class ClaimedEdge[R: Mapping[str, object]]:
    """One typed edge claimed by a Hypothesis's causal subgraph.

    `role` distinguishes the edge's place in the §3 verdict
    pattern. `source` and `target` are graph-position strings —
    `'do(arm)'` is the conventional sentinel for an interventional
    edge's source (one per Hypothesis, derives from
    `intervention_arms`); for `link`-role edges, `source` is a
    measurement path (e.g. `'mechanism.jensen_gap'`).

    `predicted_direction` uses the comparison-side literal type
    (`'a_gt_b'`, `'a_lt_b'`, `'two_sided'`) — what
    `hypothesis_comparison_from_cells` needs at the verdict-
    decision step. The Pearl-direction (`DIRECT` / `INVERSE`) on
    `BridgeEdge` is *observed* post-hoc from stats sign at stage
    8; the predicted_direction here is *prior*.

    `tier` follows Pearl convention: `INTERVENTIONAL` for
    `do(arm)`-sourced edges (we performed the do); `ASSOCIATIONAL`
    for measurement-to-measurement edges that lack a do() at the
    source (link edges, observational adjacencies)."""
    role: BridgeRole
    source: str
    target: str
    predicted_direction: PredictedDirection
    tier: Tier
    bridge: Bridge[R]


# ============ Role factories ============

def mechanism_edge[R: Mapping[str, object]](
    *,
    target: str,
    predicted_direction: PredictedDirection,
    bridge: Bridge[R],
    source: str = 'do(arm)',
) -> ClaimedEdge[R]:
    """The mechanism edge — the central theoretical claim. Source
    defaults to the `'do(arm)'` sentinel (interventional tier).

    Hasselt 2010's claim "DDQN reduces overestimation bias" is a
    mechanism edge with `target='mechanism.jensen_gap'`,
    `predicted_direction='a_lt_b'`."""
    return ClaimedEdge(
        role='mechanism',
        source=source,
        target=target,
        predicted_direction=predicted_direction,
        tier=Tier.INTERVENTIONAL,
        bridge=bridge,
    )


def outcome_edge[R: Mapping[str, object]](
    *,
    target: str,
    predicted_direction: PredictedDirection,
    bridge: Bridge[R],
    source: str = 'do(arm)',
) -> ClaimedEdge[R]:
    """The outcome edge — does the intervention move the paper-
    level outcome? Same `do(arm)` source as the mechanism edge,
    different target (the practitioner-facing outcome path).

    For DDQN: `target='outcome.late_window_mean'`,
    `predicted_direction='a_gt_b'`."""
    return ClaimedEdge(
        role='outcome',
        source=source,
        target=target,
        predicted_direction=predicted_direction,
        tier=Tier.INTERVENTIONAL,
        bridge=bridge,
    )


def link_edge[R: Mapping[str, object]](
    *,
    source: str,
    target: str,
    predicted_direction: PredictedDirection,
    bridge: Bridge[R],
) -> ClaimedEdge[R]:
    """The link edge — does the mechanism's effect *correlate*
    with the outcome's across cells? Source and target are both
    measurement paths (no do() at the source). Default tier is
    `ASSOCIATIONAL` because we cannot intervene on the mechanism
    quantity directly without a different do() experiment.

    For DDQN: `source='mechanism.jensen_gap'`,
    `target='outcome.late_window_mean'`,
    `predicted_direction='a_gt_b'` (a stronger bias reduction
    SHOULD predict a larger return improvement)."""
    return ClaimedEdge(
        role='link',
        source=source,
        target=target,
        predicted_direction=predicted_direction,
        tier=Tier.ASSOCIATIONAL,
        bridge=bridge,
    )


def refuter_edge[R: Mapping[str, object]](
    *,
    target: str,
    predicted_direction: PredictedDirection,
    bridge: Bridge[R],
    source: str = 'do(arm)',
) -> ClaimedEdge[R]:
    """A refuter — a mechanism-shaped edge whose HELD verdict
    would *contradict* the hypothesis. Authoring a refuter
    deliberately invites falsification: if both the mechanism
    edge AND a refuter edge land HELD, the hypothesis fails its
    own consistency check.

    Example: a refuter for DDQN's bias-reduction claim might
    target an OPPOSITE quantity that the same swap should NOT
    affect; a HELD refuter signals the swap has unintended
    side-effects."""
    return ClaimedEdge(
        role='refuter',
        source=source,
        target=target,
        predicted_direction=predicted_direction,
        tier=Tier.INTERVENTIONAL,
        bridge=bridge,
    )
