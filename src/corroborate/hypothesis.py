"""Hypothesis — (intervention, edges, measurables) over a record schema R.

A Hypothesis names a CHANGE to the theory and a SET of typed
edges + pre-registered measurables. Generic in `R: Mapping[str,
object]` — the (single) record schema all measurables are typed
against.

Components:

- `intervention: Mapping[str, object]` — runtime kwargs passed to
  `functools.partial(theory, **intervention)`. Mixes HP scalars
  (γ, lr, batch_size, total_steps), config-bundle slots (Replay,
  WarmedUpdate), and mechanism swaps (`partial(bootstrap,
  greedification=double_greedify)`). The dict is the *executable*
  shape; identity at this layer is type-erased on purpose.
- `intervention_arms: tuple[Intervention, ...]` — the *typed*
  identity of mechanism swaps only. HPs are NOT here; HPs are
  cell covariates that downstream meta-regression cleaves on.
  Empty tuple → baseline arm; non-empty tuple → treatment arm
  whose `arm_key()` is the canonical fingerprint.
- `edges: tuple[claim_bridge.Bridge, ...]` — typed-edge subgraph
  claim. Each Bridge carries `source` / `target` paths,
  `intervention: DoEffect | None` (interventional vs coupling
  edge), `tier`, and per-edge `predicted_direction`. Body-less
  for the verdict-walk path (`hypothesis_subgraph_verdict`
  consumes Bridges as metadata only).
- `measurables: tuple[Measurable[R, object], ...]` — pre-
  registered measurables cell_runner persists as scalar columns.
- `predicted_direction: PredictedDirection | None` — author-declared
  sign of the predicted treatment-vs-baseline effect (top-level
  hypothesis-wide default; per-edge predicted_direction overrides
  the analyses per stratum).

Arm identity flows exclusively through `intervention_arms` — two
hypotheses with the same arms but different HP grid points share
an `arm_key()` and pair as same-arm cells; the HP difference is a
covariate, not an arm distinguisher.

Structural identity beyond arm key is recoverable from the
measurements a run produces — leaf values land at dotted topology
paths via `signature.walk_paths`, and `aggregate.leaf_signature`
projects a `RunRow.measurements` to the configurational subset
suitable as a group-by key."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

from corroborate._canonical import canonical_str
from corroborate.intervention import Intervention, combined_arm_key
from corroborate.measurable import Measurable

if TYPE_CHECKING:
    # `claim_bridge.Bridge` is the typed-edge surface for the
    # Hypothesis subgraph claim; imported under TYPE_CHECKING
    # because `claim_bridge` depends on `hypothesis.PredictedDirection`.
    # The field annotation resolves through `from __future__ import
    # annotations`.
    from corroborate.claim_bridge import Bridge as ClaimBridge

__all__ = ['Hypothesis', 'PredictedDirection', 'canonical_str']


type PredictedDirection = Literal['a_gt_b', 'a_lt_b', 'two_sided']
"""Author-declared *prior* sign of the predicted treatment-vs-
baseline effect on the primary outcome. `'a_gt_b'` predicts the
intervention's arm exceeds the baseline; `'a_lt_b'` predicts
below; `'two_sided'` predicts non-zero in either direction.
None on Hypothesis means direction is unstated (downstream
infers from a `held` flag if available).

Distinct from `causal_graph.Direction` — that's the *observed*
sign (DIRECT / INVERSE) inferred post-hoc from a stat's value.
PredictedDirection is the prior; Direction is the posterior."""


@dataclass(frozen=True, slots=True)
class Hypothesis[R: Mapping[str, object]]:
    """A research hypothesis: an intervention plus the typed
    causal subgraph (bridge edges) the intervention claims.

    Generic in `R: Mapping[str, object]` — the (single) record
    schema all bridges are typed against. Authors using TypedDict
    for their record get typed bridge bodies (no narrowing);
    authors using plain `Mapping[str, object]` continue to narrow
    at use site.

    `edges: tuple[claim_bridge.Bridge, ...]` is the typed-edge
    subgraph surface. Each Bridge carries source / target
    measurement paths, `intervention: DoEffect | None`
    (interventional contrast vs measurement-coupling), Pearl
    `tier`, and per-edge `predicted_direction`. Bridges with
    `intervention is not None` are the rung-2 mechanism /
    outcome edges; bridges with `intervention is None` are
    measurement-to-measurement coupling edges.

    `measurables: tuple[Measurable[R, object], ...]` is the
    sweep-time pre-registration channel — measurables the author
    wants computed and persisted as scalar columns on every
    RunRow without having to author a per-record body for each.
    Each entry produces a column at the measurable's bare `.name`
    in `RunRow.measurements`; the substrate controls the column-
    name namespace (a measurable named `eval_final_mean`
    lands as `eval_final_mean`). Available downstream for
    typed-edge bridges (`Bridge.target=<name>`) and analyses to
    consume directly.

    Both surfaces can coexist on one Hypothesis. The top-level
    `predicted_direction: PredictedDirection | None` is vestigial
    when `edges` is populated (each edge carries its own
    `predicted_direction`).

    The framework treats a cell as producing ONE record, even
    when the underlying machinery has internal sub-passes (RL's
    eval bursts during training, etc.). Sub-pass results are
    additional fields on the same record dict — possibly with
    different shapes (`(T,)` per-step training fields,
    `(n_bursts, K)` per-burst eval fields). Bridges read whichever
    keys they care about; the record's structure is the substrate
    author's call."""
    name: str
    intervention: Mapping[str, object]
    predicted_direction: PredictedDirection | None = None
    intervention_arms: tuple[Intervention, ...] = field(default_factory=tuple)
    edges: tuple['ClaimBridge', ...] = field(default_factory=tuple)
    measurables: tuple[Measurable[R, object], ...] = field(
        default_factory=tuple,
    )

    def edges_by_target(self, target: str) -> tuple['ClaimBridge', ...]:
        """All typed edges whose `target` matches `target`. The
        primary lookup after the role-enum subtraction — consumers
        select on the path, not on a paper-narrative name."""
        return tuple(e for e in self.edges if e.target == target)

    def intervention_edges(self) -> tuple['ClaimBridge', ...]:
        """Edges whose `source` is a `DoEffect` — the rung-2
        contrast edges that drive paired comparisons. Replaces the
        former `mechanism + outcome + refuter` role union; the
        scope-distinction (mechanism vs outcome) is recoverable
        from `target` namespace or claim-graph topology, not from
        a per-edge enum."""
        from corroborate.intervention import DoEffect
        return tuple(
            e for e in self.edges if isinstance(e.source, DoEffect)
        )

    def coupling_edges(self) -> tuple['ClaimBridge', ...]:
        """Edges whose `source` is NOT a `DoEffect` — measurement-
        to-measurement coupling edges (formerly `role='link'`).
        Tested via cross-stratum Pearson r over the per-group
        effect sizes of the source and target paths."""
        from corroborate.intervention import DoEffect
        return tuple(
            e for e in self.edges if not isinstance(e.source, DoEffect)
        )

    def arm_key(self) -> str:
        """Canonical fingerprint of the typed `intervention_arms`.

        Empty `intervention_arms` → `'baseline'`; non-empty →
        `'+'`-joined slot=replacement keys (sorted by slot_path).
        Two hypotheses with same arms but different HP grid
        points share one `arm_key()`; HP variation is a covariate,
        not an arm distinguisher."""
        return combined_arm_key(self.intervention_arms)
