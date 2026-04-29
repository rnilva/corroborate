"""Hypothesis — (intervention, [bridges]) over a record schema R.

A Hypothesis names a CHANGE to the theory and a SET of paper-level
assertions (bridges) that should hold under that change. Generic
in `R: Mapping[str, object]` — the same R the bridges are typed
against, so all bridges in a hypothesis share the record schema
(one theory → one record schema).

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
- `bridges: tuple[Bridge[R], ...]` — paper-level assertions
  applied to the resulting record. All share R.
- `predicted_direction: PredictedDirection | None` — author-declared
  sign of the predicted treatment-vs-baseline effect.

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
from corroborate.bridge import Bridge
from corroborate.intervention import Intervention, combined_arm_key

if TYPE_CHECKING:
    # `ClaimedEdge` / `BridgeRole` are the typed-subgraph surface;
    # imported under TYPE_CHECKING because `claimed_edge.py`
    # depends on `hypothesis.PredictedDirection`. The field
    # annotation is a string at runtime via
    # `from __future__ import annotations`.
    from corroborate.claimed_edge import BridgeRole, ClaimedEdge

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

    Two surfaces for declaring per-edge tests, in transition:

    - `edges: tuple[ClaimedEdge[R], ...]` — the *typed* surface.
      Each `ClaimedEdge` carries its role (mechanism / outcome /
      link / refuter), source / target measurement paths,
      predicted direction, Pearl tier, and the Bridge to run.
      The mechanism edge is the load-bearing claim;
      `mechanism_edge()` accessor returns it.
    - `bridges: tuple[Bridge[R], ...]` — the back-compat flat
      tuple. No per-bridge role, source/target, or tier. Existing
      callers; new code should populate `edges`.

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
    bridges: tuple[Bridge[R], ...] = ()
    predicted_direction: PredictedDirection | None = None
    intervention_arms: tuple[Intervention, ...] = field(default_factory=tuple)
    edges: tuple[ClaimedEdge[R], ...] = field(default_factory=tuple)

    def mechanism_edge(self) -> ClaimedEdge[R] | None:
        """Return the load-bearing mechanism edge if one is
        declared, else None.

        The §3 verdict pattern's central claim. Outcome and link
        edges test its implications; `arm_key` derives from the
        intervention that's the source of this edge."""
        for e in self.edges:
            if e.role == 'mechanism':
                return e
        return None

    def edges_by_role(
        self, role: 'BridgeRole',
    ) -> tuple['ClaimedEdge[R]', ...]:
        """All edges with the given role. Most subgraphs have one
        mechanism + one outcome + one link, but the API permits
        multiple of each (e.g., several outcome paths tested
        against the same intervention)."""
        return tuple(e for e in self.edges if e.role == role)

    def arm_key(self) -> str:
        """Canonical fingerprint of the typed `intervention_arms`.

        Empty `intervention_arms` → `'baseline'`; non-empty →
        `'+'`-joined slot=replacement keys (sorted by slot_path).
        Two hypotheses with same arms but different HP grid
        points share one `arm_key()`; HP variation is a covariate,
        not an arm distinguisher."""
        return combined_arm_key(self.intervention_arms)
