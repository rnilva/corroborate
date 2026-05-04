"""Hypothesis — (intervention, measurables) over a record schema R.

A Hypothesis names a CHANGE to the theory and a SET of pre-
registered measurables. Generic in `R: Mapping[str, object]` —
the (single) record schema all measurables are typed against.

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
- `measurables: tuple[Measurable[R, object], ...]` — pre-
  registered measurables cell_runner persists as scalar columns.
- `predicted_direction: PredictedDirection | None` — author-declared
  sign of the predicted treatment-vs-baseline effect.

Bridges live at module level (`module.BRIDGES: tuple[Bridge, ...]`),
not on the Hypothesis. The verdict path is `runner.run_module` →
`evaluate(b, cells)` per bridge.

Arm identity flows exclusively through `intervention_arms` — two
hypotheses with the same arms but different HP grid points share
an `arm_key()` and pair as same-arm cells; the HP difference is a
covariate, not an arm distinguisher.

Structural identity beyond arm key is recoverable from the
measurements a run produces — leaf values land at dotted topology
paths via `signature.walk_paths`, and `corpus.leaf_signature`
projects a `RunRow.measurements` to the configurational subset
suitable as a group-by key."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Literal, override

from corroborate._internals.canonical import canonical_str
from corroborate.core.intervention import Intervention, combined_arm_key

if TYPE_CHECKING:
    from corroborate.measurables import Measurable

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


class Hypothesis[R: Mapping[str, object]]:
    """A research hypothesis: an intervention plus pre-registered
    measurables.

    Generic in `R: Mapping[str, object]` — the (single) record
    schema all measurables are typed against. Authors using
    TypedDict for their record get typed bridge bodies (no
    narrowing); authors using plain `Mapping[str, object]`
    continue to narrow at use site.

    `measurables: tuple[Measurable[R, object], ...]` is the
    sweep-time pre-registration channel — measurables the author
    wants computed and persisted as scalar columns on every
    RunRow without having to author a per-record body for each.
    Each entry produces a column at the measurable's bare `.name`
    in `RunRow.measurements`; the substrate controls the column-
    name namespace.

    Bridges live at module level (`module.BRIDGES: tuple[Bridge, ...]`).
    The verdict path is `runner.run_module` → `evaluate(b, cells)`
    per bridge; the framework no longer has a "subgraph verdict"
    surface.

    The framework treats a cell as producing ONE record, even
    when the underlying machinery has internal sub-passes (RL's
    eval bursts during training, etc.). Sub-pass results are
    additional fields on the same record dict — possibly with
    different shapes (`(T,)` per-step training fields,
    `(n_bursts, K)` per-burst eval fields). Bridges read whichever
    keys they care about; the record's structure is the substrate
    author's call.

    **Variance.** Field access is via `@property` (not bare
    dataclass attrs) so PEP 695 inference lands `R` contravariant
    — `Measurable[R, object]` is contravariant in R (post the
    Measurable refactor), and the recursive `tuple` field is
    covariant in element. A `Hypothesis[Mapping[str, object]]`
    (framework-built generic) is therefore assignable to a slot
    expecting `Hypothesis[DQNTrajectoryRecord]` (substrate-typed),
    no `cast` needed at the substrate boundary."""

    __slots__ = (
        '_name', '_intervention', '_predicted_direction',
        '_intervention_arms', '_measurables',
    )

    _name: str
    _intervention: Mapping[str, object]
    _predicted_direction: PredictedDirection | None
    _intervention_arms: tuple[Intervention, ...]
    _measurables: tuple[Measurable[R, object], ...]

    def __init__(
        self,
        name: str,
        intervention: Mapping[str, object],
        predicted_direction: PredictedDirection | None = None,
        intervention_arms: tuple[Intervention, ...] = (),
        measurables: tuple[Measurable[R, object], ...] = (),
    ) -> None:
        self._name = name
        self._intervention = intervention
        self._predicted_direction = predicted_direction
        self._intervention_arms = intervention_arms
        self._measurables = measurables

    @property
    def name(self) -> str:
        return self._name

    @property
    def intervention(self) -> Mapping[str, object]:
        return self._intervention

    @property
    def predicted_direction(self) -> PredictedDirection | None:
        return self._predicted_direction

    @property
    def intervention_arms(self) -> tuple[Intervention, ...]:
        return self._intervention_arms

    @property
    def measurables(self) -> tuple[Measurable[R, object], ...]:
        return self._measurables

    def arm_key(self) -> str:
        """Canonical fingerprint of the typed `intervention_arms`.

        Empty `intervention_arms` → `'baseline'`; non-empty →
        `'+'`-joined slot=replacement keys (sorted by slot_path).
        Two hypotheses with same arms but different HP grid
        points share one `arm_key()`; HP variation is a covariate,
        not an arm distinguisher."""
        return combined_arm_key(self._intervention_arms)

    @override
    def __repr__(self) -> str:
        return (
            f'Hypothesis(name={self._name!r}, '
            f'intervention={self._intervention!r}, '
            f'predicted_direction={self._predicted_direction!r}, '
            f'intervention_arms={self._intervention_arms!r}, '
            f'measurables={self._measurables!r})'
        )

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hypothesis):
            return NotImplemented
        return (
            self._name == other._name
            and self._intervention == other._intervention
            and self._predicted_direction == other._predicted_direction
            and self._intervention_arms == other._intervention_arms
            and self._measurables == other._measurables
        )

    @override
    def __hash__(self) -> int:
        return hash((
            self._name,
            tuple(sorted(self._intervention.items())),
            self._predicted_direction,
            self._intervention_arms,
            self._measurables,
        ))
