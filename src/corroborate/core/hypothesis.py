"""Hypothesis — the framework's typed verdict-time contract.

A Hypothesis is anything structurally exposing three attributes:

- `INTERVENTION: DoEffect` — the typed contrast (treatment +
  baseline arms as Intervention tuples on the claim graph).
- `BRIDGES: tuple[Bridge, ...]` — the authored verdict
  declarations the framework evaluates against a corpus.
- `MEASURABLES: tuple[Measurable, ...]` — typed Measurable
  instances the substrate's cell_runner pre-registers / persists
  as scalar columns on every RunRow.

Both shapes satisfy the Protocol structurally:

- **Module as hypothesis:** a Python module declaring module-level
  `INTERVENTION`, `BRIDGES`, `MEASURABLES`. Modules are Python
  objects; `getattr(module, 'INTERVENTION')` lands on the
  module-level constant.
- **Class as hypothesis:** a frozen dataclass (or any class) with
  `ClassVar` fields:
  ```python
  @dataclass(frozen=True)
  class DDQNvsVanilla:
      INTERVENTION: ClassVar[DoEffect] = DoEffect(...)
      BRIDGES: ClassVar[tuple[Bridge, ...]] = (...)
      MEASURABLES: ClassVar[tuple[Measurable, ...]] = (...)
  ```
  Multiple hypotheses can live in one file as separate classes.

`name` is NOT part of the Protocol — modules and classes carry
`__name__` from Python for free; the framework reads it
opportunistically (`getattr(h, '__name__', None)`) for cache
paths or display, but the Protocol does not require it. Arm
identity flows exclusively through `canonical_str` of the
underlying Intervention tuples (via
`DoEffect.treatment_arm_key()` / `baseline_arm_key()`); substrate-
chosen short labels are no longer part of the framework's
identity surface.

The legacy `LegacyHypothesis` dataclass coexists during the
Phase 6 migration — substrate-side sweep glue still constructs
it. Subsequent commits migrate substrate authoring to the
Protocol-conforming shape; LegacyHypothesis disappears once no
consumers remain."""
from __future__ import annotations

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    override,
    runtime_checkable,
)

from corroborate._internals.canonical import canonical_str
from corroborate.core.intervention import DoEffect, Intervention, combined_arm_key

if TYPE_CHECKING:
    from corroborate.bridge.bridge import Bridge
    from corroborate.measurables import Measurable


__all__ = [
    'Hypothesis',
    'LegacyHypothesis',
    'PredictedDirection',
    'canonical_str',
]


type PredictedDirection = Literal['a_gt_b', 'a_lt_b', 'two_sided']
"""Author-declared *prior* sign of the predicted treatment-vs-
baseline effect on the primary outcome. `'a_gt_b'` predicts the
intervention's arm exceeds the baseline; `'a_lt_b'` predicts
below; `'two_sided'` predicts non-zero in either direction.

Per-bridge metadata: `Bridge.predicted_direction` carries it for
the analysis the bridge consumes. Distinct from
`graph.causal.Direction` — that's the *observed* sign (DIRECT /
INVERSE) inferred post-hoc from a stat's value.
PredictedDirection is the prior; Direction is the posterior."""


@runtime_checkable
class Hypothesis(Protocol):
    """The framework's typed verdict-time hypothesis contract.

    Conforming objects expose three read-only attributes:

    - `INTERVENTION: DoEffect` — the typed contrast (treatment +
      baseline arms as Intervention tuples).
    - `BRIDGES: tuple[Bridge, ...]` — the authored verdict
      declarations.
    - `MEASURABLES: tuple[Measurable, ...]` — typed Measurable
      instances pre-registered for cell-runner.

    Modules and classes both satisfy the Protocol structurally
    via attribute access. The framework's verdict-time runner
    reads only `BRIDGES`; the substrate's sweep glue reads
    `INTERVENTION` (to drive paired sweep iteration) and
    `MEASURABLES` (to pre-register them on each cell)."""

    INTERVENTION: DoEffect
    BRIDGES: 'tuple[Bridge, ...]'
    MEASURABLES: 'tuple[Measurable[Mapping[str, object], object], ...]'


class LegacyHypothesis[R: Mapping[str, object]]:
    """**DEPRECATED** — sweep-time configuration object kept during
    the Phase 6 Hypothesis-Protocol migration. Substrate authoring
    will migrate to Protocol-conforming class/module declarations;
    this dataclass is removed once no consumers remain.

    Carries (name, intervention dict, intervention_arms tuple,
    measurables, predicted_direction). Today's substrate
    cell_runner reads these fields; the framework's
    `runner/sweep.py` and `runner/config_loader.py` consume them
    too. After the substrate migration, the framework's
    sweep primitive takes a `Hypothesis` Protocol-conforming
    object + the substrate's `BASE` callable directly."""

    __slots__ = (
        '_name', '_intervention', '_predicted_direction',
        '_intervention_arms', '_measurables',
    )

    _name: str
    _intervention: Mapping[str, object]
    _predicted_direction: PredictedDirection | None
    _intervention_arms: tuple[Intervention, ...]
    _measurables: tuple['Measurable[R, object]', ...]

    def __init__(
        self,
        name: str,
        intervention: Mapping[str, object],
        predicted_direction: PredictedDirection | None = None,
        intervention_arms: tuple[Intervention, ...] = (),
        measurables: tuple['Measurable[R, object]', ...] = (),
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
    def measurables(self) -> tuple['Measurable[R, object]', ...]:
        return self._measurables

    def arm_key(self) -> str:
        """Canonical fingerprint of the typed `intervention_arms`."""
        return combined_arm_key(self._intervention_arms)

    @override
    def __repr__(self) -> str:
        return (
            f'LegacyHypothesis(name={self._name!r}, '
            f'intervention={self._intervention!r}, '
            f'predicted_direction={self._predicted_direction!r}, '
            f'intervention_arms={self._intervention_arms!r}, '
            f'measurables={self._measurables!r})'
        )

    @override
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, LegacyHypothesis):
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
