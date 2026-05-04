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

For runtime-constructed Hypotheses (e.g. YAML-driven), use
`types.SimpleNamespace` with the required attributes — it
satisfies the Protocol via duck-typing.

`name` is NOT part of the Protocol — modules and classes carry
`__name__` from Python for free; the framework reads it
opportunistically (`getattr(h, '__name__', None)`) for cache
paths or display, but the Protocol does not require it. Arm
identity flows exclusively through `canonical_str` of the
underlying Intervention tuples (via
`DoEffect.treatment_arm_key()` / `baseline_arm_key()`); substrate-
chosen short labels are no longer part of the framework's
identity surface."""
from __future__ import annotations

from collections.abc import Mapping
from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    runtime_checkable,
)

from corroborate._internals.canonical import canonical_str
from corroborate.core.intervention import DoEffect

if TYPE_CHECKING:
    from corroborate.bridge.bridge import Bridge
    from corroborate.measurables import Measurable


__all__ = ['Hypothesis', 'PredictedDirection', 'canonical_str']


type PredictedDirection = Literal['a_gt_b', 'a_lt_b', 'two_sided']
"""Author-declared *prior* sign of the predicted treatment-vs-
baseline effect. `'a_gt_b'` predicts the intervention's arm
exceeds the baseline; `'a_lt_b'` predicts below; `'two_sided'`
predicts non-zero in either direction.

Per-bridge metadata: `Bridge.predicted_direction` carries it for
the analysis the bridge consumes. Distinct from
`graph.causal.Direction` — that's the *observed* sign (DIRECT /
INVERSE) inferred post-hoc from a stat's value."""


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
