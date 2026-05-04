"""Hypothesis — the framework's typed verdict-time contract.

A Hypothesis is anything structurally exposing two attributes:

- `INTERVENTION: DoEffect` — the typed contrast (treatment +
  baseline arms as Intervention tuples on the claim graph).
- `BRIDGES: tuple[Bridge, ...]` — the authored verdict
  declarations the framework evaluates against a corpus.

Both shapes satisfy the Protocol structurally:

- **Module as hypothesis:** a Python module declaring module-level
  `INTERVENTION` and `BRIDGES`. Modules are Python objects;
  `getattr(module, 'INTERVENTION')` lands on the module-level
  constant.
- **Class as hypothesis:** a frozen dataclass (or any class) with
  `ClassVar` fields:
  ```python
  @dataclass(frozen=True)
  class DDQNvsVanilla:
      INTERVENTION: ClassVar[DoEffect] = DoEffect(...)
      BRIDGES: ClassVar[tuple[Bridge, ...]] = (...)
  ```
  Multiple hypotheses can live in one file as separate classes.

For runtime-constructed Hypotheses (e.g. YAML-driven), use
`types.SimpleNamespace` with the required attributes — it
satisfies the Protocol via duck-typing.

`MEASURABLES` is NOT on the Protocol. Pre-registered measurables
are a sweep-time concern: `runner.sweep.run_intervention` takes
them as an explicit parameter, and substrates that compute
mediators post-sweep from raw traces leave the parameter empty.
Bridges that consume measurables import them by name (typed
`Measurable` instance) at module load — the registry resolves
chained dependencies; the Protocol doesn't need to repeat what
bridges already carry.

`__name__: str` is on the Protocol so the runner's typed access
(cache-path defaults / display) doesn't have to fall back to
`getattr` — both Python modules and classes carry `__name__: str`
for free, so requiring it costs no Hypothesis author anything.
Arm *identity*, distinct from `__name__`, flows exclusively
through `canonical_str` of the underlying Intervention tuples
(via `DoEffect.treatment_arm_key()` / `baseline_arm_key()`);
substrate-chosen short labels are no longer part of the
framework's identity surface."""
from __future__ import annotations

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
    - `__name__: str` — the Python identity attribute. Modules
      carry their dotted path; classes carry their bare name. The
      runner uses it for cache-path defaults and display; both
      module and class shapes carry it for free.

    Modules and classes both satisfy the Protocol structurally
    via attribute access. The framework's verdict-time runner
    reads `BRIDGES`; the substrate's sweep glue reads
    `INTERVENTION` to drive paired sweep iteration."""

    __name__: str
    INTERVENTION: DoEffect
    BRIDGES: 'tuple[Bridge, ...]'
