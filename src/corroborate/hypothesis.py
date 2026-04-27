"""Hypothesis — (intervention, [bridges]) over a record schema R.

A Hypothesis names a CHANGE to the theory (intervention) and a
SET of paper-level assertions (bridges) that should hold under
that change. Generic in `R: Mapping[str, object]` — the same R
the bridges are typed against, so all bridges in a hypothesis
share the record schema (one theory → one record schema).

Three components:

- `intervention: Mapping[str, object]` — kwargs passed to
  `functools.partial(theory, **intervention)` when the
  hypothesis is run. Values are heterogeneous (Claims, plain
  callables, primitives) — `object` is GENUINE polymorphism;
  the framework is theory-neutral.
- `bridges: tuple[Bridge[R], ...]` — the paper-level assertions
  applied to the resulting record. All share R.
- `predicted_direction: Direction | None` — author-declared sign
  of the predicted treatment-vs-baseline effect, used downstream
  by direction-aware verdict computation. None leaves direction
  inference to the consumer (e.g. derive from a `held` flag).

`mechanism_key` is the canonical structural identity — two
Hypotheses with the same `MechanismKey` make the same structural
claim about the same intervention even under cosmetic renames.
This is the anti-laundering primitive (axiom 18, axiom 19's
redundancy uses it as the intervention-similarity factor).

For v0 the framework provides only the data type. A custom
`bind()` for recursive intervention (walking nested override
dicts with TypeIs narrowing) lands when use cases require it;
v0's flat interventions are handled by `functools.partial`
directly (framework-subtraction discipline)."""
from __future__ import annotations

import types
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from corroborate.bridge import Bridge
from corroborate.claim import Claim


type Direction = Literal['a_gt_b', 'a_lt_b', 'two_sided']
"""Author-declared sign of the predicted treatment-vs-baseline
effect on the primary outcome. `'a_gt_b'` predicts the
intervention's arm exceeds the baseline; `'a_lt_b'` predicts
below; `'two_sided'` predicts non-zero in either direction.
None on Hypothesis means direction is unstated (downstream
infers from a `held` flag if available)."""


@dataclass(frozen=True, slots=True)
class InterventionKey:
    """Intervention-only projection of a `MechanismKey`. Carries
    the intervention identity WITHOUT bridge-name dependence.

    Used by causal discovery (PAPER_NOTES.md §4.3 — `arm_ddqn` as
    a binary intervention variable in a PC graph): two arms with
    identical interventions but different bridge sets should map
    to the SAME node, which `MechanismKey` (which includes
    `bridge_names`) does NOT do. `InterventionKey` is the cleaner
    primitive when only the intervention matters."""
    intervention_signature: tuple[tuple[str, str], ...]
    direction: Direction | None


@dataclass(frozen=True, slots=True)
class MechanismKey:
    """Canonical structural identity of a Hypothesis.

    Two Hypotheses with the same `MechanismKey` make the same
    structural claim about the same intervention — even under
    cosmetic name changes. The framework uses MechanismKey as
    the anti-laundering registry key and as the intervention-
    similarity factor in axiom 19's redundancy primitive.

    Three components:
    - `intervention_signature` — sorted (slot, value-canonical-
      string) pairs. Stable across instance creations of the
      same intervention.
    - `bridge_names` — frozenset of bridge names. Order-
      independent; renamed bridges produce different keys.
    - `direction` — included only when the Hypothesis declared
      one. Two Hypotheses identical except in direction get
      distinct keys (direction is a structural distinction)."""
    intervention_signature: tuple[tuple[str, str], ...]
    bridge_names: frozenset[str]
    direction: Direction | None

    def intervention_only(self) -> InterventionKey:
        """Project to `InterventionKey` — drops `bridge_names`,
        retains `intervention_signature` and `direction`. The
        intervention-only projection causal discovery wants for
        binary intervention variables."""
        return InterventionKey(
            intervention_signature=self.intervention_signature,
            direction=self.direction,
        )


@dataclass(frozen=True, slots=True)
class Hypothesis[R: Mapping[str, object]]:
    """A research hypothesis: an intervention plus the bridges
    that should hold under it.

    Generic in `R: Mapping[str, object]` — the record schema all
    bridges are typed against. Authors using TypedDict for their
    record get typed bridge bodies (no narrowing); authors using
    plain `Mapping[str, object]` continue to narrow at use site.

    `name` is a human-readable label; `mechanism_key` is the
    structural identity (don't conflate them — two hypotheses
    can share `name` and have distinct mechanism_keys, or vice
    versa; `mechanism_key` is the anti-laundering key)."""
    name: str
    intervention: Mapping[str, object]
    bridges: tuple[Bridge[R], ...] = ()
    predicted_direction: Direction | None = None

    @property
    def mechanism_key(self) -> MechanismKey:
        """Canonical structural identity. Cached implicitly via
        the frozen-dataclass property semantics — pyright infers
        the property as memoizable, but Python re-computes on
        each access; v0 doesn't cache (cheap to compute)."""
        intervention_pairs: tuple[tuple[str, str], ...] = tuple(
            sorted(
                (k, _canonical_str(v))
                for k, v in self.intervention.items()
            )
        )
        return MechanismKey(
            intervention_signature=intervention_pairs,
            bridge_names=frozenset(b.name for b in self.bridges),
            direction=self.predicted_direction,
        )


def _canonical_str(v: object) -> str:
    """Stable string form of an intervention value, used by
    `mechanism_key` to produce a hashable canonical signature.

    Handles each concrete callable kind by isinstance against the
    runtime type — `types.FunctionType`, `type`, and
    `types.BuiltinFunctionType` all carry typed `__name__: str`,
    so attribute access after narrowing is fully typed (no
    `getattr` needed). Anything else falls through to `repr()`."""
    if isinstance(v, Claim):
        return f'Claim:{v.name}'
    if isinstance(v, bool):
        # Order matters: bool is subclass of int, so this branch
        # must precede the (int, float, str) check below.
        return repr(v)
    if isinstance(v, (int, float, str)):
        return repr(v)
    if isinstance(v, types.FunctionType):
        return f'callable:{v.__name__}'
    if isinstance(v, type):
        return f'type:{v.__name__}'
    if isinstance(v, types.BuiltinFunctionType):
        return f'builtin:{v.__name__}'
    return repr(v)
