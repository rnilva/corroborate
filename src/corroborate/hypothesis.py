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

Structural identity of a hypothesis is recoverable from the
measurements its runs produce — leaf values land at dotted
topology paths via `signature.walk_paths`, and
`aggregate.leaf_signature` projects a `RunRow.measurements` to
the configurational subset suitable as a group-by key. No
separate `MechanismKey` artifact; the framework persists the
measurements and re-derives identity on demand."""
from __future__ import annotations

import functools
import types
from collections.abc import Mapping
from dataclasses import dataclass, fields, is_dataclass
from typing import Literal

from corroborate.bridge import Bridge
from corroborate.claim import FnClaim


type Direction = Literal['a_gt_b', 'a_lt_b', 'two_sided']
"""Author-declared sign of the predicted treatment-vs-baseline
effect on the primary outcome. `'a_gt_b'` predicts the
intervention's arm exceeds the baseline; `'a_lt_b'` predicts
below; `'two_sided'` predicts non-zero in either direction.
None on Hypothesis means direction is unstated (downstream
infers from a `held` flag if available)."""


@dataclass(frozen=True, slots=True)
class Hypothesis[R: Mapping[str, object]]:
    """A research hypothesis: an intervention plus the bridges
    that should hold under it.

    Generic in `R: Mapping[str, object]` — the (single) record
    schema all bridges are typed against. Authors using TypedDict
    for their record get typed bridge bodies (no narrowing);
    authors using plain `Mapping[str, object]` continue to narrow
    at use site.

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
    predicted_direction: Direction | None = None


def _canonical_str(v: object) -> str:
    """Stable string form of a leaf value, used by leaf-path
    flattening to produce a deterministic scalar fingerprint of a
    structured kwarg (a Module, a partial, an FnClaim).

    Handles each concrete callable kind by isinstance against the
    runtime type — `types.FunctionType`, `type`, and
    `types.BuiltinFunctionType` all carry typed `__name__: str`,
    so attribute access after narrowing is fully typed (no
    `getattr` needed).

    `functools.partial` is canonicalised by recursing into
    `.func` and lexicographically encoding `.keywords` (positional
    `.args` are flattened similarly). This makes baked-in slot
    parameters (`partial(linear_epsilon, anneal_steps=50_000)`)
    contribute to the fingerprint transparently — two
    independently-constructed partials with the same wrapped
    callable + same kwargs canonicalise identically across
    processes. The "bake-in" pattern is then honest: the canonical
    name records WHAT was baked in.

    Anything else falls through to `repr()`."""
    if isinstance(v, FnClaim):
        # Function-claim wrapper: short canonical form `Claim:name`.
        # Module-claims (instances of @claim-decorated classes with
        # data fields) fall through to the dataclass branch below
        # for sorted-field expansion.
        return f'Claim:{v.name}'
    if isinstance(v, bool):
        # Order matters: bool is subclass of int, so this branch
        # must precede the (int, float, str) check below.
        return repr(v)
    if isinstance(v, (int, float, str)):
        return repr(v)
    if isinstance(v, functools.partial):
        inner = _canonical_str(v.func)
        args_part = (
            ','.join(_canonical_str(a) for a in v.args) if v.args else ''
        )
        kw_part = ','.join(
            f'{k}={_canonical_str(val)}'
            for k, val in sorted(v.keywords.items())
        ) if v.keywords else ''
        bound = ';'.join(p for p in (args_part, kw_part) if p)
        return f'partial({inner};{bound})'
    if is_dataclass(v) and not isinstance(v, type):
        # Pytree-shaped leaf values (e.g. Module instances):
        # canonicalise by sorted-field expansion so two
        # configurations differing in a single leaf get distinct,
        # structured fingerprint entries. The form is process-
        # portable because field declaration order and recursive
        # `_canonical_str` are deterministic.
        body = ','.join(
            f'{f.name}={_canonical_str(getattr(v, f.name))}'
            for f in sorted(fields(v), key=lambda f: f.name)
        )
        return f'dataclass:{type(v).__name__}({body})'
    if isinstance(v, tuple):
        # Stable canonical form for tuples (e.g. `hidden=(64, 64)`).
        # Recurse so tuples-of-scalars / tuples-of-tuples stay
        # process-portable rather than relying on `repr()` which
        # is stable in CPython but less semantically explicit here.
        return '(' + ','.join(_canonical_str(item) for item in v) + ')'
    if isinstance(v, types.FunctionType):
        return f'callable:{v.__name__}'
    if isinstance(v, type):
        return f'type:{v.__name__}'
    if isinstance(v, types.BuiltinFunctionType):
        return f'builtin:{v.__name__}'
    return repr(v)
