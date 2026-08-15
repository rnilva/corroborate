"""Canonical string fingerprint for leaf values.

`canonical_str(v)` produces a deterministic, process-portable
string form of a leaf value (FnClaim, partial, primitive,
dataclass, tuple, function, type) — the implementation for
mechanism-key fingerprints, arm-key derivation, and trace-leaf
encoding.

**Default-elision discipline.** Dataclass fields and partial
kwargs whose value equals the declared default are OMITTED from
the canonical string. Two consequences:

1. Adding a parameter to a `@claim` factory (e.g. `adam` grew
   `weight_decay: float = 0.0`) does NOT invalidate existing
   arm_keys — old `partial(Claim:adam;lr=0.0001)` and new
   `partial(Claim:adam;lr=0.0001, weight_decay=0.0)` both
   canonicalise to `partial(Claim:adam;lr=0.0001)`. The schema
   evolves; the canonical form stays stable.

2. Cells from sub-sweeps that did not override a parameter
   match cells from sub-sweeps that explicitly set the parameter
   to its default. Surfaced regime-mismatch reports won't fire
   on default-vs-explicit-default differences (see
   `_dedup_diagnostics._distinguishing_columns`'s None-as-wildcard
   complement).

Lives in its own module so multiple consumers (`hypothesis.py`,
`intervention.py`) can share without an import cycle and without
importing across module-private boundaries."""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
import functools
import inspect
import types
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING

from corroborate._internals.introspection import (
    get_attr_obj,
    get_field_default,
    get_field_default_factory,
    get_param_default,
    get_partial_args,
    get_partial_keywords,
)

if TYPE_CHECKING:
    from corroborate.core.claim import FnClaim as FnClaim  # noqa: PLC0414  re-export-style for typing


def _is_at_declared_default(
    value: object, declared_default: object,
    factory_default: 'Callable[[], object] | None',
) -> bool:
    """True iff `value` equals the field's / parameter's declared
    default. Handles three cases:

    1. `declared_default is not MISSING/empty` and `value ==
       declared_default` (the common case for primitive defaults).
    2. `factory_default is not None` (default_factory case for
       dataclasses) and `value == factory_default()`.
    3. Otherwise no declared default; return False.

    Equality uses `==`, not `is`. Comparison failures (e.g.,
    numpy arrays raising on `==`) propagate as False — we err on
    the side of NOT eliding, which preserves the cell as
    distinguishing.
    """
    if declared_default is not dataclasses.MISSING \
            and declared_default is not inspect.Parameter.empty:
        try:
            return bool(value == declared_default)
        except (TypeError, ValueError):
            return False
    if factory_default is not None:
        try:
            return bool(value == factory_default())
        except (TypeError, ValueError):
            return False
    return False


def canonical_str(v: object) -> str:
    """Stable string form of a leaf value.

    Each concrete callable kind is handled by isinstance against
    the runtime type — `types.FunctionType`, `type`, and
    `types.BuiltinFunctionType` all carry typed `__name__: str` so
    attribute access after narrowing is fully typed.

    `functools.partial` is canonicalised by recursing into `.func`
    and lexicographically encoding `.keywords` (positional `.args`
    flatten similarly), so two independently-constructed partials
    with the same wrapped callable + same kwargs canonicalise
    identically across processes. Kwargs whose value equals the
    wrapped function's signature default are elided — see module
    docstring for the forward-compatibility rationale."""
    # Lazy import: `FnClaim` lives in `core.claim`; eagerly importing
    # would cycle (canonical → core.claim → core.__init__ →
    # core.hypothesis → canonical).
    from corroborate.core.claim import FnClaim
    if isinstance(v, FnClaim):
        return f'Claim:{v.name}'
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, (int, float, str)):
        return repr(v)
    if isinstance(v, functools.partial):
        inner = canonical_str(v.func)
        partial_args = get_partial_args(v)
        partial_kw = get_partial_keywords(v)
        args_part = (
            ','.join(canonical_str(a) for a in partial_args)
            if partial_args else ''
        )
        # Elide kwargs equal to the wrapped function's signature
        # default. `FnClaim` exposes `.fn` as the underlying
        # function; plain callables we inspect directly.
        sig_defaults: dict[str, object] = {}
        underlying = v.func.fn if isinstance(v.func, FnClaim) else v.func
        try:
            sig = inspect.signature(underlying)
        except (TypeError, ValueError):
            sig = None
        if sig is not None:
            for pname, param in sig.parameters.items():
                d = get_param_default(param)
                if d is not inspect.Parameter.empty:
                    sig_defaults[pname] = d
        kw_filtered = {
            k: val for k, val in partial_kw.items()
            if not _is_at_declared_default(
                val, sig_defaults.get(k, inspect.Parameter.empty), None,
            )
        }
        kw_part = ','.join(
            f'{k}={canonical_str(val)}'
            for k, val in sorted(kw_filtered.items())
        ) if kw_filtered else ''
        bound = ';'.join(p for p in (args_part, kw_part) if p)
        return f'partial({inner};{bound})'
    if is_dataclass(v) and not isinstance(v, type):
        # Elide fields equal to their declared default (or
        # default_factory()). Forward-compatible across schema
        # additions: a new field at default doesn't change the
        # canonical string.
        body_parts: list[str] = []
        for f in sorted(fields(v), key=lambda f: f.name):
            value = get_attr_obj(v, f.name)
            declared = get_field_default(f)
            factory = get_field_default_factory(f)
            if _is_at_declared_default(value, declared, factory):
                continue
            body_parts.append(f'{f.name}={canonical_str(value)}')
        body = ','.join(body_parts)
        return f'dataclass:{type(v).__name__}({body})'
    if isinstance(v, tuple):
        return '(' + ','.join(canonical_str(item) for item in v) + ')'
    if isinstance(v, types.FunctionType):
        return f'callable:{v.__name__}'
    if isinstance(v, type):
        return f'type:{v.__name__}'
    if isinstance(v, types.BuiltinFunctionType):
        return f'builtin:{v.__name__}'
    return repr(v)
