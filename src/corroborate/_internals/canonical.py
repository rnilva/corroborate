"""Canonical string fingerprint for leaf values.

`canonical_str(v)` produces a deterministic, process-portable
string form of a leaf value (FnClaim, partial, primitive,
dataclass, tuple, function, type) — the substrate for
mechanism-key fingerprints, arm-key derivation, and trace-leaf
encoding.

Lives in its own module so multiple consumers (`hypothesis.py`,
`intervention.py`) can share without an import cycle and without
importing across module-private boundaries."""
from __future__ import annotations

import functools
import types
from dataclasses import fields, is_dataclass
from typing import TYPE_CHECKING

from corroborate._internals.introspection import (
    get_attr_obj,
    get_partial_args,
    get_partial_keywords,
)

if TYPE_CHECKING:
    from corroborate.core.claim import FnClaim as FnClaim  # noqa: PLC0414  re-export-style for typing


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
    identically across processes."""
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
        kw_part = ','.join(
            f'{k}={canonical_str(val)}'
            for k, val in sorted(partial_kw.items())
        ) if partial_kw else ''
        bound = ';'.join(p for p in (args_part, kw_part) if p)
        return f'partial({inner};{bound})'
    if is_dataclass(v) and not isinstance(v, type):
        body = ','.join(
            f'{f.name}={canonical_str(get_attr_obj(v, f.name))}'
            for f in sorted(fields(v), key=lambda f: f.name)
        )
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
