"""Internal narrowing helpers — TypeIs predicates and required/
optional field accessors used by `schema.py` to narrow `object`
values from `Mapping[str, object]` payloads without `cast` or
`# type: ignore`.

Module is underscore-prefixed to signal **internal use only**
within the framework. External users should not import from here.

Public-API status: stable across the framework's lifetime; if
narrowing patterns evolve, breaking changes here ripple to all
schema-shaped consumers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TypeIs

from corroborate.core.hypothesis import PredictedDirection
from corroborate.bridge.verdict import RefutationClass, Verdict


# ============ TypeIs predicates ============

def is_list_of_object(v: object) -> TypeIs[list[object]]:
    """TypeIs predicate narrowing `object` to `list[object]`. The
    bare `isinstance(v, list)` narrow gives `list[Unknown]` under
    strict pyright; this predicate explicitly types each element
    as `object` so subsequent isinstance-narrows on items work."""
    return isinstance(v, list)


def is_mapping_str_object(v: object) -> TypeIs[Mapping[str, object]]:
    """TypeIs predicate narrowing `object` to `Mapping[str, object]`.
    The key-type constraint is a runtime invariant guaranteed by
    the source (JSON / kwargs payloads always carry string keys);
    callers narrowing values from those sources can rely on it.
    Subsequent isinstance-narrows on values work because each
    value is typed `object`."""
    return isinstance(v, Mapping)


def is_tuple_of_int(v: object) -> TypeIs[tuple[int, ...]]:
    """TypeIs predicate narrowing `object` to `tuple[int, ...]`.
    Excludes `bool` per `int`-vs-`bool` subclass relationship —
    `True/False` would otherwise pass an `isinstance(_, int)` check
    and corrupt downstream consumers expecting a true int."""
    return (
        isinstance(v, tuple)
        and all(isinstance(s, int) and not isinstance(s, bool) for s in v)
    )


# ============ Required / optional field accessors ============

def require_str(d: Mapping[str, object], key: str) -> str:
    v = d.get(key)
    if not isinstance(v, str):
        raise TypeError(f'{key!r} must be str, got {type(v).__name__}')
    return v


def optional_str(d: Mapping[str, object], key: str) -> str | None:
    v = d.get(key)
    if v is None:
        return None
    if not isinstance(v, str):
        raise TypeError(f'{key!r} must be str | None, got {type(v).__name__}')
    return v


def require_bool(d: Mapping[str, object], key: str) -> bool:
    v = d.get(key)
    if not isinstance(v, bool):
        raise TypeError(f'{key!r} must be bool, got {type(v).__name__}')
    return v


def require_int(d: Mapping[str, object], key: str) -> int:
    v = d.get(key)
    # Reject bool explicitly — `bool` is a subclass of `int` in
    # Python, so an isinstance(int) narrow would silently accept
    # True/False as 1/0 and corrupt the manifest's size_bytes
    # field.
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f'{key!r} must be int, got {type(v).__name__}')
    return v


def optional_direction(
    d: Mapping[str, object], key: str,
) -> PredictedDirection | None:
    v = d.get(key)
    if v is None:
        return None
    if v == 'a_gt_b':
        return 'a_gt_b'
    if v == 'a_lt_b':
        return 'a_lt_b'
    if v == 'two_sided':
        return 'two_sided'
    raise TypeError(f'{key!r} must be a PredictedDirection or None, got {v!r}')


def require_verdict(d: Mapping[str, object], key: str) -> Verdict:
    """Narrow a verdict-string column (parquet representation)
    back to the typed `Verdict` enum. Round-trip pair with
    `Verdict.value`. Round-trips break loudly: any unknown
    string raises `TypeError` at parse time."""
    v = d.get(key)
    if v == Verdict.HELD.value:
        return Verdict.HELD
    if v == Verdict.HELD_WITH_SCOPE_FLAG.value:
        return Verdict.HELD_WITH_SCOPE_FLAG
    if v == Verdict.NO_EFFECT.value:
        return Verdict.NO_EFFECT
    if v == Verdict.POWER_INSUFFICIENT.value:
        return Verdict.POWER_INSUFFICIENT
    if v == Verdict.INVARIANT_VIOLATION.value:
        return Verdict.INVARIANT_VIOLATION
    raise TypeError(f'{key!r} must be a Verdict value, got {v!r}')


def optional_refutation_class(
    d: Mapping[str, object], key: str,
) -> RefutationClass | None:
    """Narrow an optional refutation-class string (the parquet
    representation) back to the typed `RefutationClass` enum.
    Round-trip pair with `RefutationClass.value`."""
    v = d.get(key)
    if v is None:
        return None
    if v == RefutationClass.NULL_EFFECT.value:
        return RefutationClass.NULL_EFFECT
    if v == RefutationClass.SIGN_FLIP.value:
        return RefutationClass.SIGN_FLIP
    if v == RefutationClass.UNDERPOWERED.value:
        return RefutationClass.UNDERPOWERED
    if v == RefutationClass.TIME_BUDGET_DORMANT.value:
        return RefutationClass.TIME_BUDGET_DORMANT
    raise TypeError(
        f'{key!r} must be a RefutationClass value or None, got {v!r}'
    )


def require_str_list(d: Mapping[str, object], key: str) -> list[str]:
    v = d.get(key)
    if not is_list_of_object(v):
        raise TypeError(f'{key!r} must be list[str], got {type(v).__name__}')
    out: list[str] = []
    for item in v:
        if not isinstance(item, str):
            raise TypeError(f'{key!r} contains non-str: {type(item).__name__}')
        out.append(item)
    return out


def optional_str_list(
    d: Mapping[str, object], key: str,
) -> list[str] | None:
    """Narrow an optional list-of-strings field. None passes
    through; a present value is delegated to `require_str_list`."""
    v = d.get(key)
    if v is None:
        return None
    return require_str_list(d, key)
