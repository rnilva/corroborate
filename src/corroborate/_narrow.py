"""Internal narrowing helpers — TypeIs predicates and required/
optional field accessors used by `schema.py` and `persistence.py`
to narrow `object` values from `Mapping[str, object]` payloads
without `cast` or `# type: ignore`.

Module is underscore-prefixed to signal **internal use only**
within the framework. External users should not import from here.

Public-API status: stable across the framework's lifetime; if
narrowing patterns evolve, breaking changes here ripple to all
schema-shaped consumers."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Literal, TypeIs

from corroborate.hypothesis import Direction


# ============ TypeIs predicates ============

def is_list_of_object(v: object) -> TypeIs[list[object]]:
    """TypeIs predicate narrowing `object` to `list[object]`. The
    bare `isinstance(v, list)` narrow gives `list[Unknown]` under
    strict pyright; this predicate explicitly types each element
    as `object` so subsequent isinstance-narrows on items work."""
    return isinstance(v, list)


def is_mapping_of_object(v: object) -> TypeIs[Mapping[str, object]]:
    """Narrows to `Mapping[str, object]` for parquet-shaped dicts.
    Caller validates keys are strings via the per-field accessors."""
    return isinstance(v, Mapping)


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


def require_int(d: Mapping[str, object], key: str) -> int:
    v = d.get(key)
    # bool is subclass of int — exclude it explicitly
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(f'{key!r} must be int, got {type(v).__name__}')
    return v


def require_float(d: Mapping[str, object], key: str) -> float:
    v = d.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f'{key!r} must be float, got {type(v).__name__}')
    return float(v)


def optional_float(d: Mapping[str, object], key: str) -> float | None:
    v = d.get(key)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(f'{key!r} must be float | None, got {type(v).__name__}')
    return float(v)


def require_bool(d: Mapping[str, object], key: str) -> bool:
    v = d.get(key)
    if not isinstance(v, bool):
        raise TypeError(f'{key!r} must be bool, got {type(v).__name__}')
    return v


def require_kind(
    d: Mapping[str, object], key: str,
) -> Literal['bridge', 'invariant']:
    v = d.get(key)
    if v == 'bridge':
        return 'bridge'
    if v == 'invariant':
        return 'invariant'
    raise TypeError(f"{key!r} must be 'bridge' or 'invariant', got {v!r}")


def optional_direction(
    d: Mapping[str, object], key: str,
) -> Direction | None:
    v = d.get(key)
    if v is None:
        return None
    if v == 'a_gt_b':
        return 'a_gt_b'
    if v == 'a_lt_b':
        return 'a_lt_b'
    if v == 'two_sided':
        return 'two_sided'
    raise TypeError(f'{key!r} must be a Direction or None, got {v!r}')


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


def require_int_list(d: Mapping[str, object], key: str) -> list[int]:
    v = d.get(key)
    if not is_list_of_object(v):
        raise TypeError(f'{key!r} must be list[int], got {type(v).__name__}')
    out: list[int] = []
    for item in v:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f'{key!r} contains non-int: {type(item).__name__}')
        out.append(item)
    return out


def require_mapping(d: Mapping[str, object], key: str) -> Mapping[str, object]:
    v = d.get(key)
    if not is_mapping_of_object(v):
        raise TypeError(f'{key!r} must be a Mapping, got {type(v).__name__}')
    return v


def list_len(d: Mapping[str, object], key: str) -> int:
    v = d.get(key)
    if not is_list_of_object(v):
        raise TypeError(f'{key!r} must be a list, got {type(v).__name__}')
    return len(v)


def require_mapping_in_list(
    d: Mapping[str, object], key: str, idx: int,
) -> Mapping[str, object]:
    v = d.get(key)
    if not is_list_of_object(v):
        raise TypeError(f'{key!r} must be a list, got {type(v).__name__}')
    item = v[idx]
    if not is_mapping_of_object(item):
        raise TypeError(
            f'{key!r}[{idx}] must be a Mapping, got {type(item).__name__}'
        )
    return item


def require_stats_mapping(
    d: Mapping[str, object], key: str,
) -> Mapping[str, float | int | bool | str]:
    raw = require_mapping(d, key)
    out: dict[str, float | int | bool | str] = {}
    for k, v in raw.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float, str)):
            out[k] = v
        else:
            raise TypeError(
                f'{key!r}[{k!r}] must be float|int|bool|str, '
                f'got {type(v).__name__}'
            )
    return out


def require_meta_mapping(
    d: Mapping[str, object], key: str,
) -> Mapping[str, str | int | float | bool]:
    raw = require_mapping(d, key)
    out: dict[str, str | int | float | bool] = {}
    for k, v in raw.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float, str)):
            out[k] = v
        else:
            raise TypeError(
                f'{key!r}[{k!r}] must be str|int|float|bool, '
                f'got {type(v).__name__}'
            )
    return out
