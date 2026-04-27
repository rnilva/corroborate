"""Persistence — parquet round-trip for schema rows.

The parquet representation is **mostly typed**, with
heterogeneous-keyed structures (mechanism_key, facts, meta)
JSON-serialized into string columns:

- Top-level fields (id, env_name, total_steps, seed, ...) → typed
  parquet columns. Query-friendly (filter, group-by).
- Lists of strings (record_keys, reads_set, run_ids, etc.) →
  polars `List[Utf8]` — typed.
- `mechanism_key`, `facts`, `meta` → JSON strings via the
  `*_json` column suffix. Polars Struct types require fixed
  keys; meta and stats genuinely don't have those.

Each row type's parquet schema is determined by which top-level
keys are JSON-wrapped. The wrapping/unwrapping is a single pair
of helpers (`_wrap_for_parquet`, `_unwrap_from_parquet`) that
takes the JSON-key list per row type. Schema rows' `as_dict()` /
`from_dict` are the single source of truth for the in-memory
shape; persistence wraps/unwraps the JSON-bearing keys at the
parquet boundary.

All eight functions (4 row types × {write, read}) are top-level
to keep schema.py polars-free."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import polars as pl

from corroborate._json_boundary import loads as _decode_json
from corroborate._narrow import require_str
from corroborate._polars_boundary import to_dicts as _to_dicts
from corroborate.schema import (
    ArmRow,
    ComparisonRow,
    CorpusRow,
    RunRow,
)


# ============ JSON-keys per row type ============

# Each row's `as_dict()` produces a fully-nested dict. Persistence
# JSON-wraps the heterogeneous-keyed (or list-of-mixed-shape)
# entries before writing to parquet, then unwraps them on read.
# The keys listed here are the ones JSON-wrapped at the parquet
# boundary; everything else stays as a typed parquet column.

_JSON_KEYS_RUNROW: tuple[str, ...] = ('mechanism_key', 'facts', 'meta')
_JSON_KEYS_ARMROW: tuple[str, ...] = ('mechanism_key', 'facts', 'meta')
_JSON_KEYS_COMPARISONROW: tuple[str, ...] = ('mechanism_key', 'facts', 'meta')
_JSON_KEYS_CORPUSROW: tuple[str, ...] = ('facts', 'meta')


# ============ Wrap / unwrap helpers ============

def _wrap_for_parquet(
    d: Mapping[str, object],
    json_keys: tuple[str, ...],
) -> dict[str, object]:
    """JSON-wrap the listed keys' values. The keys are renamed to
    `{key}_json` and their values are `json.dumps(...)` strings.
    Other keys pass through unchanged."""
    out: dict[str, object] = {**d}
    for k in json_keys:
        if k in out:
            out[f'{k}_json'] = json.dumps(out[k])
            del out[k]
    return out


def _unwrap_from_parquet(
    d: Mapping[str, object],
    json_keys: tuple[str, ...],
) -> dict[str, object]:
    """Reverse of `_wrap_for_parquet`. Reads `{key}_json` columns,
    decodes via `_decode_json` (the framework's Any-laundering
    boundary), and replaces the `_json` form with the unwrapped
    nested form. Schema rows' `from_dict` consumes the result."""
    out: dict[str, object] = {**d}
    for k in json_keys:
        json_k = f'{k}_json'
        if json_k in out:
            out[k] = _decode_json(require_str(out, json_k))
            del out[json_k]
    return out


# ============ RunRow ============

def write_runrows(rows: Iterable[RunRow], path: Path) -> None:
    """Write RunRows to parquet. Round-trip pair: `read_runrows`."""
    records = [_wrap_for_parquet(r.as_dict(), _JSON_KEYS_RUNROW) for r in rows]
    pl.DataFrame(records).write_parquet(path)


def read_runrows(path: Path) -> list[RunRow]:
    df = pl.read_parquet(path)
    return [
        RunRow.from_dict(_unwrap_from_parquet(d, _JSON_KEYS_RUNROW))
        for d in _to_dicts(df)
    ]


# ============ ArmRow ============

def write_armrows(rows: Iterable[ArmRow], path: Path) -> None:
    records = [_wrap_for_parquet(r.as_dict(), _JSON_KEYS_ARMROW) for r in rows]
    pl.DataFrame(records).write_parquet(path)


def read_armrows(path: Path) -> list[ArmRow]:
    df = pl.read_parquet(path)
    return [
        ArmRow.from_dict(_unwrap_from_parquet(d, _JSON_KEYS_ARMROW))
        for d in _to_dicts(df)
    ]


# ============ ComparisonRow ============

def write_comparisonrows(rows: Iterable[ComparisonRow], path: Path) -> None:
    records = [
        _wrap_for_parquet(r.as_dict(), _JSON_KEYS_COMPARISONROW)
        for r in rows
    ]
    pl.DataFrame(records).write_parquet(path)


def read_comparisonrows(path: Path) -> list[ComparisonRow]:
    df = pl.read_parquet(path)
    return [
        ComparisonRow.from_dict(_unwrap_from_parquet(d, _JSON_KEYS_COMPARISONROW))
        for d in _to_dicts(df)
    ]


# ============ CorpusRow ============

def write_corpusrows(rows: Iterable[CorpusRow], path: Path) -> None:
    records = [_wrap_for_parquet(r.as_dict(), _JSON_KEYS_CORPUSROW) for r in rows]
    pl.DataFrame(records).write_parquet(path)


def read_corpusrows(path: Path) -> list[CorpusRow]:
    df = pl.read_parquet(path)
    return [
        CorpusRow.from_dict(_unwrap_from_parquet(d, _JSON_KEYS_CORPUSROW))
        for d in _to_dicts(df)
    ]
