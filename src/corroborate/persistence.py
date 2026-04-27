"""Persistence — parquet round-trip for schema rows.

Each row type has paired `write_*_parquet` / `read_*_parquet`
functions. The parquet representation is **mostly typed**, with
heterogeneous-keyed dicts (meta, FactRow.stats) JSON-serialized
into string columns. Rationale:

- Top-level fields (id, env_name, total_steps, ...) become typed
  parquet columns — query-friendly (filter, group-by).
- Lists of structs with consistent shape (record_keys, reads_set,
  bridge_names) become polars `List[Utf8]` — typed.
- Heterogeneous-keyed dicts (meta with arbitrary keys per row,
  FactRow.stats with arbitrary stat names per fact) become
  JSON strings — preserves shape losslessly without forcing
  polars' Struct-with-fixed-keys assumption.
- Lists of structs with VARYING per-element keys (facts, where
  each fact's stats vary) become JSON strings of the whole list
  — same rationale.

This is a v0 design choice: simple, lossless, parquet-native for
top-level queries. A future tightening could decompose facts into
a sibling parquet (relational denormalization) for fact-level
filter/group, at the cost of join complexity at read time.

All eight functions (4 row types × {write, read}) are top-level
to keep schema.py polars-free."""
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path

import polars as pl

from corroborate._json_boundary import loads as _decode_json
from corroborate._polars_boundary import to_dicts as _to_dicts

from corroborate._narrow import (
    is_list_of_object,
    is_mapping_of_object,
    optional_direction,
    optional_float,
    optional_str,
    require_bool,
    require_float,
    require_int,
    require_int_list,
    require_kind,
    require_str,
    require_str_list,
)
from corroborate.hypothesis import MechanismKey
from corroborate.schema import (
    ArmRow,
    ComparisonRow,
    CorpusRow,
    FactRow,
    RunRow,
)


# ============ Helpers: JSON encoding for heterogeneous dicts ============

def _meta_to_json(meta: Mapping[str, str | int | float | bool]) -> str:
    """Serialize a meta dict to JSON. Keys are strings; values
    are scalar primitives — both natively JSON-encodable."""
    return json.dumps(dict(meta), sort_keys=True)


def _stats_to_json(stats: Mapping[str, float | int | bool | str]) -> str:
    return json.dumps(dict(stats), sort_keys=True)


def _meta_from_json(s: str) -> dict[str, str | int | float | bool]:
    raw = _decode_json(s)
    if not is_mapping_of_object(raw):
        raise TypeError(f'meta JSON must decode to a mapping, got {type(raw).__name__}')
    out: dict[str, str | int | float | bool] = {}
    for k, v in raw.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float, str)):
            out[k] = v
        else:
            raise TypeError(
                f'meta[{k!r}] must be str|int|float|bool, got {type(v).__name__}'
            )
    return out


def _stats_from_json(s: str) -> dict[str, float | int | bool | str]:
    raw = _decode_json(s)
    if not is_mapping_of_object(raw):
        raise TypeError(f'stats JSON must decode to a mapping, got {type(raw).__name__}')
    out: dict[str, float | int | bool | str] = {}
    for k, v in raw.items():
        if isinstance(v, bool):
            out[k] = v
        elif isinstance(v, (int, float, str)):
            out[k] = v
        else:
            raise TypeError(
                f'stats[{k!r}] must be float|int|bool|str, got {type(v).__name__}'
            )
    return out


# ============ FactRow JSON round-trip (used inside row encodings) ============

def _fact_to_record(f: FactRow) -> dict[str, object]:
    """Per-fact parquet record. Stats become JSON to handle
    heterogeneous keys."""
    return {
        'name': f.name,
        'kind': f.kind,
        'targets': list(f.targets),
        'reads': sorted(f.reads),
        'verdict': f.verdict,
        'natural_strength': f.natural_strength,
        'delta_i': f.delta_i,
        'evidentiary_level': f.evidentiary_level,
        'stats_json': _stats_to_json(f.stats),
        'intervention_signature': sorted(f.intervention_signature),
    }


def _fact_from_record(d: Mapping[str, object]) -> FactRow:
    return FactRow(
        name=require_str(d, 'name'),
        kind=require_kind(d, 'kind'),
        targets=tuple(require_str_list(d, 'targets')),
        reads=frozenset(require_str_list(d, 'reads')),
        verdict=require_str(d, 'verdict'),
        natural_strength=require_float(d, 'natural_strength'),
        delta_i=require_float(d, 'delta_i'),
        evidentiary_level=require_str(d, 'evidentiary_level'),
        stats=_stats_from_json(require_str(d, 'stats_json')),
        intervention_signature=frozenset(
            require_str_list(d, 'intervention_signature')
        ),
    )


def _facts_list_from_json(s: str) -> tuple[FactRow, ...]:
    raw = _decode_json(s)
    if not is_list_of_object(raw):
        raise TypeError(f'facts JSON must decode to a list, got {type(raw).__name__}')
    out: list[FactRow] = []
    for item in raw:
        if not is_mapping_of_object(item):
            raise TypeError(f'facts entry must be a mapping, got {type(item).__name__}')
        out.append(_fact_from_record(item))
    return tuple(out)


def _facts_to_json(facts: Iterable[FactRow]) -> str:
    return json.dumps([_fact_to_record(f) for f in facts])


# ============ MechanismKey JSON round-trip ============

def _mechanism_key_to_json(mk: MechanismKey) -> str:
    return json.dumps({
        'intervention_signature': [
            {'slot': s, 'value': v}
            for s, v in mk.intervention_signature
        ],
        'bridge_names': sorted(mk.bridge_names),
        'direction': mk.direction,
    })


def _mechanism_key_from_json(s: str) -> MechanismKey:
    raw = _decode_json(s)
    if not is_mapping_of_object(raw):
        raise TypeError(
            f'mechanism_key JSON must decode to a mapping, got {type(raw).__name__}'
        )
    sig_raw = raw.get('intervention_signature')
    if not is_list_of_object(sig_raw):
        raise TypeError('mechanism_key.intervention_signature must be a list')
    pairs: list[tuple[str, str]] = []
    for entry in sig_raw:
        if not is_mapping_of_object(entry):
            raise TypeError('intervention_signature entry must be a mapping')
        slot = entry.get('slot')
        value = entry.get('value')
        if not isinstance(slot, str) or not isinstance(value, str):
            raise TypeError('intervention_signature entry must have str slot and value')
        pairs.append((slot, value))
    return MechanismKey(
        intervention_signature=tuple(pairs),
        bridge_names=frozenset(require_str_list(raw, 'bridge_names')),
        direction=optional_direction(raw, 'direction'),
    )


# ============ RunRow ============

def _runrow_to_record(row: RunRow) -> dict[str, object]:
    return {
        'id': row.id,
        'parent_id': row.parent_id,
        'intervention_name': row.intervention_name,
        'cycle_id': row.cycle_id,
        'timestamp': row.timestamp,
        'env_name': row.env_name,
        'total_steps': row.total_steps,
        'seed': row.seed,
        'mechanism_key_json': _mechanism_key_to_json(row.mechanism_key),
        'primary_outcome_summary': row.primary_outcome_summary,
        'record_keys': list(row.record_keys),
        'facts_json': _facts_to_json(row.facts),
        'reads_set': sorted(row.reads_set),
        'verdict': row.verdict,
        'meta_json': _meta_to_json(row.meta),
    }


def _runrow_from_record(d: Mapping[str, object]) -> RunRow:
    return RunRow(
        id=require_str(d, 'id'),
        parent_id=optional_str(d, 'parent_id'),
        intervention_name=require_str(d, 'intervention_name'),
        cycle_id=optional_str(d, 'cycle_id'),
        timestamp=require_str(d, 'timestamp'),
        env_name=require_str(d, 'env_name'),
        total_steps=require_int(d, 'total_steps'),
        seed=require_int(d, 'seed'),
        mechanism_key=_mechanism_key_from_json(require_str(d, 'mechanism_key_json')),
        primary_outcome_summary=require_float(d, 'primary_outcome_summary'),
        record_keys=tuple(require_str_list(d, 'record_keys')),
        facts=_facts_list_from_json(require_str(d, 'facts_json')),
        reads_set=frozenset(require_str_list(d, 'reads_set')),
        verdict=require_str(d, 'verdict'),
        meta=_meta_from_json(require_str(d, 'meta_json')),
    )


def write_runrows(rows: Iterable[RunRow], path: Path) -> None:
    """Write RunRows to parquet at `path`. Returns nothing; raises
    on I/O failure or schema inconsistency."""
    records = [_runrow_to_record(r) for r in rows]
    pl.DataFrame(records).write_parquet(path)


def read_runrows(path: Path) -> list[RunRow]:
    df = pl.read_parquet(path)
    out: list[RunRow] = []
    for d in _to_dicts(df):
        out.append(_runrow_from_record(d))
    return out


# ============ ArmRow ============

def _armrow_to_record(row: ArmRow) -> dict[str, object]:
    return {
        'id': row.id,
        'intervention_name': row.intervention_name,
        'env_name': row.env_name,
        'cycle_id': row.cycle_id,
        'timestamp': row.timestamp,
        'mechanism_key_json': _mechanism_key_to_json(row.mechanism_key),
        'run_ids': list(row.run_ids),
        'seeds': list(row.seeds),
        'n': row.n,
        'arm_mean': row.arm_mean,
        'arm_sd': row.arm_sd,
        'facts_json': _facts_to_json(row.facts),
        'reads_set': sorted(row.reads_set),
        'meta_json': _meta_to_json(row.meta),
    }


def _armrow_from_record(d: Mapping[str, object]) -> ArmRow:
    return ArmRow(
        id=require_str(d, 'id'),
        intervention_name=require_str(d, 'intervention_name'),
        env_name=require_str(d, 'env_name'),
        cycle_id=optional_str(d, 'cycle_id'),
        timestamp=require_str(d, 'timestamp'),
        mechanism_key=_mechanism_key_from_json(require_str(d, 'mechanism_key_json')),
        run_ids=tuple(require_str_list(d, 'run_ids')),
        seeds=tuple(require_int_list(d, 'seeds')),
        n=require_int(d, 'n'),
        arm_mean=require_float(d, 'arm_mean'),
        arm_sd=require_float(d, 'arm_sd'),
        facts=_facts_list_from_json(require_str(d, 'facts_json')),
        reads_set=frozenset(require_str_list(d, 'reads_set')),
        meta=_meta_from_json(require_str(d, 'meta_json')),
    )


def write_armrows(rows: Iterable[ArmRow], path: Path) -> None:
    records = [_armrow_to_record(r) for r in rows]
    pl.DataFrame(records).write_parquet(path)


def read_armrows(path: Path) -> list[ArmRow]:
    df = pl.read_parquet(path)
    out: list[ArmRow] = []
    for d in _to_dicts(df):
        out.append(_armrow_from_record(d))
    return out


# ============ ComparisonRow ============

def _comparisonrow_to_record(row: ComparisonRow) -> dict[str, object]:
    return {
        'id': row.id,
        'parent_id': row.parent_id,
        'intervention_name': row.intervention_name,
        'env_name': row.env_name,
        'cycle_id': row.cycle_id,
        'timestamp': row.timestamp,
        'treatment_arm_id': row.treatment_arm_id,
        'baseline_arm_id': row.baseline_arm_id,
        'mechanism_key_json': _mechanism_key_to_json(row.mechanism_key),
        'predicted_direction': row.predicted_direction,
        'n_treatment': row.n_treatment,
        'n_baseline': row.n_baseline,
        'arm_a_mean': row.arm_a_mean,
        'arm_a_sd': row.arm_a_sd,
        'arm_b_mean': row.arm_b_mean,
        'arm_b_sd': row.arm_b_sd,
        'effect_size_g': row.effect_size_g,
        'se': row.se,
        'derived_q': row.derived_q,
        'delta_i_population': row.delta_i_population,
        'verdict': row.verdict,
        'refutation_class': row.refutation_class,
        'adequately_powered': row.adequately_powered,
        'facts_json': _facts_to_json(row.facts),
        'reads_set': sorted(row.reads_set),
        'meta_json': _meta_to_json(row.meta),
    }


def _comparisonrow_from_record(d: Mapping[str, object]) -> ComparisonRow:
    return ComparisonRow(
        id=require_str(d, 'id'),
        parent_id=optional_str(d, 'parent_id'),
        intervention_name=require_str(d, 'intervention_name'),
        env_name=require_str(d, 'env_name'),
        cycle_id=optional_str(d, 'cycle_id'),
        timestamp=require_str(d, 'timestamp'),
        treatment_arm_id=require_str(d, 'treatment_arm_id'),
        baseline_arm_id=require_str(d, 'baseline_arm_id'),
        mechanism_key=_mechanism_key_from_json(require_str(d, 'mechanism_key_json')),
        predicted_direction=optional_direction(d, 'predicted_direction'),
        n_treatment=require_int(d, 'n_treatment'),
        n_baseline=require_int(d, 'n_baseline'),
        arm_a_mean=require_float(d, 'arm_a_mean'),
        arm_a_sd=require_float(d, 'arm_a_sd'),
        arm_b_mean=require_float(d, 'arm_b_mean'),
        arm_b_sd=require_float(d, 'arm_b_sd'),
        effect_size_g=optional_float(d, 'effect_size_g'),
        se=optional_float(d, 'se'),
        derived_q=optional_float(d, 'derived_q'),
        delta_i_population=require_float(d, 'delta_i_population'),
        verdict=require_str(d, 'verdict'),
        refutation_class=optional_str(d, 'refutation_class'),
        adequately_powered=require_bool(d, 'adequately_powered'),
        facts=_facts_list_from_json(require_str(d, 'facts_json')),
        reads_set=frozenset(require_str_list(d, 'reads_set')),
        meta=_meta_from_json(require_str(d, 'meta_json')),
    )


def write_comparisonrows(rows: Iterable[ComparisonRow], path: Path) -> None:
    records = [_comparisonrow_to_record(r) for r in rows]
    pl.DataFrame(records).write_parquet(path)


def read_comparisonrows(path: Path) -> list[ComparisonRow]:
    df = pl.read_parquet(path)
    out: list[ComparisonRow] = []
    for d in _to_dicts(df):
        out.append(_comparisonrow_from_record(d))
    return out


# ============ CorpusRow ============

def _corpusrow_to_record(row: CorpusRow) -> dict[str, object]:
    return {
        'id': row.id,
        'name': row.name,
        'cycle_id': row.cycle_id,
        'timestamp': row.timestamp,
        'comparison_ids': list(row.comparison_ids),
        'n_comparisons': row.n_comparisons,
        'facts_json': _facts_to_json(row.facts),
        'reads_set': sorted(row.reads_set),
        'meta_json': _meta_to_json(row.meta),
    }


def _corpusrow_from_record(d: Mapping[str, object]) -> CorpusRow:
    return CorpusRow(
        id=require_str(d, 'id'),
        name=require_str(d, 'name'),
        cycle_id=optional_str(d, 'cycle_id'),
        timestamp=require_str(d, 'timestamp'),
        comparison_ids=tuple(require_str_list(d, 'comparison_ids')),
        n_comparisons=require_int(d, 'n_comparisons'),
        facts=_facts_list_from_json(require_str(d, 'facts_json')),
        reads_set=frozenset(require_str_list(d, 'reads_set')),
        meta=_meta_from_json(require_str(d, 'meta_json')),
    )


def write_corpusrows(rows: Iterable[CorpusRow], path: Path) -> None:
    records = [_corpusrow_to_record(r) for r in rows]
    pl.DataFrame(records).write_parquet(path)


def read_corpusrows(path: Path) -> list[CorpusRow]:
    df = pl.read_parquet(path)
    out: list[CorpusRow] = []
    for d in _to_dicts(df):
        out.append(_corpusrow_from_record(d))
    return out
