"""Schema — typed row dataclasses for corroborate's four-row corpus.

The corpus has four levels:

- `RunRow` — per-cell evidence (one (env, seed) execution).
  Source of truth.
- `ArmRow` — per-(hypothesis, env) aggregate (across seeds).
- `ComparisonRow` — per (treatment_arm, baseline_arm) pair on one
  env. Carries Hedges' g, SE, derived_q, etc. (populated by the
  statistics module — step 5 — at construction time).
- `CorpusRow` — across-comparison aggregate (e.g. the Pearson-r
  link bridge in §3.5 of PAPER_NOTES.md).

Each level is a frozen-dataclass summary. Schema rows are
**non-generic**: they store derived summaries (mechanism_key,
facts, scalars), not the live record. The framework's record
schema parameter `R` lives in `Bridge[R]` / `Hypothesis[R]` /
`Measurable[R, T]` where it actually constrains behavior; on
schema rows it would have been phantom (no field uses it) and
phantom-R breaks `Self` in `from_dict` classmethods. The honest
shape is to keep `R` where it carries information and erase at
the schema boundary — a row is a record-derived summary, not a
record.

Lineage is explicit via `*_id` fields:
- `RunRow.id` → referenced by `ArmRow.run_ids`
- `ArmRow.id` → referenced by `ComparisonRow.{treatment,baseline}_arm_id`
- `ComparisonRow.id` → referenced by `CorpusRow.comparison_ids`

`FactRow` is the per-bridge atomic verdict carried by `RunRow.facts`
and aggregated upward via admit-rate and overlap-weighted ΔI.

Each row exposes `as_dict() -> dict[str, object]` and a `from_dict`
classmethod for round-tripping. Parquet persistence (polars-based)
lives in `persistence.py` (step 2.7b)."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Self

from corroborate._narrow import (
    is_list_of_object,
    is_mapping_of_object,
    list_len,
    optional_direction,
    optional_float,
    optional_refutation_class,
    optional_str,
    require_bool,
    require_float,
    require_int,
    require_int_list,
    require_kind,
    require_mapping,
    require_mapping_in_list,
    require_meta_mapping,
    require_stats_mapping,
    require_str,
    require_str_list,
    require_verdict,
)
from corroborate.hypothesis import Direction, MechanismKey
from corroborate.verdict import RefutationClass, Verdict


# ============ Default factories (typed) ============

# `field(default_factory=_empty_meta)` produces `dict[Unknown, Unknown]`
# under strict pyright because bare `dict` is the unparameterized
# constructor. Typed module-level factories supply the correct
# parameterization, so pyright infers each field's runtime type
# matches its declared type.

def _empty_meta() -> dict[str, str | int | float | bool]:
    return {}


# ============ TraceRow — raw per-cell observation store ============

# A trace leaf is either a scalar (HP value, summary scalar) or a
# 1-D trajectory list (per-step or per-burst sequence). Higher-
# dimensional arrays (e.g. `(total_steps, batch_size, n_actions)`
# for `online_q_values`) are NOT in v0's trace store — substrate
# authors who need them either reduce to scalar/1-D before yield
# or wait for a future shape extension. Dropping them keeps the
# parquet schema columnar (one type per column) and the writer
# logic flat — no nested-list wrangling, no JSON columns.

type TraceLeaf = (
    str | int | float | bool
    | list[float] | list[int] | list[bool] | list[str]
)


@dataclass(frozen=True, slots=True)
class TraceRow:
    """One cell's raw observation: HPs (path-keyed scalars) +
    per-step trajectories (1-D lists) + provenance.

    The trace store is the v9-`traces.parquet` analog: low-
    derivation, queryable, re-usable for post-hoc bridge
    re-evaluation. The hypothesis-record store (`RunRow`) sits
    above this, carrying framework verdicts derived from these
    leaves.

    `id` matches the corresponding `RunRow.id` so the two stores
    join on a single column. Linking by UUID (not by hypothesis
    name as v9 did) lets two cells of the same hypothesis remain
    distinguishable.

    `leaves` is path-keyed: HPs are dotted topology paths
    (`bootstrap.gamma`, `optimizer.inner.lr`); trajectories are
    flat author-chosen return-dict keys (`reward`, `loss`,
    `td_error`). Mixed scalar/list values per parquet column
    type — the type tells you which kind a path is.

    No `evidence__` / `binding__` namespace prefixes: paths
    encode origin via topology (dotted) vs. author-key (flat),
    and parquet column types disambiguate scalar vs. trajectory."""
    id: str
    cycle_id: str | None
    timestamp: str
    leaves: Mapping[str, TraceLeaf] = field(
        default_factory=lambda: {},
    )

    def as_dict(self) -> dict[str, object]:
        """Flat top-level dict for parquet: provenance fields +
        each leaf-path becomes its own top-level key. The writer
        feeds this directly to `pl.DataFrame` — no JSON wrapping,
        no nested structs."""
        out: dict[str, object] = {
            'id': self.id,
            'cycle_id': self.cycle_id,
            'timestamp': self.timestamp,
        }
        for path, value in self.leaves.items():
            out[path] = value
        return out

    @classmethod
    def from_row_dict(cls, d: Mapping[str, object]) -> Self:
        """Reverse of `as_dict`: split provenance fields from
        leaf-path columns. Any column not in the typed-provenance
        set is treated as a leaf. Null-padded columns (paths the
        row didn't carry — polars fills missing columns with None
        when rows have heterogeneous keys) are skipped."""
        provenance: frozenset[str] = frozenset(
            ('id', 'cycle_id', 'timestamp')
        )
        leaves: dict[str, TraceLeaf] = {}
        for k, v in d.items():
            if k in provenance:
                continue
            if v is None:
                # Polars null-pads columns this row didn't write.
                continue
            leaves[k] = _coerce_trace_leaf(v)
        return cls(
            id=require_str(d, 'id'),
            cycle_id=optional_str(d, 'cycle_id'),
            timestamp=require_str(d, 'timestamp'),
            leaves=leaves,
        )


def _coerce_trace_leaf(value: object) -> TraceLeaf:
    """Narrow a parquet-decoded object to `TraceLeaf` at the
    persistence boundary. Scalars pass through; lists are accepted
    as `list[float | int | bool | str]` (polars decodes List columns
    to Python lists)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, list):
        return _coerce_trace_list(value)
    raise TypeError(
        f'unsupported TraceRow leaf type: {type(value).__name__}',
    )


def _coerce_trace_list(
    value: list[object],
) -> list[float] | list[int] | list[bool] | list[str]:
    """Narrow a list to one of the four homogeneous shapes
    `TraceLeaf` allows. Empty lists are returned as `list[float]`
    by convention (polars produces typed empty lists at read; we
    canonicalise to float-list)."""
    if not value:
        return []
    first = value[0]
    if isinstance(first, bool):
        return [bool(v) for v in value]
    if isinstance(first, int):
        return [int(v) for v in value]
    if isinstance(first, float):
        return [float(v) for v in value]
    if isinstance(first, str):
        return [str(v) for v in value]
    raise TypeError(
        f'unsupported TraceRow list element type: '
        f'{type(first).__name__}',
    )


# ============ FactRow (non-generic — record-agnostic) ============

@dataclass(frozen=True, slots=True)
class FactRow:
    """One bridge or invariant verdict carried within a `RunRow`.

    Stats are scalar primitives only; rich data (arrays, nested
    structures) lives on the underlying record.

    `targets` is the **canonical** ordered declaration of what
    record fields the fact reads. `reads` is a **derived
    property** (frozenset over targets) used by axiom 19's
    redundancy primitive's reads-set Jaccard — derived rather
    than stored so direction information in `targets` cannot be
    silently dropped by an inconsistent caller. `intervention_
    signature` is the leaf-flattened form of the parent
    hypothesis's intervention; lets the redundancy primitive's
    intervention factor activate at the fact level.

    Note: which *record* produced this fact is not carried as a
    field on FactRow. The Hypothesis structure (primary `bridges`
    vs secondary `bridges_e`) and the bridge `targets` are the
    framework-honest record-identity primitives — `targets`
    already encodes which keys this fact read. Adding a
    `data_source` field would denormalise + import substrate-
    specific vocabulary (e.g. RL's 'train' / 'eval' phases) into
    the schema layer."""
    name: str
    kind: Literal['bridge', 'invariant']
    targets: tuple[str, ...]
    verdict: Verdict
    natural_strength: float
    delta_i: float
    evidentiary_level: str
    stats: Mapping[str, float | int | bool | str]
    intervention_signature: frozenset[str] = field(default_factory=frozenset)

    @property
    def reads(self) -> frozenset[str]:
        """Frozenset over `targets`. Derived (not stored) so
        ordered direction information in `targets` cannot be
        accidentally dropped — callers can't construct a FactRow
        with `reads != frozenset(targets)`."""
        return frozenset(self.targets)

    def as_dict(self) -> dict[str, object]:
        return {
            'name': self.name,
            'kind': self.kind,
            'targets': list(self.targets),
            'verdict': self.verdict.value,
            'natural_strength': self.natural_strength,
            'delta_i': self.delta_i,
            'evidentiary_level': self.evidentiary_level,
            'stats': {**self.stats},
            'intervention_signature': sorted(self.intervention_signature),
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        return cls(
            name=require_str(d, 'name'),
            kind=require_kind(d, 'kind'),
            targets=tuple(require_str_list(d, 'targets')),
            verdict=require_verdict(d, 'verdict'),
            natural_strength=require_float(d, 'natural_strength'),
            delta_i=require_float(d, 'delta_i'),
            evidentiary_level=require_str(d, 'evidentiary_level'),
            stats=require_stats_mapping(d, 'stats'),
            intervention_signature=frozenset(
                require_str_list(d, 'intervention_signature')
            ),
        )


# ============ RunRow ============

@dataclass(frozen=True, slots=True)
class RunRow:
    """Per-cell evidence — one (env, seed) execution under one
    hypothesis. The lowest-level row, source of truth for upper
    aggregations.

    `mechanism_key` carries the canonical structural identity from
    the hypothesis; downstream consumers dedup runs by
    (mechanism_key, env_name, seed)."""
    id: str
    parent_id: str | None
    intervention_name: str
    cycle_id: str | None
    timestamp: str
    env_name: str
    total_steps: int
    seed: int
    mechanism_key: MechanismKey
    primary_outcome_summary: float
    record_keys: tuple[str, ...]
    facts: tuple[FactRow, ...]
    reads_set: frozenset[str]
    verdict: Verdict
    meta: Mapping[str, str | int | float | bool] = field(default_factory=_empty_meta)

    def as_dict(self) -> dict[str, object]:
        return {
            'id': self.id,
            'parent_id': self.parent_id,
            'intervention_name': self.intervention_name,
            'cycle_id': self.cycle_id,
            'timestamp': self.timestamp,
            'env_name': self.env_name,
            'total_steps': self.total_steps,
            'seed': self.seed,
            'mechanism_key': _mechanism_key_as_dict(self.mechanism_key),
            'primary_outcome_summary': self.primary_outcome_summary,
            'record_keys': list(self.record_keys),
            'facts': [f.as_dict() for f in self.facts],
            'reads_set': sorted(self.reads_set),
            'verdict': self.verdict.value,
            'meta': {**self.meta},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        return cls(
            id=require_str(d, 'id'),
            parent_id=optional_str(d, 'parent_id'),
            intervention_name=require_str(d, 'intervention_name'),
            cycle_id=optional_str(d, 'cycle_id'),
            timestamp=require_str(d, 'timestamp'),
            env_name=require_str(d, 'env_name'),
            total_steps=require_int(d, 'total_steps'),
            seed=require_int(d, 'seed'),
            mechanism_key=_mechanism_key_from_dict(
                require_mapping(d, 'mechanism_key')
            ),
            primary_outcome_summary=require_float(d, 'primary_outcome_summary'),
            record_keys=tuple(require_str_list(d, 'record_keys')),
            facts=tuple(
                FactRow.from_dict(require_mapping_in_list(d, 'facts', i))
                for i in range(list_len(d, 'facts'))
            ),
            reads_set=frozenset(require_str_list(d, 'reads_set')),
            verdict=require_verdict(d, "verdict"),
            meta=require_meta_mapping(d, 'meta'),
        )


# ============ ArmRow ============

@dataclass(frozen=True, slots=True)
class ArmRow:
    """Per-(hypothesis, env) aggregate across seeds. Treatment and
    baseline arms are constructed separately; ComparisonRow pairs
    them by env."""
    id: str
    intervention_name: str
    env_name: str
    cycle_id: str | None
    timestamp: str
    mechanism_key: MechanismKey
    run_ids: tuple[str, ...]
    seeds: tuple[int, ...]
    n: int
    arm_mean: float
    arm_sd: float
    facts: tuple[FactRow, ...]
    reads_set: frozenset[str]
    meta: Mapping[str, str | int | float | bool] = field(default_factory=_empty_meta)

    def as_dict(self) -> dict[str, object]:
        return {
            'id': self.id,
            'intervention_name': self.intervention_name,
            'env_name': self.env_name,
            'cycle_id': self.cycle_id,
            'timestamp': self.timestamp,
            'mechanism_key': _mechanism_key_as_dict(self.mechanism_key),
            'run_ids': list(self.run_ids),
            'seeds': list(self.seeds),
            'n': self.n,
            'arm_mean': self.arm_mean,
            'arm_sd': self.arm_sd,
            'facts': [f.as_dict() for f in self.facts],
            'reads_set': sorted(self.reads_set),
            'meta': {**self.meta},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        return cls(
            id=require_str(d, 'id'),
            intervention_name=require_str(d, 'intervention_name'),
            env_name=require_str(d, 'env_name'),
            cycle_id=optional_str(d, 'cycle_id'),
            timestamp=require_str(d, 'timestamp'),
            mechanism_key=_mechanism_key_from_dict(
                require_mapping(d, 'mechanism_key')
            ),
            run_ids=tuple(require_str_list(d, 'run_ids')),
            seeds=tuple(require_int_list(d, 'seeds')),
            n=require_int(d, 'n'),
            arm_mean=require_float(d, 'arm_mean'),
            arm_sd=require_float(d, 'arm_sd'),
            facts=tuple(
                FactRow.from_dict(require_mapping_in_list(d, 'facts', i))
                for i in range(list_len(d, 'facts'))
            ),
            reads_set=frozenset(require_str_list(d, 'reads_set')),
            meta=require_meta_mapping(d, 'meta'),
        )


# ============ ComparisonRow ============

@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """Per (treatment_arm, baseline_arm) comparison on one env.

    Statistical fields (`effect_size_g`, `se`, `derived_q`,
    `delta_i_population`, `verdict`, `refutation_class`,
    `adequately_powered`) are populated by the statistics module
    (step 5). v0's schema declares them; aggregation factories
    land later."""
    id: str
    parent_id: str | None
    intervention_name: str
    env_name: str
    cycle_id: str | None
    timestamp: str
    treatment_arm_id: str
    baseline_arm_id: str
    mechanism_key: MechanismKey
    predicted_direction: Direction | None
    n_treatment: int
    n_baseline: int
    arm_a_mean: float
    arm_a_sd: float
    arm_b_mean: float
    arm_b_sd: float
    effect_size_g: float | None
    se: float | None
    derived_q: float | None
    delta_i_population: float
    verdict: Verdict
    refutation_class: RefutationClass | None
    adequately_powered: bool
    facts: tuple[FactRow, ...]
    reads_set: frozenset[str]
    meta: Mapping[str, str | int | float | bool] = field(default_factory=_empty_meta)

    def as_dict(self) -> dict[str, object]:
        return {
            'id': self.id,
            'parent_id': self.parent_id,
            'intervention_name': self.intervention_name,
            'env_name': self.env_name,
            'cycle_id': self.cycle_id,
            'timestamp': self.timestamp,
            'treatment_arm_id': self.treatment_arm_id,
            'baseline_arm_id': self.baseline_arm_id,
            'mechanism_key': _mechanism_key_as_dict(self.mechanism_key),
            'predicted_direction': self.predicted_direction,
            'n_treatment': self.n_treatment,
            'n_baseline': self.n_baseline,
            'arm_a_mean': self.arm_a_mean,
            'arm_a_sd': self.arm_a_sd,
            'arm_b_mean': self.arm_b_mean,
            'arm_b_sd': self.arm_b_sd,
            'effect_size_g': self.effect_size_g,
            'se': self.se,
            'derived_q': self.derived_q,
            'delta_i_population': self.delta_i_population,
            'verdict': self.verdict.value,
            'refutation_class': (
                self.refutation_class.value
                if self.refutation_class is not None
                else None
            ),
            'adequately_powered': self.adequately_powered,
            'facts': [f.as_dict() for f in self.facts],
            'reads_set': sorted(self.reads_set),
            'meta': {**self.meta},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        return cls(
            id=require_str(d, 'id'),
            parent_id=optional_str(d, 'parent_id'),
            intervention_name=require_str(d, 'intervention_name'),
            env_name=require_str(d, 'env_name'),
            cycle_id=optional_str(d, 'cycle_id'),
            timestamp=require_str(d, 'timestamp'),
            treatment_arm_id=require_str(d, 'treatment_arm_id'),
            baseline_arm_id=require_str(d, 'baseline_arm_id'),
            mechanism_key=_mechanism_key_from_dict(
                require_mapping(d, 'mechanism_key')
            ),
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
            verdict=require_verdict(d, "verdict"),
            refutation_class=optional_refutation_class(d, 'refutation_class'),
            adequately_powered=require_bool(d, 'adequately_powered'),
            facts=tuple(
                FactRow.from_dict(require_mapping_in_list(d, 'facts', i))
                for i in range(list_len(d, 'facts'))
            ),
            reads_set=frozenset(require_str_list(d, 'reads_set')),
            meta=require_meta_mapping(d, 'meta'),
        )


# ============ CorpusRow ============

@dataclass(frozen=True, slots=True)
class CorpusRow:
    """Across-comparison aggregate. The corpus-level summary
    consumed by link bridges (e.g. §3.5's
    `Pearson r(stat_q, stat_f)` across envs) and by axiom 19's
    redundancy primitive (G as the latest-wins fact register)."""
    id: str
    name: str
    cycle_id: str | None
    timestamp: str
    comparison_ids: tuple[str, ...]
    n_comparisons: int
    facts: tuple[FactRow, ...]
    reads_set: frozenset[str]
    meta: Mapping[str, str | int | float | bool] = field(default_factory=_empty_meta)

    def as_dict(self) -> dict[str, object]:
        return {
            'id': self.id,
            'name': self.name,
            'cycle_id': self.cycle_id,
            'timestamp': self.timestamp,
            'comparison_ids': list(self.comparison_ids),
            'n_comparisons': self.n_comparisons,
            'facts': [f.as_dict() for f in self.facts],
            'reads_set': sorted(self.reads_set),
            'meta': {**self.meta},
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, object]) -> Self:
        return cls(
            id=require_str(d, 'id'),
            name=require_str(d, 'name'),
            cycle_id=optional_str(d, 'cycle_id'),
            timestamp=require_str(d, 'timestamp'),
            comparison_ids=tuple(require_str_list(d, 'comparison_ids')),
            n_comparisons=require_int(d, 'n_comparisons'),
            facts=tuple(
                FactRow.from_dict(require_mapping_in_list(d, 'facts', i))
                for i in range(list_len(d, 'facts'))
            ),
            reads_set=frozenset(require_str_list(d, 'reads_set')),
            meta=require_meta_mapping(d, 'meta'),
        )


# ============ Mechanism key serialization ============

def _mechanism_key_as_dict(mk: MechanismKey) -> dict[str, object]:
    return {
        'intervention_signature': [
            {'slot': slot, 'value': value}
            for slot, value in mk.intervention_signature
        ],
        'bridge_names': sorted(mk.bridge_names),
        'direction': mk.direction,
    }


def _mechanism_key_from_dict(d: Mapping[str, object]) -> MechanismKey:
    sig_raw = d.get('intervention_signature')
    if not is_list_of_object(sig_raw):
        raise TypeError(
            f"mechanism_key.intervention_signature must be a list, "
            f"got {type(sig_raw).__name__}"
        )
    pairs: list[tuple[str, str]] = []
    for entry in sig_raw:
        if not is_mapping_of_object(entry):
            raise TypeError(
                f"intervention_signature entry must be a mapping, "
                f"got {type(entry).__name__}"
            )
        slot = entry.get('slot')
        value = entry.get('value')
        if not isinstance(slot, str) or not isinstance(value, str):
            raise TypeError(
                f"intervention_signature entry must have str slot and str value"
            )
        pairs.append((slot, value))
    return MechanismKey(
        intervention_signature=tuple(pairs),
        bridge_names=frozenset(require_str_list(d, 'bridge_names')),
        direction=optional_direction(d, 'direction'),
    )


