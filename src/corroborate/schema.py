"""Schema — typed row dataclasses for corroborate's four-row corpus.

The corpus has four levels:

- `RunRow` — per-cell evidence (one (env, seed) execution).
  Source of truth.
- `ArmRow` — per-(intervention, env) aggregate (across seeds).
- `ComparisonRow` — per (treatment_arm, baseline_arm) pair on one
  env. Carries Hedges' g, SE, derived_q, etc.
- `CorpusRow` — across-comparison aggregate (e.g. the Pearson-r
  link bridge in §3.5 of PAPER_NOTES.md).

Each row splits into a **framework-typed surface** (closed-set
enums, lineage IDs, framework-controlled provenance) and an
**open surface** — `measurements: Mapping[str, MeasurementLeaf]`,
path-keyed scalar leaves. HPs land at dotted topology paths
(`gamma`, `optimizer.inner.lr`); bridge/invariant results at
`bridge.<name>.verdict`/`bridge.<name>.stats.<key>` and
`invariant.<name>.verdict`; substrate-named outcome reductions
under their own keys (e.g. `outcome.late_window_mean`).

No JSON-wrapped struct columns, no `evidence__`/`binding__`
namespace prefixes. Persistence is flat columnar parquet — every
measurement is its own typed column, queryable directly.

`as_dict()` returns a flat top-level dict (provenance/typed
fields plus each measurement at top-level, unprefixed).
`from_row_dict(d)` reverses: provenance fields by name, the rest
into `measurements`. Skip None-valued columns (polars null-pads
when rows have heterogeneous keys).

Lineage is explicit via `*_id` fields:
- `RunRow.id` → referenced by `ArmRow.run_ids`
- `ArmRow.id` → referenced by `ComparisonRow.{treatment,baseline}_arm_id`
- `ComparisonRow.id` → referenced by `CorpusRow.comparison_ids`"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Self

from corroborate._narrow import (
    optional_direction,
    optional_refutation_class,
    optional_str,
    require_bool,
    require_str,
    require_str_list,
    require_verdict,
)
from corroborate.hypothesis import Direction
from corroborate.verdict import RefutationClass, Verdict


# ============ Measurement leaf type ============

type MeasurementLeaf = str | int | float | bool
"""One scalar measurement on a row. The framework persists each
leaf as its own typed parquet column."""


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


# ============ Measurement coercion (shared row helper) ============

def _coerce_measurement_leaf(value: object) -> MeasurementLeaf:
    """Narrow a parquet-decoded object to `MeasurementLeaf` at
    the persistence boundary. Polars decodes typed columns back to
    `bool` / `int` / `float` / `str` — anything else is a schema
    contract violation (loud error)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    raise TypeError(
        f'unsupported measurement leaf type: {type(value).__name__}',
    )


def _flatten_measurements(
    out: dict[str, object],
    measurements: Mapping[str, MeasurementLeaf],
) -> None:
    """Lift each measurement entry to a top-level key in `out`.
    Shared between row `as_dict()` implementations — keeps the
    flattening rule in one place."""
    for path, value in measurements.items():
        out[path] = value


# ============ RunRow ============

@dataclass(frozen=True, slots=True)
class RunRow:
    """Per-cell evidence — one (env, seed) execution under one
    hypothesis. The lowest-level row, source of truth for upper
    aggregations.

    Framework-typed surface: lineage IDs, cycle/timestamp,
    aggregate `verdict`. Open surface: `measurements` carrying
    HP values at dotted topology paths, bridge/invariant results
    under `bridge.<name>.*` / `invariant.<name>.*`, outcome
    reductions under substrate-named keys (e.g.
    `outcome.late_window_mean`), and substrate metadata
    (`env_name`, `seed`, `total_steps`, `intervention_name`)."""
    id: str
    parent_id: str | None
    cycle_id: str | None
    timestamp: str
    verdict: Verdict
    measurements: Mapping[str, MeasurementLeaf] = field(
        default_factory=lambda: {},
    )

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            'id': self.id,
            'parent_id': self.parent_id,
            'cycle_id': self.cycle_id,
            'timestamp': self.timestamp,
            'verdict': self.verdict.value,
        }
        _flatten_measurements(out, self.measurements)
        return out

    @classmethod
    def from_row_dict(cls, d: Mapping[str, object]) -> Self:
        provenance: frozenset[str] = frozenset(
            ('id', 'parent_id', 'cycle_id', 'timestamp', 'verdict')
        )
        measurements: dict[str, MeasurementLeaf] = {}
        for k, v in d.items():
            if k in provenance:
                continue
            if v is None:
                continue
            measurements[k] = _coerce_measurement_leaf(v)
        return cls(
            id=require_str(d, 'id'),
            parent_id=optional_str(d, 'parent_id'),
            cycle_id=optional_str(d, 'cycle_id'),
            timestamp=require_str(d, 'timestamp'),
            verdict=require_verdict(d, 'verdict'),
            measurements=measurements,
        )


# ============ ArmRow ============

@dataclass(frozen=True, slots=True)
class ArmRow:
    """Per-(intervention, env) aggregate across seeds. Treatment
    and baseline arms are constructed separately; ComparisonRow
    pairs them by env.

    Framework-typed surface: lineage IDs (`run_ids`),
    cycle/timestamp. Open surface: `measurements` carrying HP
    fingerprint (the configurational keys shared across the arm's
    runs), per-bridge admit-rates, arm-level outcome statistics
    (`outcome.<m>.arm_mean`, `outcome.<m>.arm_sd`, `n`)."""
    id: str
    cycle_id: str | None
    timestamp: str
    run_ids: tuple[str, ...]
    measurements: Mapping[str, MeasurementLeaf] = field(
        default_factory=lambda: {},
    )

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            'id': self.id,
            'cycle_id': self.cycle_id,
            'timestamp': self.timestamp,
            'run_ids': list(self.run_ids),
        }
        _flatten_measurements(out, self.measurements)
        return out

    @classmethod
    def from_row_dict(cls, d: Mapping[str, object]) -> Self:
        provenance: frozenset[str] = frozenset(
            ('id', 'cycle_id', 'timestamp', 'run_ids')
        )
        measurements: dict[str, MeasurementLeaf] = {}
        for k, v in d.items():
            if k in provenance:
                continue
            if v is None:
                continue
            measurements[k] = _coerce_measurement_leaf(v)
        return cls(
            id=require_str(d, 'id'),
            cycle_id=optional_str(d, 'cycle_id'),
            timestamp=require_str(d, 'timestamp'),
            run_ids=tuple(require_str_list(d, 'run_ids')),
            measurements=measurements,
        )


# ============ ComparisonRow ============

@dataclass(frozen=True, slots=True)
class ComparisonRow:
    """Per (treatment_arm, baseline_arm) comparison on one env.

    Statistical fields (`predicted_direction`, `verdict`,
    `refutation_class`, `adequately_powered`) are framework-typed.
    Per-arm stats (`n_treatment`, `n_baseline`, effect sizes, SEs)
    live in `measurements` under `outcome.<m>.*` / `bridge.<name>.*`
    keys."""
    id: str
    parent_id: str | None
    cycle_id: str | None
    timestamp: str
    treatment_arm_id: str
    baseline_arm_id: str
    predicted_direction: Direction | None
    verdict: Verdict
    refutation_class: RefutationClass | None
    adequately_powered: bool
    measurements: Mapping[str, MeasurementLeaf] = field(
        default_factory=lambda: {},
    )

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            'id': self.id,
            'parent_id': self.parent_id,
            'cycle_id': self.cycle_id,
            'timestamp': self.timestamp,
            'treatment_arm_id': self.treatment_arm_id,
            'baseline_arm_id': self.baseline_arm_id,
            'predicted_direction': self.predicted_direction,
            'verdict': self.verdict.value,
            'refutation_class': (
                self.refutation_class.value
                if self.refutation_class is not None
                else None
            ),
            'adequately_powered': self.adequately_powered,
        }
        _flatten_measurements(out, self.measurements)
        return out

    @classmethod
    def from_row_dict(cls, d: Mapping[str, object]) -> Self:
        provenance: frozenset[str] = frozenset((
            'id', 'parent_id', 'cycle_id', 'timestamp',
            'treatment_arm_id', 'baseline_arm_id',
            'predicted_direction', 'verdict', 'refutation_class',
            'adequately_powered',
        ))
        measurements: dict[str, MeasurementLeaf] = {}
        for k, v in d.items():
            if k in provenance:
                continue
            if v is None:
                continue
            measurements[k] = _coerce_measurement_leaf(v)
        return cls(
            id=require_str(d, 'id'),
            parent_id=optional_str(d, 'parent_id'),
            cycle_id=optional_str(d, 'cycle_id'),
            timestamp=require_str(d, 'timestamp'),
            treatment_arm_id=require_str(d, 'treatment_arm_id'),
            baseline_arm_id=require_str(d, 'baseline_arm_id'),
            predicted_direction=optional_direction(d, 'predicted_direction'),
            verdict=require_verdict(d, 'verdict'),
            refutation_class=optional_refutation_class(d, 'refutation_class'),
            adequately_powered=require_bool(d, 'adequately_powered'),
            measurements=measurements,
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
    measurements: Mapping[str, MeasurementLeaf] = field(
        default_factory=lambda: {},
    )

    def as_dict(self) -> dict[str, object]:
        out: dict[str, object] = {
            'id': self.id,
            'name': self.name,
            'cycle_id': self.cycle_id,
            'timestamp': self.timestamp,
            'comparison_ids': list(self.comparison_ids),
        }
        _flatten_measurements(out, self.measurements)
        return out

    @classmethod
    def from_row_dict(cls, d: Mapping[str, object]) -> Self:
        provenance: frozenset[str] = frozenset(
            ('id', 'name', 'cycle_id', 'timestamp', 'comparison_ids')
        )
        measurements: dict[str, MeasurementLeaf] = {}
        for k, v in d.items():
            if k in provenance:
                continue
            if v is None:
                continue
            measurements[k] = _coerce_measurement_leaf(v)
        return cls(
            id=require_str(d, 'id'),
            name=require_str(d, 'name'),
            cycle_id=optional_str(d, 'cycle_id'),
            timestamp=require_str(d, 'timestamp'),
            comparison_ids=tuple(require_str_list(d, 'comparison_ids')),
            measurements=measurements,
        )
