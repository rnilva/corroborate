"""Schema — typed row dataclasses for corroborate's corpus.

Two levels (mirror v9's traces.parquet + measurements.parquet,
v10's HypothesisRunRow + HypothesisComparisonRow):

- `RunRow` — per-cell evidence (one (env, seed) execution).
  Source of truth.
- `ComparisonRow` — per (treatment, baseline) pair on one env.
  Carries Hedges' g, SE, derived_q, etc. Materialized view of
  RunRows; re-derivable on demand.

Plus the per-cell raw observation store, split across two
backends:

- `TraceRow.leaves` — scalars + 1-D series, persisted to parquet
  (one column per leaf path, polars-queryable).
- `TraceRow.arrays` — multi-dim numpy arrays (2-D+), persisted
  to zarr keyed by `{cell_id}/{array_name}`. Lazy-loaded on
  access. Bridges that need full Q-tensors / sample-index
  matrices read from `arrays` (not `leaves`).

The split is at the dimensionality boundary. Parquet handles
flat tabular and 1-D-list columns well; deeply-nested-list
columns (`list<list<list<float>>>`) work technically but suffer
from Python list materialization overhead and opaque queries.
zarr is the right format for dense multi-dim arrays.

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
- `RunRow.id` → referenced by `TraceRow.id` (1:1 join)
- `ComparisonRow.{treatment,baseline}_arm_id` is vestigial (an
  ArmRow id) and is empty-string for paired-by-seed comparisons.
  The N:1 RunRow→ComparisonRow lineage is currently implicit
  via paired_comparison_from_runs; field rename to run_ids tuple
  is a follow-up."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from corroborate.hypothesis import Hypothesis
from dataclasses import dataclass, field
from typing import Self

import numpy as np
import numpy.typing as npt

from corroborate._narrow import (
    optional_direction,
    optional_refutation_class,
    optional_str,
    require_bool,
    require_str,
    require_verdict,
)
from corroborate.hypothesis import Direction
from corroborate.statistics import PooledStats
from corroborate.verdict import RefutationClass, Verdict


type ArrayLeaf = npt.NDArray[np.floating] | npt.NDArray[np.integer] | npt.NDArray[np.bool_]
"""Multi-dim array leaf — what TraceRow.arrays carries. Shape
and dtype determined by the producer; no constraints at the type
level. Persisted to zarr by `write_tracerows`."""


# ============ Measurement leaf type ============

type MeasurementLeaf = str | int | float | bool
"""One scalar measurement on a row. The framework persists each
leaf as its own typed parquet column."""


# ============ TraceRow — raw per-cell observation store ============

# A trace leaf is a scalar (leaf value, summary scalar) OR a list
# of trace leaves at any nesting depth. Recursive type captures
# any-dim trajectory: 1-D for per-step series (`reward`, `loss`),
# 2-D for batch-per-step (`sample_indices` is `(steps, batch)`),
# 3-D for value-per-action-per-batch-per-step (`online_q_values`
# is `(steps, batch, n_actions)`). Polars handles arbitrary-depth
# nested `List` columns natively — `arr.tolist()` produces the
# nested Python form regardless of dimensionality.

type TraceLeaf = str | int | float | bool | list[TraceLeaf]


@dataclass(frozen=True, slots=True)
class TraceRow:
    """One cell's raw observation: configurational leaves +
    multi-dim arrays + provenance.

    The trace store is the v9-`traces.parquet` analog: low-
    derivation, queryable, re-usable for post-hoc bridge
    re-evaluation. The hypothesis-record store (`RunRow`) sits
    above this, carrying framework verdicts derived from these
    leaves.

    `id` matches the corresponding `RunRow.id` so the two stores
    join on a single column. Linking by UUID (not by hypothesis
    name as v9 did) lets two cells of the same hypothesis remain
    distinguishable.

    Two backends, one TraceRow:

    - `leaves` — scalars + 1-D per-step series, persisted to
      parquet. Path-keyed: configurational leaves at dotted
      topology paths (`bootstrap.gamma`, `optimizer.inner.lr`);
      per-step trajectories at flat author-chosen return-dict
      keys (`reward`, `loss`, `online_max_q_per_step`). Scalar
      columns persist as `Float64`/`Int64`/`Utf8`/`Boolean`;
      1-D trajectory columns persist as `List[<scalar>]`. (The
      `TraceLeaf` type technically allows deeper nesting for
      backward-compat with pre-zarr trace stores; new code should
      use `arrays` for 2-D+.)
    - `arrays` — multi-dim numpy arrays, persisted to zarr keyed
      by `{cell_id}/{array_name}`. Used for full Q-tensors,
      sample-index matrices, eval-burst trajectories — anything
      shaped `(steps, batch, n_actions)` or similar. Lazy-loaded
      from disk on access; in-memory rows constructed from
      `run_dqn_arm` carry materialised numpy arrays directly.

    No `evidence__` / `binding__` namespace prefixes: paths
    encode origin via topology (dotted) vs. author-key (flat)."""
    id: str
    cycle_id: str | None
    timestamp: str
    leaves: Mapping[str, TraceLeaf] = field(
        default_factory=lambda: {},
    )
    arrays: Mapping[str, ArrayLeaf] = field(
        default_factory=lambda: {},
    )

    def as_dict(self) -> dict[str, object]:
        """Flat top-level dict for parquet: provenance fields +
        each leaf-path becomes its own top-level key. The writer
        feeds this directly to `pl.DataFrame` — no JSON wrapping,
        no nested structs.

        `arrays` are NOT included in this dict — they go to zarr
        via `persistence.write_tracerows`'s `zarr_path` argument."""
        out: dict[str, object] = {
            'id': self.id,
            'cycle_id': self.cycle_id,
            'timestamp': self.timestamp,
        }
        for path, value in self.leaves.items():
            out[path] = value
        return out

    @classmethod
    def from_row_dict(
        cls,
        d: Mapping[str, object],
        *,
        arrays: Mapping[str, ArrayLeaf] | None = None,
    ) -> Self:
        """Reverse of `as_dict`: split provenance fields from
        leaf-path columns. Any column not in the typed-provenance
        set is treated as a leaf. Null-padded columns (paths the
        row didn't carry — polars fills missing columns with None
        when rows have heterogeneous keys) are skipped.

        `arrays` is supplied separately by the caller (e.g.
        `read_tracerows` reads from zarr and passes it in). If
        omitted, the resulting TraceRow has no array data — fine
        for tests / scalar-only workflows."""
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
            arrays=arrays if arrays is not None else {},
        )


def _coerce_trace_leaf(value: object) -> TraceLeaf:
    """Narrow a parquet-decoded object to `TraceLeaf` at the
    persistence boundary. Scalars pass through; lists recurse so
    arbitrary-depth nested lists (`(total_steps, batch, n_actions)`
    coming back from polars as `list[list[list[float]]]`) reconstruct
    correctly."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    if isinstance(value, list):
        return [_coerce_trace_leaf(item) for item in value]
    raise TypeError(
        f'unsupported TraceRow leaf type: {type(value).__name__}',
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


# ============ FactRow — typed projection of a BridgeResult ============

@dataclass(frozen=True, slots=True)
class FactRow:
    """One per-bridge / per-invariant fact attached to a hypothesis
    comparison. The typed projection of a `BridgeResult` after a
    cell evaluates its bridges + composition-discovered invariants.

    Carries the verdict-oriented information `compute_R_info` and
    the redundancy primitive will read:

    - `name` — bridge / invariant identifier (matches
      `BridgeResult.name`).
    - `reads` — leaf record-key fingerprint (the union of the
      bridge's `targets` and the transitive `reads` of any
      registered measurables it consumes via `evaluate_with_
      measurables`). Computed by `aggregate.fact_from_bridge_
      result`.
    - `verdict` — strongly-typed `Verdict`.
    - `natural_strength` — continuous evidence strength in [0, 1],
      derived from `BridgeResult.stats` (ρ, partial correlations,
      threshold margins, ATE — whatever the bridge reports).
    - `delta_i` — verdict-oriented information gain in bits,
      `1 - H₂(q_oriented)` where `q_oriented = 0.5 ± 0.5 *
      natural_strength` depending on verdict polarity.
    - `evidentiary_level` — coarse causal-tier label
      ('correlational' / 'causal_one_sided' / 'refuted').
    - `refutation_class` — `RefutationClass | None` (only set on
      REJECT-style verdicts where the comparison-level diagnostic
      could distinguish null vs underpowered)."""
    name: str
    reads: frozenset[str]
    verdict: Verdict
    natural_strength: float
    delta_i: float
    evidentiary_level: str
    refutation_class: RefutationClass | None = None


# ============ GroupStats — per-stratum summary ============

@dataclass(frozen=True, slots=True)
class GroupStats:
    """Per-stratum (paired Hedges' g + se + verdict) summary, one
    per `group_by`-value when `HypothesisComparisonRow.from_cells`
    runs in stratified mode.

    `group_value` carries whatever value the `group_by` column had
    for this stratum (e.g. `'CartPole-v1'` when `group_by=
    'env_name'`). Heterogeneous Python types are intentional —
    different substrates use different group identities."""
    group_value: object
    n_pairs: int
    arm_a_mean: float | None
    arm_a_sd: float | None
    arm_b_mean: float | None
    arm_b_sd: float | None
    effect_size_g: float | None
    se: float | None
    derived_q: float | None
    delta_i: float
    verdict: Verdict
    refutation_class: RefutationClass | None
    adequately_powered: bool


# ============ HypothesisComparisonRow — canonical aggregator ============

@dataclass(frozen=True, slots=True)
class HypothesisComparisonRow:
    """The canonical per-hypothesis comparison row. Materialized by
    `from_cells` from per-cell `RunRow`s; never authored by hand.

    Compresses the per-(env, leaf-sig) `paired_comparison_from_
    runs` + cross-env `random_effects_summary` thread into one
    object. When `group_by` is None, single-group mode: per-arm
    stats + Hedges' g over the paired Δ distribution. When
    `group_by` is set, stratified mode: `per_group` carries one
    `GroupStats` per stratum and `pooled` carries the random-
    effects pooled summary; the row's top-level `effect_size_g`
    mirrors `pooled.pooled_g`.

    `pair_by` and `group_by` are recorded on the row so consumers
    know how the aggregation was performed.

    `facts` is the deduped union of bridge / invariant facts
    across treatment cells (one FactRow per name; verdict folded
    by majority, natural_strength by mean). `reads_set` is the
    union of fact reads — the input to the redundancy primitive."""
    id: str
    parent_id: str | None
    cycle_id: str | None
    timestamp: str
    intervention_name: str
    treatment_run_ids: tuple[str, ...]
    baseline_run_ids: tuple[str, ...]
    predicted_direction: Direction | None
    pair_by: tuple[str, ...]
    group_by: str | None

    # Single-group / overall stats.
    arm_a_n: int
    arm_a_mean: float | None
    arm_a_sd: float | None
    arm_b_n: int
    arm_b_mean: float | None
    arm_b_sd: float | None
    effect_size_g: float | None
    se: float | None
    derived_q: float | None
    delta_i_population: float
    adequately_powered: bool
    verdict: Verdict
    refutation_class: RefutationClass | None

    # Stratified mode (empty / None when group_by is None).
    per_group: tuple[GroupStats, ...]
    pooled: PooledStats | None

    # Fact union + reads.
    facts: tuple[FactRow, ...]
    reads_set: frozenset[str]

    # Diagnostics.
    n_dropped_unpaired: int

    @classmethod
    def from_cells(
        cls,
        h: 'Hypothesis[Mapping[str, object]]',
        treatment_runs: Sequence['RunRow'],
        baseline_runs: Sequence['RunRow'],
        *,
        outcome_path: str,
        pair_by: tuple[str, ...],
        group_by: str | None = None,
        alpha: float = 0.05,
        power: float = 0.8,
        cycle_id: str | None = None,
        timestamp: str | None = None,
    ) -> 'HypothesisComparisonRow':
        """Canonical constructor — never call `__init__` directly.
        Delegates to `corroborate.aggregate.
        hypothesis_comparison_from_cells` (lazy import avoids the
        schema → aggregate cycle).

        See `hypothesis_comparison_from_cells` for parameter
        semantics."""
        from corroborate.aggregate import (
            hypothesis_comparison_from_cells,
        )
        return hypothesis_comparison_from_cells(
            h, treatment_runs, baseline_runs,
            outcome_path=outcome_path,
            pair_by=pair_by, group_by=group_by,
            alpha=alpha, power=power,
            cycle_id=cycle_id, timestamp=timestamp,
        )


