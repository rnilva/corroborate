"""Schema — typed row dataclasses for corroborate's corpus.

Two persistence stores:

- `RunRow` — per-cell evidence (one (env, seed) execution).
  Source of truth.
- `TraceRow.leaves` — scalars + N-D series, persisted to parquet
  (one column per leaf path, polars-queryable). Multi-dim
  arrays live in nested-list columns; the streaming reader
  (`iter_trace_records`) keeps memory bounded regardless of
  inner shape, so the older zarr backend was subtracted.

The cross-arm aggregation surface (paired Hedges' g + per-group +
pooled) lives in `corroborate.analyses.paired_comparison`, not
here — the analysis is ephemeral / not persisted.

Each row splits into a **framework-typed surface** (closed-set
enums, lineage IDs, framework-controlled provenance) and an
**open surface** — `measurements: Mapping[str, MeasurementLeaf]`,
path-keyed scalar leaves. HPs land at dotted topology paths
(`gamma`, `optimizer.inner.lr`); bridge/invariant results at
`bridge.<name>.verdict`/`bridge.<name>.stats.<key>` and
`invariant.<name>.verdict`; substrate-named outcome reductions
under their own keys (e.g. `late_window_mean`).

No JSON-wrapped struct columns, no `evidence__`/`binding__`
namespace prefixes. Persistence is flat columnar parquet — every
measurement is its own typed column, queryable directly.

`as_dict()` returns a flat top-level dict (provenance/typed
fields plus each measurement at top-level, unprefixed).
`from_row_dict(d)` reverses: provenance fields by name, the rest
into `measurements`. Skip None-valued columns (polars null-pads
when rows have heterogeneous keys)."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from typing import Self

import numpy as np
import numpy.typing as npt

from corroborate._internals.narrow import (
    optional_str,
    require_str,
    require_verdict,
)
from corroborate.bridge.verdict import Verdict


# ============ Measurement leaf type ============

type MeasurementLeaf = str | int | float | bool
"""One scalar measurement on a row. The framework persists each
leaf as its own typed parquet column."""


# ============ TraceRow — raw per-cell observation store ============

# A trace leaf is a scalar (leaf value, summary scalar), a numpy
# ndarray (any dim, write-side: cell_runner emits these directly
# from JAX so polars infers narrow dtype — `List(Float32)`,
# `List(Int32)` — at DataFrame construction), or a list of trace
# leaves at any nesting depth (read-side: parquet decodes
# `List[...]` columns back to nested Python lists).
#
# Both shapes coexist: in-memory rows from `cell_runner` carry
# ndarrays; rows reconstructed via `from_row_dict` after a parquet
# round-trip carry lists. Consumers that want a uniform shape
# should `np.asarray(...)` on access.

type TraceLeaf = (
    str | int | float | bool
    | npt.NDArray[np.floating] | npt.NDArray[np.integer] | npt.NDArray[np.bool_]
    | Sequence[TraceLeaf]
)
# `Sequence[TraceLeaf]` (covariant) rather than `list[TraceLeaf]`
# (invariant) so that `list[list[float]]` and `list[list[list[
# float]]]` and so on can be assigned through the recursive shape
# without requiring callers to up-cast each layer. polars decodes
# nested-list columns to plain Python `list`s, which satisfy
# `Sequence`; numpy ndarrays already cover the dense-array branch.


@dataclass(frozen=True, slots=True)
class TraceRow:
    """One cell's raw observation: leaves + provenance.

    The trace store is the v9-`traces.parquet` analog: low-
    derivation, queryable, re-usable for post-hoc bridge
    re-evaluation. The hypothesis-record store (`RunRow`) sits
    above this, carrying framework verdicts derived from these
    leaves.

    `id` matches the corresponding `RunRow.id` so the two stores
    join on a single column. Linking by UUID (not by hypothesis
    name as v9 did) lets two cells of the same hypothesis remain
    distinguishable.

    `leaves` — scalars + N-D series, all persisted to parquet via
    polars' nested-list columns. Path-keyed: configurational
    leaves at dotted topology paths (`bootstrap.gamma`,
    `optimizer.inner.lr`); per-step trajectories at flat author-
    chosen return-dict keys (`reward`, `loss`,
    `online_max_q_per_step`, `pearson_stats`). Scalar columns
    persist as `Float32/64`/`Int32/64`/`Utf8`/`Boolean`; 1-D
    trajectory columns as `List[<scalar>]`; 2-D+ arrays as
    nested `List[List[...]]` or `List[Array(<scalar>, shape=N)]`
    when polars detects a uniform inner shape.

    No `evidence__` / `binding__` namespace prefixes: paths
    encode origin via topology (dotted) vs. author-key (flat)."""
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
    def from_row_dict(
        cls,
        d: Mapping[str, object],
    ) -> Self:
        """Reverse of `as_dict`: split provenance fields from
        leaf-path columns. Any column not in the typed-provenance
        set is treated as a leaf. Null-padded columns (paths the
        row didn't carry — polars fills missing columns with None
        when rows have heterogeneous keys) are skipped."""
        leaves: dict[str, TraceLeaf] = {}
        for k, v in d.items():
            if k in _TRACE_ROW_TYPED_FIELDS:
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
    aggregate `verdict`, `arm_key` (the canonical fingerprint of
    the run's intervention arms — load-bearing identity for
    pairing). Open surface: `measurements` carrying HP values at
    dotted topology paths, bridge/invariant results under
    `bridge.<name>.*` / `invariant.<name>.*`, outcome reductions
    under substrate-named keys (e.g. `late_window_mean`),
    and substrate metadata (`env_name`, `seed`, `total_steps`,
    `intervention_name`).

    `arm_key` defaults to `'baseline'` so hand-constructed
    fixtures and old parquets without the column read as the
    baseline arm. Production write paths populate it from
    `Hypothesis.arm_key()`.

    Older parquets may carry a `claim_graph_signature` column
    (legacy program-structural fingerprint); `from_row_dict`
    silently absorbs it into `measurements` if encountered. The
    column is no longer authored — the per-cell program identity
    is derivable from the leaves the path-keyed convention already
    flattens, and a denormalised hash didn't earn its keep."""
    id: str
    parent_id: str | None
    cycle_id: str | None
    timestamp: str
    verdict: Verdict
    arm_key: str = 'baseline'
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
            'arm_key': self.arm_key,
        }
        _flatten_measurements(out, self.measurements)
        return out

    @classmethod
    def from_row_dict(cls, d: Mapping[str, object]) -> Self:
        measurements: dict[str, MeasurementLeaf] = {}
        for k, v in d.items():
            if k in _RUN_ROW_TYPED_FIELDS or k == 'claim_graph_signature':
                continue
            if v is None:
                continue
            # `RunRow` is the *scalar* row store; per-burst trace
            # arrays (`mc_return`, `predicted_q_at_start`, …) joined
            # in by the runner are list-typed and don't belong here.
            # Skipping them keeps RunRow's measurements scalar and
            # leaves the array form available on the source DataFrame
            # for analyses that consume it directly.
            if isinstance(v, list):
                continue
            measurements[k] = _coerce_measurement_leaf(v)
        arm_key_v = d.get('arm_key')
        arm_key = arm_key_v if isinstance(arm_key_v, str) else 'baseline'
        return cls(
            id=require_str(d, 'id'),
            parent_id=optional_str(d, 'parent_id'),
            cycle_id=optional_str(d, 'cycle_id'),
            timestamp=require_str(d, 'timestamp'),
            verdict=require_verdict(d, 'verdict'),
            arm_key=arm_key,
            measurements=measurements,
        )


# ============ Lineage / typed-field constants ============

# Typed dataclass field names on each row class. Auto-derived
# (single source of truth — adding a typed field to RunRow / TraceRow
# automatically updates the `from_row_dict` skip set, the runner's
# content-equality dedup, and any downstream consumer that imports
# these constants).
_TRACE_ROW_TYPED_FIELDS: frozenset[str] = frozenset(
    f.name for f in fields(TraceRow) if f.name != 'leaves'
)
_RUN_ROW_TYPED_FIELDS: frozenset[str] = frozenset(
    f.name for f in fields(RunRow) if f.name != 'measurements'
)

# Lineage subset across both row classes — UUIDs, lineage IDs,
# timestamp. Provenance bookkeeping that the runner ignores when
# checking whether two rows describe the same scientific cell
# (`runner._dedup_by_content`).
LINEAGE_FIELDS: frozenset[str] = (
    _TRACE_ROW_TYPED_FIELDS | _RUN_ROW_TYPED_FIELDS
) - {'verdict', 'arm_key'}


# ============ StratumG[K] — parametric per-stratum effect size ============

@dataclass(frozen=True, slots=True)
class StratumG[K]:
    """Per-stratum paired Hedges' g result. `K` is the stratum-id
    type — `str` for env-strata, `tuple[str, int]` for (env,
    burst), or whatever the substrate uses.

    Per-stratum analyses (`paired_g_pooled`, `paired_g_per_burst`,
    panel meta-regression) produce sequences of these. The
    sibling `StratumObservation` in `corroborate.stats.meta_regression`
    extends the shape with a `covariates: Mapping[str, float]`
    field for covariate-bearing meta-regression."""
    stratum_id: K
    g: float
    se: float
    n_pairs: int


# Convenience aliases for the two common shapes.
type PerEnvG = StratumG[str]
type PerBurstStratum = StratumG[tuple[str, int]]


