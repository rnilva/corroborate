"""Trajectory-resolved partial Spearman mediation — `dynamic_partial_spearman`.

Sibling to `corroborate.analyses.spearman.partial_spearman`. Where
the static primitive iterates one observation per (cell, burst)
and Fisher-z-pools across strata, this primitive iterates one ρ
per BURST INDEX and surfaces the trajectory plus a
`TimeAggregationStatus` enum that flags three pathologies of
burst-pooled aggregate mediation:

  - **SIGN_FLIP_DETECTED** — `ρ(arm, outcome)` flips sign across
    bursts AT NON-TRIVIAL MAGNITUDE. The Fisher-z burst-pool of
    opposing-sign per-burst ρ's is a Simpson's-paradox artifact;
    the primitive returns NaN for BOTH `rho_marginal_pooled` and
    `rho_partial_pooled` to force consumers off the aggregate
    (the partial is computed on the same suspect bursts as the
    marginal — if the marginal is incoherent, the partial inherits
    that). Sign-flips at noise-level |ρ| (below
    `sign_flip_min_abs_rho`) are treated as sampling noise, NOT
    as flips.
  - **WEAK_TIME_VARYING** — sign-consistent but
    `max(|ρ|) / min(|ρ|) > weak_time_varying_ratio` across
    NON-NOISE-LEVEL bursts (|ρ| ≥ `sign_flip_min_abs_rho`);
    the aggregate hides where the effect is concentrated.
  - **UNDERPOWERED_BURSTS** — every burst has `n < min_n_per_burst`;
    diagnosis itself is unreliable, but aggregates are produced
    on a best-effort basis.

Inputs are PER-BURST `List(Float64)` columns on `cells` (a
`pl.DataFrame`), produced by the substrate's `_per_burst`
measurables (see e.g. `bootstrap_gap_magnitude_per_burst` in
`corroborate_rl.dqn.measurables`). Each cell carries an array of
length `n_bursts` at the named columns. The primitive aligns by
burst index across cells within a stratum; cells with shorter
trajectories still contribute their prefix (the per-burst valid
count `n_per_burst[b]` grows as the longer cells continue
contributing past shorter cells' tails — "ragged tail"
semantics, the less-information-losing form vs truncating all
cells to the shortest stratum length).

Granularity contract:
  - `mediator_per_burst`, `outcome_per_burst`: str column names on
    `cells` carrying `List[Float64]` of length `n_bursts`, OR
    `Measurable[..., NDArray]` instances (mirroring
    `partial_spearman`'s lazy-evaluation pattern — the framework
    reads the Measurable's cached column if present, else
    evaluates against the cell record). Scalar columns trigger a
    structural raise via the `_as_float_list` shape contract.
  - `arm_field`: str column name on `cells` carrying a per-cell
    string arm tag. The arm is numerically encoded (sorted-unique
    → integer code) for the rank-based Spearman; only the
    *partition* between arms matters for ρ.

Per-burst computation: at burst index `b`, collect one (arm,
outcome[b], mediator[b]) row per cell; drop cells where any of the
three is NaN or missing the b-th entry. With `n_b ≥ min_n_per_burst`
the primitive computes:
  - `rho_marginal[b]` via `graph.discovery._spearman_marginal`
  - `rho_partial[b]` via `graph.discovery.partial_spearman_rho`
    (closed-form first-order partial Spearman)

Bursts that fail the floor contribute NaN to the trajectory; the
aggregate Fisher-z pool skips them. Pool weights: `(n_per_burst[b]
− 3)` for `rho_marginal_pooled` (matches `stratified_spearman_rho`
weighting) and `(n_per_burst[b] − 4)` for `rho_partial_pooled`
(matches `stratified_partial_spearman_rho`, closed-form first-
order partial df).

This is the canonical primitive for any mediation analysis on RL
training trajectories. The static `partial_spearman` should NOT be
used on per-burst data when burst dynamics are non-monotonic —
see `findings_per_burst_mediation_trajectory` and
`DYNAMIC_MEDIATION_DESIGN.md` for the empirical motivation.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
import numpy.typing as npt
import polars as pl

from corroborate._internals.polars import to_dicts as _to_dicts
from corroborate.analyses._cell_value import evaluate_per_burst_source
from corroborate.bridge.analysis import analysis
from corroborate.graph.discovery import (
    _spearman_marginal as _graph_spearman_marginal,  # pyright: ignore[reportPrivateUsage]
    partial_spearman_rho,
)
from corroborate.measurables import Measurable
from corroborate.stats import fisher_z_pool


type _PerBurstMeasurable = Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
]
type _ColumnOrMeasurable = str | _PerBurstMeasurable


class TimeAggregationStatus(Enum):
    """Diagnostic enum for trajectory-resolved mediation.

    The trajectory analogue of `mediation_dowhy`'s
    `LinearityStatus`. Surfaces "burst-pool aggregate is incoherent
    on this trajectory" as a typed value rather than a runtime
    gotcha. Consumers gate their verdict on this status before
    reading `rho_*_pooled`."""
    CONSISTENT_DIRECTION = auto()
    """All bursts agree in sign on `rho_marginal` at non-trivial
    magnitude; aggregate is a coherent estimator of the average
    effect."""

    SIGN_FLIP_DETECTED = auto()
    """At least one burst's `rho_marginal` opposes the majority
    sign at magnitude above the noise floor
    (`sign_flip_min_abs_rho`). Both pooled estimates are NaN by
    construction — the pool over sign-opposing bursts is a
    Simpson's-paradox artifact, and the partial inherits the
    same suspect support."""

    WEAK_TIME_VARYING = auto()
    """Sign-consistent (above noise floor) but `max(|ρ|) / min(|ρ|)
    > weak_time_varying_ratio` across the non-noise-level valid
    bursts; the aggregate hides where the effect is concentrated.
    Pooled values produced but flagged."""

    UNDERPOWERED_BURSTS = auto()
    """Per-burst `n` is below `min_n_per_burst` for every burst —
    the trajectory itself is too noisy to diagnose."""


# Stratum identity is a hashable tuple of the values at
# `stratify_by` columns, in declaration order. `tuple[object,
# ...]` is the upper bound because polars cells can carry str /
# int / float at stratify keys (e.g. `env_name: str`, `gamma:
# float`).
type Stratum = tuple[object, ...]


@dataclass(frozen=True, slots=True)
class DynamicMediationResult:
    """Trajectory-resolved partial Spearman mediation result.

    Parallel shape to `PartialSpearmanResult` but per-burst-shaped.
    `burst_steps` is the burst-index axis (0..n_bursts-1); the
    framework doesn't try to recover wall-clock training-step
    counts at the primitive level (that would require trace-store
    introspection that's substrate-specific).

    `rho_marginal[b]` and `rho_partial[b]` are NaN at bursts where
    fewer than `min_n_per_burst` cells contributed — consumers
    must filter NaN when reading the trajectory.

    Both `rho_marginal_pooled` AND `rho_partial_pooled` are NaN
    when `aggregation_status` is SIGN_FLIP_DETECTED — the pool
    over sign-opposing bursts is a structural Simpson's-paradox
    artifact, not an estimate. The partial pool inherits the same
    suspect support as the marginal, so the framework refuses to
    report either. For WEAK_TIME_VARYING / UNDERPOWERED_BURSTS
    the pools are still computed (best-effort) but the status flag
    warns consumers that the aggregate may not represent the
    trajectory."""
    burst_steps: tuple[int, ...]
    rho_marginal: tuple[float, ...]
    rho_partial: tuple[float, ...]
    n_per_burst: tuple[int, ...]
    rho_marginal_pooled: float
    rho_partial_pooled: float
    aggregation_status: TimeAggregationStatus
    mediator_name: str
    outcome_name: str
    arm_field: str

    @property
    def n_bursts(self) -> int:
        return len(self.burst_steps)


def _marginal_spearman(
    x: npt.NDArray[np.float64], y: npt.NDArray[np.float64],
) -> float:
    """Marginal Spearman ρ. Thin wrapper around
    `graph.discovery._spearman_marginal` that drops the p-value
    (we only need ρ here; the Fisher-z pool computes its own pooled
    p-stat). Kept as a private helper to centralize the import and
    return-shape narrowing in one place."""
    rho, _ = _graph_spearman_marginal(x, y)
    return rho


def _encode_arm(arms: Sequence[str]) -> npt.NDArray[np.float64]:
    """Map a sequence of string arm labels to a float64 vector of
    integer codes via sorted-unique. Spearman ρ is invariant under
    monotone transformations of either variable, so the specific
    encoding doesn't matter — only the *partition* between arms.

    Sorted-unique keeps the encoding deterministic across
    re-orderings of the cells list (vs `dict.fromkeys` which leaks
    insertion order)."""
    unique = sorted(set(arms))
    code: dict[str, int] = {a: i for i, a in enumerate(unique)}
    return np.asarray([code[a] for a in arms], dtype=np.float64)


def _stratum_key(
    cell: Mapping[str, object], stratify_by: tuple[str, ...],
) -> Stratum | None:
    """Build the stratum-key tuple for `cell`. Returns None when
    any key is missing or null — the cell is dropped from
    analysis (matches the static primitive's behaviour)."""
    key: list[object] = []
    for k in stratify_by:
        if k not in cell:
            return None
        v = cell[k]
        if v is None:
            return None
        key.append(v)
    return tuple(key)


def _as_float_list(value: object) -> list[float] | None:
    """Coerce a per-burst column cell to a list of floats. The
    column is `List(Float64)` after polars `to_dicts`; the cell
    value is therefore a `list[float | None]`. Returns None when
    the value isn't list-shaped (silent structural mismatch — the
    bridge author passed a scalar column name where per-burst was
    expected)."""
    if not isinstance(value, list):
        return None
    # Cells inside the list may be None (polars null inside list) —
    # surface as NaN so the burst-level NaN filter sees them.
    # `list[object]` upper bound on the polars list-cell value
    # because polars stores heterogeneous null+float arrays.
    items: list[object] = list(value)
    out: list[float] = []
    for v in items:
        if v is None:
            out.append(float('nan'))
        elif isinstance(v, bool):
            out.append(float(v))
        elif isinstance(v, (int, float)):
            out.append(float(v))
        else:
            out.append(float('nan'))
    return out


def _resolve_per_burst(
    cell: Mapping[str, object],
    source: _ColumnOrMeasurable,
) -> list[float] | None:
    """Resolve a per-burst source to `list[float]`, dispatching on
    whether the caller passed a column name (str) or a Measurable
    instance. Mirrors `partial_spearman`'s lazy-evaluation pattern:

      - str → read the named `List(Float64)` column from the cell
        record via `_as_float_list`.
      - Measurable → cache-first via `evaluate_per_burst_source`;
        falls back to evaluating the Measurable against the raw
        record if the cache column isn't present.

    Returns None on shape mismatch so the calling stratum-loop can
    skip the cell silently — the same behaviour as the column-name
    path's `_as_float_list` returning None for non-list inputs."""
    if isinstance(source, str):
        return _as_float_list(cell.get(source))
    arr = evaluate_per_burst_source(source, cell)
    if arr.size == 0:
        return None
    return [float(v) for v in arr]


def _source_name(source: _ColumnOrMeasurable) -> str:
    """Stable provenance label for a per-burst source — the column
    name (str input) or the Measurable's `.name` attribute."""
    return source if isinstance(source, str) else source.name


def _classify_status(
    rho_marginal: Sequence[float],
    n_per_burst: Sequence[int],
    min_n_per_burst: int,
    weak_time_varying_ratio: float,
    sign_flip_min_abs_rho: float,
) -> TimeAggregationStatus:
    """Determine the `TimeAggregationStatus` from the trajectory.

    Order of checks matters:
      1. UNDERPOWERED_BURSTS — every burst below min n.
      2. SIGN_FLIP_DETECTED — at least one valid burst has sign
         opposite to the majority of valid bursts, with both
         opposing-sign and majority-sign bursts at |ρ| >=
         `sign_flip_min_abs_rho`. Bursts at noise-level magnitude
         are excluded from the sign analysis — opposite signs at
         |ρ| ≈ 0 are sampling noise, not structural flips.
      3. WEAK_TIME_VARYING — sign-consistent but |ρ| varies more
         than `weak_time_varying_ratio` across NON-NOISE-LEVEL
         valid bursts (|ρ| ≥ `sign_flip_min_abs_rho`). Excluding
         noise bursts makes the ratio robust to a single near-zero
         burst inflating the max/min spread.
      4. CONSISTENT_DIRECTION — otherwise.

    Bursts with NaN ρ or `n < min_n_per_burst` are excluded from
    the sign/magnitude analysis; they're already absent from
    the aggregate pool."""
    valid_rhos: list[float] = [
        r for r, n in zip(rho_marginal, n_per_burst)
        if not math.isnan(r) and n >= min_n_per_burst
    ]
    if not valid_rhos:
        return TimeAggregationStatus.UNDERPOWERED_BURSTS

    # Sign-flip detection at the noise floor: a burst only counts
    # as evidence of a flip if its |ρ| exceeds the noise threshold.
    above_floor = [r for r in valid_rhos if abs(r) >= sign_flip_min_abs_rho]
    n_pos = sum(1 for r in above_floor if r > 0)
    n_neg = sum(1 for r in above_floor if r < 0)
    if n_pos > 0 and n_neg > 0:
        # Both signs present among non-noise-level bursts — sign-
        # flip regardless of which dominates. The aggregate is
        # structurally suspect even if one direction dominates 9:1.
        return TimeAggregationStatus.SIGN_FLIP_DETECTED

    # Sign-consistent path (within the noise floor). Check
    # magnitude variation across non-noise bursts only — a single
    # near-zero burst shouldn't drag the framework into
    # WEAK_TIME_VARYING when the rest of the trajectory is
    # well-behaved.
    if len(above_floor) < 2:
        # One or zero bursts above the noise floor — no magnitude
        # trajectory to flag. Includes the "all-noise" case, which
        # is CONSISTENT_DIRECTION by default (the noise IS
        # consistent in shape, even if uninformative).
        return TimeAggregationStatus.CONSISTENT_DIRECTION
    abs_rhos = [abs(r) for r in above_floor]
    rho_max = max(abs_rhos)
    rho_min = min(abs_rhos)
    if rho_min > 0.0 and rho_max / rho_min > weak_time_varying_ratio:
        return TimeAggregationStatus.WEAK_TIME_VARYING
    return TimeAggregationStatus.CONSISTENT_DIRECTION


@analysis
def dynamic_partial_spearman(
    cells: pl.DataFrame,
    *,
    arm_field: str = 'arm_key',
    mediator_per_burst: _ColumnOrMeasurable,
    outcome_per_burst: _ColumnOrMeasurable,
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    min_n_per_burst: int = 5,
    weak_time_varying_ratio: float = 2.0,
    sign_flip_min_abs_rho: float = 0.05,
) -> Mapping[Stratum, DynamicMediationResult]:
    """Trajectory-resolved partial Spearman mediation per stratum.

    For each stratum (tuple of values at `stratify_by` columns):
    align the per-burst arrays across cells, compute per-burst
    marginal Spearman `ρ(arm, outcome_per_burst[b])` and partial
    Spearman `ρ(arm, outcome | mediator)[b]` via the closed-form
    first-order partial. Pool across bursts with Fisher-z, weighted
    by `(n_per_burst[b] − 3)` for the marginal pool and
    `(n_per_burst[b] − 4)` for the partial pool (closed-form
    partial df). Compute the `TimeAggregationStatus` from the
    trajectory.

    `min_n_per_burst` floors each burst's per-cell count; bursts
    below the floor contribute NaN to the trajectory and are
    excluded from the pool. `weak_time_varying_ratio` is the
    `max(|ρ|)/min(|ρ|)` threshold across non-noise-level valid
    bursts that triggers the WEAK_TIME_VARYING status.
    `sign_flip_min_abs_rho` is the noise-floor magnitude below
    which a per-burst ρ is treated as sampling noise rather than
    structural signal — bursts below the floor neither flip nor
    drive the WEAK ratio.

    `mediator_per_burst` and `outcome_per_burst` may be column
    names (str) for cells that materialise the per-burst array as
    a `List(Float64)` column, OR `Measurable[..., NDArray]`
    instances for cells where the array is computed lazily from
    raw trace columns. The Measurable path uses
    `evaluate_per_burst_source`'s cache-first dispatch (read the
    cached column if present, else evaluate against the raw
    record) — same pattern as the static `partial_spearman`.

    Returns a `Mapping[Stratum, DynamicMediationResult]`. Strata
    where no cell contributes (missing arm tag, malformed per-burst
    columns, ...) are absent from the result.

    Both `rho_marginal_pooled` AND `rho_partial_pooled` are NaN
    when status is SIGN_FLIP_DETECTED — the pool over sign-
    opposing per-burst ρ's is a Simpson's-paradox artifact, and
    the partial inherits the same suspect support.

    The input is a `pl.DataFrame` (the canonical corpus shape after
    cache materialisation). Per-burst columns are read as
    `List(Float64)`; scalar columns named at `mediator_per_burst` /
    `outcome_per_burst` cause every cell to be dropped (the
    `_as_float_list` shape contract returns None for non-list
    inputs).

    Per-burst alignment is "ragged tail": `n_bursts` is the
    longest trajectory in the stratum, and at each burst index
    only cells whose trajectory reaches that index contribute.
    Shorter cells drop off as their trajectory ends, so
    `n_per_burst[b]` is non-increasing in `b` (subject to NaN
    handling). This is the less-information-losing semantics — vs
    truncating ALL cells to the shortest stratum length, which
    discards every late-training burst a single short cell can
    cause."""
    # Group cells by stratum. Each entry is the list of cell-dicts
    # contributing to that stratum.
    by_stratum: dict[Stratum, list[Mapping[str, object]]] = {}
    for cell in _to_dicts(cells):
        key = _stratum_key(cell, stratify_by)
        if key is None:
            continue
        by_stratum.setdefault(key, []).append(cell)

    out: dict[Stratum, DynamicMediationResult] = {}
    for stratum, stratum_cells in by_stratum.items():
        result = _compute_one_stratum(
            stratum_cells,
            arm_field=arm_field,
            mediator_per_burst=mediator_per_burst,
            outcome_per_burst=outcome_per_burst,
            min_n_per_burst=min_n_per_burst,
            weak_time_varying_ratio=weak_time_varying_ratio,
            sign_flip_min_abs_rho=sign_flip_min_abs_rho,
        )
        if result is not None:
            out[stratum] = result
    return out


def _compute_one_stratum(
    cells: Sequence[Mapping[str, object]],
    *,
    arm_field: str,
    mediator_per_burst: _ColumnOrMeasurable,
    outcome_per_burst: _ColumnOrMeasurable,
    min_n_per_burst: int,
    weak_time_varying_ratio: float,
    sign_flip_min_abs_rho: float,
) -> DynamicMediationResult | None:
    """Per-stratum core. Returns None when no per-cell row has
    valid arm + per-burst columns at all."""
    # First pass: collect per-cell (arm, mediator-array, outcome-array).
    # Drop cells that don't match the shape contract.
    arms: list[str] = []
    mediator_lists: list[list[float]] = []
    outcome_lists: list[list[float]] = []
    for cell in cells:
        arm = cell.get(arm_field)
        if not isinstance(arm, str):
            continue
        med = _resolve_per_burst(cell, mediator_per_burst)
        out_arr = _resolve_per_burst(cell, outcome_per_burst)
        if med is None or out_arr is None:
            continue
        arms.append(arm)
        mediator_lists.append(med)
        outcome_lists.append(out_arr)

    if not arms:
        return None
    unique_arms = sorted(set(arms))
    if len(unique_arms) < 2:
        # Spearman needs variation in the arm axis; a single-arm
        # stratum gives ρ=NaN at every burst. Skip.
        return None

    # Ragged-tail alignment: take the LONGEST per-cell trajectory
    # in the stratum as the burst-index axis. Cells with shorter
    # trajectories contribute only their prefix — they're absent
    # from late-burst observations and the per-burst NaN filter
    # naturally excludes them. This preserves late-burst signal
    # from longer cells (truncate-to-min would discard every burst
    # past the shortest cell's tail, which is information loss in
    # the common case of one short cell in a stratum of long ones).
    n_bursts = max(
        max(len(m), len(o))
        for m, o in zip(mediator_lists, outcome_lists)
    )
    if n_bursts == 0:
        return None

    arm_codes = _encode_arm(arms)

    rho_marg: list[float] = []
    rho_part: list[float] = []
    n_per_burst: list[int] = []
    for b in range(n_bursts):
        # Collect non-NaN rows at this burst. Cells whose
        # trajectory is shorter than `b + 1` don't have an entry —
        # treated as missing (skipped), not as NaN propagation.
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for i in range(len(arms)):
            if b >= len(outcome_lists[i]) or b >= len(mediator_lists[i]):
                continue
            yv = outcome_lists[i][b]
            zv = mediator_lists[i][b]
            if math.isnan(yv) or math.isnan(zv):
                continue
            xs.append(float(arm_codes[i]))
            ys.append(yv)
            zs.append(zv)
        n_b = len(xs)
        n_per_burst.append(n_b)
        if n_b < min_n_per_burst:
            rho_marg.append(float('nan'))
            rho_part.append(float('nan'))
            continue
        x_np = np.asarray(xs, dtype=np.float64)
        y_np = np.asarray(ys, dtype=np.float64)
        z_np = np.asarray(zs, dtype=np.float64)
        r_m = _marginal_spearman(x_np, y_np)
        r_p, _ = partial_spearman_rho(x_np, y_np, z_np)
        rho_marg.append(r_m)
        rho_part.append(r_p)

    status = _classify_status(
        rho_marg, n_per_burst, min_n_per_burst,
        weak_time_varying_ratio,
        sign_flip_min_abs_rho,
    )
    # Fisher-z pool over valid bursts. Skip-NaN is handled inside
    # `fisher_z_pool`. df_offset matches the sibling primitives:
    # 3 for marginal (`stratified_spearman_rho`), 4 for closed-
    # form first-order partial (`stratified_partial_spearman_rho`).
    marg_pool, _ = fisher_z_pool(
        rho_marg, n_per_burst, df_offset=3,
    )
    part_pool, _ = fisher_z_pool(
        rho_part, n_per_burst, df_offset=4,
    )
    if status is TimeAggregationStatus.SIGN_FLIP_DETECTED:
        # NaN BOTH aggregates: the marginal pool is the
        # Simpson's-paradox artifact directly, and the partial pool
        # inherits the same suspect support (it's computed on the
        # same per-burst (xs, ys, zs) trios). If the marginal is
        # incoherent across bursts, the partial's pool isn't a
        # trustworthy summary either — consumers must read the
        # trajectory.
        marg_pool = float('nan')
        part_pool = float('nan')

    return DynamicMediationResult(
        burst_steps=tuple(range(n_bursts)),
        rho_marginal=tuple(rho_marg),
        rho_partial=tuple(rho_part),
        n_per_burst=tuple(n_per_burst),
        rho_marginal_pooled=float(marg_pool),
        rho_partial_pooled=float(part_pool),
        aggregation_status=status,
        mediator_name=_source_name(mediator_per_burst),
        outcome_name=_source_name(outcome_per_burst),
        arm_field=arm_field,
    )


__all__ = [
    'DynamicMediationResult',
    'Stratum',
    'TimeAggregationStatus',
    'dynamic_partial_spearman',
]
