"""Shared infrastructure for trajectory-resolved mediation primitives.

Houses the pieces that `dynamic_partial_spearman` (closed-form
partial-correlation magnitude) and `dynamic_pc_adjacency` (PC-style
Fisher-z CI test) BOTH need:

  - `TimeAggregationStatus` enum — the trajectory analogue of
    `mediation_dowhy`'s `LinearityStatus`. Surfaces "burst-pool
    aggregate is incoherent on this trajectory" as a typed value
    rather than a runtime gotcha.
  - `_classify_status` — the sign-flip / weak-time-varying /
    underpowered classifier with noise-floor handling. The primitive
    that paired the trajectory with the classifier (the partial-
    Spearman primitive) provides the empirical motivation; the PC-
    based primitive reuses the same classifier driven by its own
    `rho_marginal[b]` trajectory.
  - `_encode_arm` — sorted-unique str-to-int code (Spearman ρ is
    invariant under monotone transform; only the *partition*
    matters).
  - `_as_float_list` / `_resolve_per_burst` / `_source_name` —
    cell-record → per-burst array adapter. Mirrors the static
    `partial_spearman`'s cache-first dispatch pattern via
    `evaluate_per_burst_source`.
  - `_stratum_key` — stratify-by tuple builder.
  - Type aliases (`Stratum`, `_PerBurstMeasurable`,
    `_ColumnOrMeasurable`).

This module has no public API surface; primitives consume it
internally. The package `__init__.py` exposes only the typed result
dataclasses + the `@analysis`-decorated primitives + the shared
`TimeAggregationStatus` enum / `Stratum` type alias.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from enum import Enum, auto

import numpy as np
import numpy.typing as npt

from corroborate.analyses._cell_value import evaluate_per_burst_source
from corroborate.measurables import Measurable


type _PerBurstMeasurable = Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
]
type _ColumnOrMeasurable = str | _PerBurstMeasurable


# Stratum identity is a hashable tuple of the values at
# `stratify_by` columns, in declaration order. `tuple[object,
# ...]` is the upper bound because polars cells can carry str /
# int / float at stratify keys (e.g. `env_name: str`, `gamma:
# float`).
type Stratum = tuple[object, ...]


class TimeAggregationStatus(Enum):
    """Diagnostic enum for trajectory-resolved mediation.

    The trajectory analogue of `mediation_dowhy`'s
    `LinearityStatus`. Surfaces "burst-pool aggregate is incoherent
    on this trajectory" as a typed value rather than a runtime
    gotcha. Consumers gate their verdict on this status before
    reading aggregated outputs.

    Shared between `dynamic_partial_spearman` (where the trajectory
    is the per-burst marginal Spearman ρ) and `dynamic_pc_adjacency`
    (where the trajectory is the per-burst Fisher-z partial-
    correlation ρ used for the CI test). The classifier operates on
    `rho_marginal[b]` in both cases — the meaning of the underlying
    quantity is primitive-specific but the burst-pool pathology
    discipline is the same."""
    CONSISTENT_DIRECTION = auto()
    """All bursts agree in sign on `rho_marginal` at non-trivial
    magnitude; aggregate is a coherent estimator of the average
    effect."""

    SIGN_FLIP_DETECTED = auto()
    """At least one burst's `rho_marginal` opposes the majority
    sign at magnitude above the noise floor
    (`sign_flip_min_abs_rho`). Aggregate is structurally suspect —
    consumers should refuse the pooled output."""

    WEAK_TIME_VARYING = auto()
    """Sign-consistent (above noise floor) but `max(|ρ|) / min(|ρ|)
    > weak_time_varying_ratio` across the non-noise-level valid
    bursts; the aggregate hides where the effect is concentrated.
    Pooled values produced but flagged."""

    UNDERPOWERED_BURSTS = auto()
    """Per-burst `n` is below `min_n_per_burst` for every burst —
    the trajectory itself is too noisy to diagnose."""


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


def _collect_arm_and_per_burst(
    cells: Sequence[Mapping[str, object]],
    *,
    arm_field: str,
    mediator_per_burst: _ColumnOrMeasurable,
    outcome_per_burst: _ColumnOrMeasurable,
) -> tuple[list[str], list[list[float]], list[list[float]]] | None:
    """Shared first-pass cell-record extractor.

    Walks `cells` once; for each cell collects (arm-tag string,
    mediator-array, outcome-array). Cells that don't have a string
    arm value or whose mediator / outcome resolves to None
    (shape mismatch) are silently dropped — the calling primitive's
    stratum loop sees only well-shaped rows.

    Returns the three parallel lists, or None when nothing valid
    was collected. Returning None here lets the stratum-level
    `_compute_one_stratum` short-circuit cleanly (the framework
    refuses to silently emit per-burst NaN for an empty stratum)."""
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
    return arms, mediator_lists, outcome_lists


def _n_bursts(
    mediator_lists: Sequence[Sequence[float]],
    outcome_lists: Sequence[Sequence[float]],
) -> int:
    """Ragged-tail burst-axis length: max trajectory length across
    cells. Cells with shorter trajectories contribute only their
    prefix; the per-burst NaN filter naturally excludes them past
    their tail.

    Picks max(mediator_len, outcome_len) per cell to handle the
    rare case where the two are unaligned (substrate bug or
    Measurable-fallback corner case); in well-formed input the two
    are equal per cell."""
    return max(
        max(len(m), len(o))
        for m, o in zip(mediator_lists, outcome_lists)
    )


def _gather_burst_b(
    arm_codes: npt.NDArray[np.float64],
    mediator_lists: Sequence[Sequence[float]],
    outcome_lists: Sequence[Sequence[float]],
    b: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Collect (arm, outcome, mediator) triples at burst index `b`
    across cells. Cells whose trajectory is shorter than `b + 1`
    don't have an entry (treated as missing, NOT NaN
    propagation); cells with NaN at either per-burst value are
    skipped. Returns three parallel float64 arrays of equal
    length `n_b`."""
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []
    for i in range(len(arm_codes)):
        if b >= len(outcome_lists[i]) or b >= len(mediator_lists[i]):
            continue
        yv = outcome_lists[i][b]
        zv = mediator_lists[i][b]
        if math.isnan(yv) or math.isnan(zv):
            continue
        xs.append(float(arm_codes[i]))
        ys.append(yv)
        zs.append(zv)
    return (
        np.asarray(xs, dtype=np.float64),
        np.asarray(ys, dtype=np.float64),
        np.asarray(zs, dtype=np.float64),
    )


__all__ = [
    'Stratum',
    'TimeAggregationStatus',
    '_ColumnOrMeasurable',
    '_PerBurstMeasurable',
    '_as_float_list',
    '_classify_status',
    '_collect_arm_and_per_burst',
    '_encode_arm',
    '_gather_burst_b',
    '_n_bursts',
    '_resolve_per_burst',
    '_source_name',
    '_stratum_key',
]
