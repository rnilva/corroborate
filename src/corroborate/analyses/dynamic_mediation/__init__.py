"""Trajectory-resolved partial Spearman mediation — `dynamic_partial_spearman`.

Sibling to `corroborate.analyses.spearman.partial_spearman`. Where
the static primitive iterates one observation per (cell, burst)
and Fisher-z-pools across strata, this primitive iterates one ρ
per BURST INDEX and surfaces the trajectory plus a
`TimeAggregationStatus` enum that flags three pathologies of
burst-pooled aggregate mediation:

  - **SIGN_FLIP_DETECTED** — `ρ(arm, outcome)` flips sign across
    bursts. The Fisher-z burst-pool of opposing-sign per-burst ρ's
    is a Simpson's-paradox artifact; the primitive returns NaN for
    `rho_marginal_pooled` to force consumers off the aggregate.
  - **WEAK_TIME_VARYING** — sign-consistent but
    `max(|ρ|) / min(|ρ|) > weak_time_varying_ratio` across bursts;
    the aggregate hides where the effect is concentrated.
  - **UNDERPOWERED_BURSTS** — every burst has `n < min_n_per_burst`;
    diagnosis itself is unreliable, but aggregates are produced
    on a best-effort basis.

Inputs are PER-BURST `List(Float64)` columns on `cells` (a
`pl.DataFrame`), produced by the substrate's `_per_burst`
measurables (see e.g. `bootstrap_gap_magnitude_per_burst` in
`corroborate_rl.dqn.measurables`). Each cell carries an array of
length `n_bursts` at the named columns. The primitive aligns by
burst index across cells within a stratum.

Granularity contract:
  - `mediator_per_burst`, `outcome_per_burst`: str column names on
    `cells` carrying `List[Float64]` of length `n_bursts`. The
    primitive REQUIRES this shape; scalar columns trigger a
    structural raise.
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
aggregate Fisher-z pool skips them. Aggregate weights are
`(n_per_burst[b] − 4)` (closed-form partial Spearman df).

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

from scipy.stats import spearmanr

from corroborate._internals.polars import to_dicts as _to_dicts
from corroborate.bridge.analysis import analysis
from corroborate.graph.discovery import partial_spearman_rho
from corroborate.stats import fisher_z_pool


class TimeAggregationStatus(Enum):
    """Diagnostic enum for trajectory-resolved mediation.

    The trajectory analogue of `mediation_dowhy`'s
    `LinearityStatus`. Surfaces "burst-pool aggregate is incoherent
    on this trajectory" as a typed value rather than a runtime
    gotcha. Consumers gate their verdict on this status before
    reading `rho_*_pooled`."""
    CONSISTENT_DIRECTION = auto()
    """All bursts agree in sign on `rho_marginal`; aggregate is a
    coherent estimator of the average effect."""

    SIGN_FLIP_DETECTED = auto()
    """At least one burst's `rho_marginal` opposes the majority
    sign. The pooled estimate is a Simpson's-paradox artifact —
    `rho_marginal_pooled` is NaN by construction."""

    WEAK_TIME_VARYING = auto()
    """Sign-consistent but `max(|ρ|) / min(|ρ|) > weak_time_varying_ratio`
    across bursts; the aggregate hides where the effect is
    concentrated. Pooled values produced but flagged."""

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

    `rho_marginal_pooled` is NaN when `aggregation_status` is
    SIGN_FLIP_DETECTED — the pool over sign-opposing bursts is a
    structural Simpson's-paradox artifact, not an estimate. For
    WEAK_TIME_VARYING / UNDERPOWERED_BURSTS the pool is still
    computed (best-effort) but the status flag warns consumers
    that the aggregate may not represent the trajectory."""
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
    """Marginal Spearman ρ. Returns NaN when either side has zero
    variance or `n < 4` (the smallest n where scipy's
    `spearmanr` has a well-defined two-sided p-value). Matches the
    framework's existing `graph.discovery._spearman_marginal`
    semantics without crossing the private-name boundary."""
    if len(x) < 4 or float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float('nan')
    r, _ = spearmanr(x, y)
    return float(r)


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


def _classify_status(
    rho_marginal: Sequence[float],
    n_per_burst: Sequence[int],
    min_n_per_burst: int,
    weak_time_varying_ratio: float,
) -> TimeAggregationStatus:
    """Determine the `TimeAggregationStatus` from the trajectory.

    Order of checks matters:
      1. UNDERPOWERED_BURSTS — every burst below min n.
      2. SIGN_FLIP_DETECTED — at least one valid burst has sign
         opposite to the majority of valid bursts.
      3. WEAK_TIME_VARYING — sign-consistent but |ρ| varies more
         than `weak_time_varying_ratio` across valid bursts.
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

    n_pos = sum(1 for r in valid_rhos if r > 0)
    n_neg = sum(1 for r in valid_rhos if r < 0)
    if n_pos > 0 and n_neg > 0:
        # Both signs present among valid bursts — sign-flip
        # regardless of which dominates. The aggregate is
        # structurally suspect even if one direction dominates 9:1.
        return TimeAggregationStatus.SIGN_FLIP_DETECTED

    # Sign-consistent path. Check magnitude variation.
    abs_rhos = [abs(r) for r in valid_rhos if r != 0.0]
    if len(abs_rhos) < 2:
        # Single valid burst or all-zero — no magnitude trajectory
        # to flag.
        return TimeAggregationStatus.CONSISTENT_DIRECTION
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
    mediator_per_burst: str,
    outcome_per_burst: str,
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    min_n_per_burst: int = 5,
    weak_time_varying_ratio: float = 2.0,
) -> Mapping[Stratum, DynamicMediationResult]:
    """Trajectory-resolved partial Spearman mediation per stratum.

    For each stratum (tuple of values at `stratify_by` columns):
    align the per-burst arrays across cells, compute per-burst
    marginal Spearman `ρ(arm, outcome_per_burst[b])` and partial
    Spearman `ρ(arm, outcome | mediator)[b]` via the closed-form
    first-order partial. Pool across bursts with Fisher-z, weighted
    by `(n_per_burst[b] − 4)` (closed-form partial df). Compute the
    `TimeAggregationStatus` from the trajectory.

    `min_n_per_burst` floors each burst's per-cell count; bursts
    below the floor contribute NaN to the trajectory and are
    excluded from the pool. `weak_time_varying_ratio` is the
    `max(|ρ|)/min(|ρ|)` threshold across valid bursts that triggers
    the WEAK_TIME_VARYING status.

    Returns a `Mapping[Stratum, DynamicMediationResult]`. Strata
    where no cell contributes (missing arm tag, malformed per-burst
    columns, ...) are absent from the result.

    The aggregate `rho_marginal_pooled` is NaN when status is
    SIGN_FLIP_DETECTED — the pool over sign-opposing per-burst ρ's
    is a Simpson's-paradox artifact, not an estimate.

    The input is a `pl.DataFrame` (the canonical corpus shape after
    cache materialisation). Per-burst columns are read as
    `List(Float64)`; scalar columns named at `mediator_per_burst` /
    `outcome_per_burst` cause every cell to be dropped (the
    `_as_float_list` shape contract returns None for non-list
    inputs)."""
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
        )
        if result is not None:
            out[stratum] = result
    return out


def _compute_one_stratum(
    cells: Sequence[Mapping[str, object]],
    *,
    arm_field: str,
    mediator_per_burst: str,
    outcome_per_burst: str,
    min_n_per_burst: int,
    weak_time_varying_ratio: float,
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
        med = _as_float_list(cell.get(mediator_per_burst))
        out_arr = _as_float_list(cell.get(outcome_per_burst))
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

    # Truncate each cell's per-burst arrays to a shared minimum
    # length so all cells contribute to the same burst-index axis.
    # Cells with shorter trajectories still contribute their prefix.
    n_bursts = min(
        min(len(m), len(o))
        for m, o in zip(mediator_lists, outcome_lists)
    )
    if n_bursts == 0:
        return None

    arm_codes = _encode_arm(arms)

    rho_marg: list[float] = []
    rho_part: list[float] = []
    n_per_burst: list[int] = []
    for b in range(n_bursts):
        # Collect non-NaN rows at this burst.
        xs: list[float] = []
        ys: list[float] = []
        zs: list[float] = []
        for i in range(len(arms)):
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
    )
    # Fisher-z pool over valid bursts. Skip-NaN is handled inside
    # `fisher_z_pool`.
    marg_pool, _ = fisher_z_pool(
        rho_marg, n_per_burst, df_offset=4,
    )
    part_pool, _ = fisher_z_pool(
        rho_part, n_per_burst, df_offset=4,
    )
    if status is TimeAggregationStatus.SIGN_FLIP_DETECTED:
        # NaN the marginal aggregate to force consumers off the
        # incoherent pool. The partial pool stays — it's a
        # separate estimand whose sign-coherence isn't determined
        # by the marginal's. (Empirically partial tends to track
        # marginal at sign-flipping envs; the partial's status is
        # a separate question, surfaced via the trajectory.)
        marg_pool = float('nan')

    return DynamicMediationResult(
        burst_steps=tuple(range(n_bursts)),
        rho_marginal=tuple(rho_marg),
        rho_partial=tuple(rho_part),
        n_per_burst=tuple(n_per_burst),
        rho_marginal_pooled=float(marg_pool),
        rho_partial_pooled=float(part_pool),
        aggregation_status=status,
        mediator_name=mediator_per_burst,
        outcome_name=outcome_per_burst,
        arm_field=arm_field,
    )


__all__ = [
    'DynamicMediationResult',
    'Stratum',
    'TimeAggregationStatus',
    'dynamic_partial_spearman',
]
