"""Trajectory-resolved partial Spearman mediation —
`dynamic_partial_spearman`.

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

See the package `__init__.py` docstring + the design doc
(`DYNAMIC_MEDIATION_DESIGN.md`) for the empirical motivation
(PacMan sign-flip, MetaMaze mid-training peak, Asterix factor-
substitution).

The shared infrastructure (status classifier, arm encoding,
per-burst extraction, ragged-tail alignment) lives in `_common.py`
— consumed by both this primitive and the sibling
`dynamic_pc_adjacency`.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from corroborate._internals.polars import to_dicts as _to_dicts
from corroborate.analyses.dynamic_mediation._common import (
    FisherZDLPool,
    Stratum,
    TimeAggregationStatus,
    _ColumnOrMeasurable,
    _classify_status,
    _collect_arm_and_per_burst,
    _encode_arm,
    _fisher_z_dl_pool,
    _gather_burst_b,
    _n_bursts,
    _source_name,
    _stratum_key,
)
from corroborate.bridge.analysis import analysis
from corroborate.graph.discovery import (
    _spearman_marginal as _graph_spearman_marginal,  # pyright: ignore[reportPrivateUsage]
    partial_spearman_rho,
)
from corroborate.stats import fisher_z_pool


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

    Both `rho_marginal_pooled` AND `rho_partial_pooled` (the
    fixed-effects Fisher-z pool, n-weighted) are NaN when
    `aggregation_status` is SIGN_FLIP_DETECTED — the pool over
    sign-opposing bursts is a structural Simpson's-paradox artifact,
    not an estimate. The partial pool inherits the same suspect
    support as the marginal, so the framework refuses to report
    either. For WEAK_TIME_VARYING / UNDERPOWERED_BURSTS the FE
    pools are still computed (best-effort) but the status flag
    warns consumers that the aggregate may not represent the
    trajectory.

    `dl_marginal` and `dl_partial` (the DerSimonian-Laird random-
    effects pool) are NEVER NaN'd by the diagnostic gate. Their
    `tau2` and `i2` ARE the quantitative signal of the
    heterogeneity that `aggregation_status` flags qualitatively:
    SIGN_FLIP → large τ², I² near 1.0; WEAK_TIME_VARYING →
    moderate τ², I² ∈ [0.5, 1.0]; CONSISTENT_DIRECTION → small
    τ², I² near 0. The DL pool's `rho_pooled` and PI bounds are
    inverse-Fisher-z'd back to ρ-units; `se_pooled` stays in
    Fisher-z units (its scale). Pair the two pools: FE for the
    point estimate under consistent-direction (its n-weighting is
    sharper at low τ²), DL for the heterogeneity diagnostic +
    point estimate under any pool-incoherent regime."""
    burst_steps: tuple[int, ...]
    rho_marginal: tuple[float, ...]
    rho_partial: tuple[float, ...]
    n_per_burst: tuple[int, ...]
    rho_marginal_pooled: float
    rho_partial_pooled: float
    dl_marginal: FisherZDLPool
    dl_partial: FisherZDLPool
    aggregation_status: TimeAggregationStatus
    mediator_name: str
    outcome_name: str
    arm_field: str

    @property
    def n_bursts(self) -> int:
        return len(self.burst_steps)


def _marginal_spearman(
    x: np.ndarray, y: np.ndarray,
) -> float:
    """Marginal Spearman ρ. Thin wrapper around
    `graph.discovery._spearman_marginal` that drops the p-value
    (we only need ρ here; the Fisher-z pool computes its own pooled
    p-stat). Kept as a private helper to centralize the import and
    return-shape narrowing in one place."""
    rho, _ = _graph_spearman_marginal(x, y)
    return rho


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
    collected = _collect_arm_and_per_burst(
        cells,
        arm_field=arm_field,
        mediator_per_burst=mediator_per_burst,
        outcome_per_burst=outcome_per_burst,
    )
    if collected is None:
        return None
    arms, mediator_lists, outcome_lists = collected
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
    n_bursts = _n_bursts(mediator_lists, outcome_lists)
    if n_bursts == 0:
        return None

    arm_codes = _encode_arm(arms)

    rho_marg: list[float] = []
    rho_part: list[float] = []
    n_per_burst: list[int] = []
    for b in range(n_bursts):
        x_np, y_np, z_np = _gather_burst_b(
            arm_codes, mediator_lists, outcome_lists, b,
        )
        n_b = int(x_np.size)
        n_per_burst.append(n_b)
        if n_b < min_n_per_burst:
            rho_marg.append(float('nan'))
            rho_part.append(float('nan'))
            continue
        r_m = _marginal_spearman(x_np, y_np)
        r_p, _ = partial_spearman_rho(x_np, y_np, z_np)
        rho_marg.append(r_m)
        rho_part.append(r_p)

    status = _classify_status(
        rho_marg, n_per_burst, min_n_per_burst,
        weak_time_varying_ratio,
        sign_flip_min_abs_rho,
    )
    # Fisher-z FE pool over valid bursts. Skip-NaN is handled inside
    # `fisher_z_pool`. df_offset matches the sibling primitives:
    # 3 for marginal (`stratified_spearman_rho`), 4 for closed-
    # form first-order partial (`stratified_partial_spearman_rho`).
    marg_pool, _ = fisher_z_pool(
        rho_marg, n_per_burst, df_offset=3,
    )
    part_pool, _ = fisher_z_pool(
        rho_part, n_per_burst, df_offset=4,
    )
    # DerSimonian-Laird random-effects pool — computed on the SAME
    # per-burst (ρ, n) pairs the FE pool uses, with the same
    # df_offset accounting. NEVER NaN'd by the diagnostic gate: its
    # τ² IS the quantitative signal of the heterogeneity that
    # SIGN_FLIP_DETECTED flags qualitatively.
    dl_marg = _fisher_z_dl_pool(rho_marg, n_per_burst, df_offset=3)
    dl_part = _fisher_z_dl_pool(rho_part, n_per_burst, df_offset=4)
    if status is TimeAggregationStatus.SIGN_FLIP_DETECTED:
        # NaN BOTH FE aggregates: the marginal pool is the
        # Simpson's-paradox artifact directly, and the partial pool
        # inherits the same suspect support (it's computed on the
        # same per-burst (xs, ys, zs) trios). If the marginal is
        # incoherent across bursts, the partial's pool isn't a
        # trustworthy summary either — consumers must read the
        # trajectory.
        # The DL pool is NOT NaN'd; its τ² + I² are the quantitative
        # heterogeneity signal — that's the point of having BOTH.
        marg_pool = float('nan')
        part_pool = float('nan')

    return DynamicMediationResult(
        burst_steps=tuple(range(n_bursts)),
        rho_marginal=tuple(rho_marg),
        rho_partial=tuple(rho_part),
        n_per_burst=tuple(n_per_burst),
        rho_marginal_pooled=float(marg_pool),
        rho_partial_pooled=float(part_pool),
        dl_marginal=dl_marg,
        dl_partial=dl_part,
        aggregation_status=status,
        mediator_name=_source_name(mediator_per_burst),
        outcome_name=_source_name(outcome_per_burst),
        arm_field=arm_field,
    )


__all__ = [
    'DynamicMediationResult',
    'dynamic_partial_spearman',
]
