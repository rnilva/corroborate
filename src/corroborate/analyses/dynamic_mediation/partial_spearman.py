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
    ClusterBootstrapInterval,
    FisherZDLPool,
    Stratum,
    TimeAggregationStatus,
    _ColumnOrMeasurable,
    _classify_status,
    _cluster_bootstrap_pool,
    _cluster_bootstrap_pool_multi,
    _collect_arm_and_per_burst_multi,
    _encode_arm,
    _fisher_z_dl_pool,
    _gather_burst_b_multi,
    _n_bursts_multi,
    _source_name,
    _stratum_key,
)
from corroborate.bridge.analysis import analysis
from corroborate.graph.discovery import (
    _spearman_marginal as _graph_spearman_marginal,  # pyright: ignore[reportPrivateUsage]
    partial_spearman_rho,
    partial_spearman_rho_multi,
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
    point estimate under any pool-incoherent regime.

    `bootstrap_marginal` / `bootstrap_partial` (the cluster
    bootstrap empirical CI) are `None` when `n_bootstrap == 0`
    (the default, fast path) and populated when `n_bootstrap > 0`.
    DL's PI bounds are *parametric* — they assume per-burst
    independence, which trajectory data violates (bursts within
    one cell share network state, dynamics, replay buffer). The
    cluster bootstrap resamples WHOLE CELLS with replacement and
    is therefore assumption-free under any within-cell
    autocorrelation structure. For publication-grade CIs reach
    for `n_bootstrap=1000`; the empirical 2.5% / 97.5% percentile
    range is the 95% CI."""
    burst_steps: tuple[int, ...]
    rho_marginal: tuple[float, ...]
    rho_partial: tuple[float, ...]
    n_per_burst: tuple[int, ...]
    rho_marginal_pooled: float
    rho_partial_pooled: float
    dl_marginal: FisherZDLPool
    dl_partial: FisherZDLPool
    aggregation_status: TimeAggregationStatus
    mediator_names: tuple[str, ...]
    outcome_name: str
    arm_field: str
    bootstrap_marginal: ClusterBootstrapInterval | None = None
    bootstrap_partial: ClusterBootstrapInterval | None = None
    n_bootstrap: int = 0

    @property
    def n_bursts(self) -> int:
        return len(self.burst_steps)

    @property
    def k_conditioning(self) -> int:
        """Number of conditioning mediators (depth of the
        conditioning set). Depth-1 (single mediator) has k=1;
        depth-≥2 multi-mediator has k≥2."""
        return len(self.mediator_names)


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
    mediator_per_burst: (
        _ColumnOrMeasurable | tuple[_ColumnOrMeasurable, ...]
    ),
    outcome_per_burst: _ColumnOrMeasurable,
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    min_n_per_burst: int = 5,
    weak_time_varying_ratio: float = 2.0,
    sign_flip_min_abs_rho: float = 0.05,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 42,
    bootstrap_alpha: float = 0.05,
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

    `mediator_per_burst` may also be a TUPLE of column-names /
    Measurables for multi-mediator depth-≥2 conditioning. Single
    `str | Measurable` → k=1 (closed-form `partial_spearman_rho`,
    Fisher-z df = n − 4). Tuple of length k → multi-Z partial
    (`partial_spearman_rho_multi` via OLS-residual regression,
    Fisher-z df = n − 3 − k). The dispatch parallels the static
    `partial_spearman`'s `conditioning` parameter. Empty tuple
    raises `ValueError` — that's the marginal test, which is
    already reported in `rho_marginal`. At k=1 the closed-form
    path is preferred over the multi path for verdict-stability
    reasons (the two differ by ~1e-2 in tie-handling drift).

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
    cause.

    `n_bootstrap` (default 0) enables a cluster-bootstrap CI on
    the DL pool. Cells are the resampling unit (each cell = one
    training trajectory = one independent unit). For each of
    `n_bootstrap` iterations, resample `n_cells` with replacement
    and recompute the DL-pooled ρ from the resampled per-burst
    trajectories; the empirical [α/2, 1 − α/2] percentile range of
    the bootstrap distribution becomes `bootstrap_marginal` /
    `bootstrap_partial`. DL's PI bounds assume per-burst
    independence — which trajectory data violates — so the
    cluster bootstrap is the methodologically-correct CI for
    publication-grade reports. `n_bootstrap=1000` is the
    recommended value; the default 0 keeps the fast path
    bit-identical to the pre-bootstrap behaviour.
    `bootstrap_seed` (default 42) makes the resample deterministic
    via `np.random.default_rng`; `bootstrap_alpha` (default 0.05 →
    95% CI) controls the percentile."""
    # Normalize mediator argument to a tuple. The empty-tuple case
    # is the marginal test (which we already report via
    # `rho_marginal`); raising here keeps the framework's two
    # surfaces typed-distinct and refuses a silently no-op
    # invocation.
    mediators_tuple: tuple[_ColumnOrMeasurable, ...]
    if isinstance(mediator_per_burst, tuple):
        mediators_tuple = mediator_per_burst
        if len(mediators_tuple) == 0:
            raise ValueError(
                'dynamic_partial_spearman: mediator_per_burst=() is '
                'the marginal test (no conditioning) — the framework '
                'already reports the marginal per-burst ρ via '
                '`rho_marginal`. Pass a non-empty mediator tuple, '
                'or use the marginal output directly.',
            )
    else:
        mediators_tuple = (mediator_per_burst,)
    k = len(mediators_tuple)

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
            mediators_per_burst=mediators_tuple,
            outcome_per_burst=outcome_per_burst,
            min_n_per_burst=min_n_per_burst,
            weak_time_varying_ratio=weak_time_varying_ratio,
            sign_flip_min_abs_rho=sign_flip_min_abs_rho,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            bootstrap_alpha=bootstrap_alpha,
            k=k,
        )
        if result is not None:
            out[stratum] = result
    return out


def _compute_one_stratum(
    cells: Sequence[Mapping[str, object]],
    *,
    arm_field: str,
    mediators_per_burst: tuple[_ColumnOrMeasurable, ...],
    outcome_per_burst: _ColumnOrMeasurable,
    min_n_per_burst: int,
    weak_time_varying_ratio: float,
    sign_flip_min_abs_rho: float,
    n_bootstrap: int,
    bootstrap_seed: int,
    bootstrap_alpha: float,
    k: int,
) -> DynamicMediationResult | None:
    """Per-stratum core. Returns None when no per-cell row has
    valid arm + per-burst columns at all.

    `k = len(mediators_per_burst)` parameterises the dispatch:
      - k = 1 → use the closed-form `partial_spearman_rho` (df =
        n − 4) for verdict stability with the static
        `partial_spearman` k=1 path.
      - k ≥ 2 → use the OLS-residual `partial_spearman_rho_multi`
        (df = n − 3 − k).

    Both paths feed into the same DL pool / cluster bootstrap with
    `df_offset = 3 + k`."""
    collected = _collect_arm_and_per_burst_multi(
        cells,
        arm_field=arm_field,
        mediators_per_burst=mediators_per_burst,
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
    n_bursts = _n_bursts_multi(mediator_lists, outcome_lists)
    if n_bursts == 0:
        return None

    arm_codes = _encode_arm(arms)
    df_offset_partial = 3 + k

    rho_marg: list[float] = []
    rho_part: list[float] = []
    n_per_burst: list[int] = []
    for b in range(n_bursts):
        if k == 1:
            # Flatten the k=1 mediator matrix into the 1-D vector
            # `partial_spearman_rho` expects; the resulting (x, y,
            # z) tuple is bit-identical to what the depth-1
            # `_gather_burst_b` returns on the equivalent
            # single-mediator panel.
            x_np, y_np, z_mat = _gather_burst_b_multi(
                arm_codes, mediator_lists, outcome_lists, b,
            )
            z_np = z_mat[:, 0] if z_mat.size else z_mat.reshape(-1)
            n_b = int(x_np.size)
            n_per_burst.append(n_b)
            if n_b < min_n_per_burst:
                rho_marg.append(float('nan'))
                rho_part.append(float('nan'))
                continue
            r_m = _marginal_spearman(x_np, y_np)
            r_p, _ = partial_spearman_rho(x_np, y_np, z_np)
        else:
            x_np, y_np, z_mat = _gather_burst_b_multi(
                arm_codes, mediator_lists, outcome_lists, b,
            )
            n_b = int(x_np.size)
            n_per_burst.append(n_b)
            if n_b < min_n_per_burst:
                rho_marg.append(float('nan'))
                rho_part.append(float('nan'))
                continue
            r_m = _marginal_spearman(x_np, y_np)
            r_p, _ = partial_spearman_rho_multi(x_np, y_np, z_mat)
        rho_marg.append(r_m)
        rho_part.append(r_p)

    status = _classify_status(
        rho_marg, n_per_burst, min_n_per_burst,
        weak_time_varying_ratio,
        sign_flip_min_abs_rho,
    )
    # Fisher-z FE pool over valid bursts. Skip-NaN is handled inside
    # `fisher_z_pool`. df_offset accounting: 3 for marginal Spearman,
    # 3 + k for the conditional pool (= 4 at k=1, 5 at k=2, ...).
    marg_pool, _ = fisher_z_pool(
        rho_marg, n_per_burst, df_offset=3,
    )
    part_pool, _ = fisher_z_pool(
        rho_part, n_per_burst, df_offset=df_offset_partial,
    )
    # DerSimonian-Laird random-effects pool — same df_offset
    # accounting as the FE pool. NEVER NaN'd by the diagnostic gate.
    dl_marg = _fisher_z_dl_pool(rho_marg, n_per_burst, df_offset=3)
    dl_part = _fisher_z_dl_pool(
        rho_part, n_per_burst, df_offset=df_offset_partial,
    )
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

    # Cluster bootstrap CI on the DL pool — opt-in via
    # `n_bootstrap > 0`. The k=1 path uses the depth-1
    # `_cluster_bootstrap_pool` (closed-form partial inside) so the
    # default behaviour remains bit-identical to pre-multi versions.
    # k≥2 uses the multi-mediator variant.
    bootstrap_marg: ClusterBootstrapInterval | None = None
    bootstrap_part: ClusterBootstrapInterval | None = None
    if n_bootstrap > 0:
        if k == 1:
            # Project the per-cell list-of-1 mediator back to the
            # depth-1 shape so the closed-form bootstrap kernel sees
            # the same (arm, single-mediator, outcome) buffer it did
            # before the multi-mediator refactor.
            flat_med: list[Sequence[float]] = [
                cell_meds[0] for cell_meds in mediator_lists
            ]
            bootstrap_marg = _cluster_bootstrap_pool(
                arm_codes=arm_codes,
                mediator_lists=flat_med,
                outcome_lists=outcome_lists,
                n_bursts=n_bursts,
                min_n_per_burst=min_n_per_burst,
                kind='marginal',
                df_offset=3,
                n_resamples=n_bootstrap,
                alpha=bootstrap_alpha,
                seed=bootstrap_seed,
            )
            bootstrap_part = _cluster_bootstrap_pool(
                arm_codes=arm_codes,
                mediator_lists=flat_med,
                outcome_lists=outcome_lists,
                n_bursts=n_bursts,
                min_n_per_burst=min_n_per_burst,
                kind='partial',
                df_offset=4,
                n_resamples=n_bootstrap,
                alpha=bootstrap_alpha,
                seed=bootstrap_seed,
            )
        else:
            bootstrap_marg = _cluster_bootstrap_pool_multi(
                arm_codes=arm_codes,
                mediator_lists=mediator_lists,
                outcome_lists=outcome_lists,
                n_bursts=n_bursts,
                min_n_per_burst=min_n_per_burst,
                kind='marginal',
                df_offset=3,
                n_resamples=n_bootstrap,
                alpha=bootstrap_alpha,
                seed=bootstrap_seed,
            )
            bootstrap_part = _cluster_bootstrap_pool_multi(
                arm_codes=arm_codes,
                mediator_lists=mediator_lists,
                outcome_lists=outcome_lists,
                n_bursts=n_bursts,
                min_n_per_burst=min_n_per_burst,
                kind='partial',
                df_offset=df_offset_partial,
                n_resamples=n_bootstrap,
                alpha=bootstrap_alpha,
                seed=bootstrap_seed,
            )

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
        mediator_names=tuple(_source_name(m) for m in mediators_per_burst),
        outcome_name=_source_name(outcome_per_burst),
        arm_field=arm_field,
        bootstrap_marginal=bootstrap_marg,
        bootstrap_partial=bootstrap_part,
        n_bootstrap=n_bootstrap,
    )


__all__ = [
    'DynamicMediationResult',
    'dynamic_partial_spearman',
]
