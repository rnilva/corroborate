"""Trajectory-resolved PC-style mediation — `dynamic_pc_adjacency`.

Sibling to `dynamic_partial_spearman`. Where the partial-Spearman
primitive reports a per-burst MAGNITUDE (rho_partial[b]) and lets
the consumer decide what counts as "mediated," this primitive runs
a per-burst FISHER-Z CI TEST (the same one PC consumes for edge
removal in `corroborate.graph.discovery.discover_adjacency`) and
reports per-burst EDGE PRESENCE:

  - `p_marginal[b]` — P-value for arm ⊥ outcome (depth-0 marginal
    Spearman CI test).
  - `p_conditional[b]` — P-value for arm ⊥ outcome | mediator
    (depth-1 closed-form partial-Spearman CI test).
  - `marginal_edge[b]` = p_marginal[b] < alpha.
  - `mediator_dseparates[b]` = marginal_edge[b] AND
    NOT(p_conditional[b] < alpha) → "full mediation at burst b":
    edge present marginally, vanishes under conditioning.
  - `direct_edge[b]` = marginal_edge[b] AND p_conditional[b] <
    alpha → "partial mediation or direct effect at burst b".

The same `TimeAggregationStatus` enum + `classify_status` machinery
from `_common.py` is reused — driven by the per-burst marginal
partial-correlation magnitude (`rho_marginal[b]`) so the
classifier's sign-flip / weak-time-varying / underpowered branches
fire on the same shape as the partial-Spearman primitive.

This primitive cross-validates `dynamic_partial_spearman` from a
different identification path. Discrepancies between
"mediator_dseparates fraction" (PC) and "rho_partial trajectory
near zero" (partial-correlation magnitude) are diagnostic of
non-linearity or identification failure — the same role that
`mediation_dowhy`'s `LinearityStatus` plays for the static
mediation siblings.

Multi-conditioner extension (`dynamic_pc_adjacency_multi`) is a
separate primitive for a future session; this primitive keeps to
depth-1 with one mediator.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from corroborate._internals.polars import to_dicts as _to_dicts
from corroborate.analyses.dynamic_mediation._common import (
    ClusterBootstrapEdgeCounts,
    ClusterBootstrapInterval,
    FisherZDLPool,
    Stratum,
    TimeAggregationStatus,
    ColumnOrMeasurable,
    classify_status,
    cluster_bootstrap_edge_counts,
    cluster_bootstrap_edge_counts_multi,
    cluster_bootstrap_pool,
    cluster_bootstrap_pool_multi,
    collect_arm_and_per_burst_multi,
    encode_arm,
    fisher_z_dl_pool,
    gather_burst_b_multi,
    n_bursts_multi,
    source_name,
    stratum_key,
)
from corroborate.bridge.analysis import analysis
from corroborate.graph.discovery import (
    _spearman_marginal as _graph_spearman_marginal,  # pyright: ignore[reportPrivateUsage]
    partial_spearman_rho,
    partial_spearman_rho_multi,
)


@dataclass(frozen=True, slots=True)
class DynamicPCResult:
    """Trajectory-resolved PC-style mediation analysis.

    At each burst `b`, the framework runs two CI tests:

      - **Marginal** (depth-0): `arm ⊥ outcome`. P-value
        `p_marginal[b]` from `scipy.stats.spearmanr` (which uses
        the t-approximation for Spearman's null distribution at
        finite n).
      - **Conditional** (depth-1): `arm ⊥ outcome | mediator`.
        P-value `p_conditional[b]` from the closed-form first-
        order partial Spearman's Fisher-z statistic (df = n − 4),
        matching `corroborate.graph.discovery.partial_spearman_rho`.

    Per-burst edge classification (using `alpha` as the rejection
    threshold):

      - `marginal_edge[b]` = `p_marginal[b] < alpha` — arm and
        outcome are NOT independent at burst b (treatment effect
        on outcome is detectable).
      - `mediator_dseparates[b]` = `marginal_edge[b]` AND
        `p_conditional[b] >= alpha` — conditioning on the mediator
        removes the edge. PC interpretation: "the mediator
        d-separates arm from outcome at burst b" → full mediation.
      - `direct_edge[b]` = `marginal_edge[b]` AND
        `p_conditional[b] < alpha` — conditioning on the mediator
        does NOT remove the edge. PC interpretation: "there's a
        direct edge from arm to outcome not blocked by the
        mediator" → partial mediation or direct effect.

    Aggregate diagnostics:

      - `aggregation_status` — shared `TimeAggregationStatus` enum
        driven by `rho_marginal[b]` (same classifier as the
        partial-Spearman sibling); flags sign-flip / weak-time-
        varying / underpowered burst trajectories.
      - `n_bursts_marginal_edge` / `n_bursts_mediator_dseparates` /
        `n_bursts_direct_edge` — raw counts across the trajectory.
        Consumers decide thresholds for "mostly-mediated" /
        "rarely-mediated" / etc.; the framework declines to
        prescribe a meta-aggregator.

    Bursts where `n_per_burst[b] < min_n_per_burst` flag as
    UNDERPOWERED at that burst: NaN p-values, NaN ρs, all three
    boolean flags False (no edge can be asserted at insufficient
    n).

    `dl_marginal` / `dl_partial` carry the DerSimonian-Laird
    random-effects pool over the per-burst (ρ, n) trajectory —
    same shape as the sibling `dynamic_partial_spearman`. The DL
    pool exposes τ² / I² / Q as the quantitative heterogeneity
    signal (the enum `aggregation_status` flags it
    qualitatively). Unlike the partial-Spearman sibling, this
    primitive doesn't expose an FE Fisher-z pool — the PC
    primitive's primary output is per-burst CI-test edge
    presence, not a pooled magnitude; the DL pool sits alongside
    as the trajectory-level magnitude / heterogeneity summary
    for consumers that want it.

    `bootstrap_marginal` / `bootstrap_partial` (the cluster
    bootstrap empirical CI on the DL pool) are `None` when
    `n_bootstrap == 0` (default fast path) and populated when
    `n_bootstrap > 0`. DL's PI bounds are parametric and assume
    per-burst independence — which trajectory data violates; the
    cluster bootstrap resamples WHOLE CELLS and is therefore
    assumption-free under any within-cell autocorrelation
    structure. `n_bootstrap=1000` is recommended for
    publication-grade CIs.

    `bootstrap_edge_counts` (`ClusterBootstrapEdgeCounts | None`)
    is the cluster-bootstrap CI on the INTEGER count triple
    (`n_bursts_marginal_edge` / `n_bursts_mediator_dseparates` /
    `n_bursts_direct_edge`). Conceptually distinct from the
    ρ-pool CIs: the count CIs answer "is the edge classification
    robust to which cells we sampled?" (a few outlier cells
    flipping per-burst CI decisions widens the interval); the
    ρ-pool CIs answer "what's the average effect magnitude under
    bootstrap resampling?". Populated alongside the ρ-pool CIs
    when `n_bootstrap > 0`; `None` on the fast path."""
    burst_steps: tuple[int, ...]
    n_per_burst: tuple[int, ...]
    p_marginal: tuple[float, ...]
    p_conditional: tuple[float, ...]
    rho_marginal: tuple[float, ...]
    rho_partial: tuple[float, ...]
    alpha: float
    aggregation_status: TimeAggregationStatus
    n_bursts_marginal_edge: int
    n_bursts_mediator_dseparates: int
    n_bursts_direct_edge: int
    dl_marginal: FisherZDLPool
    dl_partial: FisherZDLPool
    mediator_names: tuple[str, ...]
    outcome_name: str
    arm_field: str
    bootstrap_marginal: ClusterBootstrapInterval | None = None
    bootstrap_partial: ClusterBootstrapInterval | None = None
    bootstrap_edge_counts: ClusterBootstrapEdgeCounts | None = None
    n_bootstrap: int = 0

    @property
    def n_bursts(self) -> int:
        return len(self.burst_steps)

    @property
    def k_conditioning(self) -> int:
        """Number of conditioning mediators (depth of the
        conditioning set). Depth-1 (single mediator) has k=1;
        depth-≥2 multi-mediator has k≥2. The
        `n_bursts_mediator_dseparates` count's semantics
        generalises: at k=1 it asks "does THIS mediator d-separate?";
        at k≥2 it asks "does the JOINT mediator set d-separate?"."""
        return len(self.mediator_names)


@analysis
def dynamic_pc_adjacency(
    cells: pl.DataFrame,
    *,
    arm_field: str = 'arm_key',
    mediator_per_burst: (
        ColumnOrMeasurable | tuple[ColumnOrMeasurable, ...]
    ),
    outcome_per_burst: ColumnOrMeasurable,
    stratify_by: tuple[str, ...] = ('env_name', 'gamma'),
    alpha: float = 0.05,
    min_n_per_burst: int = 20,
    sign_flip_min_abs_rho: float = 0.05,
    weak_time_varying_ratio: float = 2.0,
    n_bootstrap: int = 0,
    bootstrap_seed: int = 42,
    bootstrap_alpha: float = 0.05,
) -> Mapping[Stratum, DynamicPCResult]:
    """Trajectory-resolved PC-style mediation per stratum.

    For each stratum (tuple of values at `stratify_by` columns):
    align the per-burst arrays across cells (ragged-tail
    semantics, same as the partial-Spearman sibling); at each
    burst index run two CI tests:

      - Marginal: `_spearman_marginal(arm_code, outcome_b)` —
        depth-0 Spearman CI (matches PC's depth-0 test).
      - Conditional: `partial_spearman_rho(arm_code, outcome_b,
        mediator_b)` — closed-form first-order partial Spearman
        with Fisher-z df = n − 4 (matches PC's depth-1 test with
        a single conditioner).

    Edge presence per burst is decided at `alpha`. Aggregates are
    raw counts across the trajectory; consumers downstream
    interpret them.

    `min_n_per_burst` defaults to 20 — higher than the partial-
    Spearman sibling's 5 because PC needs more samples per burst
    for stable α-level CI tests. PC's static depth-1 test at n=10
    has substantial type-I-error inflation; n=20 is the smallest
    n where the Fisher-z df=16 t-approximation lands within ~5%
    of the asymptotic normal.

    The `TimeAggregationStatus` enum is driven by the per-burst
    marginal ρ (`rho_marginal[b]`) — same classifier as
    `dynamic_partial_spearman`. Bursts below `min_n_per_burst`
    contribute NaN ρ AND False edge flags; the trajectory keeps
    them as placeholders so `n_per_burst[b]` aligns with
    `burst_steps[b]`.

    `mediator_per_burst` / `outcome_per_burst` accept str column
    names OR `Measurable[..., NDArray]` instances; same cache-first
    dispatch as the static `partial_spearman` and the sibling
    `dynamic_partial_spearman`.

    `mediator_per_burst` may also be a TUPLE of column-names /
    Measurables for multi-mediator depth-≥2 conditioning. Single
    `str | Measurable` → k=1 (closed-form `partial_spearman_rho`,
    Fisher-z df = n − 4). Tuple of length k → multi-Z
    (`partial_spearman_rho_multi`, df = n − 3 − k). The
    `mediator_dseparates` semantics generalise: at k=1 it's "this
    one mediator d-separates"; at k≥2 it's "the JOINT mediator set
    d-separates". The bootstrap edge-count CIs use the same
    machinery — depth-k just changes which CI primitive runs
    per-burst. Empty tuple raises `ValueError`.

    `n_bootstrap` (default 0) enables a cluster-bootstrap CI on
    the DL pool AND on the per-burst edge-count triple. Cells
    are the resampling unit. For each iteration, resample
    `n_cells` with replacement and recompute (a) the DL-pooled
    per-burst ρ — empirical [α/2, 1 − α/2] percentile range
    becomes `bootstrap_marginal` / `bootstrap_partial`; (b) the
    per-burst CI decisions and their summed (marg, dsep, direct)
    triple — empirical percentiles become
    `bootstrap_edge_counts` (`ClusterBootstrapEdgeCounts`). The
    count CIs and the ρ-pool CIs answer structurally distinct
    questions: the count CIs ask "is the edge classification
    robust to which cells we sampled?"; the ρ-pool CIs ask
    "what's the average effect magnitude under bootstrap
    resampling?". `n_bootstrap=1000` is the recommended
    publication-grade value; default 0 keeps the fast path
    intact. `bootstrap_seed` (default 42) → `np.random.default_rng`
    for reproducibility; `bootstrap_alpha` (default 0.05 → 95% CI)
    controls the percentile for both interval types.

    Returns a `Mapping[Stratum, DynamicPCResult]`. Strata where no
    cell contributes (missing arm tag, malformed per-burst columns,
    single-arm stratum) are absent from the result — the framework
    refuses to silently emit a NaN trajectory."""
    mediators_tuple: tuple[ColumnOrMeasurable, ...]
    if isinstance(mediator_per_burst, tuple):
        mediators_tuple = mediator_per_burst
        if len(mediators_tuple) == 0:
            raise ValueError(
                'dynamic_pc_adjacency: mediator_per_burst=() is the '
                'marginal test (no conditioning) — already reported '
                'via `p_marginal`. Pass a non-empty mediator tuple, '
                'or use the marginal output directly.',
            )
    else:
        mediators_tuple = (mediator_per_burst,)
    k = len(mediators_tuple)

    by_stratum: dict[Stratum, list[Mapping[str, object]]] = {}
    for cell in _to_dicts(cells):
        key = stratum_key(cell, stratify_by)
        if key is None:
            continue
        by_stratum.setdefault(key, []).append(cell)

    out: dict[Stratum, DynamicPCResult] = {}
    for stratum, stratum_cells in by_stratum.items():
        result = _compute_one_stratum_pc(
            stratum_cells,
            arm_field=arm_field,
            mediators_per_burst=mediators_tuple,
            outcome_per_burst=outcome_per_burst,
            alpha=alpha,
            min_n_per_burst=min_n_per_burst,
            sign_flip_min_abs_rho=sign_flip_min_abs_rho,
            weak_time_varying_ratio=weak_time_varying_ratio,
            n_bootstrap=n_bootstrap,
            bootstrap_seed=bootstrap_seed,
            bootstrap_alpha=bootstrap_alpha,
            k=k,
        )
        if result is not None:
            out[stratum] = result
    return out


def _compute_one_stratum_pc(
    cells: Sequence[Mapping[str, object]],
    *,
    arm_field: str,
    mediators_per_burst: tuple[ColumnOrMeasurable, ...],
    outcome_per_burst: ColumnOrMeasurable,
    alpha: float,
    min_n_per_burst: int,
    sign_flip_min_abs_rho: float,
    weak_time_varying_ratio: float,
    n_bootstrap: int,
    bootstrap_seed: int,
    bootstrap_alpha: float,
    k: int,
) -> DynamicPCResult | None:
    """Per-stratum core for the PC primitive. Returns None when no
    per-cell row has valid arm + per-burst columns at all.

    `k = len(mediators_per_burst)`; at k=1 the depth-1 CI machinery
    runs (closed-form `partial_spearman_rho`, Fisher-z df = n − 4)
    for bit-exact back-compat; at k≥2 the multi-Z primitive runs
    (`partial_spearman_rho_multi`, df = n − 3 − k)."""
    collected = collect_arm_and_per_burst_multi(
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
        return None

    n_bursts = n_bursts_multi(mediator_lists, outcome_lists)
    if n_bursts == 0:
        return None

    arm_codes = encode_arm(arms)
    df_offset_partial = 3 + k

    p_marg: list[float] = []
    p_cond: list[float] = []
    rho_marg: list[float] = []
    rho_part: list[float] = []
    n_per_burst: list[int] = []
    n_marg_edge = 0
    n_dsep = 0
    n_direct = 0

    for b in range(n_bursts):
        x_np, y_np, z_mat = gather_burst_b_multi(
            arm_codes, mediator_lists, outcome_lists, b,
        )
        n_b = int(x_np.size)
        n_per_burst.append(n_b)
        if n_b < min_n_per_burst:
            # Under-powered burst: no edge claim possible. NaN both
            # p-values + both ρs; the boolean counters don't
            # increment.
            p_marg.append(float('nan'))
            p_cond.append(float('nan'))
            rho_marg.append(float('nan'))
            rho_part.append(float('nan'))
            continue

        # Depth-0 CI: marginal Spearman ρ + p.
        rho_m, p_m = _graph_spearman_marginal(x_np, y_np)
        # Depth-k CI: at k=1 closed-form partial Spearman (df=n−4)
        # for bit-exact back-compat; at k≥2 multi-Z OLS-residual
        # (df = n − 3 − k).
        if k == 1:
            z_np = z_mat[:, 0] if z_mat.size else z_mat.reshape(-1)
            rho_p, p_p = partial_spearman_rho(x_np, y_np, z_np)
        else:
            rho_p, p_p = partial_spearman_rho_multi(x_np, y_np, z_mat)

        rho_marg.append(rho_m)
        rho_part.append(rho_p)
        p_marg.append(p_m)
        p_cond.append(p_p)

        # Per-burst edge classification at α.
        # NaN p-value (degenerate variance, ill-conditioned
        # partial) → no edge claim. The marginal_edge boolean
        # short-circuits to False in that case.
        marg_edge = (
            not _is_nan(p_m) and p_m < alpha
        )
        cond_edge = (
            not _is_nan(p_p) and p_p < alpha
        )
        if marg_edge:
            n_marg_edge += 1
            if cond_edge:
                n_direct += 1
            else:
                # marginal edge present AND conditional edge
                # absent (or NaN — treated as "no edge" by the
                # Fisher-z CI test's null convention). The
                # (joint) mediator set d-separates arm from
                # outcome at burst b.
                n_dsep += 1

    status = classify_status(
        rho_marg, n_per_burst, min_n_per_burst,
        weak_time_varying_ratio,
        sign_flip_min_abs_rho,
    )

    # DerSimonian-Laird random-effects pool over the per-burst
    # (ρ, n) trajectory. df_offset accounting matches the
    # primitive: 3 for marginal, 3+k for the partial pool.
    dl_marg = fisher_z_dl_pool(rho_marg, n_per_burst, df_offset=3)
    dl_part = fisher_z_dl_pool(
        rho_part, n_per_burst, df_offset=df_offset_partial,
    )

    # Cluster bootstrap on the DL pool AND on the integer edge-
    # count triple — both opt-in via `n_bootstrap > 0`. The k=1
    # path uses the depth-1 (closed-form) cluster-bootstrap kernel
    # for bit-exact back-compat; the k≥2 path uses the multi-Z
    # variant. Same cell-resampling pattern for both.
    bootstrap_marg: ClusterBootstrapInterval | None = None
    bootstrap_part: ClusterBootstrapInterval | None = None
    bootstrap_counts: ClusterBootstrapEdgeCounts | None = None
    if n_bootstrap > 0:
        if k == 1:
            flat_med: list[Sequence[float]] = [
                cell_meds[0] for cell_meds in mediator_lists
            ]
            bootstrap_marg = cluster_bootstrap_pool(
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
            bootstrap_part = cluster_bootstrap_pool(
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
            bootstrap_counts = cluster_bootstrap_edge_counts(
                arm_codes=arm_codes,
                mediator_lists=flat_med,
                outcome_lists=outcome_lists,
                n_bursts=n_bursts,
                min_n_per_burst=min_n_per_burst,
                alpha=alpha,
                n_resamples=n_bootstrap,
                bootstrap_alpha=bootstrap_alpha,
                seed=bootstrap_seed,
            )
        else:
            bootstrap_marg = cluster_bootstrap_pool_multi(
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
            bootstrap_part = cluster_bootstrap_pool_multi(
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
            bootstrap_counts = cluster_bootstrap_edge_counts_multi(
                arm_codes=arm_codes,
                mediator_lists=mediator_lists,
                outcome_lists=outcome_lists,
                n_bursts=n_bursts,
                min_n_per_burst=min_n_per_burst,
                alpha=alpha,
                n_resamples=n_bootstrap,
                bootstrap_alpha=bootstrap_alpha,
                seed=bootstrap_seed,
            )

    return DynamicPCResult(
        burst_steps=tuple(range(n_bursts)),
        n_per_burst=tuple(n_per_burst),
        p_marginal=tuple(p_marg),
        p_conditional=tuple(p_cond),
        rho_marginal=tuple(rho_marg),
        rho_partial=tuple(rho_part),
        alpha=alpha,
        aggregation_status=status,
        n_bursts_marginal_edge=n_marg_edge,
        n_bursts_mediator_dseparates=n_dsep,
        n_bursts_direct_edge=n_direct,
        dl_marginal=dl_marg,
        dl_partial=dl_part,
        mediator_names=tuple(source_name(m) for m in mediators_per_burst),
        outcome_name=source_name(outcome_per_burst),
        arm_field=arm_field,
        bootstrap_marginal=bootstrap_marg,
        bootstrap_partial=bootstrap_part,
        bootstrap_edge_counts=bootstrap_counts,
        n_bootstrap=n_bootstrap,
    )


def _is_nan(v: float) -> bool:
    """Local NaN check — `math.isnan` raises TypeError on non-float
    inputs, and pyright's narrowing for `float != float` is
    fragile. Direct comparison `v != v` is the canonical NaN
    detector and float-typed throughout."""
    return v != v


__all__ = [
    'DynamicPCResult',
    'dynamic_pc_adjacency',
]
