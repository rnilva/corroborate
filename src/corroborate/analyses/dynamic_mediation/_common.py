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
  - `FisherZDLPool` + `_fisher_z_dl_pool` — DerSimonian-Laird
    random-effects pool over per-burst Fisher-z-transformed ρ
    values. Wraps the framework's general-purpose
    `random_effects_summary` (the DL implementation lives in
    `stats.effect_size` — we reuse, never reimplement) on (z_b,
    SE_z_b) pairs and inverse-transforms the pooled estimate +
    PI bounds back to ρ-units. Exposes τ²/I²/Q as the *quantitative*
    measure of the heterogeneity that `TimeAggregationStatus` flags
    qualitatively.
  - Type aliases (`Stratum`, `_PerBurstMeasurable`,
    `_ColumnOrMeasurable`).

The `FisherZDLPool` dataclass is public API (re-exported through
the package `__init__.py`); the rest are package-internal helpers.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal

import numpy as np
import numpy.typing as npt

from corroborate.analyses._cell_value import evaluate_per_burst_source
from corroborate.graph.discovery import (
    _spearman_marginal as _graph_spearman_marginal,  # pyright: ignore[reportPrivateUsage]
    partial_spearman_rho,
    partial_spearman_rho_multi,
)
from corroborate.measurables import Measurable
from corroborate.stats import random_effects_summary


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


def _collect_arm_and_per_burst_multi(
    cells: Sequence[Mapping[str, object]],
    *,
    arm_field: str,
    mediators_per_burst: Sequence[_ColumnOrMeasurable],
    outcome_per_burst: _ColumnOrMeasurable,
) -> tuple[
    list[str], list[list[list[float]]], list[list[float]],
] | None:
    """Multi-mediator first-pass cell-record extractor.

    Same role as `_collect_arm_and_per_burst` but extracts k
    mediator trajectories per cell. Returns
    `(arms, mediator_lists, outcome_lists)` where
    `mediator_lists[i]` is a list of length `k` of per-burst
    arrays for cell `i`. A cell is dropped iff ANY mediator
    resolves to None or the outcome resolves to None — the
    multi-mediator CI test requires all k mediators to be
    well-shaped on the same cell.

    The depth-1 sibling `_collect_arm_and_per_burst` is the thin
    `k=1` case of this; both coexist because the depth-1
    extractor's flatter return shape simplifies bit-exact
    back-compat at the existing call sites."""
    arms: list[str] = []
    mediator_lists: list[list[list[float]]] = []
    outcome_lists: list[list[float]] = []
    for cell in cells:
        arm = cell.get(arm_field)
        if not isinstance(arm, str):
            continue
        out_arr = _resolve_per_burst(cell, outcome_per_burst)
        if out_arr is None:
            continue
        meds: list[list[float]] = []
        skip = False
        for m_src in mediators_per_burst:
            med = _resolve_per_burst(cell, m_src)
            if med is None:
                skip = True
                break
            meds.append(med)
        if skip:
            continue
        arms.append(arm)
        mediator_lists.append(meds)
        outcome_lists.append(out_arr)
    if not arms:
        return None
    return arms, mediator_lists, outcome_lists


def _n_bursts_multi(
    mediator_lists: Sequence[Sequence[Sequence[float]]],
    outcome_lists: Sequence[Sequence[float]],
) -> int:
    """Ragged-tail burst-axis length under multi-mediator shape.

    Each cell's mediator entry is a list of `k` per-burst arrays;
    pick the longest across (all k mediators, the outcome) for
    each cell; ragged-tail max across cells. Matches the depth-1
    `_n_bursts` semantics at `k=1`."""
    def _cell_len(meds: Sequence[Sequence[float]], o: Sequence[float]) -> int:
        med_max = max((len(m) for m in meds), default=0)
        return max(med_max, len(o))
    return max(
        _cell_len(m, o)
        for m, o in zip(mediator_lists, outcome_lists)
    )


def _gather_burst_b_multi(
    arm_codes: npt.NDArray[np.float64],
    mediator_lists: Sequence[Sequence[Sequence[float]]],
    outcome_lists: Sequence[Sequence[float]],
    b: int,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]:
    """Multi-mediator burst-b gather. Returns
    `(arm_b, outcome_b, mediator_matrix_b)` where the matrix has
    shape `(n_b, k)`. Cells that don't have a `b`-th entry in
    ANY of the k mediator arrays or the outcome are dropped (the
    multi-Z OLS-residual computation needs aligned observations
    across all conditioning vars).

    Depth-1 sibling `_gather_burst_b` returns a 1-D mediator
    array; this returns a 2-D matrix. The two are kept distinct
    so the depth-1 path bit-exactly preserves its existing
    1-D-array call sites (the closed-form `partial_spearman_rho`
    takes a 1-D z)."""
    xs: list[float] = []
    ys: list[float] = []
    n_cells = len(arm_codes)
    k = max((len(m) for m in mediator_lists), default=0)
    zs_per_k: list[list[float]] = [[] for _ in range(k)]
    for i in range(n_cells):
        if b >= len(outcome_lists[i]):
            continue
        meds_i = mediator_lists[i]
        if any(b >= len(meds_i[j]) for j in range(k)):
            continue
        yv = outcome_lists[i][b]
        z_vals = [float(meds_i[j][b]) for j in range(k)]
        if math.isnan(yv):
            continue
        if any(math.isnan(zv) for zv in z_vals):
            continue
        xs.append(float(arm_codes[i]))
        ys.append(yv)
        for j in range(k):
            zs_per_k[j].append(z_vals[j])
    x_np = np.asarray(xs, dtype=np.float64)
    y_np = np.asarray(ys, dtype=np.float64)
    if x_np.size == 0 or k == 0:
        z_mat = np.zeros((x_np.size, k), dtype=np.float64)
    else:
        z_mat = np.column_stack([
            np.asarray(col, dtype=np.float64) for col in zs_per_k
        ])
    return x_np, y_np, z_mat


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


@dataclass(frozen=True, slots=True)
class FisherZDLPool:
    """DerSimonian-Laird random-effects pool over per-burst Fisher-z
    transformed ρ values.

    The framework's existing `random_effects_summary` (in
    `stats.effect_size`) is the general-purpose DL implementation —
    this dataclass wraps the outputs after the Fisher-z transform
    has been applied to per-burst Spearman ρ's. Each (burst, ρ_b,
    n_b) triple becomes one `(z_b, SE_z_b)` pair fed to the DL
    estimator: `z_b = atanh(ρ_b)`, `SE_z_b = 1 / sqrt(n_b −
    df_offset)`. `df_offset = 3` for marginal Spearman, `df_offset
    = 4` for closed-form first-order partial Spearman (matching
    the existing `fisher_z_pool` sibling).

    Field semantics:

      - `rho_pooled` — DL-pooled estimate in ρ-units (inverse
        Fisher-z of the pooled z). The DL pool's POINT estimate;
        contrast with the FE pool's `rho_marginal_pooled` /
        `rho_partial_pooled` on the result dataclasses, which use
        n-weighted Fisher-z averaging without between-burst variance.
      - `se_pooled` — SE of the pooled estimate in Fisher-z units
        (the DL formula's natural scale; inverse-transforming SE to
        ρ-units is meaningful only at a specific ρ via the
        delta-method `(1 − ρ²) · SE_z`).
      - `tau2` — between-burst heterogeneity variance in z-units.
        The QUANTITATIVE measure of the heterogeneity that
        `TimeAggregationStatus` flags qualitatively: SIGN_FLIP →
        large τ²; WEAK_TIME_VARYING → moderate τ²; CONSISTENT
        DIRECTION → small τ² (often clipped to 0 by `max(0, ·)`).
      - `i2` — Higgins' I² fraction in [0, 1]. The proportion of
        total variance attributable to between-burst heterogeneity.
        I² ≥ 0.5 is the conventional "moderate-to-substantial"
        threshold (`stats.effect_size.I2_THRESHOLD`).
      - `q` — Cochran's Q statistic. Chi-square distributed with
        df = G − 1 under the null of no heterogeneity.
      - `rho_pi_lo`, `rho_pi_hi` — 95% Higgins-Thompson-Spiegelhalter
        prediction interval bounds INVERSE-TRANSFORMED back to
        ρ-units (`tanh(pi_z)`). Both NaN at G < 3 (PI undefined).
      - `n_bursts_used` — number of valid (non-NaN, n ≥ df_offset)
        bursts contributing.
      - `assumption_violations` — DL small-G regime warnings
        passed through from `PooledStats.assumption_violations`
        (n_cells < 5 unreliable inference, etc.).

    Behavior under `TimeAggregationStatus`:

    Unlike the FE pool (which is NaN'd under SIGN_FLIP_DETECTED
    because the n-weighted z-average is structurally meaningless on
    sign-opposing bursts), the DL pool is NEVER NaN'd by the
    diagnostic gate. Its `tau2` and `i2` ARE the quantitative
    signal of the heterogeneity the enum flags. Under SIGN_FLIP
    expect τ² large and I² near 1.0; under WEAK_TIME_VARYING
    expect moderate τ² and I² ∈ [0.5, 1.0]; under CONSISTENT
    DIRECTION expect small τ² and I² near 0. Consumers read the
    DL pool's heterogeneity statistics as first-class metrics
    rather than treating the enum as the only diagnostic.

    Bursts with `n_b < df_offset + 1` (would give non-positive
    Fisher-z weight) are dropped from the pool; if fewer than 2
    bursts remain valid, the pool returns NaN for all numeric
    fields and `n_bursts_used` = (number that DID survive the n
    filter, possibly 0 or 1)."""
    rho_pooled: float
    se_pooled: float
    tau2: float
    i2: float
    q: float
    rho_pi_lo: float
    rho_pi_hi: float
    n_bursts_used: int
    assumption_violations: tuple[str, ...] = ()


def _fisher_z_dl_pool(
    rhos: Sequence[float],
    ns: Sequence[int],
    df_offset: int,
) -> FisherZDLPool:
    """DerSimonian-Laird random-effects pool over per-burst
    Fisher-z transformed ρ values.

    For each burst with non-NaN ρ_b and `n_b > df_offset`:

      - z_b = atanh(ρ_b) (clipped to |ρ| ≤ 0.999999 to avoid the
        atanh asymptote — matches `fisher_z_pool`'s convention).
      - SE_z_b = 1 / sqrt(n_b − df_offset).

    The (z_b, SE_z_b) pairs feed into `random_effects_summary`
    (the framework's general DL primitive in `stats.effect_size`).
    The pooled z-estimate + PI bounds are inverse-Fisher-z'd back
    to ρ-units; τ², I², and Q stay in z-units (their
    interpretation as heterogeneity measures is scale-free).

    Returns a NaN-filled `FisherZDLPool` when fewer than 2 bursts
    are valid (DL needs G ≥ 2 for τ² estimation; PI needs G ≥ 3).
    `n_bursts_used` reflects the actual count fed into the DL
    pool — the caller can distinguish "no valid bursts" from
    "DL undefined at G < 2"."""
    z_se_pairs: list[tuple[float, float]] = []
    n_valid = 0
    for r, n in zip(rhos, ns):
        if math.isnan(r):
            continue
        if n - df_offset < 1:
            continue
        r_c = max(-0.999999, min(0.999999, r))
        z = 0.5 * math.log((1 + r_c) / (1 - r_c))
        se_z = 1.0 / math.sqrt(float(n - df_offset))
        z_se_pairs.append((z, se_z))
        n_valid += 1

    pooled = random_effects_summary(z_se_pairs)
    # `random_effects_summary` returns NaN-filled `PooledStats` when
    # n < 2. Propagate that through with the rho-side inverse
    # transforms also as NaN.
    if math.isnan(pooled.pooled_g):
        return FisherZDLPool(
            rho_pooled=float('nan'),
            se_pooled=float('nan'),
            tau2=float('nan'),
            i2=float('nan'),
            q=float('nan'),
            rho_pi_lo=float('nan'),
            rho_pi_hi=float('nan'),
            n_bursts_used=n_valid,
            assumption_violations=pooled.assumption_violations,
        )
    rho_pooled = math.tanh(pooled.pooled_g)
    rho_pi_lo = (
        math.tanh(pooled.pi_lo) if not math.isnan(pooled.pi_lo)
        else float('nan')
    )
    rho_pi_hi = (
        math.tanh(pooled.pi_hi) if not math.isnan(pooled.pi_hi)
        else float('nan')
    )
    return FisherZDLPool(
        rho_pooled=rho_pooled,
        se_pooled=pooled.se_pooled,
        tau2=pooled.tau2,
        i2=pooled.I2,
        q=pooled.Q,
        rho_pi_lo=rho_pi_lo,
        rho_pi_hi=rho_pi_hi,
        n_bursts_used=n_valid,
        assumption_violations=pooled.assumption_violations,
    )


@dataclass(frozen=True, slots=True)
class ClusterBootstrapInterval:
    """Empirical CI from a cluster bootstrap over cells.

    DerSimonian-Laird's PI bounds are *parametric*: they assume
    per-burst observations are independent. In trajectory data
    bursts within one cell share network state, dynamics, and
    replay buffer — they are NOT independent. The cluster
    bootstrap is the standard fix: resample whole cells (each
    cell = one training trajectory = one independent unit) with
    replacement, recompute the per-burst ρ + pooled estimate per
    resample, and take the empirical [α/2, 1−α/2] percentile
    range as the CI. This is *assumption-free* under any
    within-cell autocorrelation structure — the resampling
    preserves whatever dependence exists.

    Field semantics:

      - `rho_lower` / `rho_upper` — empirical α/2 and 1 − α/2
        percentiles of the pooled ρ across `n_resamples`
        cell-resampled panels.
      - `rho_median` — median of the bootstrap distribution; a
        more robust point estimate than mean when the
        distribution is asymmetric.
      - `n_resamples` — number of bootstrap iterations.
      - `alpha` — significance level (default 0.05 → 95% CI).
      - `seed` — RNG seed for reproducibility.

    See Pustejovsky & Tipton (2022) on CHE/RVE and Deen & de
    Rooij (2020) on ClusterBootstrap for the methodological
    background. Distinguishes from DL's parametric PI (over-
    confident under within-cell autocorrelation); pair the two
    pools: DL for the heterogeneity diagnostic, cluster bootstrap
    for the publication-grade CI."""
    rho_lower: float
    rho_upper: float
    rho_median: float
    n_resamples: int
    alpha: float
    seed: int


def _pool_rhos_dl(
    rhos: Sequence[float], ns: Sequence[int], df_offset: int,
) -> float:
    """Compute the DL-pooled ρ from a per-burst (ρ, n) trajectory
    via `_fisher_z_dl_pool`. Returns the inverse-Fisher-z'd point
    estimate (`rho_pooled`). NaN propagates when DL is undefined
    (fewer than 2 valid bursts). Bootstrap iterations that
    resample to a degenerate panel (e.g. all-same-cell after
    replacement) get NaN, which the percentile reducer filters
    out."""
    dl = _fisher_z_dl_pool(rhos, ns, df_offset)
    return dl.rho_pooled


def _per_burst_rhos_from_subset(
    *,
    arm_codes: npt.NDArray[np.float64],
    mediator_lists: Sequence[Sequence[float]],
    outcome_lists: Sequence[Sequence[float]],
    cell_idx: npt.NDArray[np.intp],
    n_bursts: int,
    min_n_per_burst: int,
    kind: Literal['marginal', 'partial'],
) -> tuple[list[float], list[int]]:
    """Compute per-burst ρ + n on a subset of cells (given by
    `cell_idx`). The subset is built by indexing the parallel
    arrays; resampling-with-replacement repeats the same cell
    multiple times in `cell_idx`, which is the cluster-bootstrap
    semantics (a duplicated cell contributes all its bursts to
    each replica).

    `kind` selects which ρ to compute: 'marginal' calls
    `_spearman_marginal(arm, outcome)`; 'partial' calls
    `partial_spearman_rho(arm, outcome, mediator)`. Same
    primitives the non-bootstrap `_compute_one_stratum` path uses
    — guarantees the bootstrap distribution centres on the
    point estimate by construction."""
    # Build subsetted arm-code vector + per-burst lookups.
    # Using lists-of-lists indexed by `cell_idx` to mirror the
    # `_gather_burst_b` shape without materialising NumPy 2-D
    # matrices (mediator_lists / outcome_lists are ragged-length
    # in the general case).
    sub_arm_codes = arm_codes[cell_idx]
    sub_mediator: list[Sequence[float]] = [
        mediator_lists[int(i)] for i in cell_idx
    ]
    sub_outcome: list[Sequence[float]] = [
        outcome_lists[int(i)] for i in cell_idx
    ]

    rho_list: list[float] = []
    n_list: list[int] = []
    for b in range(n_bursts):
        x_np, y_np, z_np = _gather_burst_b(
            sub_arm_codes, sub_mediator, sub_outcome, b,
        )
        n_b = int(x_np.size)
        n_list.append(n_b)
        if n_b < min_n_per_burst:
            rho_list.append(float('nan'))
            continue
        # Spearman is invariant to monotone transforms but degenerate
        # when arm has no variance — the cluster bootstrap CAN sample
        # all-treatment / all-baseline cells when n_cells is small.
        # Guard explicitly: NaN-out the burst when single-arm.
        if float(np.std(x_np)) == 0.0:
            rho_list.append(float('nan'))
            continue
        if kind == 'marginal':
            r, _ = _graph_spearman_marginal(x_np, y_np)
        else:
            r, _ = partial_spearman_rho(x_np, y_np, z_np)
        rho_list.append(float(r))
    return rho_list, n_list


def _per_burst_rhos_from_subset_multi(
    *,
    arm_codes: npt.NDArray[np.float64],
    mediator_lists: Sequence[Sequence[Sequence[float]]],
    outcome_lists: Sequence[Sequence[float]],
    cell_idx: npt.NDArray[np.intp],
    n_bursts: int,
    min_n_per_burst: int,
    kind: Literal['marginal', 'partial'],
) -> tuple[list[float], list[int]]:
    """Multi-mediator sibling of `_per_burst_rhos_from_subset`.

    `kind='marginal'` ignores the mediator matrix; `kind='partial'`
    calls `partial_spearman_rho_multi` on the (n_b, k) z-matrix.
    For k=1 the result differs from the closed-form
    `partial_spearman_rho` only in tie-handling drift (matches the
    static `partial_spearman` dispatch rule: k=1 → closed-form;
    k≥2 → OLS-residual). The dispatch-by-k decision lives at the
    caller (the public primitive), not here — this helper does
    multi only.

    Mirrors the depth-1 `_per_burst_rhos_from_subset`'s zero-arm-
    variance NaN guard for the bootstrap path (cluster-resampling
    can give all-treatment / all-baseline replicas at small n)."""
    sub_arm_codes = arm_codes[cell_idx]
    sub_mediator: list[Sequence[Sequence[float]]] = [
        mediator_lists[int(i)] for i in cell_idx
    ]
    sub_outcome: list[Sequence[float]] = [
        outcome_lists[int(i)] for i in cell_idx
    ]

    rho_list: list[float] = []
    n_list: list[int] = []
    for b in range(n_bursts):
        x_np, y_np, z_mat = _gather_burst_b_multi(
            sub_arm_codes, sub_mediator, sub_outcome, b,
        )
        n_b = int(x_np.size)
        n_list.append(n_b)
        if n_b < min_n_per_burst:
            rho_list.append(float('nan'))
            continue
        if float(np.std(x_np)) == 0.0:
            rho_list.append(float('nan'))
            continue
        if kind == 'marginal':
            r, _ = _graph_spearman_marginal(x_np, y_np)
        else:
            r, _ = partial_spearman_rho_multi(x_np, y_np, z_mat)
        rho_list.append(float(r))
    return rho_list, n_list


def _cluster_bootstrap_pool(
    *,
    arm_codes: npt.NDArray[np.float64],
    mediator_lists: Sequence[Sequence[float]],
    outcome_lists: Sequence[Sequence[float]],
    n_bursts: int,
    min_n_per_burst: int,
    kind: Literal['marginal', 'partial'],
    df_offset: int,
    n_resamples: int,
    alpha: float,
    seed: int,
) -> ClusterBootstrapInterval:
    """Cluster bootstrap CI over the DL pool. Cells are the
    resampling unit (each cell = one training trajectory = one
    independent unit); bursts within a cell stay together. For
    each of `n_resamples` iterations:

      1. Sample `n_cells` cell indices with replacement.
      2. Recompute per-burst ρ (and n) from the resampled panel.
      3. Pool the per-burst ρ via DL → one bootstrap-replica ρ.

    The empirical [α/2, 1 − α/2] percentile range of the
    bootstrap-replica ρ distribution is the CI; the median is the
    point estimate (more robust than mean under asymmetric
    bootstrap distributions). NaN replicas (bootstrap panels too
    degenerate to compute DL — e.g. all-same-cell or single-arm
    under replacement) are filtered before the percentile call so
    they don't bias the bounds.

    Deterministic given `seed` via `np.random.default_rng`."""
    n_cells = arm_codes.size
    rng = np.random.default_rng(seed)
    replicas: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n_cells, size=n_cells)
        rhos, ns = _per_burst_rhos_from_subset(
            arm_codes=arm_codes,
            mediator_lists=mediator_lists,
            outcome_lists=outcome_lists,
            cell_idx=idx,
            n_bursts=n_bursts,
            min_n_per_burst=min_n_per_burst,
            kind=kind,
        )
        r = _pool_rhos_dl(rhos, ns, df_offset)
        if not math.isnan(r):
            replicas.append(r)
    if not replicas:
        # Every replica was degenerate (extreme small-stratum
        # boundary). Return NaN bounds — consumers see a NaN
        # interval and know to expand the panel.
        return ClusterBootstrapInterval(
            rho_lower=float('nan'),
            rho_upper=float('nan'),
            rho_median=float('nan'),
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        )
    arr = np.asarray(replicas, dtype=np.float64)
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    rho_lower = float(np.percentile(arr, lo_q))
    rho_upper = float(np.percentile(arr, hi_q))
    rho_median = float(np.median(arr))
    return ClusterBootstrapInterval(
        rho_lower=rho_lower,
        rho_upper=rho_upper,
        rho_median=rho_median,
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
    )


def _cluster_bootstrap_pool_multi(
    *,
    arm_codes: npt.NDArray[np.float64],
    mediator_lists: Sequence[Sequence[Sequence[float]]],
    outcome_lists: Sequence[Sequence[float]],
    n_bursts: int,
    min_n_per_burst: int,
    kind: Literal['marginal', 'partial'],
    df_offset: int,
    n_resamples: int,
    alpha: float,
    seed: int,
) -> ClusterBootstrapInterval:
    """Multi-mediator cluster bootstrap CI over the DL pool. Same
    cell-resampling pattern as `_cluster_bootstrap_pool` but
    `kind='partial'` recomputes ρ via `partial_spearman_rho_multi`
    on the (n_b, k) z-matrix per burst per replica. `df_offset`
    should be `3 + k` to match the multi-Z Fisher-z df accounting
    inside `_fisher_z_dl_pool`.

    The marginal `kind='marginal'` path is identical to the
    depth-1 sibling (mediators don't enter the marginal ρ); we
    keep the parameter so consumers don't fork their bootstrap
    plumbing by depth."""
    n_cells = arm_codes.size
    rng = np.random.default_rng(seed)
    replicas: list[float] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n_cells, size=n_cells)
        rhos, ns = _per_burst_rhos_from_subset_multi(
            arm_codes=arm_codes,
            mediator_lists=mediator_lists,
            outcome_lists=outcome_lists,
            cell_idx=idx,
            n_bursts=n_bursts,
            min_n_per_burst=min_n_per_burst,
            kind=kind,
        )
        r = _pool_rhos_dl(rhos, ns, df_offset)
        if not math.isnan(r):
            replicas.append(r)
    if not replicas:
        return ClusterBootstrapInterval(
            rho_lower=float('nan'),
            rho_upper=float('nan'),
            rho_median=float('nan'),
            n_resamples=n_resamples,
            alpha=alpha,
            seed=seed,
        )
    arr = np.asarray(replicas, dtype=np.float64)
    lo_q = 100.0 * (alpha / 2.0)
    hi_q = 100.0 * (1.0 - alpha / 2.0)
    rho_lower = float(np.percentile(arr, lo_q))
    rho_upper = float(np.percentile(arr, hi_q))
    rho_median = float(np.median(arr))
    return ClusterBootstrapInterval(
        rho_lower=rho_lower,
        rho_upper=rho_upper,
        rho_median=rho_median,
        n_resamples=n_resamples,
        alpha=alpha,
        seed=seed,
    )


@dataclass(frozen=True, slots=True)
class ClusterBootstrapEdgeCounts:
    """Empirical CIs on PC edge-classification counts from a cluster
    bootstrap over cells.

    Conceptually DISTINCT from `ClusterBootstrapInterval` on the
    ρ-pool: the ρ-pool interval answers "what's the average effect
    magnitude under bootstrap resampling?" (a continuous quantity);
    the edge-count interval answers "is the edge classification
    *robust* to which cells we sampled?" (an integer-count
    quantity over the per-burst Fisher-z CI decision). A wide CI
    on the dsep / direct / marginal counts means the verdict is
    driven by which subset of cells happened to land in the panel
    — a few outlier cells flip the per-burst CI test from "edge
    present" to "edge absent" at multiple bursts, so the count
    triple drifts across resamples.

    Each count's lower / upper bound is the empirical α/2 and
    1 − α/2 percentile of the count across `n_resamples`
    cell-resampled panels. `median` is the bootstrap
    distribution's median (robust integer point estimate). Counts
    on a resample replica are *that replica's* per-burst CI
    decisions, recomputed from the resampled per-burst (ρ, n)
    trajectory — identical machinery to the non-bootstrap path,
    so the bootstrap distribution centres on the original count
    by construction.

    Assumption-free under any within-cell autocorrelation
    structure — the cluster bootstrap resamples whole cells, so
    bursts within one cell stay together. Same methodological
    foundation as `ClusterBootstrapInterval` (Deen & de Rooij
    2020; cluster-robust SE under within-cluster dependence).
    """
    # Marginal-edge count CI: how many bursts the marginal CI test
    # rejects at α across the bootstrap distribution.
    marg_lower: int
    marg_median: int
    marg_upper: int
    # Mediator-d-separates count CI: how many bursts have marginal
    # edge present AND conditional edge absent.
    dsep_lower: int
    dsep_median: int
    dsep_upper: int
    # Direct-edge count CI: how many bursts have both marginal and
    # conditional edges present.
    direct_lower: int
    direct_median: int
    direct_upper: int
    # Provenance — mirrors ClusterBootstrapInterval.
    n_resamples: int
    alpha: float
    seed: int


def _per_burst_edge_counts_from_subset(
    *,
    arm_codes: npt.NDArray[np.float64],
    mediator_lists: Sequence[Sequence[float]],
    outcome_lists: Sequence[Sequence[float]],
    cell_idx: npt.NDArray[np.intp],
    n_bursts: int,
    min_n_per_burst: int,
    alpha: float,
) -> tuple[int, int, int]:
    """Recompute the PC edge-classification count triple on a
    resampled subset of cells. Reuses the SAME primitives the
    non-bootstrap path uses (`_spearman_marginal` for the depth-0
    p-value via scipy's `spearmanr` t-approximation;
    `partial_spearman_rho` for the depth-1 p-value via the
    closed-form Fisher-z df=n-4 normal CDF). Calling the same
    primitives guarantees the bootstrap distribution centres on
    the original count by construction — we'd diverge if we
    re-derived p-values from ρ analytically.

    Returns `(n_marg, n_dsep, n_direct)` for this subset. Bursts
    with `n_b < min_n_per_burst` contribute nothing (no edge can
    be asserted). Single-arm bursts (zero variance on arm under
    resampling) contribute nothing — `_spearman_marginal` returns
    (NaN, NaN) at zero variance."""
    sub_arm_codes = arm_codes[cell_idx]
    sub_mediator: list[Sequence[float]] = [
        mediator_lists[int(i)] for i in cell_idx
    ]
    sub_outcome: list[Sequence[float]] = [
        outcome_lists[int(i)] for i in cell_idx
    ]
    n_marg, n_dsep, n_direct = 0, 0, 0
    for b in range(n_bursts):
        x_np, y_np, z_np = _gather_burst_b(
            sub_arm_codes, sub_mediator, sub_outcome, b,
        )
        n_b = int(x_np.size)
        if n_b < min_n_per_burst:
            continue
        # Skip degenerate single-arm bursts under resampling.
        if float(np.std(x_np)) == 0.0:
            continue
        _, p_m = _graph_spearman_marginal(x_np, y_np)
        if math.isnan(p_m) or p_m >= alpha:
            continue
        n_marg += 1
        _, p_p = partial_spearman_rho(x_np, y_np, z_np)
        # NaN p-value (degenerate variance, ill-conditioned
        # partial) → treated as "no conditional edge" by the
        # Fisher-z CI test's null convention → dsep at this burst.
        if math.isnan(p_p) or p_p >= alpha:
            n_dsep += 1
        else:
            n_direct += 1
    return n_marg, n_dsep, n_direct


def _per_burst_edge_counts_from_subset_multi(
    *,
    arm_codes: npt.NDArray[np.float64],
    mediator_lists: Sequence[Sequence[Sequence[float]]],
    outcome_lists: Sequence[Sequence[float]],
    cell_idx: npt.NDArray[np.intp],
    n_bursts: int,
    min_n_per_burst: int,
    alpha: float,
) -> tuple[int, int, int]:
    """Multi-mediator sibling of `_per_burst_edge_counts_from_subset`.

    Same machinery + same primitives the multi-mediator non-
    bootstrap path uses (`_spearman_marginal` for the depth-0 p,
    `partial_spearman_rho_multi` for the depth-k p with Fisher-z
    df = n − 3 − k). Calling the same primitives guarantees the
    bootstrap distribution centres on the original count by
    construction."""
    sub_arm_codes = arm_codes[cell_idx]
    sub_mediator: list[Sequence[Sequence[float]]] = [
        mediator_lists[int(i)] for i in cell_idx
    ]
    sub_outcome: list[Sequence[float]] = [
        outcome_lists[int(i)] for i in cell_idx
    ]
    n_marg, n_dsep, n_direct = 0, 0, 0
    for b in range(n_bursts):
        x_np, y_np, z_mat = _gather_burst_b_multi(
            sub_arm_codes, sub_mediator, sub_outcome, b,
        )
        n_b = int(x_np.size)
        if n_b < min_n_per_burst:
            continue
        if float(np.std(x_np)) == 0.0:
            continue
        _, p_m = _graph_spearman_marginal(x_np, y_np)
        if math.isnan(p_m) or p_m >= alpha:
            continue
        n_marg += 1
        _, p_p = partial_spearman_rho_multi(x_np, y_np, z_mat)
        if math.isnan(p_p) or p_p >= alpha:
            n_dsep += 1
        else:
            n_direct += 1
    return n_marg, n_dsep, n_direct


def _cluster_bootstrap_edge_counts_multi(
    *,
    arm_codes: npt.NDArray[np.float64],
    mediator_lists: Sequence[Sequence[Sequence[float]]],
    outcome_lists: Sequence[Sequence[float]],
    n_bursts: int,
    min_n_per_burst: int,
    alpha: float,
    n_resamples: int,
    bootstrap_alpha: float,
    seed: int,
) -> ClusterBootstrapEdgeCounts:
    """Multi-mediator sibling of `_cluster_bootstrap_edge_counts`.

    Cell-resampling pattern identical; per-replica edge-count
    triple is recomputed via the multi-Z CI primitive. The "joint
    mediator set d-separates arm from outcome" interpretation is
    the depth-k generalisation of the depth-1 semantics."""
    n_cells = arm_codes.size
    rng = np.random.default_rng(seed)
    marg_counts: list[int] = []
    dsep_counts: list[int] = []
    direct_counts: list[int] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n_cells, size=n_cells)
        n_m, n_d, n_dir = _per_burst_edge_counts_from_subset_multi(
            arm_codes=arm_codes,
            mediator_lists=mediator_lists,
            outcome_lists=outcome_lists,
            cell_idx=idx,
            n_bursts=n_bursts,
            min_n_per_burst=min_n_per_burst,
            alpha=alpha,
        )
        marg_counts.append(n_m)
        dsep_counts.append(n_d)
        direct_counts.append(n_dir)

    if not marg_counts:
        return ClusterBootstrapEdgeCounts(
            marg_lower=0, marg_median=0, marg_upper=0,
            dsep_lower=0, dsep_median=0, dsep_upper=0,
            direct_lower=0, direct_median=0, direct_upper=0,
            n_resamples=n_resamples,
            alpha=bootstrap_alpha,
            seed=seed,
        )

    lo_q = 100.0 * (bootstrap_alpha / 2.0)
    hi_q = 100.0 * (1.0 - bootstrap_alpha / 2.0)

    def _ci(counts: list[int]) -> tuple[int, int, int]:
        arr = np.asarray(counts, dtype=np.float64)
        lo = int(round(float(np.percentile(arr, lo_q))))
        hi = int(round(float(np.percentile(arr, hi_q))))
        med = int(round(float(np.median(arr))))
        return (lo, med, hi)

    m_lo, m_med, m_hi = _ci(marg_counts)
    d_lo, d_med, d_hi = _ci(dsep_counts)
    dir_lo, dir_med, dir_hi = _ci(direct_counts)

    return ClusterBootstrapEdgeCounts(
        marg_lower=m_lo, marg_median=m_med, marg_upper=m_hi,
        dsep_lower=d_lo, dsep_median=d_med, dsep_upper=d_hi,
        direct_lower=dir_lo, direct_median=dir_med, direct_upper=dir_hi,
        n_resamples=n_resamples,
        alpha=bootstrap_alpha,
        seed=seed,
    )


def _cluster_bootstrap_edge_counts(
    *,
    arm_codes: npt.NDArray[np.float64],
    mediator_lists: Sequence[Sequence[float]],
    outcome_lists: Sequence[Sequence[float]],
    n_bursts: int,
    min_n_per_burst: int,
    alpha: float,
    n_resamples: int,
    bootstrap_alpha: float,
    seed: int,
) -> ClusterBootstrapEdgeCounts:
    """Cluster bootstrap CI on the PC edge-classification count
    triple. Cells are the resampling unit (each cell = one
    training trajectory = one independent unit); bursts within a
    cell stay together.

    For each of `n_resamples` iterations:
      1. Sample `n_cells` cell indices with replacement.
      2. Recompute per-burst CI decisions on the resampled subset
         using the SAME machinery (`_spearman_marginal` +
         `partial_spearman_rho`) the non-bootstrap path uses; sum
         to a (n_marg, n_dsep, n_direct) triple.
      3. Collect.

    Then take empirical [`bootstrap_alpha`/2, 1 − `bootstrap_alpha`/2]
    percentiles for EACH count separately and the median across
    replicas. Integer outputs throughout — we round the
    percentile result to the nearest integer (the percentile
    interpolation can land between integer counts).

    Sibling to `_cluster_bootstrap_pool` (the ρ-pool variant);
    same cell-resampling pattern, different inner computation. The
    two are independent — consumers that want both pay for both
    via two passes over the resampled panels.

    Deterministic given `seed` via `np.random.default_rng`."""
    n_cells = arm_codes.size
    rng = np.random.default_rng(seed)
    marg_counts: list[int] = []
    dsep_counts: list[int] = []
    direct_counts: list[int] = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n_cells, size=n_cells)
        n_m, n_d, n_dir = _per_burst_edge_counts_from_subset(
            arm_codes=arm_codes,
            mediator_lists=mediator_lists,
            outcome_lists=outcome_lists,
            cell_idx=idx,
            n_bursts=n_bursts,
            min_n_per_burst=min_n_per_burst,
            alpha=alpha,
        )
        marg_counts.append(n_m)
        dsep_counts.append(n_d)
        direct_counts.append(n_dir)

    if not marg_counts:
        # Defensive: n_resamples == 0 would yield empty lists.
        # Surface zeros — the caller's gate (`n_bootstrap > 0`)
        # is the typed contract.
        return ClusterBootstrapEdgeCounts(
            marg_lower=0, marg_median=0, marg_upper=0,
            dsep_lower=0, dsep_median=0, dsep_upper=0,
            direct_lower=0, direct_median=0, direct_upper=0,
            n_resamples=n_resamples,
            alpha=bootstrap_alpha,
            seed=seed,
        )

    lo_q = 100.0 * (bootstrap_alpha / 2.0)
    hi_q = 100.0 * (1.0 - bootstrap_alpha / 2.0)

    def _ci(counts: list[int]) -> tuple[int, int, int]:
        arr = np.asarray(counts, dtype=np.float64)
        lo = int(round(float(np.percentile(arr, lo_q))))
        hi = int(round(float(np.percentile(arr, hi_q))))
        med = int(round(float(np.median(arr))))
        return (lo, med, hi)

    m_lo, m_med, m_hi = _ci(marg_counts)
    d_lo, d_med, d_hi = _ci(dsep_counts)
    dir_lo, dir_med, dir_hi = _ci(direct_counts)

    return ClusterBootstrapEdgeCounts(
        marg_lower=m_lo, marg_median=m_med, marg_upper=m_hi,
        dsep_lower=d_lo, dsep_median=d_med, dsep_upper=d_hi,
        direct_lower=dir_lo, direct_median=dir_med, direct_upper=dir_hi,
        n_resamples=n_resamples,
        alpha=bootstrap_alpha,
        seed=seed,
    )


__all__ = [
    'ClusterBootstrapEdgeCounts',
    'ClusterBootstrapInterval',
    'FisherZDLPool',
    'Stratum',
    'TimeAggregationStatus',
    '_ColumnOrMeasurable',
    '_PerBurstMeasurable',
    '_as_float_list',
    '_classify_status',
    '_cluster_bootstrap_edge_counts',
    '_cluster_bootstrap_edge_counts_multi',
    '_cluster_bootstrap_pool',
    '_cluster_bootstrap_pool_multi',
    '_collect_arm_and_per_burst',
    '_collect_arm_and_per_burst_multi',
    '_encode_arm',
    '_fisher_z_dl_pool',
    '_gather_burst_b',
    '_gather_burst_b_multi',
    '_n_bursts',
    '_n_bursts_multi',
    '_resolve_per_burst',
    '_source_name',
    '_stratum_key',
]
