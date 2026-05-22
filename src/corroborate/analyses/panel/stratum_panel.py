"""`stratum_panel` — the unified per-stratum primitive.

Builds ONE per-stratum panel carrying all the per-stratum
statistics that downstream stratified analyses need:

- per-stratum, per-arm cell counts (`n_treatment`, `n_baseline`)
- per-stratum, per-arm means + stds for each named measurable
- per-stratum Δ (treatment mean − baseline mean) per measurable
- per-stratum within-stratum Spearman r for each measurable pair

Downstream consumers (`stratified_arm_diff_pooled`,
`stratum_effect_panel`, the `stratum_*_dowhy` family) take this
panel as a fixture (resolved by parameter injection per the
@analysis fixture pattern) and produce specialized aggregates.
No more scattered per-panel-build code.

(Note: the earlier `dl_pool` / `fisher_z_pool` / `panel_partial`
@analysis wrappers in `panel_consumers.py` were deleted in phase
B.4 — they had zero bridge consumers and `fisher_z_pool`'s
stateless implementation was promoted to `stats.fisher_z_pool`.
Bridges wanting DerSimonian-Laird pooling on the panel should
fixture `stratified_arm_diff_pooled` directly, which already
embeds that path.)

**Why this exists.** The previous design scattered cell→panel
build logic across 8+ analyses (each rolling its own
stratify+aggregate+pool), made per-stratum data asymmetrically
exposed (some primitives expose `per_stratum`, others only pool),
and blocked the natural `scope_from_panel(upstream, ...)` pattern
because per-stratum results weren't first-class. Centralizing
into one fixture closes all three gaps.

**Performance.** Pre-computes within-stratum Spearman for all
M*(M-1)/2 measurable pairs. For typical M (3-12), this is cheap
relative to the rank-transform cost. Lazy computation is
possible (compute on demand) but eager pre-computation simplifies
the result type and matches the panel-as-primitive principle —
the panel IS the closure of per-stratum statistics.

**Substrate-neutral.** This primitive knows nothing about RL,
envs, or DDQN. Substrate authors pass `stratify_by` +
`measurables` and get a panel back.

**Scope of unification.** This panel is CROSS-ARM — it assumes
two arms (treatment, baseline) and computes per-arm means/stds
+ within-stratum across-arm Spearman. Analyses on a single
cohort without an arm distinction (e.g. `stratified_spearman` on
all cells regardless of arm) do NOT migrate to this panel —
they're conceptually arm-agnostic and live in
`graph.discovery.stratified_spearman_rho` separately. The
unification targets the cross-arm family
(`stratified_arm_diff_pooled`, `stratum_effect_panel`, the
`stratum_*_dowhy` family)."""
from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from corroborate.analyses._cell_value import resolve_value
from corroborate.bridge.analysis import analysis


type StratumAggregator = Literal['mean', 'median']

# Normalize measurable pair to a sorted tuple for order-insensitive
# lookup. `pair_key('a', 'b') == pair_key('b', 'a')`. Module-public
# so consumer fixtures (panel_consumers.py) can use the same key
# convention.
def pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted((a, b)))  # type: ignore[return-value]


# Backward-compatible alias kept private — only internal callers
# rely on it.
_pair_key = pair_key


@dataclass(frozen=True, slots=True)
class StratumPanel:
    """Unified per-stratum panel.

    Indexed by `strata` (the per-stratum-identifier tuples in
    order). Each per-stratum field is a parallel tuple indexed
    the same way.

    `measurables` is the column-name closure available for
    statistics. Per-measurable per-arm statistics live in
    `means_*` / `stds_*` / `deltas`. Per-pair statistics live in
    `spearman_within` (keyed by `_pair_key(x, y)`).

    `n_strata` / `cohen_d(measurable)` / `cohen_se(measurable)`
    are derived view methods.
    """
    stratify_by: tuple[str, ...]
    strata: tuple[tuple[object, ...], ...]
    measurables: tuple[str, ...]
    treatment_arm: str
    baseline_arm: str
    aggregator: StratumAggregator

    # Per-stratum, per-arm CELL counts (regardless of which
    # measurables have finite values on those cells).
    n_treatment: tuple[int, ...]
    n_baseline: tuple[int, ...]

    means_treatment: Mapping[str, tuple[float, ...]]
    means_baseline: Mapping[str, tuple[float, ...]]
    stds_treatment: Mapping[str, tuple[float, ...]]
    stds_baseline: Mapping[str, tuple[float, ...]]

    # Per-stratum, per-arm, per-measurable FINITE-VALUE counts.
    # `n_treatment_per_measurable[m][i]` is the number of cells in
    # (stratum i, treatment arm) where measurable `m` has a finite
    # value. This is what the existing `_cohen_d_indep_samples`
    # used for n in its SE formula — preserving verdicts across
    # migrations requires using these counts (not the total cell
    # counts) in downstream Cohen's d / SE computations.
    n_treatment_per_measurable: Mapping[str, tuple[int, ...]]
    n_baseline_per_measurable: Mapping[str, tuple[int, ...]]

    # Per-stratum within-stratum Spearman r for each measurable
    # pair. Pooled over both arms (rank-transform of all cells in
    # the stratum, regardless of arm — i.e., the marginal r).
    spearman_within: Mapping[tuple[str, str], tuple[float, ...]]

    @property
    def n_strata(self) -> int:
        return len(self.strata)

    @property
    def deltas(self) -> Mapping[str, tuple[float, ...]]:
        """Per-stratum Δ = mean_treatment − mean_baseline for
        each measurable."""
        return {
            m: tuple(
                self.means_treatment[m][i] - self.means_baseline[m][i]
                for i in range(self.n_strata)
            )
            for m in self.measurables
        }

    def cohen_d(self, measurable: str) -> tuple[float, ...]:
        """Per-stratum independent-samples Cohen's d for
        `measurable`. Uses the simple-mean-variance form
        `d = (μ_t − μ_b) / sqrt((σ_t² + σ_b²) / 2)` matching the
        existing `_cohen_d_indep_samples` in
        `stratified_arm_diff_pooled` — migration-compatible.

        n_t / n_b are the FINITE-VALUE counts for the measurable
        (not total cell counts) — matching the existing semantics.
        NaN where pooled SD is zero or finite-n < 2 on either arm."""
        if measurable not in self.measurables:
            raise KeyError(
                f'cohen_d({measurable!r}): not in panel '
                f'measurables {self.measurables!r}',
            )
        out: list[float] = []
        for i in range(self.n_strata):
            n_t = self.n_treatment_per_measurable[measurable][i]
            n_b = self.n_baseline_per_measurable[measurable][i]
            if n_t < 2 or n_b < 2:
                out.append(float('nan'))
                continue
            s_t = self.stds_treatment[measurable][i]
            s_b = self.stds_baseline[measurable][i]
            pooled_var = (s_t ** 2 + s_b ** 2) / 2.0
            if pooled_var <= 0 or math.isnan(pooled_var):
                out.append(float('nan'))
                continue
            pooled_sd = math.sqrt(pooled_var)
            d = (
                self.means_treatment[measurable][i]
                - self.means_baseline[measurable][i]
            ) / pooled_sd
            out.append(d)
        return tuple(out)

    def cohen_se(self, measurable: str) -> tuple[float, ...]:
        """Per-stratum SE of Cohen's d for `measurable` using the
        Hedges 1981 independent-samples form
        `SE ≈ sqrt((n_t+n_b)/(n_t n_b) + d²/(2(n_t+n_b−2)))` —
        migration-compatible with `_cohen_d_indep_samples`.

        n_t / n_b are FINITE-VALUE counts for the measurable.
        NaN where finite-n < 2."""
        ds = self.cohen_d(measurable)
        out: list[float] = []
        for i in range(self.n_strata):
            n_t = self.n_treatment_per_measurable[measurable][i]
            n_b = self.n_baseline_per_measurable[measurable][i]
            if n_t < 2 or n_b < 2 or math.isnan(ds[i]):
                out.append(float('nan'))
                continue
            n_sum = n_t + n_b
            n_prod = n_t * n_b
            df_se = n_sum - 2
            if n_prod == 0 or df_se <= 0:
                out.append(float('nan'))
                continue
            se_sq = n_sum / n_prod + ds[i] ** 2 / (2.0 * df_se)
            out.append(math.sqrt(max(se_sq, 0.0)))
        return tuple(out)


def _stratum_mean(values: list[float], aggregator: StratumAggregator) -> float:
    if not values:
        return float('nan')
    if aggregator == 'mean':
        return float(np.mean(values))
    return float(np.median(values))


def _stratum_std(values: list[float]) -> float:
    if len(values) < 2:
        return float('nan')
    return float(np.std(values, ddof=1))


@analysis
def stratum_panel(
    cells: pl.DataFrame | Iterable[Mapping[str, object]],
    *,
    measurables: tuple[str, ...],
    treatment_arm: str,
    baseline_arm: str,
    stratify_by: tuple[str, ...] = ('env_name',),
    arm_field: str = 'arm_key',
    min_seeds_per_arm: int = 3,
    aggregator: StratumAggregator = 'mean',
) -> StratumPanel:
    """Build the unified per-stratum panel.

    Canonical input is `pl.DataFrame` (e.g. `Panel.cells`, the
    runner's scope-filtered cells frame, or any caller-built
    DataFrame). `Iterable[Mapping[str, object]]` is accepted as
    a back-compat fallback for synthetic test cells + ad-hoc
    dict lists; converted to DataFrame at entry via
    `corroborate.data.kernel.cells_to_dataframe`.

    Algorithm:
    1. Bucket cells by (arm, stratum_key).
    2. Find strata where BOTH arms have ≥ `min_seeds_per_arm`.
    3. For each valid stratum: compute per-arm means, stds, n for
       each `measurable`.
    4. For each measurable pair: compute within-stratum Spearman r
       on the union of both arms' cells (marginal r).

    `measurables` resolves via `resolve_value` — measurable
    registry first, then cell field-path read.

    `aggregator='mean'` (default) gives `mean(X|T) - mean(X|B)` Δ.
    `aggregator='median'` gives median Δ — sibling for outlier-
    robust analyses; affects `means_*` slots, std slots stay as
    arithmetic sample SD."""
    # Canonical: pl.DataFrame in, per-cell-loop algorithm.
    # Iterable[Mapping] fallback for back-compat with synthetic
    # tests + non-Panel ad-hoc callers.
    from corroborate.data.kernel import cells_to_dataframe
    if not isinstance(cells, pl.DataFrame):
        cells = cells_to_dataframe(cells)
    cells = cells.iter_rows(named=True)
    per_arm_stratum: dict[
        tuple[str, tuple[object, ...]], list[Mapping[str, object]],
    ] = defaultdict(list)
    for c in cells:
        arm = c.get(arm_field)
        if not isinstance(arm, str):
            continue
        if arm not in (treatment_arm, baseline_arm):
            continue
        sk = tuple(c.get(k) for k in stratify_by)
        per_arm_stratum[(arm, sk)].append(c)

    all_strata: set[tuple[object, ...]] = set()
    for _, sk in per_arm_stratum.keys():
        all_strata.add(sk)

    valid_strata: list[tuple[object, ...]] = []
    for sk in sorted(all_strata, key=lambda s: tuple(repr(v) for v in s)):
        t_cells = per_arm_stratum.get((treatment_arm, sk), [])
        b_cells = per_arm_stratum.get((baseline_arm, sk), [])
        if len(t_cells) >= min_seeds_per_arm and len(b_cells) >= min_seeds_per_arm:
            valid_strata.append(sk)

    strata = tuple(valid_strata)

    # Per-measurable, per-arm value lists per stratum.
    def _values(
        cs: list[Mapping[str, object]], m: str,
    ) -> list[float]:
        out: list[float] = []
        for c in cs:
            try:
                v = resolve_value(c, m)
            except (KeyError, TypeError, ValueError):
                continue
            if not math.isnan(v):
                out.append(v)
        return out

    means_t: dict[str, list[float]] = {m: [] for m in measurables}
    means_b: dict[str, list[float]] = {m: [] for m in measurables}
    stds_t: dict[str, list[float]] = {m: [] for m in measurables}
    stds_b: dict[str, list[float]] = {m: [] for m in measurables}
    n_t_per_m: dict[str, list[int]] = {m: [] for m in measurables}
    n_b_per_m: dict[str, list[int]] = {m: [] for m in measurables}
    n_t_list: list[int] = []
    n_b_list: list[int] = []

    # Also need raw within-stratum value lists for Spearman pair r
    # (rank-transform on union of cells from both arms).
    per_stratum_raw: list[dict[str, list[float]]] = []

    for sk in strata:
        t_cells = per_arm_stratum.get((treatment_arm, sk), [])
        b_cells = per_arm_stratum.get((baseline_arm, sk), [])
        n_t_list.append(len(t_cells))
        n_b_list.append(len(b_cells))
        all_cells = t_cells + b_cells
        stratum_raw: dict[str, list[float]] = {}
        for m in measurables:
            t_vals = _values(t_cells, m)
            b_vals = _values(b_cells, m)
            n_t_per_m[m].append(len(t_vals))
            n_b_per_m[m].append(len(b_vals))
            means_t[m].append(_stratum_mean(t_vals, aggregator))
            means_b[m].append(_stratum_mean(b_vals, aggregator))
            stds_t[m].append(_stratum_std(t_vals))
            stds_b[m].append(_stratum_std(b_vals))
            stratum_raw[m] = _values(all_cells, m)
        per_stratum_raw.append(stratum_raw)

    # Compute per-stratum within-stratum Spearman for each pair.
    spearman_within: dict[tuple[str, str], list[float]] = {}
    for i, m1 in enumerate(measurables):
        for m2 in measurables[i:]:  # include self-pair (=1.0) for completeness
            key = _pair_key(m1, m2)
            if key in spearman_within:
                continue
            per_stratum_r: list[float] = []
            for stratum_raw in per_stratum_raw:
                v1 = stratum_raw[m1]
                v2 = stratum_raw[m2]
                n = min(len(v1), len(v2))
                if n < 4:
                    per_stratum_r.append(float('nan'))
                    continue
                a1 = np.asarray(v1[:n], dtype=np.float64)
                a2 = np.asarray(v2[:n], dtype=np.float64)
                if float(np.std(a1)) == 0 or float(np.std(a2)) == 0:
                    per_stratum_r.append(float('nan'))
                    continue
                if m1 == m2:
                    per_stratum_r.append(1.0)
                    continue
                r, _ = spearmanr(a1, a2)
                per_stratum_r.append(float(r))
            spearman_within[key] = per_stratum_r

    return StratumPanel(
        stratify_by=stratify_by,
        strata=strata,
        measurables=measurables,
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        aggregator=aggregator,
        n_treatment=tuple(n_t_list),
        n_baseline=tuple(n_b_list),
        means_treatment={m: tuple(means_t[m]) for m in measurables},
        means_baseline={m: tuple(means_b[m]) for m in measurables},
        stds_treatment={m: tuple(stds_t[m]) for m in measurables},
        stds_baseline={m: tuple(stds_b[m]) for m in measurables},
        n_treatment_per_measurable={
            m: tuple(n_t_per_m[m]) for m in measurables
        },
        n_baseline_per_measurable={
            m: tuple(n_b_per_m[m]) for m in measurables
        },
        spearman_within={
            k: tuple(v) for k, v in spearman_within.items()
        },
    )


__all__ = [
    'StratumPanel',
    'stratum_panel',
]
