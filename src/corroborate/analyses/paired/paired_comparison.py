"""`paired_comparison` — the canonical paired-comparison analysis.

Pairs cells on `pair_by` (e.g. `('seed',)`), optionally stratifies
by `group_by` (e.g. `'env_name'`) for random-effects pooling, and
computes Hedges' g + se across paired Δs on `outcome_path`.

Result type `PairedComparisonResult` carries:

- per-arm summary stats (`arm_a_*`, `arm_b_*`)
- overall `effect_size_g` + `se` (single-group) or pooled g/se
  (stratified mode)
- `per_group: tuple[GroupStats, ...]` when `group_by` is set
- `pooled: PooledStats | None`
- diagnostic `n_dropped_unpaired`

Verdict logic does NOT live here — bridges author thresholds via
`holds_when` against this result. The framework supplies the
typed numerical surface; the bridge author decides what HELD /
NO_EFFECT / etc. means for their claim.

Consumed two ways:
1. As a bridge fixture (parameter named `paired_comparison`)
   resolved by the framework's fixture-injection.
2. Directly: `paired_comparison(cells, treatment_arm=...,
   baseline_arm=..., ...)` — the registered analysis is callable
   like any other function."""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from corroborate._internals.polars import to_dicts
from corroborate.bridge.analysis import analysis
from corroborate.core.hypothesis import PredictedDirection
from corroborate.corpus.schema import MeasurementLeaf
from corroborate.stats import (
    PooledStats,
    hedges_g_paired,
    random_effects_summary,
)


@dataclass(frozen=True, slots=True)
class GroupStats:
    """Per-stratum (paired Hedges' g + se + count) summary, one
    per `group_by`-value when `paired_comparison` runs in
    stratified mode.

    `group_value` carries whatever value the `group_by` column had
    for this stratum (e.g. `'CartPole-v1'` when
    `group_by='env_name'`). Heterogeneous Python types are
    intentional — different substrates use different group
    identities.

    Verdict-deriving fields removed in Phase 2 — bridges author
    threshold logic via `holds_when` and consume this slim
    summary as a fixture component."""
    group_value: object
    n_pairs: int
    arm_a_mean: float | None
    arm_a_sd: float | None
    arm_b_mean: float | None
    arm_b_sd: float | None
    effect_size_g: float | None
    se: float | None


@dataclass(frozen=True, slots=True)
class PairedComparisonResult:
    """Output of `paired_comparison`. Ephemeral — not persisted on
    its own; consumed by bridges as a fixture or by downstream
    analyses (e.g. `meta_regress_panel`).

    Carries raw paired stats and provenance (which arms were
    contrasted, on which `pair_by` axis, with what
    `predicted_direction`); the verdict is bridge-author-derived
    via `holds_when` against this result."""
    treatment_arm: str
    baseline_arm: str
    predicted_direction: PredictedDirection | None
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

    # Stratified mode (empty / None when group_by is None).
    per_group: tuple[GroupStats, ...]
    pooled: PooledStats | None

    # Diagnostics.
    n_dropped_unpaired: int


def _cell_pair_key(
    cell: Mapping[str, object],
    pair_by: tuple[str, ...],
) -> tuple[MeasurementLeaf, ...]:
    """Read the tuple of values at `pair_by` keys off a cell.
    Loud error if any key is missing or non-scalar."""
    out: list[MeasurementLeaf] = []
    for k in pair_by:
        v = cell.get(k)
        if v is None:
            raise TypeError(
                f"paired_comparison: cell missing scalar pair-key "
                f"{k!r} (pair_by={pair_by!r})",
            )
        if not isinstance(v, (str, int, float, bool)):
            raise TypeError(
                f"paired_comparison: cell {k!r} value is "
                f"non-scalar: {v!r}",
            )
        out.append(v)
    return tuple(out)


def _cell_outcome(
    cell: Mapping[str, object], outcome_path: str,
) -> float:
    """Read a scalar outcome off a cell; loud error if absent or
    non-numeric."""
    v = cell.get(outcome_path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(
            f"paired_comparison: cell missing scalar "
            f"{outcome_path!r} value",
        )
    return float(v)


def _pair_cells_by_key(
    treatment: Sequence[Mapping[str, object]],
    baseline: Sequence[Mapping[str, object]],
    *,
    pair_by: tuple[str, ...],
    group_label: object | None = None,
) -> tuple[
    list[tuple[MeasurementLeaf, ...]],
    dict[tuple[MeasurementLeaf, ...], Mapping[str, object]],
    dict[tuple[MeasurementLeaf, ...], Mapping[str, object]],
    int,
]:
    """Index treatment + baseline cells by `pair_by`-key; return
    paired keys (sorted), the two key→cell dicts, and the count of
    unpaired-and-dropped cells. Raises on duplicate pair-keys
    within either arm — silent dedup would mask a misconfigured
    slice."""
    def _index(
        cells: Sequence[Mapping[str, object]], side: str,
    ) -> dict[tuple[MeasurementLeaf, ...], Mapping[str, object]]:
        out: dict[
            tuple[MeasurementLeaf, ...], Mapping[str, object],
        ] = {}
        for c in cells:
            pk = _cell_pair_key(c, pair_by)
            if pk in out:
                tag = (
                    f' for group {group_label!r}'
                    if group_label is not None else ''
                )
                raise ValueError(
                    f'paired_comparison: duplicate pair_by='
                    f'{pair_by!r} key {pk!r} in {side}{tag}',
                )
            out[pk] = c
        return out

    t_by_key = _index(treatment, 'treatment')
    b_by_key = _index(baseline, 'baseline')
    paired = sorted(t_by_key.keys() & b_by_key.keys())
    n_dropped = (
        (len(t_by_key) - len(paired))
        + (len(b_by_key) - len(paired))
    )
    return paired, t_by_key, b_by_key, n_dropped


def _per_group_stats(
    group_value: object,
    treatment_cells: Sequence[Mapping[str, object]],
    baseline_cells: Sequence[Mapping[str, object]],
    *,
    outcome_path: str,
    pair_by: tuple[str, ...],
) -> tuple[GroupStats | None, int]:
    """Pair within one group → GroupStats. Returns
    (GroupStats | None, n_dropped_unpaired). None when no pairs
    survive the outcome-finite filter."""
    paired, t_by, b_by, n_dropped = _pair_cells_by_key(
        treatment_cells, baseline_cells,
        pair_by=pair_by, group_label=group_value,
    )
    a_values: list[float] = []
    b_values: list[float] = []
    deltas: list[float] = []
    for pk in paired:
        a = _cell_outcome(t_by[pk], outcome_path)
        b = _cell_outcome(b_by[pk], outcome_path)
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        a_values.append(a)
        b_values.append(b)
        deltas.append(a - b)

    n_pairs = len(deltas)
    if n_pairs == 0:
        return None, n_dropped

    a_mean = float(sum(a_values) / n_pairs)
    b_mean = float(sum(b_values) / n_pairs)
    a_sd: float | None
    b_sd: float | None
    if n_pairs > 1:
        a_var = sum((v - a_mean) ** 2 for v in a_values) / (n_pairs - 1)
        b_var = sum((v - b_mean) ** 2 for v in b_values) / (n_pairs - 1)
        a_sd = math.sqrt(a_var)
        b_sd = math.sqrt(b_var)
    else:
        a_sd = b_sd = None

    g, se = (
        hedges_g_paired(deltas) if n_pairs >= 2
        else (float('nan'), float('nan'))
    )
    g_safe: float | None = None if math.isnan(g) else float(g)
    se_safe: float | None = None if math.isnan(se) else float(se)

    return GroupStats(
        group_value=group_value,
        n_pairs=n_pairs,
        arm_a_mean=a_mean, arm_a_sd=a_sd,
        arm_b_mean=b_mean, arm_b_sd=b_sd,
        effect_size_g=g_safe, se=se_safe,
    ), n_dropped


@analysis(reads=())
def paired_comparison(
    cells: pl.DataFrame,
    *,
    treatment_arm: str,
    baseline_arm: str,
    outcome_path: str,
    pair_by: tuple[str, ...],
    group_by: str | None = None,
    arm_field: str = 'arm_key',
    predicted_direction: PredictedDirection | None = None,
) -> PairedComparisonResult:
    """The canonical paired-comparison analysis.

    Pairs cells on `pair_by` (e.g. `('seed',)`), optionally
    stratifies by `group_by` (e.g. `'env_name'`) for random-effects
    pooling, computes Hedges' g + se across paired Δs on
    `outcome_path`. Returns raw paired stats; the verdict is
    bridge-author-controlled.

    `treatment_arm` / `baseline_arm` are matched against
    `cell[arm_field]` to partition the corpus. `arm_field`
    defaults to `'arm_key'`; legacy parquets keyed on the older
    `'intervention_name'` column can pass that explicitly.

    `predicted_direction` is recorded on the result for downstream
    consumers' sign tests.

    Raises:
    - `ValueError` when `treatment_arm == baseline_arm` (HPO-smuggle).
    - `ValueError` when no cells match either arm.
    - `ValueError` on duplicate `pair_by` keys within an arm
      within a group (silent dedup would mask a misconfigured
      slice).
    - `KeyError` when a cell is missing the `group_by` value.
    - `TypeError` when a cell is missing/non-scalar at `pair_by`
      keys or `outcome_path`."""
    rows = to_dicts(cells)
    if treatment_arm == baseline_arm:
        raise ValueError(
            f'paired_comparison: treatment_arm and baseline_arm '
            f'share value {treatment_arm!r}; the comparison would '
            f'be self-against-self (HPO-smuggle indicator).',
        )
    if not pair_by:
        raise ValueError(
            'paired_comparison: pair_by must be non-empty',
        )

    treatment_cells: list[Mapping[str, object]] = []
    baseline_cells: list[Mapping[str, object]] = []
    for c in rows:
        arm = c.get(arm_field)
        if arm == treatment_arm:
            treatment_cells.append(c)
        elif arm == baseline_arm:
            baseline_cells.append(c)

    if not treatment_cells:
        raise ValueError(
            f'paired_comparison: no rows with '
            f'{arm_field}={treatment_arm!r}',
        )
    if not baseline_cells:
        raise ValueError(
            f'paired_comparison: no rows with '
            f'{arm_field}={baseline_arm!r}',
        )

    if group_by is None:
        gs, n_dropped = _per_group_stats(
            group_value=None,
            treatment_cells=treatment_cells,
            baseline_cells=baseline_cells,
            outcome_path=outcome_path,
            pair_by=pair_by,
        )
        if gs is None:
            return PairedComparisonResult(
                treatment_arm=treatment_arm,
                baseline_arm=baseline_arm,
                predicted_direction=predicted_direction,
                pair_by=pair_by, group_by=group_by,
                arm_a_n=0, arm_a_mean=None, arm_a_sd=None,
                arm_b_n=0, arm_b_mean=None, arm_b_sd=None,
                effect_size_g=None, se=None,
                per_group=(), pooled=None,
                n_dropped_unpaired=n_dropped,
            )
        return PairedComparisonResult(
            treatment_arm=treatment_arm,
            baseline_arm=baseline_arm,
            predicted_direction=predicted_direction,
            pair_by=pair_by, group_by=group_by,
            arm_a_n=gs.n_pairs, arm_a_mean=gs.arm_a_mean,
            arm_a_sd=gs.arm_a_sd,
            arm_b_n=gs.n_pairs, arm_b_mean=gs.arm_b_mean,
            arm_b_sd=gs.arm_b_sd,
            effect_size_g=gs.effect_size_g, se=gs.se,
            per_group=(), pooled=None,
            n_dropped_unpaired=n_dropped,
        )

    # Stratified mode.
    treatment_groups: dict[object, list[Mapping[str, object]]] = {}
    baseline_groups: dict[object, list[Mapping[str, object]]] = {}
    for c in treatment_cells:
        v = c.get(group_by)
        if v is None:
            raise KeyError(
                f'paired_comparison: cell missing group_by key '
                f'{group_by!r}',
            )
        treatment_groups.setdefault(v, []).append(c)
    for c in baseline_cells:
        v = c.get(group_by)
        if v is None:
            raise KeyError(
                f'paired_comparison: cell missing group_by key '
                f'{group_by!r}',
            )
        baseline_groups.setdefault(v, []).append(c)

    all_keys = sorted(
        treatment_groups.keys() | baseline_groups.keys(),
        key=lambda k: repr(k),
    )

    per_group: list[GroupStats] = []
    g_se_pairs: list[tuple[float, float]] = []
    n_dropped = 0
    all_a: list[float] = []
    all_b: list[float] = []

    for gkey in all_keys:
        t_g = treatment_groups.get(gkey, [])
        b_g = baseline_groups.get(gkey, [])
        if not t_g or not b_g:
            n_dropped += len(t_g) + len(b_g)
            continue
        gs, dropped = _per_group_stats(
            group_value=gkey,
            treatment_cells=t_g,
            baseline_cells=b_g,
            outcome_path=outcome_path,
            pair_by=pair_by,
        )
        n_dropped += dropped
        if gs is None:
            continue
        per_group.append(gs)
        if (gs.effect_size_g is not None
                and gs.se is not None
                and not math.isnan(gs.effect_size_g)
                and not math.isnan(gs.se)):
            g_se_pairs.append((gs.effect_size_g, gs.se))
        if gs.arm_a_mean is not None:
            all_a.extend([gs.arm_a_mean] * gs.n_pairs)
        if gs.arm_b_mean is not None:
            all_b.extend([gs.arm_b_mean] * gs.n_pairs)

    pooled = random_effects_summary(g_se_pairs)
    arm_n = sum(gs.n_pairs for gs in per_group)
    if math.isnan(pooled.pooled_g) or math.isnan(pooled.se_pooled):
        effect_g: float | None = None
        se_top: float | None = None
    else:
        effect_g = float(pooled.pooled_g)
        se_top = float(pooled.se_pooled)

    return PairedComparisonResult(
        treatment_arm=treatment_arm,
        baseline_arm=baseline_arm,
        predicted_direction=predicted_direction,
        pair_by=pair_by, group_by=group_by,
        arm_a_n=arm_n,
        arm_a_mean=(
            float(sum(all_a) / arm_n) if arm_n > 0 else None
        ),
        arm_a_sd=None,
        arm_b_n=arm_n,
        arm_b_mean=(
            float(sum(all_b) / arm_n) if arm_n > 0 else None
        ),
        arm_b_sd=None,
        effect_size_g=effect_g,
        se=se_top,
        per_group=tuple(per_group),
        pooled=pooled,
        n_dropped_unpaired=n_dropped,
    )


__all__ = [
    'GroupStats',
    'PairedComparisonResult',
    'paired_comparison',
]
