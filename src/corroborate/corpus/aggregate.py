"""Aggregate — sweep → HypothesisComparisonRow hand-off.

The framework's sweep emits `list[RunRow]`; downstream consumers
need per-(treatment, baseline) comparison rows. This module
provides the typed factory functions for that hand-off.

The framework is *substrate-agnostic*: the outcome path being
aggregated is supplied by the caller as `outcome_path: str` on
every entry point. The framework reads `measurements[outcome_path]`
off each run; the resulting `HypothesisComparisonRow` carries
typed `effect_size_g` / `se` / `derived_q` / `pooled` fields.
v0's RL substrate authors `late_window_mean`; other substrates or
other outcomes pass their own keys.

Structure:

- `leaf_signature(measurements)` — the configurational fingerprint
  used as a group-by key. Filters out registered-measurable
  outputs and per-cell metadata, returns sorted (path, str) pairs.
  "Leaf" because each entry is a non-recursive scalar claim of
  the configured composition (RL practice calls these
  hyperparameters; the framework name is `leaf`).
- `hypothesis_comparison_from_cells(...)` — the canonical typed
  cross-arm aggregator. Paired by `pair_by`, optionally
  stratified by `group_by` for random-effects pooling. Consumed
  by `HypothesisComparisonRow.from_cells` and the verdict-walk
  (`hypothesis_subgraph_verdict`)."""
from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from corroborate.core.hypothesis import Hypothesis, PredictedDirection
from corroborate.corpus.schema import (
    GroupStats,
    HypothesisComparisonRow,
    MeasurementLeaf,
    RunRow,
)
from corroborate.stats import (
    adequately_powered_paired,
    delta_i_from_q,
    derived_q_from_g_se,
    hedges_g_paired,
    random_effects_summary,
    random_effects_verdict,
    verdict_from_paired_stats,
)
from corroborate.bridge.verdict import Verdict


# ============ Leaf-signature projection ============

# Always-excluded framework-typed metadata. These are
# substrate-AGNOSTIC: every RunRow carries `intervention_name`
# (the Hypothesis name), so it's never a configurational leaf.
_FRAMEWORK_EXCLUDED_KEYS: frozenset[str] = frozenset({
    'intervention_name',
})


def leaf_signature(
    measurements: Mapping[str, MeasurementLeaf],
    *,
    exogenous_keys: frozenset[str] = frozenset(),
) -> tuple[tuple[str, str], ...]:
    """The configurational fingerprint — leaf-only subset of
    `measurements` as a sorted (path, str-canonical-value) tuple.
    Hashable; suitable as a group-by key.

    Filters out:
    - Every registered-measurable name (the registry is the
      single source of truth post-Phase-5; substrate-paper-
      narrative prefixes are gone).
    - The framework-typed `intervention_name` (always excluded).
    - Substrate-supplied exogenous keys: keys the substrate
      declared via `Annotated[T, Exogenous]` on its `@claim`'s
      kwargs. Caller passes those names as `exogenous_keys` (e.g.
      `frozenset({'env_name', 'seed', 'total_steps'})` for the RL
      substrate). The framework does NOT hardcode RL key names.

    What remains is the configurational leaves at their dotted
    topology paths. "Leaf" rather than "HP": a leaf-regime kwarg
    is a non-recursive scalar claim of the configured composition,
    observed at composition time."""
    from corroborate.measurables import registered_names
    excluded = (
        _FRAMEWORK_EXCLUDED_KEYS
        | exogenous_keys
        | frozenset(registered_names())
    )
    return tuple(sorted(
        (k, str(v))
        for k, v in measurements.items()
        if k not in excluded
    ))


# ============ Paired-by-seed comparison (v9-port stats) ============

def _resolved_timestamp(timestamp: str | None) -> str:
    return (
        timestamp
        if timestamp is not None
        else datetime.now(UTC).isoformat(timespec='seconds')
    )


def _run_intervention_name(run: RunRow) -> str:
    v = run.measurements.get('intervention_name')
    if not isinstance(v, str):
        raise TypeError(
            f"RunRow {run.id!r} missing 'intervention_name' measurement"
        )
    return v


def _pair_runs_by_key(
    treatment: Sequence[RunRow], baseline: Sequence[RunRow],
    *,
    pair_by: tuple[str, ...],
    dup_check: bool = False,
    group_label: object | None = None,
) -> tuple[
    list[tuple[MeasurementLeaf, ...]],
    dict[tuple[MeasurementLeaf, ...], RunRow],
    dict[tuple[MeasurementLeaf, ...], RunRow],
    int,
]:
    """Index treatment + baseline runs by `pair_by`-key; return
    paired keys (sorted), the two key→RunRow dicts, and the count
    of unpaired-and-dropped runs.

    `dup_check=True` raises `ValueError` on duplicate pair-keys
    within either arm — silent dedup would mask a misconfigured
    slice."""
    def _index(
        runs: Sequence[RunRow], side: str,
    ) -> dict[tuple[MeasurementLeaf, ...], RunRow]:
        out: dict[tuple[MeasurementLeaf, ...], RunRow] = {}
        for r in runs:
            pk = _run_pair_key(r, pair_by)
            if dup_check and pk in out:
                tag = (
                    f' for group {group_label!r}'
                    if group_label is not None else ''
                )
                raise ValueError(
                    f'duplicate pair_by={pair_by!r} key {pk!r} '
                    f'in {side}{tag}',
                )
            out[pk] = r
        return out

    t_by_key = _index(treatment, 'treatment')
    b_by_key = _index(baseline, 'baseline')
    paired = sorted(t_by_key.keys() & b_by_key.keys())
    n_dropped = (
        (len(t_by_key) - len(paired))
        + (len(b_by_key) - len(paired))
    )
    return paired, t_by_key, b_by_key, n_dropped


def _run_pair_key(
    run: RunRow, pair_by: tuple[str, ...],
) -> tuple[MeasurementLeaf, ...]:
    """Read the tuple of measurement values at `pair_by` keys.
    Used as a hashable index for paired comparisons. Loud error
    if any key is missing or non-scalar."""
    out: list[MeasurementLeaf] = []
    for k in pair_by:
        v = run.measurements.get(k)
        if v is None:
            raise TypeError(
                f"RunRow {run.id!r} missing scalar pair-key "
                f"{k!r} (pair_by={pair_by!r})",
            )
        out.append(v)
    return tuple(out)


def _run_outcome(
    run: RunRow,
    outcome_path: str,
) -> float:
    """Read a scalar outcome measurement off a run; loud error if
    absent or non-numeric. The substrate authors `outcome_path` as
    the path-keyed measurement; the framework reads it back here."""
    v = run.measurements.get(outcome_path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(
            f"RunRow {run.id!r} missing scalar {outcome_path!r} "
            f"measurement"
        )
    return float(v)


# ============ HypothesisComparisonRow.from_cells ============

def _partition_runs_by(
    runs: Sequence[RunRow], group_by: str,
) -> dict[object, list[RunRow]]:
    """Partition runs by the value of measurement key `group_by`.
    Loud KeyError when a run is missing the key."""
    out: dict[object, list[RunRow]] = {}
    for r in runs:
        if group_by not in r.measurements:
            raise KeyError(
                f'run {r.id!r} missing group_by key {group_by!r}',
            )
        out.setdefault(r.measurements[group_by], []).append(r)
    return out


def _per_group_stats(
    h: 'Hypothesis[Mapping[str, object]]',
    group_value: object,
    treatment_runs: Sequence[RunRow],
    baseline_runs: Sequence[RunRow],
    *,
    outcome_path: str,
    pair_by: tuple[str, ...],
    alpha: float,
    power: float,
    predicted_direction: PredictedDirection | None = None,
) -> tuple[GroupStats | None, int]:
    """Pair within one group, compute Hedges' g + verdict +
    GroupStats. Returns (GroupStats | None, n_dropped_unpaired).
    None when no pairs survive the outcome-finite filter.

    `predicted_direction` overrides `h.predicted_direction` for
    the sign test when the caller knows a per-edge prior (e.g.
    a typed `claim_bridge.Bridge` carries `predicted_direction='a_lt_b'`
    on its mechanism edge but `'a_gt_b'` on its outcome edge). When
    None, falls back to `h.predicted_direction`.

    Raises ValueError on duplicate pair_by keys within an arm —
    silent dedup would hide a misconfigured slice."""
    paired, t_by_pkey, b_by_pkey, n_dropped = _pair_runs_by_key(
        treatment_runs, baseline_runs,
        pair_by=pair_by, dup_check=True, group_label=group_value,
    )

    a_values: list[float] = []
    b_values: list[float] = []
    deltas: list[float] = []
    for pk in paired:
        a = _run_outcome(t_by_pkey[pk], outcome_path)
        b = _run_outcome(b_by_pkey[pk], outcome_path)
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
    q = derived_q_from_g_se(g, se)
    di = delta_i_from_q(q)
    effective_direction = (
        predicted_direction if predicted_direction is not None
        else h.predicted_direction
    )
    verdict, refutation, is_powered = verdict_from_paired_stats(
        g, se, n_pairs,
        predicted_direction=effective_direction,
        alpha=alpha, power=power,
    )

    g_safe: float | None = None if math.isnan(g) else float(g)
    se_safe: float | None = None if math.isnan(se) else float(se)
    q_safe: float | None = None if math.isnan(q) else float(q)

    return GroupStats(
        group_value=group_value,
        n_pairs=n_pairs,
        arm_a_mean=a_mean, arm_a_sd=a_sd,
        arm_b_mean=b_mean, arm_b_sd=b_sd,
        effect_size_g=g_safe, se=se_safe, derived_q=q_safe,
        delta_i=di, verdict=verdict,
        refutation_class=refutation,
        adequately_powered=is_powered,
    ), n_dropped


def hypothesis_comparison_from_cells(
    h: 'Hypothesis[Mapping[str, object]]',
    treatment_runs: Sequence[RunRow],
    baseline_runs: Sequence[RunRow],
    *,
    outcome_path: str,
    pair_by: tuple[str, ...],
    group_by: str | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    cycle_id: str | None = None,
    timestamp: str | None = None,
    baseline_h: 'Hypothesis[Mapping[str, object]] | None' = None,
    predicted_direction: PredictedDirection | None = None,
) -> HypothesisComparisonRow:
    """The canonical cross-arm aggregator. Builds one
    HypothesisComparisonRow from per-cell `RunRow`s.

    `pair_by` identifies a (treatment, baseline) pair within a
    stratum (e.g. `('seed',)` for DQN). REQUIRED for paired tests.

    `group_by` partitions runs into strata before pairing (e.g.
    `'env_name'`). When set, the row carries `per_group:
    tuple[GroupStats, ...]` AND `pooled: PooledStats` from
    DerSimonian-Laird random-effects pooling across strata; the
    top-level `effect_size_g` mirrors `pooled.pooled_g`.

    When `group_by` is None, single-group mode: per-arm stats and
    Hedges' g over the paired Δ distribution; `per_group=()`,
    `pooled=None`.

    `baseline_h` carries the typed identity of the baseline arm
    (`intervention_arms` tuple). Default None means baseline is
    the empty-arms baseline (`arm_key()='baseline'`). The two arm
    keys MUST differ — equal keys raise ValueError as a
    HPO-smuggle indicator (the comparison would be self-against-
    self, signal-free).

    `predicted_direction` overrides the hypothesis-level prior
    (`h.predicted_direction`) for the sign test. The verdict-walk
    path through `hypothesis_subgraph_verdict` passes the per-edge
    `claim_bridge.Bridge.predicted_direction` here so each edge's
    sign test uses its own prior (mechanism predicts a_lt_b,
    outcome predicts a_gt_b — they can't share one direction).
    When None, falls back to `h.predicted_direction`.

    Raises:
    - `ValueError` on empty arms or empty `pair_by`.
    - `ValueError` when treatment and baseline arm keys match
      (HPO-smuggle indicator).
    - `ValueError` on duplicate `pair_by` keys within an arm
      within a group (silent dedup would mask a misconfigured
      slice).
    - `KeyError` when a run is missing the `group_by` measurement
      key (should never happen for a correctly-typed corpus).

    Drops unmatched pairs silently; the count lands on
    `row.n_dropped_unpaired` so consumers see the gap."""
    if not treatment_runs:
        raise ValueError(
            'hypothesis_comparison_from_cells: treatment_runs empty',
        )
    if not baseline_runs:
        raise ValueError(
            'hypothesis_comparison_from_cells: baseline_runs empty',
        )
    if not pair_by:
        raise ValueError(
            'hypothesis_comparison_from_cells: pair_by must be non-empty',
        )

    treatment_arm_key = h.arm_key()
    baseline_arm_key = (
        baseline_h.arm_key() if baseline_h is not None else 'baseline'
    )
    if treatment_arm_key == baseline_arm_key:
        raise ValueError(
            f'hypothesis_comparison_from_cells: treatment and baseline '
            f'share arm_key {treatment_arm_key!r}; the comparison would '
            f'be self-against-self (HPO-smuggle indicator). Treatment '
            f'and baseline must differ in their `intervention_arms`.',
        )

    # Arm-key consistency check: every run in `treatment_runs`
    # should carry `treatment_arm_key`; same for baseline. The
    # framework's load-bearing promise: HPO variation does NOT
    # change arm_key, so mixed-arm runs in one arm-list signal a
    # pairing-data bug. The default 'baseline' value on legacy
    # RunRows passes the baseline check transparently.
    t_arms = {r.arm_key for r in treatment_runs}
    if t_arms != {treatment_arm_key}:
        raise ValueError(
            f'hypothesis_comparison_from_cells: treatment_runs carry '
            f'mixed arm_keys {sorted(t_arms)!r}; expected all to be '
            f'{treatment_arm_key!r}. Pairing inconsistency.',
        )
    b_arms = {r.arm_key for r in baseline_runs}
    if b_arms != {baseline_arm_key}:
        raise ValueError(
            f'hypothesis_comparison_from_cells: baseline_runs carry '
            f'mixed arm_keys {sorted(b_arms)!r}; expected all to be '
            f'{baseline_arm_key!r}. Pairing inconsistency.',
        )

    intervention_name = _run_intervention_name(treatment_runs[0])

    effective_direction = (
        predicted_direction if predicted_direction is not None
        else h.predicted_direction
    )

    if group_by is None:
        # Single-group mode.
        gs, n_dropped = _per_group_stats(
            h, group_value=None,
            treatment_runs=treatment_runs,
            baseline_runs=baseline_runs,
            outcome_path=outcome_path,
            pair_by=pair_by, alpha=alpha, power=power,
            predicted_direction=effective_direction,
        )
        if gs is None:
            return HypothesisComparisonRow(
                id=str(uuid.uuid4()),
                parent_id=None,
                cycle_id=cycle_id,
                timestamp=_resolved_timestamp(timestamp),
                intervention_name=intervention_name,
                treatment_arm_key=treatment_arm_key,
                baseline_arm_key=baseline_arm_key,
                treatment_run_ids=tuple(r.id for r in treatment_runs),
                baseline_run_ids=tuple(r.id for r in baseline_runs),
                predicted_direction=effective_direction,
                pair_by=pair_by,
                group_by=group_by,
                arm_a_n=0, arm_a_mean=None, arm_a_sd=None,
                arm_b_n=0, arm_b_mean=None, arm_b_sd=None,
                effect_size_g=None, se=None, derived_q=None,
                delta_i_population=0.0,
                adequately_powered=False,
                verdict=Verdict.POWER_INSUFFICIENT,
                refutation_class=None,
                per_group=(), pooled=None,
                n_dropped_unpaired=n_dropped,
            )
        return HypothesisComparisonRow(
            id=str(uuid.uuid4()),
            parent_id=None,
            cycle_id=cycle_id,
            timestamp=_resolved_timestamp(timestamp),
            intervention_name=intervention_name,
            treatment_arm_key=treatment_arm_key,
            baseline_arm_key=baseline_arm_key,
            treatment_run_ids=tuple(r.id for r in treatment_runs),
            baseline_run_ids=tuple(r.id for r in baseline_runs),
            predicted_direction=effective_direction,
            pair_by=pair_by,
            group_by=group_by,
            arm_a_n=gs.n_pairs, arm_a_mean=gs.arm_a_mean,
            arm_a_sd=gs.arm_a_sd,
            arm_b_n=gs.n_pairs, arm_b_mean=gs.arm_b_mean,
            arm_b_sd=gs.arm_b_sd,
            effect_size_g=gs.effect_size_g, se=gs.se,
            derived_q=gs.derived_q,
            delta_i_population=gs.delta_i,
            adequately_powered=gs.adequately_powered,
            verdict=gs.verdict,
            refutation_class=gs.refutation_class,
            per_group=(), pooled=None,
            n_dropped_unpaired=n_dropped,
        )

    # Stratified mode.
    treatment_groups = _partition_runs_by(treatment_runs, group_by)
    baseline_groups = _partition_runs_by(baseline_runs, group_by)
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
            h, group_value=gkey,
            treatment_runs=t_g,
            baseline_runs=b_g,
            outcome_path=outcome_path,
            pair_by=pair_by, alpha=alpha, power=power,
            predicted_direction=effective_direction,
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
    verdict_p, refutation_p = random_effects_verdict(
        pooled, predicted_direction=effective_direction,
    )

    arm_n = sum(gs.n_pairs for gs in per_group)
    if math.isnan(pooled.pooled_g) or math.isnan(pooled.se_pooled):
        effect_g: float | None = None
        se_top: float | None = None
        derived_q: float | None = None
        delta_i_pop = 0.0
        adequately_powered = False
    else:
        effect_g = float(pooled.pooled_g)
        se_top = float(pooled.se_pooled)
        # Use the pooled n_cells as n for power assessment — the
        # appropriate n for "is the pooled estimate detectable at
        # this number of cells?" rather than total pair count.
        adequately_powered = adequately_powered_paired(
            effect_g, pooled.n_cells, alpha=alpha, power=power,
        )
        q_val = derived_q_from_g_se(effect_g, se_top)
        derived_q = None if math.isnan(q_val) else float(q_val)
        delta_i_pop = (
            delta_i_from_q(q_val) if not math.isnan(q_val) else 0.0
        )

    return HypothesisComparisonRow(
        id=str(uuid.uuid4()),
        parent_id=None,
        cycle_id=cycle_id,
        timestamp=_resolved_timestamp(timestamp),
        intervention_name=intervention_name,
        treatment_arm_key=treatment_arm_key,
        baseline_arm_key=baseline_arm_key,
        treatment_run_ids=tuple(r.id for r in treatment_runs),
        baseline_run_ids=tuple(r.id for r in baseline_runs),
        predicted_direction=effective_direction,
        pair_by=pair_by,
        group_by=group_by,
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
        derived_q=derived_q,
        delta_i_population=delta_i_pop,
        adequately_powered=adequately_powered,
        verdict=verdict_p,
        refutation_class=refutation_p,
        per_group=tuple(per_group),
        pooled=pooled,
        n_dropped_unpaired=n_dropped,
    )
