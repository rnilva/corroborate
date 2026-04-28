"""Aggregate — sweep → ComparisonRow hand-off.

The framework's sweep emits `list[RunRow]`; downstream consumers
need per-(treatment, baseline) ComparisonRows. This module
provides the typed factory functions for that hand-off.

The framework is *substrate-agnostic*: the outcome path being
aggregated is supplied by the caller as `outcome_path: str` on
every entry point. The framework reads `measurements[outcome_path]`
off each run, writes `{outcome_path}.effect_size_g` / `se` /
`derived_q` / etc. on the resulting ComparisonRow. v0's RL
substrate authors `outcome.late_window_mean`; other substrates or
other outcomes pass their own keys.

Structure:

- `leaf_signature(measurements)` — the configurational fingerprint
  used as a group-by key. Filters out outcome/bridge/invariant
  paths and per-cell metadata keys, returns sorted (path, str)
  pairs. "Leaf" because each entry is a non-recursive scalar
  claim of the configured composition (RL practice calls these
  hyperparameters; the framework name is `leaf`).
- `aggregate_cell_verdict(verdicts)` — Popperian aggregation over
  per-bridge verdicts (any single refutation refutes; INVARIANT_
  VIOLATION dominates).
- `paired_comparison_from_runs(treatment_runs, baseline_runs, *,
  outcome_path, predicted_direction, ...)` — paired-by-seed Δ on
  `outcome_path`; computes Hedges' g + SE + Popperian verdict.
- `link_pearson_across_envs(mechanism_comparisons,
  outcome_comparisons, *, mechanism_path, outcome_path, ...)` —
  cross-env Pearson r between mechanism and outcome effect sizes."""
from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from corroborate.hypothesis import Direction
from corroborate.schema import ComparisonRow, MeasurementLeaf, RunRow
from corroborate.verdict import RefutationClass, Verdict


# ============ Leaf-signature projection ============

# Output-side prefixes — paths whose values are observed at run
# time, not authored at composition. Filtered out of the
# configurational fingerprint.
_OUTPUT_PREFIXES: tuple[str, ...] = ('outcome.', 'bridge.', 'invariant.')

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
    - Output paths (`outcome.`/`bridge.`/`invariant.`).
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
    excluded = _FRAMEWORK_EXCLUDED_KEYS | exogenous_keys
    return tuple(sorted(
        (k, str(v))
        for k, v in measurements.items()
        if not any(k.startswith(p) for p in _OUTPUT_PREFIXES)
        and k not in excluded
    ))


# ============ Cell-level verdict aggregator (shared) ============

def aggregate_cell_verdict(verdicts: tuple[Verdict, ...]) -> Verdict:
    """Cell-level verdict from per-bridge verdicts. Popperian
    aggregation: any single refutation refutes.

    Precedence (highest first):
    1. Any `Verdict.INVARIANT_VIOLATION` (a tautological-tagged
       gap exceeded its scope-commitment threshold; the run sat
       outside the theorem's domain — outcome verdicts are out of
       scope per axiom 18).
    2. Any `Verdict.NO_EFFECT` → NO_EFFECT (one bridge refuted
       is enough; the hypothesis as a whole is refuted under this
       cell — Popperian falsification).
    3. All `Verdict.HELD` → HELD.
    4. Otherwise (mixed HELD + POWER_INSUFFICIENT) →
       `POWER_INSUFFICIENT` (cannot tell).
    Empty input → `POWER_INSUFFICIENT` (no test was performed)."""
    if not verdicts:
        return Verdict.POWER_INSUFFICIENT
    if any(v is Verdict.INVARIANT_VIOLATION for v in verdicts):
        return Verdict.INVARIANT_VIOLATION
    if any(v is Verdict.NO_EFFECT for v in verdicts):
        return Verdict.NO_EFFECT
    if all(v is Verdict.HELD for v in verdicts):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


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


def _run_pair_key(
    run: RunRow, pair_by: tuple[str, ...],
) -> tuple[MeasurementLeaf, ...]:
    """Read the tuple of measurement values at `pair_by` keys.
    Used as a hashable index for paired comparisons. Loud error
    if any key is missing or non-scalar."""
    out: list[MeasurementLeaf] = []
    for k in pair_by:
        v = run.measurements.get(k)
        if v is None or not isinstance(v, (str, int, float, bool)):
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


def paired_comparison_from_runs(
    treatment_runs: Sequence[RunRow],
    baseline_runs: Sequence[RunRow],
    *,
    outcome_path: str,
    pair_by: tuple[str, ...],
    predicted_direction: Direction | None,
    alpha: float = 0.05,
    power: float = 0.8,
    cycle_id: str | None = None,
    timestamp: str | None = None,
    extra_measurements: Mapping[str, MeasurementLeaf] | None = None,
) -> ComparisonRow:
    """Paired-by-`pair_by` ComparisonRow.

    Substrate-agnostic: `pair_by` names the measurement keys that
    identify a matched (treatment, baseline) pair. For RL,
    `pair_by=('seed',)` (when grouping is per-env) or
    `pair_by=('env_name', 'seed')` (when one comparison spans
    multiple envs). For non-RL substrates, whatever the matching
    axis is.

    Pairs by tuple-of-values at `pair_by`, drops unmatched pairs,
    computes Δ_i = treatment_i − baseline_i across pairs, fits
    Hedges' g + SE on the Δ distribution, derives MDE and verdict
    via `corroborate.statistics`. Same shape as v9's paired
    comparison (dialectic/hypothesis.py:332-350).

    Treatment and baseline must share at least one pair-key value
    for a valid comparison; if the intersection is empty, returns
    a POWER_INSUFFICIENT row with NaN stats."""
    from corroborate.statistics import (
        delta_i_from_q,
        derived_q_from_g_se,
        hedges_g_paired,
        verdict_from_paired_stats,
    )

    if not treatment_runs or not baseline_runs:
        raise ValueError(
            'paired_comparison_from_runs requires non-empty '
            'treatment_runs and baseline_runs'
        )
    if not pair_by:
        raise ValueError(
            'pair_by must be non-empty — the substrate must name '
            'the measurement key(s) that identify matched pairs.'
        )

    # Index by pair-key tuple; compute Δ over the intersection.
    treatment_by_key = {
        _run_pair_key(r, pair_by): r for r in treatment_runs
    }
    baseline_by_key = {
        _run_pair_key(r, pair_by): r for r in baseline_runs
    }
    paired_keys = sorted(
        treatment_by_key.keys() & baseline_by_key.keys()
    )

    deltas: list[float] = [
        _run_outcome(treatment_by_key[k], outcome_path)
        - _run_outcome(baseline_by_key[k], outcome_path)
        for k in paired_keys
    ]
    n_pairs = len(deltas)

    g, se = hedges_g_paired(deltas) if n_pairs >= 2 else (
        float('nan'), float('nan'),
    )
    q = derived_q_from_g_se(g, se)
    delta_i = delta_i_from_q(q)
    verdict, refutation_class, is_powered = verdict_from_paired_stats(
        g, se, n_pairs,
        predicted_direction=predicted_direction,
        alpha=alpha, power=power,
    )

    treatment_intervention = _run_intervention_name(treatment_runs[0])

    measurements: dict[str, MeasurementLeaf] = {
        'intervention_name': treatment_intervention,
        'n_treatment': len(treatment_runs),
        'n_baseline': len(baseline_runs),
        'n_pairs': n_pairs,
    }
    if not math.isnan(g):
        measurements[f'{outcome_path}.effect_size_g'] = g
    if not math.isnan(se):
        measurements[f'{outcome_path}.se'] = se
    if not math.isnan(q):
        measurements[f'{outcome_path}.derived_q'] = q
    measurements[f'{outcome_path}.delta_i_population'] = delta_i

    if extra_measurements is not None:
        measurements.update(extra_measurements)

    return ComparisonRow(
        id=str(uuid.uuid4()),
        parent_id=None,
        cycle_id=cycle_id,
        timestamp=_resolved_timestamp(timestamp),
        treatment_arm_id='',  # paired path doesn't materialise arms
        baseline_arm_id='',
        predicted_direction=predicted_direction,
        verdict=verdict,
        refutation_class=refutation_class,
        adequately_powered=is_powered,
        measurements=measurements,
    )


# ============ Cross-group link verdict (PAPER_NOTES.md §3.5) ============

def link_pearson_across_groups(
    mechanism_comparisons: Sequence[ComparisonRow],
    outcome_comparisons: Sequence[ComparisonRow],
    *,
    mechanism_path: str,
    outcome_path: str,
    group_by: str,
    alpha: float = 0.05,
    power: float = 0.8,
    cycle_id: str | None = None,
    timestamp: str | None = None,
    extra_measurements: Mapping[str, MeasurementLeaf] | None = None,
) -> ComparisonRow:
    """Cross-group link verdict — Pearson r between mechanism Δ
    and outcome Δ across groups.

    PAPER_NOTES.md §3.5: 'a methodological intervention's mechanism
    fingerprint should covary with its outcome fingerprint across
    [groups].' For RL, the group is `env_name`. For non-RL
    substrates, whatever single key identifies a group across
    treatment/baseline pairs.

    `group_by` is the substrate-named measurement key — caller
    passes `'env_name'` for RL or e.g. `'patient_id'` for clinical.
    Pairs `mechanism_comparisons[g]` and `outcome_comparisons[g]`
    by their `measurements[group_by]`; mismatches drop quietly
    (only fully-paired groups contribute to the Pearson r).

    Returns a ComparisonRow tagged at the cross-group granularity;
    its measurements include the Pearson r as the effect-size
    statistic, a derived MDE-style verdict on r vs 0, and the
    `group_by` key recorded as `'group_by': <key>`."""
    from corroborate.statistics import (
        adequately_powered_paired,
        delta_i_from_q,
        derived_q_from_g_se,
    )

    def _group_id(c: ComparisonRow) -> str:
        v = c.measurements.get(group_by)
        if not isinstance(v, str):
            raise TypeError(
                f"ComparisonRow {c.id!r} missing scalar string "
                f"measurement {group_by!r} for grouping",
            )
        return v

    mech_by_group = {_group_id(c): c for c in mechanism_comparisons}
    out_by_group = {_group_id(c): c for c in outcome_comparisons}
    paired_groups = sorted(mech_by_group.keys() & out_by_group.keys())

    mech_gs: list[float] = []
    out_gs: list[float] = []
    for g_id in paired_groups:
        mg = mech_by_group[g_id].measurements.get(mechanism_path)
        og = out_by_group[g_id].measurements.get(outcome_path)
        if isinstance(mg, (int, float)) and isinstance(og, (int, float)):
            if not (math.isnan(float(mg)) or math.isnan(float(og))):
                mech_gs.append(float(mg))
                out_gs.append(float(og))

    n = len(mech_gs)
    if n < 3:
        # Pearson's r needs at least 3 paired observations to be
        # well-defined; return a POWER_INSUFFICIENT row.
        underpowered_measurements: dict[str, MeasurementLeaf] = {
            'intervention_name': 'link',
            'n_paired_groups': n,
            'group_by': group_by,
        }
        if extra_measurements is not None:
            underpowered_measurements.update(extra_measurements)
        return ComparisonRow(
            id=str(uuid.uuid4()), parent_id=None, cycle_id=cycle_id,
            timestamp=_resolved_timestamp(timestamp),
            treatment_arm_id='', baseline_arm_id='',
            predicted_direction='a_gt_b',
            verdict=Verdict.POWER_INSUFFICIENT,
            refutation_class=RefutationClass.UNDERPOWERED,
            adequately_powered=False,
            measurements=underpowered_measurements,
        )

    # Pearson r with one-sided test (predicted positive).
    r = float(_pearson(mech_gs, out_gs))
    # SE under H0 (Fisher z-transform approximation): SE_r = 1/sqrt(n-3).
    se_r = 1.0 / math.sqrt(n - 3) if n > 3 else float('nan')
    q = derived_q_from_g_se(r, se_r) if not math.isnan(se_r) else float('nan')
    delta_i = delta_i_from_q(q) if not math.isnan(q) else 0.0
    is_powered = adequately_powered_paired(
        r, n, alpha=alpha, power=power, alternative='larger',
    )

    if not is_powered:
        verdict = Verdict.POWER_INSUFFICIENT
        refutation = RefutationClass.UNDERPOWERED
    elif r > 0:
        verdict = Verdict.HELD
        refutation = None
    else:
        verdict = Verdict.NO_EFFECT
        refutation = RefutationClass.SIGN_FLIP

    measurements: dict[str, MeasurementLeaf] = {
        'intervention_name': 'link',
        'n_paired_groups': n,
        'group_by': group_by,
        'link.pearson_r': r,
        'link.se': se_r,
        'link.derived_q': q,
        'link.delta_i_population': delta_i,
    }
    if extra_measurements is not None:
        measurements.update(extra_measurements)

    return ComparisonRow(
        id=str(uuid.uuid4()), parent_id=None, cycle_id=cycle_id,
        timestamp=_resolved_timestamp(timestamp),
        treatment_arm_id='', baseline_arm_id='',
        predicted_direction='a_gt_b',
        verdict=verdict,
        refutation_class=refutation,
        adequately_powered=is_powered,
        measurements=measurements,
    )


# Backward-compat alias for the old name (RL substrate uses it).
link_pearson_across_envs = link_pearson_across_groups


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson r — population formula. Returns 0.0 for zero-
    variance inputs (degenerate; deferred verdict)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return num / (sx * sy)
