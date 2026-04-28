"""Aggregate — typed sweep → ArmRow → ComparisonRow hand-off.

The framework's sweep emits `list[RunRow]`; downstream consumers
need per-(intervention, env) ArmRows and per-(treatment, baseline)
ComparisonRows. This module provides the typed factory functions
for that hand-off.

The framework is *substrate-agnostic*: the outcome path being
aggregated is supplied by the caller as `outcome_path: str` on
every entry point. The framework reads `measurements[outcome_path]`
off each run, writes `{outcome_path}.arm_mean` / `arm_sd` /
`effect_size_g` / etc. on the resulting Arm/ComparisonRow. v0's
RL substrate authors `outcome.late_window_mean`; other substrates
or other outcomes pass their own keys.

Structure:

- `aggregate_cell_verdict(verdicts)` — Popperian aggregation over
  per-bridge verdicts (any single refutation refutes; INVARIANT_
  VIOLATION dominates).
- `arm_from_runs(runs, *, outcome_path, intervention_name,
  env_name, ...)` produces an ArmRow from a homogeneous list of
  RunRows. Computes arm-level statistics for `outcome_path`
  (NaN-aware), aggregates per-bridge admit-rates across runs, and
  forwards the leaf-only subset of measurements (the
  configurational fingerprint).
- `comparison_from_arms(treatment, baseline, *, outcome_path,
  predicted_direction, ...)` produces a ComparisonRow with per-arm
  stats for `outcome_path` threaded through `measurements`
  (`{outcome_path}.arm_a_mean` etc.). Stats populated by
  `_default_statistics_stub` for the unpaired path; the paired
  path uses real Hedges' g via `paired_comparison_from_runs`.
- `aggregate_runs(runs, *, outcome_path)` is the convenience entry
  point: groups runs by (intervention_name, env_name,
  leaf_signature), produces one ArmRow per group.
- `paired_comparison_from_runs(treatment_runs, baseline_runs, *,
  outcome_path, predicted_direction, ...)` — paired-by-seed Δ on
  `outcome_path`; computes Hedges' g + SE + Popperian verdict.
- `link_pearson_across_envs(mechanism_comparisons,
  outcome_comparisons, *, mechanism_path, outcome_path, ...)` —
  cross-env Pearson r between mechanism and outcome effect sizes.
- `leaf_signature(measurements)` — the configurational fingerprint
  used as a group-by key. Filters out outcome/bridge/invariant
  paths and per-cell metadata keys, returns sorted (path, str)
  pairs. "Leaf" because each entry is a non-recursive scalar
  claim of the configured composition (RL practice calls these
  hyperparameters; the framework name is `leaf`)."""
from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from corroborate.hypothesis import Direction
from corroborate.schema import ArmRow, ComparisonRow, MeasurementLeaf, RunRow
from corroborate.verdict import RefutationClass, Verdict


# ============ Leaf-signature projection ============

# Output-side prefixes — paths whose values are observed at run
# time, not authored at composition. Filtered out of the
# configurational fingerprint.
_OUTPUT_PREFIXES: tuple[str, ...] = ('outcome.', 'bridge.', 'invariant.')

# Substrate-supplied per-cell metadata that varies independently of
# configuration (seed, env_name) or restates a leaf already in the
# fingerprint (total_steps appears in both metadata and as a leaf).
_NON_LEAF_KEYS: frozenset[str] = frozenset({
    'env_name', 'seed', 'total_steps', 'intervention_name',
})


def leaf_signature(
    measurements: Mapping[str, MeasurementLeaf],
) -> tuple[tuple[str, str], ...]:
    """The configurational fingerprint — leaf-only subset of
    `measurements` as a sorted (path, str-canonical-value) tuple.
    Hashable; suitable as a group-by key.

    Filters out output paths (`outcome.`/`bridge.`/`invariant.`)
    and per-cell metadata (`env_name`, `seed`, `total_steps`,
    `intervention_name`). What remains is the configurational
    leaves at their dotted topology paths.

    "Leaf" rather than "HP": a leaf-regime kwarg is a non-recursive
    scalar claim of the configured composition, observed at
    composition time. RL practice calls these hyperparameters; the
    framework's term is `leaf` since the same shape covers any
    non-RL configuration too."""
    return tuple(sorted(
        (k, str(v))
        for k, v in measurements.items()
        if not any(k.startswith(p) for p in _OUTPUT_PREFIXES)
        and k not in _NON_LEAF_KEYS
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


# ============ ArmRow factory ============

def _empty_meta() -> dict[str, MeasurementLeaf]:
    return {}


def _resolved_timestamp(timestamp: str | None) -> str:
    return (
        timestamp
        if timestamp is not None
        else datetime.now(UTC).isoformat(timespec='seconds')
    )


def _outcome_summary(
    measurements: Mapping[str, MeasurementLeaf],
    outcome_path: str,
) -> float:
    """Read the outcome at `outcome_path` as a float, returning NaN
    if absent or non-numeric. Cells without the key contribute NaN
    to the arm aggregate."""
    v = measurements.get(outcome_path)
    if v is None:
        return float('nan')
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, (int, float)):
        return float(v)
    return float('nan')


def _bridge_admit_rates(
    runs: Sequence[RunRow],
) -> dict[str, MeasurementLeaf]:
    """For each `bridge.<name>.verdict` / `invariant.<name>.verdict`
    path that appears in any run's measurements, compute the
    admit-rate across runs (n_held / n_with_verdict) and surface
    it as `<prefix><name>.admit_rate`. Runs missing a verdict for
    a given name contribute neither numerator nor denominator."""
    counts: dict[str, dict[str, int]] = {}
    for run in runs:
        for k, v in run.measurements.items():
            if not (
                k.startswith('bridge.') or k.startswith('invariant.')
            ):
                continue
            if not k.endswith('.verdict'):
                continue
            slot = counts.setdefault(k, {'n': 0, 'n_held': 0})
            slot['n'] += 1
            if v == Verdict.HELD.value:
                slot['n_held'] += 1
    out: dict[str, MeasurementLeaf] = {}
    for verdict_path, c in counts.items():
        # `bridge.<name>.verdict` → `bridge.<name>.admit_rate`
        base = verdict_path.removesuffix('.verdict')
        admit_rate = c['n_held'] / c['n'] if c['n'] > 0 else 0.0
        out[f'{base}.admit_rate'] = admit_rate
    return out


def arm_from_runs(
    runs: Sequence[RunRow],
    *,
    outcome_path: str,
    intervention_name: str,
    env_name: str,
    cycle_id: str | None = None,
    timestamp: str | None = None,
    extra_measurements: Mapping[str, MeasurementLeaf] | None = None,
) -> ArmRow:
    """Construct an ArmRow from a list of RunRows. Runs must be
    homogeneous in `intervention_name` + `env_name` + HP signature
    (caller is responsible for grouping); this function trusts
    that contract.

    Computes `{outcome_path}.arm_mean` and `{outcome_path}.arm_sd`
    (NaN-aware) from each run's `measurements[outcome_path]`.
    Intersects the HP subset of measurements across the runs and
    forwards each common (k, v) entry. Per-bridge admit-rates land
    at `bridge.<name>.admit_rate` / `invariant.<name>.admit_rate`."""
    if not runs:
        raise ValueError('arm_from_runs requires ≥1 RunRow')

    summaries = [_outcome_summary(r.measurements, outcome_path) for r in runs]
    n = len(summaries)
    # NaN-aware mean/sd: a run with no terminated episode in the
    # late window legitimately yields NaN. `sum / n` would poison
    # the whole arm with one NaN; instead compute over only finite
    # summaries. `n` stays the total run count for provenance.
    finite_summaries = [s for s in summaries if not math.isnan(s)]
    n_finite = len(finite_summaries)
    if n_finite == 0:
        arm_mean = float('nan')
        arm_sd = float('nan')
    else:
        arm_mean = sum(finite_summaries) / n_finite
        if n_finite > 1:
            var = sum(
                (s - arm_mean) ** 2 for s in finite_summaries
            ) / (n_finite - 1)
            arm_sd = math.sqrt(var)
        else:
            arm_sd = 0.0

    # Leaf-subset shared across all runs in the arm. Authors who
    # follow the grouping contract get identical leaf signatures
    # for all members; we still intersect defensively (a run with
    # a bridge-set-difference in measurements doesn't poison the
    # arm).
    common_leaves = _intersect_leaf_measurements(runs)

    measurements: dict[str, MeasurementLeaf] = {
        'env_name': env_name,
        'intervention_name': intervention_name,
        'n': n,
        f'{outcome_path}.arm_mean': arm_mean,
        f'{outcome_path}.arm_sd': arm_sd,
        **common_leaves,
        **_bridge_admit_rates(runs),
        **(extra_measurements or _empty_meta()),
    }

    return ArmRow(
        id=str(uuid.uuid4()),
        cycle_id=cycle_id,
        timestamp=_resolved_timestamp(timestamp),
        run_ids=tuple(r.id for r in runs),
        measurements=measurements,
    )


def _intersect_leaf_measurements(
    runs: Sequence[RunRow],
) -> dict[str, MeasurementLeaf]:
    """Intersection of configurational-leaf measurements across
    runs. A path that appears with the SAME value in every run
    survives; anything else is dropped. Uses the same filter as
    `leaf_signature` — projection is leaf keys only."""
    if not runs:
        return {}
    head = {
        k: v for k, v in runs[0].measurements.items()
        if not any(k.startswith(p) for p in _OUTPUT_PREFIXES)
        and k not in _NON_LEAF_KEYS
    }
    out: dict[str, MeasurementLeaf] = {}
    for k, v in head.items():
        if all(r.measurements.get(k) == v for r in runs[1:]):
            out[k] = v
    return out


# ============ ComparisonRow factory ============

@dataclass(frozen=True, slots=True)
class _StatisticsStub:
    """Stub statistics — what the step-5 MDE+power module will
    replace with real Hedges' g + SE + derived q. Each field
    matches the corresponding ComparisonRow.measurements path or
    the typed `verdict` / `refutation_class` / `adequately_powered`
    fields one-to-one."""
    effect_size_g: float | None
    se: float | None
    derived_q: float | None
    delta_i_population: float
    verdict: Verdict
    refutation_class: RefutationClass | None
    adequately_powered: bool


def _default_statistics_stub(
    treatment: ArmRow, baseline: ArmRow,
) -> _StatisticsStub:
    """Unpaired stub — returns None / POWER_INSUFFICIENT for arms
    aggregated independently. The paired-by-seed flow uses
    `paired_comparison_from_runs` (below), which doesn't go
    through this stub. v0's per-arm aggregation path keeps this
    so consumers that legitimately want unpaired comparison can
    still build a ComparisonRow without statistics."""
    del treatment, baseline
    return _StatisticsStub(
        effect_size_g=None,
        se=None,
        derived_q=None,
        delta_i_population=0.0,
        verdict=Verdict.POWER_INSUFFICIENT,
        refutation_class=None,
        adequately_powered=False,
    )


def _arm_env_name(arm: ArmRow) -> str:
    """Read `env_name` measurement off an arm; loud error if
    absent (the arm-builder always writes it)."""
    v = arm.measurements.get('env_name')
    if not isinstance(v, str):
        raise TypeError(
            f"ArmRow {arm.id!r} missing 'env_name' measurement"
        )
    return v


def _arm_intervention_name(arm: ArmRow) -> str:
    v = arm.measurements.get('intervention_name')
    if not isinstance(v, str):
        raise TypeError(
            f"ArmRow {arm.id!r} missing 'intervention_name' measurement"
        )
    return v


def _arm_n(arm: ArmRow) -> int:
    v = arm.measurements.get('n')
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(
            f"ArmRow {arm.id!r} missing 'n' measurement"
        )
    return v


def _arm_outcome_stats(
    arm: ArmRow, outcome_path: str,
) -> tuple[float, float]:
    """Read (arm_mean, arm_sd) for `outcome_path` off an arm. Loud
    error if absent (arm-builder always writes them when given
    matching `outcome_path`)."""
    mean_key = f'{outcome_path}.arm_mean'
    sd_key = f'{outcome_path}.arm_sd'
    mean_v = arm.measurements.get(mean_key)
    sd_v = arm.measurements.get(sd_key)
    if isinstance(mean_v, bool) or not isinstance(mean_v, (int, float)):
        raise TypeError(
            f'ArmRow {arm.id!r} missing {mean_key!r} measurement'
        )
    if isinstance(sd_v, bool) or not isinstance(sd_v, (int, float)):
        raise TypeError(
            f'ArmRow {arm.id!r} missing {sd_key!r} measurement'
        )
    return float(mean_v), float(sd_v)


def comparison_from_arms(
    treatment: ArmRow,
    baseline: ArmRow,
    *,
    outcome_path: str,
    predicted_direction: Direction | None,
    cycle_id: str | None = None,
    timestamp: str | None = None,
    extra_measurements: Mapping[str, MeasurementLeaf] | None = None,
    statistics_fn: object = None,
) -> ComparisonRow:
    """Construct a ComparisonRow from a (treatment, baseline) arm
    pair. The arms must share `env_name` (caller is responsible
    for matching).

    Per-arm stats land in `measurements` under
    `{outcome_path}.arm_a_*` / `{outcome_path}.arm_b_*` paths plus
    `n_treatment` / `n_baseline` / `intervention_name` / `env_name`.
    Stats fields populated by `_default_statistics_stub` for v0;
    the paired-by-seed flow uses real `paired_comparison_from_runs`.

    `statistics_fn` is the pluggable MDE+power computation;
    `None` uses the v0 stub. Type is `object` because the
    function signature isn't carried at the framework level
    (callers can use the stub or their own — the typed contract
    is the `_StatisticsStub` shape).

    `predicted_direction` is taken from the treatment's
    `Hypothesis.predicted_direction` and threaded through; the
    caller passes it explicitly because ArmRow doesn't carry it."""
    treatment_env = _arm_env_name(treatment)
    baseline_env = _arm_env_name(baseline)
    if treatment_env != baseline_env:
        raise ValueError(
            f'env_name mismatch: treatment={treatment_env!r} '
            f'vs baseline={baseline_env!r}'
        )

    stats = (
        statistics_fn(treatment, baseline)
        if callable(statistics_fn)
        else _default_statistics_stub(treatment, baseline)
    )
    if not isinstance(stats, _StatisticsStub):
        raise TypeError(
            f'statistics_fn must return a _StatisticsStub, '
            f'got {type(stats).__name__}'
        )

    arm_a_mean, arm_a_sd = _arm_outcome_stats(treatment, outcome_path)
    arm_b_mean, arm_b_sd = _arm_outcome_stats(baseline, outcome_path)
    n_treatment = _arm_n(treatment)
    n_baseline = _arm_n(baseline)

    measurements: dict[str, MeasurementLeaf] = {
        'env_name': treatment_env,
        'intervention_name': _arm_intervention_name(treatment),
        'n_treatment': n_treatment,
        'n_baseline': n_baseline,
        f'{outcome_path}.arm_a_mean': arm_a_mean,
        f'{outcome_path}.arm_a_sd': arm_a_sd,
        f'{outcome_path}.arm_b_mean': arm_b_mean,
        f'{outcome_path}.arm_b_sd': arm_b_sd,
        f'{outcome_path}.delta_i_population': stats.delta_i_population,
    }
    if stats.effect_size_g is not None:
        measurements[f'{outcome_path}.effect_size_g'] = stats.effect_size_g
    if stats.se is not None:
        measurements[f'{outcome_path}.se'] = stats.se
    if stats.derived_q is not None:
        measurements[f'{outcome_path}.derived_q'] = stats.derived_q

    if extra_measurements is not None:
        measurements.update(extra_measurements)

    return ComparisonRow(
        id=str(uuid.uuid4()),
        parent_id=None,
        cycle_id=cycle_id,
        timestamp=_resolved_timestamp(timestamp),
        treatment_arm_id=treatment.id,
        baseline_arm_id=baseline.id,
        predicted_direction=predicted_direction,
        verdict=stats.verdict,
        refutation_class=stats.refutation_class,
        adequately_powered=stats.adequately_powered,
        measurements=measurements,
    )


# ============ Convenience: aggregate runs into arms ============

def _run_intervention_name(run: RunRow) -> str:
    v = run.measurements.get('intervention_name')
    if not isinstance(v, str):
        raise TypeError(
            f"RunRow {run.id!r} missing 'intervention_name' measurement"
        )
    return v


def _run_env_name(run: RunRow) -> str:
    v = run.measurements.get('env_name')
    if not isinstance(v, str):
        raise TypeError(
            f"RunRow {run.id!r} missing 'env_name' measurement"
        )
    return v


def aggregate_runs(
    runs: Iterable[RunRow], *, outcome_path: str,
) -> list[ArmRow]:
    """Group `runs` by (intervention_name, env_name, leaf_signature),
    build one ArmRow per group. Convenience for the common
    dialectic-loop case where a sweep produces a flat list of cells
    across multiple (hypothesis, env) combinations.

    Grouping uses `leaf_signature(run.measurements)` rather than a
    declared `MechanismKey` artifact — two runs with identical
    configurational leaves on the same intervention/env land in the
    same arm, even if they were authored as distinct Hypothesis
    instances.

    `outcome_path` is the substrate-supplied measurement key whose
    arm-level mean/sd will be computed (passed through to
    `arm_from_runs`)."""
    by_key: dict[
        tuple[str, str, tuple[tuple[str, str], ...]],
        list[RunRow],
    ] = {}
    for r in runs:
        key = (
            _run_intervention_name(r),
            _run_env_name(r),
            leaf_signature(r.measurements),
        )
        by_key.setdefault(key, []).append(r)

    out: list[ArmRow] = []
    for (intervention_name, env_name, _), group in by_key.items():
        out.append(arm_from_runs(
            group,
            outcome_path=outcome_path,
            intervention_name=intervention_name,
            env_name=env_name,
            cycle_id=group[0].cycle_id,
        ))
    return out


# ============ Paired-by-seed comparison (v9-port stats) ============

def _run_seed(run: RunRow) -> int:
    v = run.measurements.get('seed')
    if isinstance(v, bool) or not isinstance(v, int):
        raise TypeError(
            f"RunRow {run.id!r} missing 'seed' measurement"
        )
    return v


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
    predicted_direction: Direction | None,
    alpha: float = 0.05,
    power: float = 0.8,
    cycle_id: str | None = None,
    timestamp: str | None = None,
    extra_measurements: Mapping[str, MeasurementLeaf] | None = None,
) -> ComparisonRow:
    """Paired-by-seed ComparisonRow.

    Pairs treatment and baseline runs by `(env_name, seed)`, drops
    unmatched pairs, computes Δ_i = treatment_i − baseline_i across
    pairs, fits Hedges' g + SE on the Δ distribution, derives MDE
    and verdict via `corroborate.statistics`. Same shape as
    v9's paired comparison (dialectic/hypothesis.py:332-350).

    All runs must share `env_name`. Treatment and baseline must
    share at least one seed for a valid comparison; if pairs is
    empty, returns a POWER_INSUFFICIENT row with NaN stats."""
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

    env_names = {_run_env_name(r) for r in (*treatment_runs, *baseline_runs)}
    if len(env_names) != 1:
        raise ValueError(
            f'env_name mismatch across runs: {env_names!r}'
        )
    env_name = next(iter(env_names))

    # Index by seed; compute Δ over the intersection.
    treatment_by_seed = {_run_seed(r): r for r in treatment_runs}
    baseline_by_seed = {_run_seed(r): r for r in baseline_runs}
    paired_seeds = sorted(treatment_by_seed.keys() & baseline_by_seed.keys())

    deltas: list[float] = [
        _run_outcome(treatment_by_seed[s], outcome_path)
        - _run_outcome(baseline_by_seed[s], outcome_path)
        for s in paired_seeds
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
        'env_name': env_name,
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


# ============ Cross-env link verdict (PAPER_NOTES.md §3.5) ============

def link_pearson_across_envs(
    mechanism_comparisons: Sequence[ComparisonRow],
    outcome_comparisons: Sequence[ComparisonRow],
    *,
    mechanism_path: str,
    outcome_path: str,
    alpha: float = 0.05,
    power: float = 0.8,
    cycle_id: str | None = None,
    timestamp: str | None = None,
    extra_measurements: Mapping[str, MeasurementLeaf] | None = None,
) -> ComparisonRow:
    """Cross-env link verdict — Pearson r between mechanism Δ and
    outcome Δ across envs.

    PAPER_NOTES.md §3.5: 'a methodological intervention's mechanism
    fingerprint should covary with its outcome fingerprint across
    envs.' If DDQN reduces the Jensen gap on env A and improves
    the return on env A (and likewise for env B, ...), the Pearson
    correlation across envs of (mechanism_g, outcome_g) is positive.
    Negative or zero correlation means mechanism reduction doesn't
    track outcome improvement — a different relationship than
    ddqn-helps-on-A AND ddqn-helps-on-B taken separately.

    Inputs are aligned per env: `mechanism_comparisons[i]` and
    `outcome_comparisons[i]` should refer to the same env. The
    function pairs by env_name; mismatches raise.

    Returns a ComparisonRow tagged at the cross-env granularity
    (env_name='cross_env_link'); its measurements include the
    Pearson r as the effect-size statistic and a derived MDE-style
    verdict on r vs 0."""
    from corroborate.statistics import (
        adequately_powered_paired,
        delta_i_from_q,
        derived_q_from_g_se,
    )

    # Pair by env_name.
    mech_by_env = {
        _comparison_env(c): c for c in mechanism_comparisons
    }
    out_by_env = {
        _comparison_env(c): c for c in outcome_comparisons
    }
    paired_envs = sorted(mech_by_env.keys() & out_by_env.keys())

    mech_gs: list[float] = []
    out_gs: list[float] = []
    for env in paired_envs:
        mg = mech_by_env[env].measurements.get(mechanism_path)
        og = out_by_env[env].measurements.get(outcome_path)
        if isinstance(mg, (int, float)) and isinstance(og, (int, float)):
            if not (math.isnan(float(mg)) or math.isnan(float(og))):
                mech_gs.append(float(mg))
                out_gs.append(float(og))

    n = len(mech_gs)
    if n < 3:
        # Pearson's r needs at least 3 paired observations to be
        # well-defined; return a POWER_INSUFFICIENT row.
        underpowered_measurements: dict[str, MeasurementLeaf] = {
            'env_name': 'cross_env_link',
            'intervention_name': 'link',
            'n_paired_envs': n,
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
        'env_name': 'cross_env_link',
        'intervention_name': 'link',
        'n_paired_envs': n,
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


def _comparison_env(c: ComparisonRow) -> str:
    """Read env_name off a ComparisonRow's measurements."""
    v = c.measurements.get('env_name')
    if not isinstance(v, str):
        raise TypeError(
            f"ComparisonRow {c.id!r} missing 'env_name' measurement"
        )
    return v


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
