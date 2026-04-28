"""Aggregate — typed sweep → ArmRow → ComparisonRow hand-off.

The framework's sweep emits `list[RunRow]`; downstream consumers
need per-(intervention, env) ArmRows and per-(treatment, baseline)
ComparisonRows. This module provides the typed factory functions
for that hand-off.

Structure:

- `aggregate_cell_verdict(verdicts)` — Popperian aggregation over
  per-bridge verdicts (any single refutation refutes; INVARIANT_
  VIOLATION dominates).
- `arm_from_runs(runs, *, intervention_name, env_name, ...)`
  produces an ArmRow from a homogeneous list of RunRows. Computes
  arm-level outcome statistics from each run's
  `outcome.late_window_mean` measurement (NaN-aware), aggregates
  per-bridge admit-rates across runs, and forwards the HP-only
  subset of measurements (the configurational fingerprint).
- `comparison_from_arms(treatment, baseline, *, predicted_direction,
  ...)` produces a ComparisonRow with per-arm stats threaded
  through `measurements` (`outcome.<m>.arm_a_mean` etc.). Stat
  fields populated by `_default_statistics_stub` for v0; step 5
  replaces it with Hedges' g + power machinery.
- `aggregate_runs(runs)` is the convenience entry point: groups
  runs by (intervention_name, env_name, hp_signature), produces
  one ArmRow per group.
- `hp_signature(measurements)` — the configuration fingerprint
  used as a group-by key. Filters out outcome/bridge/invariant
  paths and substrate-metadata keys, returns sorted (path, str)
  pairs.

Step 5's MDE+power statistics module will replace the stub with
real `Hedges_g`, `SE_g`, `derived_q`, `RefutationClass` selection.
The shape stays — only `_default_statistics_stub` swaps out."""
from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from corroborate.hypothesis import Direction
from corroborate.schema import ArmRow, ComparisonRow, MeasurementLeaf, RunRow
from corroborate.verdict import RefutationClass, Verdict


# ============ HP-signature projection ============

_NON_HP_PREFIXES: tuple[str, ...] = ('outcome.', 'bridge.', 'invariant.')
_NON_HP_KEYS: frozenset[str] = frozenset({
    'env_name', 'seed', 'total_steps', 'intervention_name',
})


def hp_signature(
    measurements: Mapping[str, MeasurementLeaf],
) -> tuple[tuple[str, str], ...]:
    """The configuration fingerprint — HP-only subset of
    `measurements` as a sorted (path, str-canonical-value) tuple.
    Hashable; suitable as a group-by key.

    Filters out paths under `outcome.`/`bridge.`/`invariant.`
    prefixes and substrate-metadata keys (`env_name`, `seed`,
    `total_steps`, `intervention_name`). What remains is the HP
    leaves at their dotted topology paths plus any other author-
    chosen scalar measurement that isn't a result/metadata."""
    return tuple(sorted(
        (k, str(v))
        for k, v in measurements.items()
        if not any(k.startswith(p) for p in _NON_HP_PREFIXES)
        and k not in _NON_HP_KEYS
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


def _outcome_summary(measurements: Mapping[str, MeasurementLeaf]) -> float:
    """Read `outcome.late_window_mean` as a float, returning NaN
    if absent or non-numeric. Cell runners write a substrate-named
    outcome key; v0's RL substrate uses `outcome.late_window_mean`.
    Cells without that key contribute NaN to the arm aggregate."""
    v = measurements.get('outcome.late_window_mean')
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

    Computes `outcome.late_window_mean.arm_mean` and `arm_sd`
    (NaN-aware) from each run's outcome measurement. Intersects
    the HP subset of measurements across the runs and forwards
    each common (k, v) entry. Per-bridge admit-rates land at
    `bridge.<name>.admit_rate` / `invariant.<name>.admit_rate`."""
    if not runs:
        raise ValueError('arm_from_runs requires ≥1 RunRow')

    summaries = [_outcome_summary(r.measurements) for r in runs]
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

    # HP-subset shared across all runs in the arm. Authors who
    # follow the grouping contract get identical HP signatures for
    # all members; we still intersect defensively (a run with a
    # bridge-set-difference in measurements doesn't poison the arm).
    common_hp = _intersect_hp_measurements(runs)

    measurements: dict[str, MeasurementLeaf] = {
        'env_name': env_name,
        'intervention_name': intervention_name,
        'n': n,
        'outcome.late_window_mean.arm_mean': arm_mean,
        'outcome.late_window_mean.arm_sd': arm_sd,
        **common_hp,
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


def _intersect_hp_measurements(
    runs: Sequence[RunRow],
) -> dict[str, MeasurementLeaf]:
    """Intersection of HP measurements across runs. A path that
    appears with the SAME value in every run survives; anything
    else is dropped. Uses `hp_signature` indirectly via the
    same NON_HP filter — the projection is HP keys only."""
    if not runs:
        return {}
    head_hp = {
        k: v for k, v in runs[0].measurements.items()
        if not any(k.startswith(p) for p in _NON_HP_PREFIXES)
        and k not in _NON_HP_KEYS
    }
    out: dict[str, MeasurementLeaf] = {}
    for k, v in head_hp.items():
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
    """v0 placeholder for step 5's MDE+power machinery. Returns
    None for the stat fields, POWER_INSUFFICIENT verdict, no
    refutation_class. Step 5 replaces this function with real
    Hedges' g computation."""
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


def _arm_outcome_stats(arm: ArmRow) -> tuple[float, float]:
    """Read (arm_mean, arm_sd) for `outcome.late_window_mean` off
    an arm. Loud error if absent (arm-builder always writes them)."""
    mean_v = arm.measurements.get('outcome.late_window_mean.arm_mean')
    sd_v = arm.measurements.get('outcome.late_window_mean.arm_sd')
    if isinstance(mean_v, bool) or not isinstance(mean_v, (int, float)):
        raise TypeError(
            f"ArmRow {arm.id!r} missing "
            f"'outcome.late_window_mean.arm_mean' measurement"
        )
    if isinstance(sd_v, bool) or not isinstance(sd_v, (int, float)):
        raise TypeError(
            f"ArmRow {arm.id!r} missing "
            f"'outcome.late_window_mean.arm_sd' measurement"
        )
    return float(mean_v), float(sd_v)


def comparison_from_arms(
    treatment: ArmRow,
    baseline: ArmRow,
    *,
    predicted_direction: Direction | None,
    cycle_id: str | None = None,
    timestamp: str | None = None,
    extra_measurements: Mapping[str, MeasurementLeaf] | None = None,
    statistics_fn: object = None,
) -> ComparisonRow:
    """Construct a ComparisonRow from a (treatment, baseline) arm
    pair. The arms must share `env_name` (caller is responsible
    for matching).

    Per-arm stats land in `measurements` under `outcome.<m>.arm_a_*`
    / `outcome.<m>.arm_b_*` paths plus `n_treatment` / `n_baseline`
    / `intervention_name` / `env_name`. Step 5 fills in real
    `outcome.<m>.effect_size_g` / `se` / `derived_q` /
    `delta_i_population`; v0 stub leaves them None / 0.0.

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

    arm_a_mean, arm_a_sd = _arm_outcome_stats(treatment)
    arm_b_mean, arm_b_sd = _arm_outcome_stats(baseline)
    n_treatment = _arm_n(treatment)
    n_baseline = _arm_n(baseline)

    measurements: dict[str, MeasurementLeaf] = {
        'env_name': treatment_env,
        'intervention_name': _arm_intervention_name(treatment),
        'n_treatment': n_treatment,
        'n_baseline': n_baseline,
        'outcome.late_window_mean.arm_a_mean': arm_a_mean,
        'outcome.late_window_mean.arm_a_sd': arm_a_sd,
        'outcome.late_window_mean.arm_b_mean': arm_b_mean,
        'outcome.late_window_mean.arm_b_sd': arm_b_sd,
        'outcome.late_window_mean.delta_i_population': stats.delta_i_population,
    }
    if stats.effect_size_g is not None:
        measurements['outcome.late_window_mean.effect_size_g'] = stats.effect_size_g
    if stats.se is not None:
        measurements['outcome.late_window_mean.se'] = stats.se
    if stats.derived_q is not None:
        measurements['outcome.late_window_mean.derived_q'] = stats.derived_q

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


def aggregate_runs(runs: Iterable[RunRow]) -> list[ArmRow]:
    """Group `runs` by (intervention_name, env_name, hp_signature),
    build one ArmRow per group. Convenience for the common
    dialectic-loop case where a sweep produces a flat list of
    cells across multiple (hypothesis, env) combinations.

    Grouping uses `hp_signature(run.measurements)` rather than a
    declared `MechanismKey` artifact — two runs with identical HP
    settings on the same intervention/env land in the same arm,
    even if they were authored as distinct Hypothesis instances."""
    by_key: dict[
        tuple[str, str, tuple[tuple[str, str], ...]],
        list[RunRow],
    ] = {}
    for r in runs:
        key = (
            _run_intervention_name(r),
            _run_env_name(r),
            hp_signature(r.measurements),
        )
        by_key.setdefault(key, []).append(r)

    out: list[ArmRow] = []
    for (intervention_name, env_name, _), group in by_key.items():
        out.append(arm_from_runs(
            group,
            intervention_name=intervention_name,
            env_name=env_name,
            cycle_id=group[0].cycle_id,
        ))
    return out
