"""Aggregate — typed sweep → ArmRow → ComparisonRow hand-off.

The framework's sweep emits `list[RunRow]`; downstream consumers
need per-(hypothesis, env) ArmRows and per-(treatment, baseline)
ComparisonRows. This module provides the typed factory functions
for that hand-off — porting v10's `HypothesisComparisonRow.from_
cells` pattern with the actual statistics computation deferred
to step 5.

Structure:

- `arm_from_runs(runs, *, intervention_name, env_name, ...)`
  produces an ArmRow from a homogeneous list of RunRows (same
  intervention_name + env_name + mechanism_key). Computes
  `arm_mean` and `arm_sd` from `primary_outcome_summary` across
  cells. Aggregates `facts` by name (admit-rate per fact).
- `comparison_from_arms(treatment, baseline, *, predicted_direction,
  ...)` produces a ComparisonRow with per-arm stats threaded
  through. Stat fields (effect_size_g, se, derived_q,
  delta_i_population, refutation_class, adequately_powered) are
  populated by `_default_statistics_stub` for v0; step 5 replaces
  it with Hedges' g + power machinery.
- `aggregate_runs(runs)` is the convenience entry point: groups
  runs by (intervention_name, env_name, mechanism_key), produces
  one ArmRow per group.

Step 5's MDE+power statistics module will replace the stub with
real `Hedges_g`, `SE_g`, `derived_q`, `RefutationClass` selection.
The shape stays — only `_default_statistics_stub` swaps out."""
from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from corroborate.hypothesis import Direction, MechanismKey
from corroborate.schema import ArmRow, ComparisonRow, FactRow, RunRow
from corroborate.verdict import RefutationClass, Verdict


# ============ ArmRow factory ============

def _empty_meta() -> dict[str, str | int | float | bool]:
    return {}


def arm_from_runs(
    runs: Sequence[RunRow],
    *,
    intervention_name: str,
    env_name: str,
    mechanism_key: MechanismKey,
    cycle_id: str | None = None,
    timestamp: str | None = None,
    meta: Mapping[str, str | int | float | bool] | None = None,
) -> ArmRow:
    """Construct an ArmRow from a list of RunRows. The runs must
    be homogeneous in `intervention_name`, `env_name`, and
    `mechanism_key` (caller is responsible for grouping); this
    function trusts that contract.

    Computes `arm_mean` and `arm_sd` from each run's
    `primary_outcome_summary`. Aggregates facts by name and
    converts per-cell verdict counts into the ArmRow's facts
    list (admit-rate becomes `natural_strength` placeholder
    until step 5 fills in real q values)."""
    if not runs:
        raise ValueError('arm_from_runs requires ≥1 RunRow')

    summaries = [r.primary_outcome_summary for r in runs]
    n = len(summaries)
    arm_mean = sum(summaries) / n
    if n > 1:
        var = sum((s - arm_mean) ** 2 for s in summaries) / (n - 1)
        arm_sd = math.sqrt(var)
    else:
        arm_sd = 0.0

    facts = _aggregate_facts_by_name(runs)
    reads_set: frozenset[str] = frozenset()
    for f in facts:
        reads_set = reads_set | f.reads

    resolved_timestamp = (
        timestamp
        if timestamp is not None
        else datetime.now(UTC).isoformat(timespec='seconds')
    )

    return ArmRow(
        id=str(uuid.uuid4()),
        intervention_name=intervention_name,
        env_name=env_name,
        cycle_id=cycle_id,
        timestamp=resolved_timestamp,
        mechanism_key=mechanism_key,
        run_ids=tuple(r.id for r in runs),
        seeds=tuple(r.seed for r in runs),
        n=n,
        arm_mean=arm_mean,
        arm_sd=arm_sd,
        facts=facts,
        reads_set=reads_set,
        meta={**(meta or _empty_meta())},
    )


def _aggregate_facts_by_name(runs: Sequence[RunRow]) -> tuple[FactRow, ...]:
    """Group facts across runs by `name`, then collapse each group
    into one fact carrying the per-cell admit-rate as
    `natural_strength` (placeholder until step 5 derives real q
    from raw stats). Verdict at the arm level: the majority verdict
    if any, else the most-common; for v0, `HELD` if all admit,
    `NO_EFFECT` if all reject, else `POWER_INSUFFICIENT`."""
    by_name: dict[str, list[FactRow]] = {}
    for run in runs:
        for f in run.facts:
            by_name.setdefault(f.name, []).append(f)

    out: list[FactRow] = []
    for name, group in by_name.items():
        n = len(group)
        n_held = sum(1 for f in group if f.verdict is Verdict.HELD)
        n_rejected = sum(1 for f in group if f.verdict is Verdict.NO_EFFECT)
        admit_rate = n_held / n if n > 0 else 0.0
        if n_held == n:
            verdict = Verdict.HELD
        elif n_rejected == n:
            verdict = Verdict.NO_EFFECT
        else:
            verdict = Verdict.POWER_INSUFFICIENT

        first = group[0]
        merged_reads: frozenset[str] = frozenset()
        for f in group:
            merged_reads = merged_reads | f.reads
        out.append(FactRow(
            name=name,
            kind=first.kind,
            targets=first.targets,
            reads=merged_reads,
            verdict=verdict,
            natural_strength=admit_rate,
            delta_i=0.0,
            evidentiary_level=first.evidentiary_level,
            stats={**first.stats},
            intervention_signature=first.intervention_signature,
        ))
    return tuple(out)


# ============ ComparisonRow factory ============

@dataclass(frozen=True, slots=True)
class _StatisticsStub:
    """Stub statistics — what the step-5 MDE+power module will
    replace with real Hedges' g + SE + derived q. Each field
    matches the corresponding ComparisonRow field one-to-one."""
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


def comparison_from_arms(
    treatment: ArmRow,
    baseline: ArmRow,
    *,
    predicted_direction: Direction | None,
    cycle_id: str | None = None,
    timestamp: str | None = None,
    meta: Mapping[str, str | int | float | bool] | None = None,
    statistics_fn: object = None,
) -> ComparisonRow:
    """Construct a ComparisonRow from a (treatment, baseline) arm
    pair. The arms must share `env_name` (caller is responsible
    for matching).

    `statistics_fn` is the pluggable MDE+power computation;
    `None` uses the v0 stub. Step 5's statistics module passes
    its own implementation here. Type is `object` because the
    function signature `Callable[[ArmRow, ArmRow], _StatisticsStub]`
    isn't carried at the framework level (callers can use the
    stub or their own — the typed contract is the
    `_StatisticsStub` shape).

    `predicted_direction` is taken from the treatment's
    `Hypothesis.predicted_direction` and threaded through; the
    caller passes it explicitly because ArmRow doesn't carry it."""
    if treatment.env_name != baseline.env_name:
        raise ValueError(
            f'env_name mismatch: treatment={treatment.env_name!r} '
            f'vs baseline={baseline.env_name!r}'
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

    # Merge facts from both arms (admit-rate-weighted union).
    facts = _merge_arm_facts(treatment, baseline)
    reads_set: frozenset[str] = frozenset()
    for f in facts:
        reads_set = reads_set | f.reads

    resolved_timestamp = (
        timestamp
        if timestamp is not None
        else datetime.now(UTC).isoformat(timespec='seconds')
    )

    return ComparisonRow(
        id=str(uuid.uuid4()),
        parent_id=None,
        intervention_name=treatment.intervention_name,
        env_name=treatment.env_name,
        cycle_id=cycle_id,
        timestamp=resolved_timestamp,
        treatment_arm_id=treatment.id,
        baseline_arm_id=baseline.id,
        mechanism_key=treatment.mechanism_key,
        predicted_direction=predicted_direction,
        n_treatment=treatment.n,
        n_baseline=baseline.n,
        arm_a_mean=treatment.arm_mean,
        arm_a_sd=treatment.arm_sd,
        arm_b_mean=baseline.arm_mean,
        arm_b_sd=baseline.arm_sd,
        effect_size_g=stats.effect_size_g,
        se=stats.se,
        derived_q=stats.derived_q,
        delta_i_population=stats.delta_i_population,
        verdict=stats.verdict,
        refutation_class=stats.refutation_class,
        adequately_powered=stats.adequately_powered,
        facts=facts,
        reads_set=reads_set,
        meta={**(meta or _empty_meta())},
    )


def _merge_arm_facts(
    treatment: ArmRow, baseline: ArmRow,
) -> tuple[FactRow, ...]:
    """Combine arm-level facts. v0: prefer treatment's fact when
    a name appears on both arms (treatment is what the hypothesis
    is actually testing); pure baseline-only facts are appended.
    Step 6 may refine this when comparison-level fact aggregation
    has clearer semantics."""
    by_name: dict[str, FactRow] = {}
    for f in baseline.facts:
        by_name[f.name] = f
    for f in treatment.facts:  # treatment overwrites baseline on name conflict
        by_name[f.name] = f
    return tuple(by_name.values())


# ============ Convenience: aggregate runs into arms ============

def aggregate_runs(runs: Iterable[RunRow]) -> list[ArmRow]:
    """Group `runs` by (intervention_name, env_name,
    mechanism_key), build one ArmRow per group. Convenience for
    the common dialectic-loop case where a sweep produces a
    flat list of cells across multiple (hypothesis, env)
    combinations."""
    by_key: dict[tuple[str, str, MechanismKey], list[RunRow]] = {}
    for r in runs:
        key = (r.intervention_name, r.env_name, r.mechanism_key)
        by_key.setdefault(key, []).append(r)

    out: list[ArmRow] = []
    for (intervention_name, env_name, mechanism_key), group in by_key.items():
        out.append(arm_from_runs(
            group,
            intervention_name=intervention_name,
            env_name=env_name,
            mechanism_key=mechanism_key,
            cycle_id=group[0].cycle_id,
        ))
    return out
