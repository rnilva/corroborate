"""Tests for `aggregate` — sweep → ArmRow → ComparisonRow hand-off.

The framework's typed factory functions ported from v10's
`HypothesisComparisonRow.from_cells` pattern. Tests verify the
arm-construction grouping, comparison construction with stat
stubs, and the convenience `aggregate_runs` entry point."""
from __future__ import annotations

from corroborate.aggregate import (
    aggregate_runs,
    arm_from_runs,
    comparison_from_arms,
)
from corroborate.hypothesis import MechanismKey
from corroborate.schema import ArmRow, FactRow, RunRow
from corroborate.verdict import Verdict


# ============ Fixtures ============

def _sample_mechanism_key() -> MechanismKey:
    return MechanismKey(
        intervention_signature=(
            ('greedification', 'callable:double_greedify'),
        ),
        bridge_names=frozenset({'mechanism', 'outcome'}),
        direction=None,
    )


def _make_runrow(
    *,
    intervention_name: str,
    env_name: str,
    seed: int,
    primary: float,
    fact_verdict: Verdict = Verdict.HELD,
    mechanism_key: MechanismKey | None = None,
) -> RunRow:
    mk = mechanism_key if mechanism_key is not None else _sample_mechanism_key()
    fact = FactRow(
        name='outcome',
        kind='bridge',
        targets=('final_return',),
        reads=frozenset({'final_return'}),
        verdict=fact_verdict,
        natural_strength=0.0,
        delta_i=0.0,
        evidentiary_level='correlational',
        stats={},
    )
    return RunRow(
        id=f'run-{intervention_name}-{env_name}-{seed}',
        parent_id=None,
        intervention_name=intervention_name,
        cycle_id=None,
        timestamp='2026-04-27T10:00:00Z',
        env_name=env_name,
        total_steps=1000,
        seed=seed,
        mechanism_key=mk,
        primary_outcome_summary=primary,
        record_keys=('final_return',),
        facts=(fact,),
        reads_set=frozenset({'final_return'}),
        verdict=fact_verdict,
    )


# ============ arm_from_runs ============

def test_arm_from_runs_computes_mean_and_sd() -> None:
    runs = [
        _make_runrow(intervention_name='ddqn', env_name='e', seed=i, primary=p)
        for i, p in enumerate([10.0, 12.0, 14.0])
    ]
    arm = arm_from_runs(
        runs,
        intervention_name='ddqn',
        env_name='e',
        mechanism_key=_sample_mechanism_key(),
    )
    assert arm.n == 3
    assert arm.arm_mean == 12.0  # (10 + 12 + 14) / 3
    # Sample SD with ddof=1: sqrt(((10-12)^2 + 0 + (14-12)^2) / 2) = 2.0
    assert abs(arm.arm_sd - 2.0) < 1e-9


def test_arm_from_runs_n_one_yields_zero_sd() -> None:
    runs = [_make_runrow(intervention_name='h', env_name='e', seed=0, primary=42.0)]
    arm = arm_from_runs(
        runs,
        intervention_name='h',
        env_name='e',
        mechanism_key=_sample_mechanism_key(),
    )
    assert arm.arm_sd == 0.0


def test_arm_from_runs_collects_run_ids_and_seeds() -> None:
    runs = [
        _make_runrow(intervention_name='h', env_name='e', seed=i, primary=float(i))
        for i in [0, 1, 2]
    ]
    arm = arm_from_runs(
        runs,
        intervention_name='h',
        env_name='e',
        mechanism_key=_sample_mechanism_key(),
    )
    assert arm.run_ids == ('run-h-e-0', 'run-h-e-1', 'run-h-e-2')
    assert arm.seeds == (0, 1, 2)


def test_arm_from_runs_aggregates_facts_by_admit_rate() -> None:
    """Per-fact admit-rate becomes the arm fact's natural_strength.
    All-HELD → arm fact verdict = HELD; mixed → POWER_INSUFFICIENT;
    all-rejected → NO_EFFECT."""
    runs_all_held = [
        _make_runrow(
            intervention_name='h', env_name='e', seed=i, primary=1.0,
            fact_verdict=Verdict.HELD,
        )
        for i in range(3)
    ]
    arm = arm_from_runs(
        runs_all_held,
        intervention_name='h', env_name='e',
        mechanism_key=_sample_mechanism_key(),
    )
    assert len(arm.facts) == 1
    assert arm.facts[0].verdict is Verdict.HELD
    assert arm.facts[0].natural_strength == 1.0  # 3/3 admit rate

    runs_mixed = [
        _make_runrow(intervention_name='h', env_name='e', seed=0, primary=1.0,
                     fact_verdict=Verdict.HELD),
        _make_runrow(intervention_name='h', env_name='e', seed=1, primary=1.0,
                     fact_verdict=Verdict.NO_EFFECT),
    ]
    arm = arm_from_runs(
        runs_mixed,
        intervention_name='h', env_name='e',
        mechanism_key=_sample_mechanism_key(),
    )
    assert arm.facts[0].verdict is Verdict.POWER_INSUFFICIENT
    assert arm.facts[0].natural_strength == 0.5


def test_arm_from_runs_empty_input_raises() -> None:
    try:
        arm_from_runs(
            [], intervention_name='h', env_name='e',
            mechanism_key=_sample_mechanism_key(),
        )
        raise AssertionError('expected ValueError')
    except ValueError:
        pass


# ============ comparison_from_arms ============

def test_comparison_from_arms_carries_arm_stats() -> None:
    treatment = arm_from_runs(
        [_make_runrow(intervention_name='ddqn', env_name='e',
                      seed=i, primary=float(i + 10)) for i in range(3)],
        intervention_name='ddqn', env_name='e',
        mechanism_key=_sample_mechanism_key(),
    )
    baseline = arm_from_runs(
        [_make_runrow(intervention_name='vanilla', env_name='e',
                      seed=i, primary=float(i)) for i in range(3)],
        intervention_name='vanilla', env_name='e',
        mechanism_key=_sample_mechanism_key(),
    )
    cmp = comparison_from_arms(
        treatment, baseline,
        predicted_direction='a_gt_b',
    )
    assert cmp.intervention_name == 'ddqn'
    assert cmp.treatment_arm_id == treatment.id
    assert cmp.baseline_arm_id == baseline.id
    assert cmp.arm_a_mean == treatment.arm_mean
    assert cmp.arm_b_mean == baseline.arm_mean
    assert cmp.predicted_direction == 'a_gt_b'


def test_comparison_from_arms_default_stub_yields_power_insufficient() -> None:
    """The v0 stub returns None for stats and POWER_INSUFFICIENT
    verdict — step 5's MDE+power module replaces this with real
    Hedges' g + power machinery."""
    arm_t = arm_from_runs(
        [_make_runrow(intervention_name='t', env_name='e',
                      seed=0, primary=1.0)],
        intervention_name='t', env_name='e',
        mechanism_key=_sample_mechanism_key(),
    )
    arm_b = arm_from_runs(
        [_make_runrow(intervention_name='b', env_name='e',
                      seed=0, primary=0.0)],
        intervention_name='b', env_name='e',
        mechanism_key=_sample_mechanism_key(),
    )
    cmp = comparison_from_arms(arm_t, arm_b, predicted_direction=None)
    assert cmp.effect_size_g is None
    assert cmp.se is None
    assert cmp.derived_q is None
    assert cmp.delta_i_population == 0.0
    assert cmp.verdict is Verdict.POWER_INSUFFICIENT
    assert cmp.refutation_class is None
    assert cmp.adequately_powered is False


def test_comparison_from_arms_env_mismatch_raises() -> None:
    arm_t = arm_from_runs(
        [_make_runrow(intervention_name='t', env_name='env_a',
                      seed=0, primary=1.0)],
        intervention_name='t', env_name='env_a',
        mechanism_key=_sample_mechanism_key(),
    )
    arm_b = arm_from_runs(
        [_make_runrow(intervention_name='b', env_name='env_b',
                      seed=0, primary=0.0)],
        intervention_name='b', env_name='env_b',
        mechanism_key=_sample_mechanism_key(),
    )
    try:
        _ = comparison_from_arms(arm_t, arm_b, predicted_direction=None)
        raise AssertionError('expected ValueError on env mismatch')
    except ValueError:
        pass


# ============ aggregate_runs ============

def test_aggregate_runs_groups_by_intervention_env_mechanism() -> None:
    """Runs from different (intervention, env) combinations
    produce distinct ArmRows; runs from the same combination
    coalesce."""
    runs = [
        # (ddqn, env_a) × 2 seeds
        _make_runrow(intervention_name='ddqn', env_name='env_a',
                     seed=0, primary=1.0),
        _make_runrow(intervention_name='ddqn', env_name='env_a',
                     seed=1, primary=2.0),
        # (vanilla, env_a) × 2 seeds
        _make_runrow(intervention_name='vanilla', env_name='env_a',
                     seed=0, primary=3.0),
        _make_runrow(intervention_name='vanilla', env_name='env_a',
                     seed=1, primary=4.0),
        # (ddqn, env_b) × 1 seed
        _make_runrow(intervention_name='ddqn', env_name='env_b',
                     seed=0, primary=5.0),
    ]
    arms = aggregate_runs(runs)
    assert len(arms) == 3

    by_key: dict[tuple[str, str], ArmRow] = {
        (a.intervention_name, a.env_name): a for a in arms
    }
    assert by_key[('ddqn', 'env_a')].n == 2
    assert by_key[('vanilla', 'env_a')].n == 2
    assert by_key[('ddqn', 'env_b')].n == 1


def test_aggregate_runs_distinct_mechanism_keys_split_arms() -> None:
    """Runs with the SAME intervention_name + env but DIFFERENT
    mechanism_keys (e.g. different bridge sets) produce distinct
    arms — the mechanism_key is part of the grouping key."""
    mk_a = _sample_mechanism_key()
    mk_b = MechanismKey(
        intervention_signature=mk_a.intervention_signature,
        bridge_names=frozenset({'different', 'bridge_set'}),
        direction=mk_a.direction,
    )
    runs = [
        _make_runrow(intervention_name='h', env_name='e',
                     seed=0, primary=1.0, mechanism_key=mk_a),
        _make_runrow(intervention_name='h', env_name='e',
                     seed=0, primary=2.0, mechanism_key=mk_b),
    ]
    arms = aggregate_runs(runs)
    assert len(arms) == 2
