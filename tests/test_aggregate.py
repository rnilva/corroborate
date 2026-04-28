"""Tests for `aggregate` — sweep → ArmRow → ComparisonRow hand-off.

The framework's typed factory functions. Tests verify the
arm-construction grouping (now via `hp_signature` rather than
`MechanismKey`), comparison construction with stat stubs, and the
convenience `aggregate_runs` entry point."""
from __future__ import annotations

from corroborate.aggregate import (
    aggregate_cell_verdict,
    aggregate_runs,
    arm_from_runs,
    comparison_from_arms,
    hp_signature,
)
from corroborate.schema import ArmRow, MeasurementLeaf, RunRow
from corroborate.verdict import Verdict


# ============ Fixtures ============

def _make_runrow(
    *,
    intervention_name: str,
    env_name: str,
    seed: int,
    primary: float,
    extra_hp: dict[str, MeasurementLeaf] | None = None,
    bridge_verdict: Verdict | None = Verdict.HELD,
) -> RunRow:
    measurements: dict[str, MeasurementLeaf] = {
        'intervention_name': intervention_name,
        'env_name': env_name,
        'seed': seed,
        'total_steps': 1000,
        'outcome.late_window_mean': primary,
    }
    if extra_hp is not None:
        measurements.update(extra_hp)
    if bridge_verdict is not None:
        measurements['bridge.outcome.verdict'] = bridge_verdict.value
    cell_verdict = (
        bridge_verdict if bridge_verdict is not None
        else Verdict.POWER_INSUFFICIENT
    )
    return RunRow(
        id=f'run-{intervention_name}-{env_name}-{seed}',
        parent_id=None,
        cycle_id=None,
        timestamp='2026-04-27T10:00:00Z',
        verdict=cell_verdict,
        measurements=measurements,
    )


# ============ aggregate_cell_verdict ============

def test_aggregate_cell_verdict_empty_yields_power_insufficient() -> None:
    assert aggregate_cell_verdict(()) is Verdict.POWER_INSUFFICIENT


def test_aggregate_cell_verdict_invariant_violation_dominates() -> None:
    assert aggregate_cell_verdict(
        (Verdict.HELD, Verdict.INVARIANT_VIOLATION),
    ) is Verdict.INVARIANT_VIOLATION
    assert aggregate_cell_verdict(
        (Verdict.NO_EFFECT, Verdict.INVARIANT_VIOLATION),
    ) is Verdict.INVARIANT_VIOLATION


def test_aggregate_cell_verdict_no_effect_dominates_held() -> None:
    assert aggregate_cell_verdict(
        (Verdict.HELD, Verdict.NO_EFFECT),
    ) is Verdict.NO_EFFECT


def test_aggregate_cell_verdict_all_held_yields_held() -> None:
    assert aggregate_cell_verdict(
        (Verdict.HELD, Verdict.HELD),
    ) is Verdict.HELD


def test_aggregate_cell_verdict_mixed_held_power_insufficient() -> None:
    assert aggregate_cell_verdict(
        (Verdict.HELD, Verdict.POWER_INSUFFICIENT),
    ) is Verdict.POWER_INSUFFICIENT


# ============ hp_signature ============

def test_hp_signature_filters_outcome_and_metadata() -> None:
    measurements: dict[str, MeasurementLeaf] = {
        'gamma': 0.99,
        'optimizer.inner.lr': 0.001,
        'env_name': 'CartPole-v1',
        'seed': 0,
        'total_steps': 1000,
        'intervention_name': 'h',
        'outcome.late_window_mean': 100.0,
        'bridge.x.verdict': 'held',
        'invariant.y.verdict': 'held',
    }
    sig = hp_signature(measurements)
    keys = [k for k, _ in sig]
    assert 'gamma' in keys
    assert 'optimizer.inner.lr' in keys
    assert 'env_name' not in keys
    assert 'seed' not in keys
    assert 'total_steps' not in keys
    assert 'intervention_name' not in keys
    assert not any(k.startswith('outcome.') for k in keys)
    assert not any(k.startswith('bridge.') for k in keys)
    assert not any(k.startswith('invariant.') for k in keys)


def test_hp_signature_is_sorted_and_hashable() -> None:
    sig_a = hp_signature({'b': 1, 'a': 2})
    sig_b = hp_signature({'a': 2, 'b': 1})
    assert sig_a == sig_b
    # Hashable as a dict key.
    _: dict[tuple[tuple[str, str], ...], int] = {sig_a: 0}


# ============ arm_from_runs ============

def test_arm_from_runs_computes_outcome_mean_and_sd() -> None:
    runs = [
        _make_runrow(
            intervention_name='ddqn', env_name='e', seed=i, primary=p,
        )
        for i, p in enumerate([10.0, 12.0, 14.0])
    ]
    arm = arm_from_runs(
        runs, intervention_name='ddqn', env_name='e',
    )
    assert arm.measurements['n'] == 3
    assert arm.measurements['outcome.late_window_mean.arm_mean'] == 12.0
    sd = arm.measurements['outcome.late_window_mean.arm_sd']
    assert isinstance(sd, float)
    assert abs(sd - 2.0) < 1e-9


def test_arm_from_runs_n_one_yields_zero_sd() -> None:
    runs = [
        _make_runrow(
            intervention_name='h', env_name='e', seed=0, primary=42.0,
        ),
    ]
    arm = arm_from_runs(runs, intervention_name='h', env_name='e')
    assert arm.measurements['outcome.late_window_mean.arm_sd'] == 0.0


def test_arm_from_runs_collects_run_ids() -> None:
    runs = [
        _make_runrow(
            intervention_name='h', env_name='e', seed=i, primary=float(i),
        )
        for i in [0, 1, 2]
    ]
    arm = arm_from_runs(runs, intervention_name='h', env_name='e')
    assert arm.run_ids == ('run-h-e-0', 'run-h-e-1', 'run-h-e-2')


def test_arm_from_runs_aggregates_bridge_admit_rate() -> None:
    """`bridge.<name>.admit_rate` aggregates across runs."""
    runs_all_held = [
        _make_runrow(
            intervention_name='h', env_name='e', seed=i, primary=1.0,
            bridge_verdict=Verdict.HELD,
        )
        for i in range(3)
    ]
    arm = arm_from_runs(
        runs_all_held, intervention_name='h', env_name='e',
    )
    assert arm.measurements['bridge.outcome.admit_rate'] == 1.0

    runs_mixed = [
        _make_runrow(
            intervention_name='h', env_name='e', seed=0, primary=1.0,
            bridge_verdict=Verdict.HELD,
        ),
        _make_runrow(
            intervention_name='h', env_name='e', seed=1, primary=1.0,
            bridge_verdict=Verdict.NO_EFFECT,
        ),
    ]
    arm = arm_from_runs(
        runs_mixed, intervention_name='h', env_name='e',
    )
    assert arm.measurements['bridge.outcome.admit_rate'] == 0.5


def test_arm_from_runs_forwards_common_hp() -> None:
    """HP measurements common to all runs surface on the ArmRow."""
    runs = [
        _make_runrow(
            intervention_name='h', env_name='e', seed=i, primary=float(i),
            extra_hp={'gamma': 0.99, 'lr': 0.001},
        )
        for i in range(2)
    ]
    arm = arm_from_runs(runs, intervention_name='h', env_name='e')
    assert arm.measurements['gamma'] == 0.99
    assert arm.measurements['lr'] == 0.001


def test_arm_from_runs_empty_input_raises() -> None:
    try:
        arm_from_runs([], intervention_name='h', env_name='e')
        raise AssertionError('expected ValueError')
    except ValueError:
        pass


# ============ comparison_from_arms ============

def test_comparison_from_arms_carries_arm_stats() -> None:
    treatment = arm_from_runs(
        [
            _make_runrow(
                intervention_name='ddqn', env_name='e',
                seed=i, primary=float(i + 10),
            )
            for i in range(3)
        ],
        intervention_name='ddqn', env_name='e',
    )
    baseline = arm_from_runs(
        [
            _make_runrow(
                intervention_name='vanilla', env_name='e',
                seed=i, primary=float(i),
            )
            for i in range(3)
        ],
        intervention_name='vanilla', env_name='e',
    )
    cmp = comparison_from_arms(
        treatment, baseline, predicted_direction='a_gt_b',
    )
    assert cmp.treatment_arm_id == treatment.id
    assert cmp.baseline_arm_id == baseline.id
    assert cmp.predicted_direction == 'a_gt_b'
    assert cmp.measurements['intervention_name'] == 'ddqn'
    # Treatment outcome mean threaded into arm_a_mean.
    expected_a_mean = treatment.measurements['outcome.late_window_mean.arm_mean']
    assert cmp.measurements['outcome.late_window_mean.arm_a_mean'] == expected_a_mean


def test_comparison_from_arms_default_stub_yields_power_insufficient() -> None:
    """The v0 stub returns POWER_INSUFFICIENT verdict and
    placeholder None stats."""
    arm_t = arm_from_runs(
        [_make_runrow(
            intervention_name='t', env_name='e', seed=0, primary=1.0,
        )],
        intervention_name='t', env_name='e',
    )
    arm_b = arm_from_runs(
        [_make_runrow(
            intervention_name='b', env_name='e', seed=0, primary=0.0,
        )],
        intervention_name='b', env_name='e',
    )
    cmp = comparison_from_arms(arm_t, arm_b, predicted_direction=None)
    assert cmp.verdict is Verdict.POWER_INSUFFICIENT
    assert cmp.refutation_class is None
    assert cmp.adequately_powered is False
    # Stub leaves None stats out of measurements.
    assert 'outcome.late_window_mean.effect_size_g' not in cmp.measurements
    assert 'outcome.late_window_mean.se' not in cmp.measurements
    assert cmp.measurements['outcome.late_window_mean.delta_i_population'] == 0.0


def test_comparison_from_arms_env_mismatch_raises() -> None:
    arm_t = arm_from_runs(
        [_make_runrow(
            intervention_name='t', env_name='env_a', seed=0, primary=1.0,
        )],
        intervention_name='t', env_name='env_a',
    )
    arm_b = arm_from_runs(
        [_make_runrow(
            intervention_name='b', env_name='env_b', seed=0, primary=0.0,
        )],
        intervention_name='b', env_name='env_b',
    )
    try:
        _ = comparison_from_arms(arm_t, arm_b, predicted_direction=None)
        raise AssertionError('expected ValueError on env mismatch')
    except ValueError:
        pass


# ============ aggregate_runs ============

def _arm_intervention(arm: ArmRow) -> str:
    v = arm.measurements['intervention_name']
    assert isinstance(v, str)
    return v


def _arm_env(arm: ArmRow) -> str:
    v = arm.measurements['env_name']
    assert isinstance(v, str)
    return v


def _arm_n(arm: ArmRow) -> int:
    v = arm.measurements['n']
    assert isinstance(v, int) and not isinstance(v, bool)
    return v


def test_aggregate_runs_groups_by_intervention_env_hp() -> None:
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
        (_arm_intervention(a), _arm_env(a)): a for a in arms
    }
    assert _arm_n(by_key[('ddqn', 'env_a')]) == 2
    assert _arm_n(by_key[('vanilla', 'env_a')]) == 2
    assert _arm_n(by_key[('ddqn', 'env_b')]) == 1


def test_aggregate_runs_distinct_hp_signatures_split_arms() -> None:
    """Runs with the SAME intervention_name + env but DIFFERENT
    HP signatures (e.g. different `gamma`) produce distinct
    arms — `hp_signature` is part of the grouping key."""
    runs = [
        _make_runrow(
            intervention_name='h', env_name='e', seed=0, primary=1.0,
            extra_hp={'gamma': 0.99},
        ),
        _make_runrow(
            intervention_name='h', env_name='e', seed=0, primary=2.0,
            extra_hp={'gamma': 0.95},
        ),
    ]
    arms = aggregate_runs(runs)
    assert len(arms) == 2
