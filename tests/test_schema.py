"""Tests for schema row dataclasses + as_dict/from_dict round-trip.

Verifies each row type round-trips losslessly through its dict
representation. Strict pyright passes throughout — type narrowing
in from_dict happens via TypeIs predicates and isinstance checks,
not cast or `# type: ignore`."""
from __future__ import annotations

from corroborate.hypothesis import MechanismKey
from corroborate.verdict import RefutationClass, Verdict
from corroborate.schema import (
    ArmRow,
    ComparisonRow,
    CorpusRow,
    FactRow,
    RunRow,
)


# ============ Fixtures ============

def _sample_mechanism_key() -> MechanismKey:
    return MechanismKey(
        intervention_signature=(
            ('greedification', 'callable:double_greedify'),
            ('lr', '0.001'),
        ),
        bridge_names=frozenset({'mechanism', 'outcome'}),
        direction='a_gt_b',
    )


def _sample_fact() -> FactRow:
    return FactRow(
        name='mechanism',
        kind='bridge',
        targets=('max_q_late',),
        
        verdict=Verdict.HELD,
        natural_strength=0.85,
        delta_i=0.42,
        evidentiary_level='causal_one_sided',
        stats={'rho': 0.85, 'n': 15, 'powered': True, 'estimand': 'E[Y|do(X)]'},
        intervention_signature=frozenset({'callable:double_greedify'}),
    )


# ============ FactRow round-trip ============

def test_fact_row_as_dict_keys() -> None:
    f = _sample_fact()
    d = f.as_dict()
    assert d['name'] == 'mechanism'
    assert d['kind'] == 'bridge'
    assert d['targets'] == ['max_q_late']
    assert d['verdict'] == 'held'
    assert d['natural_strength'] == 0.85


def test_fact_row_round_trip() -> None:
    f = _sample_fact()
    d = f.as_dict()
    f2 = FactRow.from_dict(d)
    assert f == f2


def test_fact_row_invariant_kind_round_trip() -> None:
    f = FactRow(
        name='q_bounded',
        kind='invariant',
        targets=('q_max',),
        
        verdict=Verdict.HELD,
        natural_strength=1.0,
        delta_i=1.0,
        evidentiary_level='causal_bridged',
        stats={'kind': 'tautological', 'of_claim': 'vanilla_greedify'},
    )
    f2 = FactRow.from_dict(f.as_dict())
    assert f == f2
    assert f2.kind == 'invariant'


def test_fact_row_from_dict_rejects_bad_kind() -> None:
    bad: dict[str, object] = {
        'name': 'x', 'kind': 'unknown_kind', 'targets': [],
        'reads': [], 'verdict': 'held',
        'natural_strength': 0.0, 'delta_i': 0.0,
        'evidentiary_level': 'correlational', 'stats': {},
        'intervention_signature': [],
    }
    try:
        FactRow.from_dict(bad)
        raise AssertionError('expected TypeError')
    except TypeError:
        pass


# ============ RunRow round-trip ============

def test_run_row_round_trip() -> None:
    run: RunRow = RunRow(
        id='run-1',
        parent_id=None,
        intervention_name='dqn_with_double_greedify',
        cycle_id='cycle-7',
        timestamp='2026-04-27T10:00:00Z',
        env_name='CartPole-v1',
        total_steps=30_000,
        seed=42,
        mechanism_key=_sample_mechanism_key(),
        primary_outcome_summary=120.5,
        record_keys=('q_max', 'epsilon', 'ep_return'),
        facts=(_sample_fact(),),
        reads_set=frozenset({'max_q_late', 'final_return'}),
        verdict=Verdict.HELD,
        meta={'agent': 'sweep', 'cycle': 7, 'wall_time_s': 120.5},
    )
    d = run.as_dict()
    run2: RunRow = RunRow.from_dict(d)
    assert run == run2


def test_run_row_default_meta_round_trip() -> None:
    run: RunRow = RunRow(
        id='run-2',
        parent_id=None,
        intervention_name='vanilla',
        cycle_id=None,
        timestamp='2026-04-27T10:00:00Z',
        env_name='CartPole-v1',
        total_steps=10_000,
        seed=0,
        mechanism_key=_sample_mechanism_key(),
        primary_outcome_summary=100.0,
        record_keys=(),
        facts=(),
        reads_set=frozenset(),
        verdict=Verdict.HELD,
    )
    d = run.as_dict()
    run2: RunRow = RunRow.from_dict(d)
    assert run == run2
    assert run2.meta == {}


# ============ ArmRow round-trip ============

def test_arm_row_round_trip() -> None:
    arm: ArmRow = ArmRow(
        id='arm-1',
        intervention_name='ddqn',
        env_name='Asterix-MinAtar',
        cycle_id='cycle-7',
        timestamp='2026-04-27T10:00:00Z',
        mechanism_key=_sample_mechanism_key(),
        run_ids=('run-1', 'run-2', 'run-3'),
        seeds=(0, 1, 2),
        n=3,
        arm_mean=42.5,
        arm_sd=3.1,
        facts=(_sample_fact(),),
        reads_set=frozenset({'max_q_late'}),
        meta={'cycle': 7},
    )
    d = arm.as_dict()
    arm2: ArmRow = ArmRow.from_dict(d)
    assert arm == arm2


# ============ ComparisonRow round-trip ============

def test_comparison_row_round_trip_full() -> None:
    cmp: ComparisonRow = ComparisonRow(
        id='cmp-1',
        parent_id=None,
        intervention_name='ddqn_vs_vanilla',
        env_name='Asterix-MinAtar',
        cycle_id='cycle-7',
        timestamp='2026-04-27T10:00:00Z',
        treatment_arm_id='arm-ddqn',
        baseline_arm_id='arm-vanilla',
        mechanism_key=_sample_mechanism_key(),
        predicted_direction='a_gt_b',
        n_treatment=15,
        n_baseline=15,
        arm_a_mean=42.5,
        arm_a_sd=3.1,
        arm_b_mean=39.0,
        arm_b_sd=4.2,
        effect_size_g=0.91,
        se=0.32,
        derived_q=0.93,
        delta_i_population=0.66,
        verdict=Verdict.HELD,
        refutation_class=None,
        adequately_powered=True,
        facts=(_sample_fact(),),
        reads_set=frozenset({'max_q_late', 'final_return'}),
        meta={'mde_d': 0.5},
    )
    d = cmp.as_dict()
    cmp2: ComparisonRow = ComparisonRow.from_dict(d)
    assert cmp == cmp2


def test_comparison_row_round_trip_with_optional_nones() -> None:
    """Stat fields can be None when underpowered or pre-statistics."""
    cmp: ComparisonRow = ComparisonRow(
        id='cmp-2',
        parent_id=None,
        intervention_name='underpowered',
        env_name='Acrobot-v1',
        cycle_id=None,
        timestamp='2026-04-27T10:00:00Z',
        treatment_arm_id='arm-t',
        baseline_arm_id='arm-b',
        mechanism_key=_sample_mechanism_key(),
        predicted_direction=None,
        n_treatment=5,
        n_baseline=5,
        arm_a_mean=-200.0,
        arm_a_sd=0.0,
        arm_b_mean=-180.0,
        arm_b_sd=10.0,
        effect_size_g=None,
        se=None,
        derived_q=None,
        delta_i_population=0.0,
        verdict=Verdict.POWER_INSUFFICIENT,
        refutation_class=RefutationClass.UNDERPOWERED,
        adequately_powered=False,
        facts=(),
        reads_set=frozenset(),
    )
    d = cmp.as_dict()
    cmp2: ComparisonRow = ComparisonRow.from_dict(d)
    assert cmp == cmp2
    assert cmp2.effect_size_g is None
    assert cmp2.refutation_class is RefutationClass.UNDERPOWERED


# ============ CorpusRow round-trip ============

def test_corpus_row_round_trip() -> None:
    corpus: CorpusRow = CorpusRow(
        id='corpus-1',
        name='ddqn_link_bridge',
        cycle_id='cycle-7',
        timestamp='2026-04-27T10:00:00Z',
        comparison_ids=('cmp-1', 'cmp-2', 'cmp-3'),
        n_comparisons=3,
        facts=(
            FactRow(
                name='hasselt_link',
                kind='bridge',
                targets=('mechanism', 'outcome'),
                
                verdict=Verdict.POWER_INSUFFICIENT,
                natural_strength=0.28,
                delta_i=0.06,
                evidentiary_level='correlational',
                stats={'pearson_r': 0.28, 'p': 0.28, 'n_envs': 17},
            ),
        ),
        reads_set=frozenset({'mechanism', 'outcome'}),
        meta={'cycle': 7},
    )
    d = corpus.as_dict()
    corpus2: CorpusRow = CorpusRow.from_dict(d)
    assert corpus == corpus2


# ============ Collection composition ============

def test_run_row_homogeneous_collection() -> None:
    """RunRows compose into a typed list — basic discipline that
    a `list[RunRow]` is well-typed and pyright tracks element
    types through the collection."""
    runs: list[RunRow] = []
    runs.append(RunRow(
        id='r1', parent_id=None, intervention_name='h',
        cycle_id=None, timestamp='2026-04-27T10:00:00Z',
        env_name='env', total_steps=10, seed=0,
        mechanism_key=_sample_mechanism_key(),
        primary_outcome_summary=0.0, record_keys=(),
        facts=(), reads_set=frozenset(), verdict=Verdict.HELD,
    ))
    assert len(runs) == 1


# ============ MechanismKey serialization ============

def test_mechanism_key_round_trip_preserves_direction() -> None:
    """MechanismKey serializes/deserializes through dict form
    losslessly; direction is preserved."""
    mk = _sample_mechanism_key()
    f = FactRow(
        name='fact', kind='bridge',
        targets=('x',),
        verdict=Verdict.HELD, natural_strength=1.0, delta_i=1.0,
        evidentiary_level='correlational', stats={},
    )
    run: RunRow = RunRow(
        id='r', parent_id=None, intervention_name='h',
        cycle_id=None, timestamp='t',
        env_name='e', total_steps=10, seed=0,
        mechanism_key=mk,
        primary_outcome_summary=0.0, record_keys=(),
        facts=(f,), reads_set=frozenset(), verdict=Verdict.HELD,
    )
    run2: RunRow = RunRow.from_dict(run.as_dict())
    assert run2.mechanism_key == mk
    assert run2.mechanism_key.direction == 'a_gt_b'


def test_mechanism_key_round_trip_no_direction() -> None:
    mk = MechanismKey(
        intervention_signature=(),
        bridge_names=frozenset(),
        direction=None,
    )
    f = FactRow(
        name='fact', kind='bridge',
        targets=(),
        verdict=Verdict.HELD, natural_strength=0.0, delta_i=0.0,
        evidentiary_level='', stats={},
    )
    run: RunRow = RunRow(
        id='r', parent_id=None, intervention_name='h',
        cycle_id=None, timestamp='t',
        env_name='e', total_steps=10, seed=0,
        mechanism_key=mk,
        primary_outcome_summary=0.0, record_keys=(),
        facts=(f,), reads_set=frozenset(), verdict=Verdict.HELD,
    )
    run2: RunRow = RunRow.from_dict(run.as_dict())
    assert run2.mechanism_key.direction is None
