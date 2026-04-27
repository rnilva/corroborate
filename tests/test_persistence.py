"""Tests for parquet round-trip across all four row types.

Each row type has paired write/read functions; the test pattern
is: construct row → write to tmp parquet → read back → assert
equality. Verifies the framework's persistence layer is lossless
through real parquet I/O (not just dict round-trip)."""
from __future__ import annotations

from pathlib import Path

from corroborate.hypothesis import MechanismKey
from corroborate.verdict import RefutationClass, Verdict
from corroborate.persistence import (
    read_armrows,
    read_comparisonrows,
    read_corpusrows,
    read_runrows,
    write_armrows,
    write_comparisonrows,
    write_corpusrows,
    write_runrows,
)
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


def _sample_runrow() -> RunRow:
    return RunRow(
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


# ============ RunRow round-trip ============

def test_runrow_parquet_round_trip_single(tmp_path: Path) -> None:
    path = tmp_path / 'runs.parquet'
    rows = [_sample_runrow()]
    write_runrows(rows, path)
    assert path.exists()

    loaded = read_runrows(path)
    assert len(loaded) == 1
    assert loaded[0] == rows[0]


def test_runrow_parquet_round_trip_multiple(tmp_path: Path) -> None:
    rows = [
        _sample_runrow(),
        RunRow(
            id='run-2',
            parent_id='run-1',
            intervention_name='vanilla',
            cycle_id=None,
            timestamp='2026-04-27T11:00:00Z',
            env_name='Acrobot-v1',
            total_steps=10_000,
            seed=0,
            mechanism_key=_sample_mechanism_key(),
            primary_outcome_summary=-200.0,
            record_keys=(),
            facts=(),
            reads_set=frozenset(),
            verdict=Verdict.POWER_INSUFFICIENT,
        ),
    ]
    path = tmp_path / 'runs.parquet'
    write_runrows(rows, path)
    loaded = read_runrows(path)
    assert loaded == rows


def test_runrow_parquet_with_no_facts(tmp_path: Path) -> None:
    """Empty facts tuple round-trips losslessly."""
    row = RunRow(
        id='no-facts',
        parent_id=None,
        intervention_name='vanilla',
        cycle_id=None,
        timestamp='t',
        env_name='env',
        total_steps=10,
        seed=0,
        mechanism_key=_sample_mechanism_key(),
        primary_outcome_summary=0.0,
        record_keys=(),
        facts=(),
        reads_set=frozenset(),
        verdict=Verdict.HELD,
    )
    path = tmp_path / 'runs.parquet'
    write_runrows([row], path)
    assert read_runrows(path) == [row]


# ============ ArmRow round-trip ============

def test_armrow_parquet_round_trip(tmp_path: Path) -> None:
    arm = ArmRow(
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
    path = tmp_path / 'arms.parquet'
    write_armrows([arm], path)
    assert read_armrows(path) == [arm]


# ============ ComparisonRow round-trip ============

def test_comparisonrow_parquet_round_trip_full(tmp_path: Path) -> None:
    cmp = ComparisonRow(
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
    path = tmp_path / 'comparisons.parquet'
    write_comparisonrows([cmp], path)
    assert read_comparisonrows(path) == [cmp]


def test_comparisonrow_parquet_with_optional_nones(tmp_path: Path) -> None:
    """Underpowered comparison: stat fields are None. Round-trip
    preserves None vs explicit values."""
    cmp = ComparisonRow(
        id='cmp-underpowered',
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
    path = tmp_path / 'comparisons.parquet'
    write_comparisonrows([cmp], path)
    loaded = read_comparisonrows(path)
    assert loaded == [cmp]
    assert loaded[0].effect_size_g is None
    assert loaded[0].refutation_class is RefutationClass.UNDERPOWERED


# ============ CorpusRow round-trip ============

def test_corpusrow_parquet_round_trip(tmp_path: Path) -> None:
    corpus = CorpusRow(
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
    path = tmp_path / 'corpus.parquet'
    write_corpusrows([corpus], path)
    assert read_corpusrows(path) == [corpus]


# ============ Empty collections ============

def test_empty_facts_via_parquet(tmp_path: Path) -> None:
    """A row with empty facts/reads_set/intervention_signature
    must round-trip without losing the empty vs None distinction."""
    row = _sample_runrow()
    row_no_facts = RunRow(
        id=row.id, parent_id=row.parent_id,
        intervention_name=row.intervention_name,
        cycle_id=row.cycle_id, timestamp=row.timestamp,
        env_name=row.env_name, total_steps=row.total_steps,
        seed=row.seed,
        mechanism_key=row.mechanism_key,
        primary_outcome_summary=row.primary_outcome_summary,
        record_keys=row.record_keys,
        facts=(),  # empty
        reads_set=frozenset(),
        verdict=row.verdict,
        meta={},  # empty meta
    )
    path = tmp_path / 'runs.parquet'
    write_runrows([row_no_facts], path)
    loaded = read_runrows(path)
    assert loaded == [row_no_facts]
    assert loaded[0].facts == ()
    assert loaded[0].reads_set == frozenset()
    assert dict(loaded[0].meta) == {}
