"""`load_runs` — producer files in, closed-form columns out.

Fixture run directories are constructed from closed forms: every
evaluation return is `base(condition) + checkpoint/10 +
(eval_seed - 101) + (pair_key - 7)`, so the loader's derived
per-run aggregates — and the claim bridge evaluated against the
resulting DataFrame — have exact expected values computed from
the same construction parameters (all increments are binary-exact
fractions — equality assertions are exact, not tolerance-based).

The loader is a reader, not a gatekeeper: structural malformation
(duplicate ids, conflicting column values, path escapes) raises a
plain ValueError; study-design questions (isolation, pairing) are
the admission gates' business, exercised here only through the
end-to-end pooling test and in depth in `test_claim_bridge.py`.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import polars as pl
import pytest

from corroborate.analyses.paired.arm_mean_diff import (
    ArmMeanDiffResult,
    arm_mean_diff,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge, evaluate
from corroborate.bridge.verdict import Verdict
from corroborate.core.intervention import DoEffect
from corroborate.data import config_columns, load_runs

_BASELINE_ENT = 0.0
_TREATMENT_ENT = 0.01
_BASELINE_BASE = -100.0
_TREATMENT_BASE = -90.0
_CHECKPOINTS = (10, 20)
_EVAL_SEEDS = (101, 102)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        '\n'.join(json.dumps(row, sort_keys=True) for row in rows) + '\n',
        encoding='utf-8',
    )


def _return_value(
    *, base: float, checkpoint: int, eval_seed: int, pair_key: int,
) -> float:
    """The fixture's closed form — the single source both the
    file construction and the test expectations derive from."""
    return base + checkpoint / 10.0 + (eval_seed - 101) + (pair_key - 7)


def _expected_return_mean_at(
    *, base: float, checkpoint: int, pair_key: int,
) -> float:
    """Per-checkpoint mean over the two evaluation seeds:
    base + checkpoint/10 + 0.5 + (pair_key - 7)."""
    return base + checkpoint / 10.0 + 0.5 + (pair_key - 7)


def _expected_return_mean(*, base: float, pair_key: int) -> float:
    """Final-checkpoint mean over the two evaluation seeds:
    base + 2.0 + 0.5 + (pair_key - 7)."""
    return _expected_return_mean_at(
        base=base, checkpoint=_CHECKPOINTS[-1], pair_key=pair_key,
    )


def _expected_return_auc(*, base: float, pair_key: int) -> float:
    """Trapezoid over checkpoint means (base + c/10 + 0.5 + Δ) at
    c ∈ {10, 20}, normalised by the span: the midpoint at c=15."""
    return base + 1.5 + 0.5 + (pair_key - 7)


def _make_runs(
    root: Path,
    *,
    training_seeds: tuple[int, ...] = (7, 9),
    checkpoints_by_arm: dict[float, tuple[int, ...]] | None = None,
    write_provenance: bool = True,
    write_evaluations: bool = True,
    extra_run_fields: dict[str, object] | None = None,
    extra_config_fields: dict[str, object] | None = None,
) -> None:
    runs: list[dict[str, object]] = []
    evaluations: list[dict[str, object]] = []
    for training_seed in training_seeds:
        for ent_coef, base in (
            (_BASELINE_ENT, _BASELINE_BASE),
            (_TREATMENT_ENT, _TREATMENT_BASE),
        ):
            run_id = f'seed-{training_seed:04d}__ent-{ent_coef:g}'
            config_relative = f'configs/{run_id}.json'
            _write_json(
                root / config_relative,
                {
                    'algorithm': {
                        'name': 'PPO',
                        'ent_coef': ent_coef,
                        'gamma': 0.99,
                    },
                    'environment': {'id': 'MountainCar-v0'},
                    'training': {'seed': training_seed, 'timesteps': 20},
                    **(extra_config_fields or {}),
                },
            )
            runs.append({
                'run_id': run_id,
                'config_path': config_relative,
                **(extra_run_fields or {}),
            })
            grid = (
                checkpoints_by_arm.get(ent_coef, _CHECKPOINTS)
                if checkpoints_by_arm is not None
                else _CHECKPOINTS
            )
            for checkpoint in grid:
                for eval_seed in _EVAL_SEEDS:
                    evaluations.append({
                        'run_id': run_id,
                        'checkpoint': checkpoint,
                        'eval_seed': eval_seed,
                        'return': _return_value(
                            base=base,
                            checkpoint=checkpoint,
                            eval_seed=eval_seed,
                            pair_key=training_seed,
                        ),
                    })
    _write_jsonl(root / 'runs.jsonl', runs)
    if write_evaluations:
        _write_jsonl(root / 'evaluations.jsonl', evaluations)
    if write_provenance:
        _write_json(
            root / 'provenance.json',
            {
                'producer': 'sbx-ppo',
                'command': 'python train.py --study fixture',
            },
        )


# ============ closed-form derivation ============


def test_load_runs_derives_closed_form_columns(tmp_path: Path) -> None:
    _make_runs(tmp_path)
    df = load_runs(tmp_path)

    assert df.height == 4
    for row in df.to_dicts():
        pair_key = row['training.seed']
        assert isinstance(pair_key, int)
        ent = row['algorithm.ent_coef']
        base = _BASELINE_BASE if ent == _BASELINE_ENT else _TREATMENT_BASE
        # Closed form from the fixture's construction parameters.
        assert row['return_mean'] == _expected_return_mean(
            base=base, pair_key=pair_key,
        )
        assert row['return_auc'] == _expected_return_auc(
            base=base, pair_key=pair_key,
        )
        # The evaluation trajectory lands as one scalar column per
        # checkpoint, alongside the final-mean/AUC projections.
        for checkpoint in _CHECKPOINTS:
            assert row[
                f'return_mean_at_{checkpoint}'
            ] == _expected_return_mean_at(
                base=base, checkpoint=checkpoint, pair_key=pair_key,
            )
        # Configuration lands at its dotted leaf paths.
        assert row['algorithm.name'] == 'PPO'
        assert row['environment.id'] == 'MountainCar-v0'
        assert row['id'] == (
            f"seed-{pair_key:04d}__ent-{ent:g}"
        )
        assert row['program'] == 'sbx-ppo'
        assert row['corpus'] == tmp_path.name


def test_corpus_name_defaults_to_directory_and_is_overridable(
    tmp_path: Path,
) -> None:
    _make_runs(tmp_path / 'batch_a')
    assert (
        load_runs(tmp_path / 'batch_a')['corpus'].unique().to_list()
        == ['batch_a']
    )
    named = load_runs(tmp_path / 'batch_a', corpus='pilot-1')
    assert named['corpus'].unique().to_list() == ['pilot-1']


def test_single_checkpoint_auc_reduces_to_mean(tmp_path: Path) -> None:
    _make_runs(
        tmp_path,
        checkpoints_by_arm={
            _BASELINE_ENT: (20,), _TREATMENT_ENT: (20,),
        },
    )
    df = load_runs(tmp_path)
    for row in df.to_dicts():
        assert row['return_auc'] == row['return_mean']
        assert row['return_mean_at_20'] == row['return_mean']


def test_ragged_checkpoint_grids_null_pad(tmp_path: Path) -> None:
    """Runs evaluated on different checkpoint grids load without a
    declared uniform extent: each run's aggregates are computed
    over its own grid, and absent trajectory cells are null — the
    loader holds no authority over what the extent should be."""
    _make_runs(
        tmp_path,
        checkpoints_by_arm={
            _BASELINE_ENT: _CHECKPOINTS, _TREATMENT_ENT: (10,),
        },
    )
    df = load_runs(tmp_path)
    for row in df.to_dicts():
        pair_key = row['training.seed']
        assert isinstance(pair_key, int)
        if row['algorithm.ent_coef'] == _BASELINE_ENT:
            assert row['return_mean'] == _expected_return_mean(
                base=_BASELINE_BASE, pair_key=pair_key,
            )
            assert row['return_mean_at_20'] is not None
        else:
            # Final checkpoint of the shorter grid IS checkpoint 10.
            assert row['return_mean'] == _expected_return_mean_at(
                base=_TREATMENT_BASE, checkpoint=10, pair_key=pair_key,
            )
            assert row['return_auc'] == row['return_mean']
            assert row['return_mean_at_20'] is None


def test_non_numeric_evaluation_fields_are_not_outcomes(
    tmp_path: Path,
) -> None:
    _make_runs(tmp_path)
    evaluations_path = tmp_path / 'evaluations.jsonl'
    rows = [
        json.loads(line)
        for line in evaluations_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    for row in rows:
        row['note'] = 'greedy rollout'
    _write_jsonl(evaluations_path, rows)

    df = load_runs(tmp_path)
    assert 'note_mean' not in df.columns
    assert 'note' not in df.columns


def test_missing_optional_files_load_without_their_columns(
    tmp_path: Path,
) -> None:
    """Only runs.jsonl is required: without evaluations there are
    no derived outcome columns, without provenance no `program` —
    absence is not an error, it is just fewer columns."""
    _make_runs(
        tmp_path, write_evaluations=False, write_provenance=False,
    )
    df = load_runs(tmp_path)
    assert df.height == 4
    assert 'return_mean' not in df.columns
    assert 'program' not in df.columns
    assert 'algorithm.ent_coef' in df.columns


# ============ structural malformation raises ============


def test_duplicate_run_id_raises(tmp_path: Path) -> None:
    _make_runs(tmp_path)
    runs_path = tmp_path / 'runs.jsonl'
    rows = [
        json.loads(line)
        for line in runs_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    rows.append(dict(rows[0]))
    _write_jsonl(runs_path, rows)

    with pytest.raises(ValueError, match='duplicate run_id'):
        load_runs(tmp_path)


def test_duplicate_evaluation_record_raises(tmp_path: Path) -> None:
    _make_runs(tmp_path)
    evaluations_path = tmp_path / 'evaluations.jsonl'
    rows = [
        json.loads(line)
        for line in evaluations_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    rows.append(dict(rows[0]))
    _write_jsonl(evaluations_path, rows)

    with pytest.raises(ValueError, match='duplicate evaluation record'):
        load_runs(tmp_path)


def test_evaluation_for_unknown_run_raises(tmp_path: Path) -> None:
    _make_runs(tmp_path)
    evaluations_path = tmp_path / 'evaluations.jsonl'
    rows = [
        json.loads(line)
        for line in evaluations_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    stray = dict(rows[0])
    stray['run_id'] = 'ghost-run'
    rows.append(stray)
    _write_jsonl(evaluations_path, rows)

    with pytest.raises(ValueError, match='ghost-run'):
        load_runs(tmp_path)


def test_config_path_traversal_rejected(tmp_path: Path) -> None:
    """A hostile run record must not read outside its directory."""
    _make_runs(tmp_path)
    runs_path = tmp_path / 'runs.jsonl'
    rows = [
        json.loads(line)
        for line in runs_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
    ]
    rows[0]['config_path'] = '../escape'
    _write_jsonl(runs_path, rows)

    with pytest.raises(ValueError, match='unsafe run-relative path'):
        load_runs(tmp_path)


def test_conflicting_column_values_raise_and_equal_values_pass(
    tmp_path: Path,
) -> None:
    """Producers often stamp the same fact on both the run record
    and the configuration; a repeated column is tolerated only
    when the values agree — the one case where accepting it loses
    nothing."""
    _make_runs(tmp_path, extra_run_fields={'environment.id': 'Pendulum-v1'})
    with pytest.raises(ValueError, match="conflicting values for column"):
        load_runs(tmp_path)

    agreeing = tmp_path / 'agreeing'
    _make_runs(agreeing, extra_run_fields={'environment.id': 'MountainCar-v0'})
    df = load_runs(agreeing)
    assert df['environment.id'].unique().to_list() == ['MountainCar-v0']


def test_producer_column_shadowing_derived_column_raises(
    tmp_path: Path,
) -> None:
    """A configuration leaf named like a derived aggregate would
    silently shadow or be shadowed; conflicting values raise
    instead."""
    _make_runs(tmp_path, extra_config_fields={'return_mean': 42.0})
    with pytest.raises(ValueError, match="conflicting values for column"):
        load_runs(tmp_path)


# ============ the live-evidence loop, end to end ============


_ENTROPY_COEFFICIENT_EFFECT = DoEffect.from_values(
    source='algorithm.ent_coef',
    reference=_BASELINE_ENT,
    treatment=_TREATMENT_ENT,
)


@claim_bridge(
    source=_ENTROPY_COEFFICIENT_EFFECT,
    target='return_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    pair_by=('training.seed',),
    predicted_direction='a_lt_b',
)
def _entropy_bonus_improves_return(
    arm_mean_diff: ArmMeanDiffResult,
) -> Verdict:
    if arm_mean_diff.mean_diff > 0:
        return Verdict.HELD
    return Verdict.NO_EFFECT


def test_config_columns_derives_the_leaf_registry(tmp_path: Path) -> None:
    """The leaf registry is read off the artifact that already
    exists — the resolved-config files — never authored: every
    dotted path the configs flatten to, and nothing else."""
    _make_runs(tmp_path)
    assert config_columns(tmp_path) == frozenset({
        'algorithm.name', 'algorithm.ent_coef', 'algorithm.gamma',
        'environment.id', 'training.seed', 'training.timesteps',
    })


def test_batches_of_the_same_study_pool(tmp_path: Path) -> None:
    """Evidence is a live record: two batches of seeds loaded
    separately concatenate into one run set, and the same claim
    evaluates against the pool — the run-more-seeds workflow the
    hypothesis layer exists for. Closed form: the pair diff
    cancels every term except the condition bases, so mean_diff =
    -90 - (-100) = 10 exactly at n = 4 per condition."""
    _make_runs(tmp_path / 'batch_a', training_seeds=(7, 9))
    _make_runs(tmp_path / 'batch_b', training_seeds=(11, 13))
    pooled = pl.concat(
        [
            load_runs(tmp_path / 'batch_a'),
            load_runs(tmp_path / 'batch_b'),
        ],
        how='diagonal',
    )

    evaluation = evaluate(
        _entropy_bonus_improves_return,
        pooled,
        leaves=config_columns(tmp_path / 'batch_a'),
    )
    assert evaluation.verdict is Verdict.HELD
    assert evaluation.blocked_by is None
    result = evaluation.analysis_results['arm_mean_diff']
    assert isinstance(result, ArmMeanDiffResult)
    assert result.n_treatment == 4
    assert result.n_baseline == 4
    assert result.mean_diff == _TREATMENT_BASE - _BASELINE_BASE


def test_loaded_frame_agrees_with_dict_rows_through_an_analysis(
    tmp_path: Path,
) -> None:
    """`Analysis.__call__` materialises either cells shape, so the
    loaded DataFrame and its dict rows agree exactly through a
    registered analysis. Welch closed form on the (7, 9) fixture:
    per-condition final means `base + 2.5 + (pair_key - 7)` →
    variance 2 at n = 2 per condition → SE = sqrt(2), df = 2."""
    _make_runs(tmp_path)
    df = load_runs(tmp_path)
    labelled = df.with_columns(
        pl.when(pl.col('algorithm.ent_coef') == _TREATMENT_ENT)
        .then(pl.lit('treatment'))
        .otherwise(pl.lit('baseline'))
        .alias('arm_key'),
    )
    from_frame = arm_mean_diff(
        labelled,
        source='return_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('training.seed',),
    )
    from_rows = arm_mean_diff(
        labelled.to_dicts(),
        source='return_mean',
        treatment_arm='treatment',
        baseline_arm='baseline',
        pair_by=('training.seed',),
    )
    # repr equality: total over every dataclass field (none are
    # repr=False) and NaN-tolerant, where `==` would fail on the
    # legitimately-NaN pairing diagnostic (n_paired < 5 here).
    assert repr(from_frame) == repr(from_rows)
    assert from_frame.mean_diff == _TREATMENT_BASE - _BASELINE_BASE
    assert from_frame.mean_diff_se == pytest.approx(
        math.sqrt(2.0), rel=1e-12,
    )
    assert from_frame.welch_df == pytest.approx(2.0, rel=1e-12)
