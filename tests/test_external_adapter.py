"""External-study adapter — fail-closed verification + Panel hand-off.

Fixture bundles are constructed from closed forms: every
evaluation return is `base(condition) + checkpoint/10 +
(eval_seed - 101) + (pair_key - 7)`, so the adapter's derived
per-seeded-run aggregates, and the analysis run on the resulting
`Panel`, have exact expected values computed from the same
construction parameters (all increments are binary-exact
fractions — equality assertions are exact, not tolerance-based).
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

import pytest

from corroborate.analyses.paired.arm_mean_diff import arm_mean_diff
from corroborate.data import (
    AdapterReceipt,
    BundleValidationError,
    CheckStatus,
    adapt_study,
    seal_bundle,
)
from corroborate.data._bundle_io import sha256_file

_BASELINE_ARM = 'entropy_0'
_TREATMENT_ARM = 'entropy_positive'
_BASELINE_BASE = -100.0
_TREATMENT_BASE = -90.0
_CHECKPOINTS = (10, 20)
_EVAL_SEEDS = (101, 102)
_STUDY_ID = 'fixture-study'


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
    bundle construction and the test expectations derive from."""
    return base + checkpoint / 10.0 + (eval_seed - 101) + (pair_key - 7)


def _expected_return_mean(*, base: float, pair_key: int) -> float:
    """Final-checkpoint mean over the two evaluation seeds:
    base + 2.0 + 0.5 + (pair_key - 7)."""
    return base + _CHECKPOINTS[-1] / 10.0 + 0.5 + (pair_key - 7)


def _expected_return_auc(*, base: float, pair_key: int) -> float:
    """Trapezoid over checkpoint means (base + c/10 + 0.5 + Δ) at
    c ∈ {10, 20}, normalised by the span: the midpoint at c=15."""
    return base + 1.5 + 0.5 + (pair_key - 7)


def _make_bundle(
    root: Path,
    *,
    training_seeds: tuple[int, ...] = (7, 9),
    treatment_gamma: float = 0.99,
    gamma_by_seed: Mapping[int, float] | None = None,
    omit_treatment: bool = False,
    duplicate_evaluation: bool = False,
    unexpected_evaluation: bool = False,
    nan_return: bool = False,
    include_protocol: bool = True,
    include_assignment: bool = True,
    omit_logical_arm_keys: bool = False,
    logical_arm_keys: Mapping[str, str] | None = None,
    protocol_pair_keys: tuple[int, ...] | None = None,
    scope: Mapping[str, object] | None = None,
) -> None:
    resolved_scope: dict[str, object] = (
        dict(scope)
        if scope is not None
        else {'env_name': 'MountainCar-v0', 'backend': 'sbx',
              'total_steps': 20}
    )
    resolved_arm_keys: dict[str, object] = (
        dict(logical_arm_keys)
        if logical_arm_keys is not None
        else {_BASELINE_ARM: 'baseline', _TREATMENT_ARM: 'entropy_bonus'}
    )
    contrast: dict[str, object] = {
        'parameter_path': 'algorithm.ent_coef',
        'baseline_arm': _BASELINE_ARM,
        'treatment_arm': _TREATMENT_ARM,
        'baseline_value': 0.0,
        'treatment_value': 0.01,
    }
    if not omit_logical_arm_keys:
        contrast['logical_arm_keys'] = resolved_arm_keys
    contract: dict[str, object] = {
        'contract_version': 1,
        'study_id': _STUDY_ID,
        'pair_by': 'training_seed',
        'pair_by_config_path': 'training.seed',
        'contrast': contrast,
        'scope': resolved_scope,
        'evaluation': {
            'checkpoints': list(_CHECKPOINTS),
            'seeds': list(_EVAL_SEEDS),
            'outcomes': ['return'],
        },
        'run_measurements': ['exploration_breadth'],
    }
    if include_assignment:
        contract['assignment'] = {
            'assurance': 'attested',
            'statement': 'condition order was shuffled before execution',
        }
    if include_protocol:
        protocol: dict[str, object] = {
            'confirmatory_pair_keys': sorted(
                protocol_pair_keys
                if protocol_pair_keys is not None
                else training_seeds,
            ),
            'paired_arms': {
                'baseline'
                if not omit_logical_arm_keys
                else _BASELINE_ARM: {'algorithm': {'ent_coef': 0.0}},
                'entropy_bonus'
                if not omit_logical_arm_keys
                else _TREATMENT_ARM: {'algorithm': {'ent_coef': 0.01}},
            },
            'evaluation': {
                'checkpoints': list(_CHECKPOINTS),
                'seeds': list(_EVAL_SEEDS),
            },
        }
        _write_json(root / 'prospective_protocol.json', protocol)
        contract['prospective_protocol'] = {
            'path': 'prospective_protocol.json',
            'sha256': sha256_file(root / 'prospective_protocol.json'),
        }
    _write_json(root / 'contract.json', contract)

    runs: list[dict[str, object]] = []
    evaluations: list[dict[str, object]] = []
    for training_seed in training_seeds:
        arms: list[tuple[str, float, float, float]] = [
            (_BASELINE_ARM, 0.0, 0.99, _BASELINE_BASE),
        ]
        if not omit_treatment:
            arms.append(
                (_TREATMENT_ARM, 0.01, treatment_gamma, _TREATMENT_BASE),
            )
        for arm, ent_coef, default_gamma, base in arms:
            gamma = (
                gamma_by_seed.get(training_seed, default_gamma)
                if gamma_by_seed is not None
                else default_gamma
            )
            run_id = f'seed-{training_seed:04d}__{arm}'
            config_relative = f'runs/{run_id}/resolved_config.json'
            _write_json(
                root / config_relative,
                {
                    'algorithm': {
                        'name': 'PPO',
                        'ent_coef': ent_coef,
                        'gamma': gamma,
                    },
                    'environment': {'id': 'MountainCar-v0'},
                    'training': {'seed': training_seed, 'timesteps': 20},
                },
            )
            runs.append({
                'run_id': run_id,
                'training_seed': training_seed,
                'physical_arm': arm,
                'config_path': config_relative,
                'complete': True,
                'exploration_breadth': 0.25 if arm == _BASELINE_ARM else 0.5,
                **resolved_scope,
            })
            for checkpoint in _CHECKPOINTS:
                for eval_seed in _EVAL_SEEDS:
                    evaluations.append({
                        'run_id': run_id,
                        'checkpoint': checkpoint,
                        'eval_seed': eval_seed,
                        'return': (
                            float('nan')
                            if nan_return
                            else _return_value(
                                base=base,
                                checkpoint=checkpoint,
                                eval_seed=eval_seed,
                                pair_key=training_seed,
                            )
                        ),
                    })
    if duplicate_evaluation:
        evaluations.append(dict(evaluations[0]))
    if unexpected_evaluation:
        unexpected = dict(evaluations[0])
        unexpected['checkpoint'] = 30
        evaluations.append(unexpected)
    _write_jsonl(root / 'runs.jsonl', runs)
    _write_jsonl(root / 'evaluations.jsonl', evaluations)
    _write_json(
        root / 'provenance.json',
        {
            'producer': 'sbx-ppo',
            'command': 'python train.py --study fixture',
        },
    )
    seal_bundle(root)


def _check_status(receipt: AdapterReceipt, code: str) -> CheckStatus:
    matches = [c for c in receipt.checks if c.code == code]
    assert matches, f'no check with code {code!r}'
    return matches[-1].status


# ============ admission + closed-form normalisation ============


def test_valid_bundle_derives_rows_and_receipt(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    study = adapt_study(tmp_path)

    assert len(study.rows) == 4
    assert study.receipt.admissible
    assert study.receipt.n_pairs == 2
    assert study.receipt.study_id == _STUDY_ID
    assert {row['arm_key'] for row in study.rows} == {
        'baseline', 'entropy_bonus',
    }
    for row in study.rows:
        pair_key = row['training_seed']
        assert isinstance(pair_key, int)
        base = (
            _BASELINE_BASE
            if row['arm_key'] == 'baseline'
            else _TREATMENT_BASE
        )
        # Closed form from the fixture's construction parameters.
        assert row['return_mean'] == _expected_return_mean(
            base=base, pair_key=pair_key,
        )
        assert row['return_auc'] == _expected_return_auc(
            base=base, pair_key=pair_key,
        )
        # Intervention value lands at its dotted leaf path.
        assert row['algorithm.ent_coef'] == (
            0.0 if row['arm_is_baseline'] is True else 0.01
        )
        assert row['env_name'] == 'MountainCar-v0'
        assert row['program'] == 'external:sbx-ppo'
        assert row['corpus'] == _STUDY_ID
    assert _check_status(study.receipt, 'assignment') is CheckStatus.ATTESTED
    assert (
        _check_status(study.receipt, 'protocol_design_match')
        is CheckStatus.VERIFIED
    )
    assert (
        _check_status(study.receipt, 'run_measurements')
        is CheckStatus.ATTESTED
    )


def test_recorded_contrast_is_receipt_bound(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    study = adapt_study(tmp_path)

    assert study.contrast.parameter_path == 'algorithm.ent_coef'
    assert study.contrast.arm_keys == ('baseline', 'entropy_bonus')
    assert study.contrast.baseline_value == 0.0
    assert study.contrast.treatment_value == 0.01
    assert study.contrast.bundle_digest == study.receipt.bundle_digest
    assert len(study.contrast.bundle_digest) == 64
    assert study.contrast.assurance is CheckStatus.ATTESTED


def test_round_trip_panel_analysis_recovers_contrast(
    tmp_path: Path,
) -> None:
    """External bundle → adapter → Panel → registered analysis
    recovers the constructed effect exactly.

    Closed form: per-condition final-checkpoint means are
    `base + 2.5 + (pair_key - 7)` for pair keys (7, 9) → baseline
    (-97.5, -95.5), treatment (-87.5, -85.5). Welch statistics:
    mean_diff = base_t - base_b = 10 exactly; per-condition sample
    variance = 2 at n = 2 per condition → SE = sqrt(2/2 + 2/2) =
    sqrt(2); Welch df = 2 (equal variances and sizes)."""
    _make_bundle(tmp_path)
    study = adapt_study(tmp_path)
    panel = study.to_panel()

    assert panel.cells.height == 4
    assert panel.sources[0].corpus == _STUDY_ID
    assert panel.sources[0].data_root == study.receipt.bundle_root

    result = arm_mean_diff(
        panel.cells.to_dicts(),
        source='return_mean',
        treatment_arm=study.contrast.treatment_key,
        baseline_arm=study.contrast.baseline_key,
        pair_by=('training_seed',),
    )
    assert result.mean_diff == _TREATMENT_BASE - _BASELINE_BASE
    assert result.n_treatment == 2
    assert result.n_baseline == 2
    assert result.mean_diff_se == pytest.approx(
        math.sqrt(2.0), rel=1e-12,
    )
    assert result.welch_df == pytest.approx(2.0, rel=1e-12)


def test_single_checkpoint_auc_reduces_to_mean(tmp_path: Path) -> None:
    _make_bundle(tmp_path, include_protocol=False)
    # Rewrite the contract + evaluations to a single checkpoint.
    contract_path = tmp_path / 'contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    contract['evaluation']['checkpoints'] = [20]
    _write_json(contract_path, contract)
    evaluations_path = tmp_path / 'evaluations.jsonl'
    kept = [
        row
        for line in evaluations_path.read_text(encoding='utf-8').splitlines()
        if line.strip()
        for row in [json.loads(line)]
        if row['checkpoint'] == 20
    ]
    _write_jsonl(evaluations_path, kept)
    seal_bundle(tmp_path)

    study = adapt_study(tmp_path)
    for row in study.rows:
        assert row['return_auc'] == row['return_mean']


# ============ fail-closed obligations ============


def test_tampered_file_fails_before_ingestion(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    victim = next(tmp_path.glob('runs/*/resolved_config.json'))
    with victim.open('ab') as stream:
        stream.write(b'tamper')

    with pytest.raises(BundleValidationError) as caught:
        adapt_study(tmp_path)
    assert 'mismatch' in str(caught.value)
    assert caught.value.checks[-1].status is CheckStatus.FAILED
    assert caught.value.checks[-1].code in {'manifest_sha256', 'manifest_size'}


def test_missing_manifest_fails_closed_with_typed_receipt(
    tmp_path: Path,
) -> None:
    _make_bundle(tmp_path)
    (tmp_path / 'manifest.json').unlink()

    with pytest.raises(BundleValidationError) as caught:
        adapt_study(tmp_path)
    assert caught.value.checks[-1].code == 'manifest_readable'
    assert caught.value.checks[-1].status is CheckStatus.FAILED


def test_malformed_contract_json_fails_closed(tmp_path: Path) -> None:
    """A broken bundle produces a typed receipt, never a raw
    JSONDecodeError from a permissive parse."""
    _make_bundle(tmp_path)
    (tmp_path / 'contract.json').write_text('{not json', encoding='utf-8')
    seal_bundle(tmp_path)

    with pytest.raises(BundleValidationError) as caught:
        adapt_study(tmp_path)
    assert caught.value.checks[-1].code == 'contract_readable'
    assert caught.value.checks[-1].status is CheckStatus.FAILED


def test_undeclared_configuration_difference_fails(tmp_path: Path) -> None:
    _make_bundle(tmp_path, treatment_gamma=0.98)

    with pytest.raises(BundleValidationError) as caught:
        adapt_study(tmp_path)
    assert 'outside algorithm.ent_coef and training.seed' in str(caught.value)


def test_common_drift_in_both_conditions_of_one_pair_fails(
    tmp_path: Path,
) -> None:
    _make_bundle(tmp_path, gamma_by_seed={9: 0.98})

    with pytest.raises(BundleValidationError) as caught:
        adapt_study(tmp_path)
    assert 'outside algorithm.ent_coef and training.seed' in str(caught.value)


def test_duplicate_logical_arm_keys_fail(tmp_path: Path) -> None:
    _make_bundle(
        tmp_path,
        logical_arm_keys={_BASELINE_ARM: 'same', _TREATMENT_ARM: 'same'},
        include_protocol=False,
    )

    with pytest.raises(BundleValidationError) as caught:
        adapt_study(tmp_path)
    assert 'logical arm keys must be distinct' in str(caught.value)


def test_incomplete_pair_fails(tmp_path: Path) -> None:
    _make_bundle(tmp_path, omit_treatment=True)

    with pytest.raises(BundleValidationError) as caught:
        adapt_study(tmp_path)
    assert 'exactly one seeded run per condition' in str(caught.value)


def test_duplicate_evaluation_record_fails(tmp_path: Path) -> None:
    _make_bundle(tmp_path, duplicate_evaluation=True)

    with pytest.raises(BundleValidationError) as caught:
        adapt_study(tmp_path)
    assert 'duplicate evaluation record' in str(caught.value)


def test_unexpected_evaluation_record_fails(tmp_path: Path) -> None:
    _make_bundle(tmp_path, unexpected_evaluation=True)

    with pytest.raises(BundleValidationError) as caught:
        adapt_study(tmp_path)
    assert 'unexpected evaluation record' in str(caught.value)


def test_non_finite_outcome_fails(tmp_path: Path) -> None:
    _make_bundle(tmp_path, nan_return=True)

    with pytest.raises(BundleValidationError) as caught:
        adapt_study(tmp_path)
    assert caught.value.checks[-1].code == 'evaluation_schema'
    assert 'must be finite' in str(caught.value)


def test_contract_scope_mutation_fails(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    contract_path = tmp_path / 'contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    contract['scope']['env_name'] = 'Acrobot-v1'
    _write_json(contract_path, contract)
    seal_bundle(tmp_path)

    with pytest.raises(BundleValidationError) as caught:
        adapt_study(tmp_path)
    assert 'differs from the contract scope' in str(caught.value)


def test_manifest_rejects_path_traversal(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    manifest_path = tmp_path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['files']['../escape'] = {'sha256': '0' * 64, 'size': 0}
    _write_json(manifest_path, manifest)

    with pytest.raises(BundleValidationError) as caught:
        adapt_study(tmp_path)
    assert 'unsafe bundle path' in str(caught.value)


def test_scope_field_colliding_with_derived_column_fails(
    tmp_path: Path,
) -> None:
    _make_bundle(
        tmp_path,
        scope={'env_name': 'MountainCar-v0', 'return_mean': 1.0},
    )

    with pytest.raises(BundleValidationError) as caught:
        adapt_study(tmp_path)
    assert caught.value.checks[-1].code == 'row_key_collision'


def test_resealed_protocol_execution_mismatch_fails(tmp_path: Path) -> None:
    _make_bundle(tmp_path, protocol_pair_keys=(8,))

    with pytest.raises(BundleValidationError) as caught:
        adapt_study(tmp_path)
    assert 'differ from the protocol' in str(caught.value)


# ============ attested / unverifiable — never silently upgraded ============


def test_absent_protocol_is_unverifiable_not_failure(tmp_path: Path) -> None:
    _make_bundle(tmp_path, include_protocol=False)
    study = adapt_study(tmp_path)

    assert study.receipt.admissible
    assert (
        _check_status(study.receipt, 'protocol')
        is CheckStatus.UNVERIFIABLE
    )


def test_absent_assignment_is_unverifiable(tmp_path: Path) -> None:
    _make_bundle(tmp_path, include_assignment=False)
    study = adapt_study(tmp_path)

    assert study.receipt.admissible
    assert (
        _check_status(study.receipt, 'assignment')
        is CheckStatus.UNVERIFIABLE
    )
    assert study.contrast.assurance is CheckStatus.UNVERIFIABLE


def test_omitted_arm_key_mapping_defaults_to_condition_names(
    tmp_path: Path,
) -> None:
    _make_bundle(tmp_path, omit_logical_arm_keys=True)
    study = adapt_study(tmp_path)

    assert {row['arm_key'] for row in study.rows} == {
        _BASELINE_ARM, _TREATMENT_ARM,
    }
    assert study.contrast.arm_keys == (_BASELINE_ARM, _TREATMENT_ARM)


def test_seal_bundle_is_deterministic(tmp_path: Path) -> None:
    _make_bundle(tmp_path)
    first = (tmp_path / 'manifest.json').read_text(encoding='utf-8')
    seal_bundle(tmp_path)
    second = (tmp_path / 'manifest.json').read_text(encoding='utf-8')
    assert first == second
