"""`corroborate_rl.sb3` — SB3 artifacts to the neutral shape.

Fixture run folders are synthesized from closed forms — a
hand-built checkpoint zip (the `data` JSON `model.save()` writes,
runtime state and cloudpickle markers included) and an
`evaluations.npz` (what `EvalCallback` writes) — so every derived
aggregate has an exact expected value, and none of it needs
stable-baselines3 installed: the leaf registry is recovered from a
constructor signature, which the tests supply as a plain class.

Closed form: the return at (checkpoint c, episode e) is
`base(gamma) + c/10 + e`, episodes e ∈ {0, 1} → per-checkpoint
mean `base + c/10 + 0.5`; checkpoints (10, 20) → final mean
`base + 2.5`, trapezoid AUC `base + 2.0` (midpoint at c = 15).
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from corroborate.analyses.paired.arm_mean_diff import (
    ArmMeanDiffResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge, evaluate
from corroborate.bridge.verdict import Verdict
from corroborate.core.intervention import DoEffect
from corroborate.data import concat_panels
from corroborate_rl.sb3 import (
    checkpoint_config,
    load_sb3_runs,
    sb3_config_columns,
)

_BASELINE_GAMMA = 0.80
_TREATMENT_GAMMA = 0.99
_BASELINE_BASE = 100.0
_TREATMENT_BASE = 110.0
_CHECKPOINTS = (10, 20)
_EPISODES = (0, 1)


class _FakeDQN:
    """Constructor signature standing in for `sb3.DQN` — the leaf
    registry is 'what the entry point accepts', so a plain class
    with the right parameters is a faithful fixture."""

    def __init__(
        self,
        policy: str,
        env: object,
        learning_rate: float = 1e-3,
        buffer_size: int = 100,
        gamma: float = 0.99,
        seed: int | None = None,
        policy_kwargs: dict[str, object] | None = None,
        train_freq: object = None,
    ) -> None:
        del policy, env, learning_rate, buffer_size, gamma, seed
        del policy_kwargs, train_freq


def _write_checkpoint(path: Path, *, gamma: float, seed: int) -> None:
    """A minimal `model.save()`-shaped zip: resolved configuration
    mixed with runtime state and cloudpickled blobs, exactly the
    mixture the signature intersection must cut through."""
    data = {
        'gamma': gamma,
        'learning_rate': 1e-3,
        'buffer_size': 5_000,
        'seed': seed,
        'policy_kwargs': {'activation': 'tanh'},
        # A constructor param SB3 could not JSON-encode — kept as
        # its opaque serialised payload so differences stay visible:
        'train_freq': {':serialized:': 'freq-blob'},
        # Runtime state — real state resumption needs, never leaves:
        'num_timesteps': 25_000,
        'exploration_rate': 0.05,
        '_n_updates': 6_000,
        # Cloudpickled entries `save()` writes for non-JSON values:
        'policy_class': {':serialized:': 'base64-noise'},
        'observation_space': {':serialized:': 'base64-noise'},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w') as archive:
        archive.writestr('data', json.dumps(data))


def _write_evaluations(path: Path, *, base: float) -> None:
    timesteps = np.array(_CHECKPOINTS, dtype=np.int64)
    results = np.array(
        [
            [base + checkpoint / 10.0 + episode for episode in _EPISODES]
            for checkpoint in _CHECKPOINTS
        ],
        dtype=np.float64,
    )
    np.savez(path, timesteps=timesteps, results=results)


def _make_sb3_runs(root: Path, *, seeds: tuple[int, ...] = (0, 1)) -> None:
    for gamma, base in (
        (_BASELINE_GAMMA, _BASELINE_BASE),
        (_TREATMENT_GAMMA, _TREATMENT_BASE),
    ):
        for seed in seeds:
            run_dir = root / f'gamma{gamma * 100:03.0f}-s{seed}'
            _write_checkpoint(
                run_dir / 'model.zip', gamma=gamma, seed=seed,
            )
            _write_evaluations(run_dir / 'evaluations.npz', base=base)


def test_checkpoint_config_intersects_constructor_signature(
    tmp_path: Path,
) -> None:
    """Configuration is what the constructor accepts: runtime
    state and cloudpickled blobs fall out; nested scalar config
    survives."""
    _write_checkpoint(tmp_path / 'model.zip', gamma=0.9, seed=7)
    config = checkpoint_config(tmp_path / 'model.zip', _FakeDQN)
    assert config == {
        'gamma': 0.9,
        'learning_rate': 1e-3,
        'buffer_size': 5_000,
        'seed': 7,
        'policy_kwargs': {'activation': 'tanh'},
        'train_freq': 'freq-blob',
    }


def test_load_sb3_runs_derives_closed_form_columns(tmp_path: Path) -> None:
    _make_sb3_runs(tmp_path)
    df = load_sb3_runs(tmp_path, _FakeDQN).cells

    assert df.height == 4
    for row in df.to_dicts():
        base = (
            _BASELINE_BASE
            if row['gamma'] == _BASELINE_GAMMA
            else _TREATMENT_BASE
        )
        assert row['return_mean'] == base + 2.5
        assert row['return_auc'] == base + 2.0
        for checkpoint in _CHECKPOINTS:
            assert row[f'return_mean_at_{checkpoint}'] == (
                base + checkpoint / 10.0 + 0.5
            )
        assert row['policy_kwargs.activation'] == 'tanh'
        assert row['corpus'] == tmp_path.name
        assert 'num_timesteps' not in row
        assert 'exploration_rate' not in row


def test_sb3_config_columns_registry(tmp_path: Path) -> None:
    _make_sb3_runs(tmp_path)
    assert sb3_config_columns(tmp_path, _FakeDQN) == frozenset({
        'gamma', 'learning_rate', 'buffer_size', 'seed',
        'policy_kwargs.activation', 'train_freq',
    })


_GAMMA_EFFECT = DoEffect.from_values(
    source='gamma',
    reference=_BASELINE_GAMMA,
    treatment=_TREATMENT_GAMMA,
)


@claim_bridge(
    source=_GAMMA_EFFECT,
    target='return_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    pair_by=('seed',),
    predicted_direction='a_gt_b',
)
def _higher_gamma_improves_sb3_return(
    arm_mean_diff: ArmMeanDiffResult,
) -> Verdict:
    if arm_mean_diff.mean_diff > 0:
        return Verdict.HELD
    return Verdict.NO_EFFECT


def test_end_to_end_value_contrast_on_sb3_artifacts(tmp_path: Path) -> None:
    """The potential-user path: a folder of ordinary SB3 outputs,
    no training-script changes, straight to a gated verdict.
    Closed form: mean_diff = 110 − 100 = 10 exactly at n = 2 per
    condition. The Panel carries the configuration registry, so
    `evaluate` needs nothing beyond the claim and the record."""
    _make_sb3_runs(tmp_path)
    panel = load_sb3_runs(tmp_path, _FakeDQN)
    assert panel.leaves == sb3_config_columns(tmp_path, _FakeDQN)

    evaluation = evaluate(_higher_gamma_improves_sb3_return, panel)
    assert evaluation.verdict is Verdict.HELD
    assert evaluation.blocked_by is None
    result = evaluation.analysis_results['arm_mean_diff']
    assert isinstance(result, ArmMeanDiffResult)
    assert result.treatment_arm == 'treatment'
    assert result.baseline_arm == 'baseline'
    assert result.mean_diff == _TREATMENT_BASE - _BASELINE_BASE


def test_unmatched_sb3_level_is_outside_declared_effect(
    tmp_path: Path,
) -> None:
    """Additional levels may accumulate in the same record without
    changing the fixed binary estimand."""
    declared = tmp_path / 'declared'
    _make_sb3_runs(declared)
    other = tmp_path / 'other' / 'gamma090-s0'
    _write_checkpoint(other / 'model.zip', gamma=0.90, seed=0)
    _write_evaluations(other / 'evaluations.npz', base=10_000.0)
    pooled = concat_panels([
        load_sb3_runs(declared, _FakeDQN),
        load_sb3_runs(tmp_path / 'other', _FakeDQN),
    ])

    evaluation = evaluate(_higher_gamma_improves_sb3_return, pooled)

    assert evaluation.verdict is Verdict.HELD
    assert evaluation.n_cells_in_scope == 4
    result = evaluation.analysis_results['arm_mean_diff']
    assert isinstance(result, ArmMeanDiffResult)
    assert result.n_treatment == 2
    assert result.n_baseline == 2
    assert result.mean_diff == _TREATMENT_BASE - _BASELINE_BASE


def test_duplicate_evaluation_timestep_raises(tmp_path: Path) -> None:
    run_dir = tmp_path / 'run-a'
    _write_checkpoint(run_dir / 'model.zip', gamma=0.9, seed=0)
    np.savez(
        run_dir / 'evaluations.npz',
        timesteps=np.array([10, 10], dtype=np.int64),
        results=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
    )
    with pytest.raises(ValueError, match='duplicate evaluation timestep'):
        load_sb3_runs(tmp_path, _FakeDQN)


def test_string_algo_without_sb3_raises_with_guidance() -> None:
    try:
        import stable_baselines3  # noqa: F401  # pyright: ignore[reportUnusedImport, reportMissingImports]
    except ImportError:
        pass
    else:
        pytest.skip('stable-baselines3 installed; error path unreachable')
    with pytest.raises(ImportError, match='stable-baselines3'):
        checkpoint_config(Path('unused.zip'), 'DQN')


def test_runs_without_evaluations_load_config_only(tmp_path: Path) -> None:
    run_dir = tmp_path / 'run-a'
    _write_checkpoint(run_dir / 'model.zip', gamma=0.9, seed=0)
    df = load_sb3_runs(tmp_path, _FakeDQN).cells
    assert df.height == 1
    assert df['gamma'].to_list() == [0.9]
    assert 'return_mean' not in df.columns


def test_stray_subdirectories_are_walked_past(tmp_path: Path) -> None:
    """Real log folders carry non-run subdirectories (tensorboard
    logs, stray tooling output); a reader skips them and raises
    only when NO subdirectory carries a checkpoint."""
    run_dir = tmp_path / 'run-a'
    _write_checkpoint(run_dir / 'model.zip', gamma=0.9, seed=0)
    (tmp_path / 'tb_logs').mkdir()
    (tmp_path / 'tb_logs' / 'events.out.tfevents.0').write_text('')
    df = load_sb3_runs(tmp_path, _FakeDQN).cells
    assert df['id'].to_list() == ['run-a']

    empty = tmp_path / 'no-runs-here'
    (empty / 'tb_logs').mkdir(parents=True)
    with pytest.raises(ValueError, match='no run directories'):
        load_sb3_runs(empty, _FakeDQN)


def test_pooling_sb3_batches_is_concat_panels(tmp_path: Path) -> None:
    """A growing record pools batch by batch; the union Panel
    keeps carrying the registry, so evaluation stays two-argument."""
    _make_sb3_runs(tmp_path / 'batch_a', seeds=(0, 1))
    _make_sb3_runs(tmp_path / 'batch_b', seeds=(2, 3))
    pooled = concat_panels([
        load_sb3_runs(tmp_path / 'batch_a', _FakeDQN),
        load_sb3_runs(tmp_path / 'batch_b', _FakeDQN),
    ])
    assert pooled.leaves == sb3_config_columns(tmp_path / 'batch_a', _FakeDQN)
    evaluation = evaluate(_higher_gamma_improves_sb3_return, pooled)
    assert evaluation.verdict is Verdict.HELD
    result = evaluation.analysis_results['arm_mean_diff']
    assert isinstance(result, ArmMeanDiffResult)
    assert result.n_treatment == 4
    assert result.n_baseline == 4


def test_ambiguous_checkpoint_series_requires_explicit_selection(
    tmp_path: Path,
) -> None:
    """A `CheckpointCallback` series without model.zip/best_model.zip
    is ambiguous — lexicographic order would bind the claim to an
    arbitrary training budget (10000 sorts before 5000). Selection
    must be deliberate."""
    run_dir = tmp_path / 'run-a'
    for steps in (5_000, 10_000, 20_000):
        _write_checkpoint(
            run_dir / f'rl_model_{steps}_steps.zip', gamma=0.9, seed=0,
        )
    with pytest.raises(ValueError, match='checkpoint='):
        load_sb3_runs(tmp_path, _FakeDQN)

    df = load_sb3_runs(
        tmp_path, _FakeDQN, checkpoint='rl_model_20000_steps.zip',
    ).cells
    assert df.height == 1
    assert 'gamma' in df.columns


def test_explicit_checkpoint_missing_in_a_run_dir_raises(
    tmp_path: Path,
) -> None:
    """A run directory carrying checkpoint zips but not the
    requested one must fail loudly: walking past it would silently
    drop the run (typically the crashed one) and hand the analysis
    a survivor-biased sample."""
    complete = tmp_path / 'a-s0'
    _write_checkpoint(
        complete / 'rl_model_20000_steps.zip', gamma=0.9, seed=0,
    )
    crashed = tmp_path / 'b-s1'
    _write_checkpoint(
        crashed / 'rl_model_10000_steps.zip', gamma=0.99, seed=1,
    )
    with pytest.raises(ValueError, match='silently excluded'):
        load_sb3_runs(
            tmp_path, _FakeDQN, checkpoint='rl_model_20000_steps.zip',
        )


def test_non_finite_terminal_episodes_stay_visible(tmp_path: Path) -> None:
    """A NaN terminal episode shrinks the finite count rather than
    silently averaging survivors; an all-NaN terminal leaves the
    terminal mean null instead of rebasing to an earlier
    checkpoint — with the attempted count retaining that the
    evaluation was tried."""
    partial = tmp_path / 'partial' / 'run-a'
    _write_checkpoint(partial / 'model.zip', gamma=0.9, seed=0)
    np.savez(
        partial / 'evaluations.npz',
        timesteps=np.array(_CHECKPOINTS, dtype=np.int64),
        results=np.array(
            [[1.0, 2.0], [3.0, float('nan')]], dtype=np.float64,
        ),
    )
    row = load_sb3_runs(tmp_path / 'partial', _FakeDQN).cells.to_dicts()[0]
    assert row['return_mean'] == 3.0
    assert row['return_terminal_n'] == 1
    assert row['return_terminal_attempted'] == 2
    # Partial-finite grids no longer qualify for a comparable AUC
    # only when a checkpoint drops out entirely; here both
    # checkpoints retain finite means.
    assert row['return_auc'] is not None

    failed = tmp_path / 'failed' / 'run-a'
    _write_checkpoint(failed / 'model.zip', gamma=0.9, seed=0)
    np.savez(
        failed / 'evaluations.npz',
        timesteps=np.array(_CHECKPOINTS, dtype=np.int64),
        results=np.array(
            [[1.0, 2.0], [float('nan'), float('nan')]], dtype=np.float64,
        ),
    )
    row = load_sb3_runs(tmp_path / 'failed', _FakeDQN).cells.to_dicts()[0]
    # Sole run in the frame: the never-derived terminal column is
    # absent outright (a multi-run frame null-pads it instead).
    assert row.get('return_mean') is None
    assert row['return_terminal_n'] == 0
    assert row['return_terminal_attempted'] == 2
    assert row.get('return_auc') is None
    assert row['return_mean_at_10'] == 1.5
