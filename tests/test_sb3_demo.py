"""End-to-end regression for the external-runs claim path.

The committed demo data is real SB3 output — `model.save()`
checkpoint zips plus `EvalCallback` `evaluations.npz` from an
actual training run — read through `corroborate_rl.sb3`. The
configuration registry comes from the algorithm constructor's
signature; `_DQN_SIGNATURE` mirrors the constructor of the
stable-baselines3 version that produced the artifacts (2.9.0),
so this test needs neither SB3 nor torch installed.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from corroborate.analyses.paired.paired_directional import (
    PairedDirectionalResult,
)
from corroborate.bridge.bridge import evaluate
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate_rl.sb3 import load_sb3_runs, sb3_config_columns

from examples.sb3_demo.sb3_claim import higher_gamma_improves_return


_RUNS = Path(__file__).parents[1] / 'examples' / 'sb3_demo' / 'runs'


class _DQN_SIGNATURE:  # noqa: N801 — stands in for the DQN class
    """Constructor signature of stable-baselines3 2.9.0's DQN —
    the version that produced the committed checkpoints. The leaf
    registry is 'what the entry point accepts', so a plain class
    with the same parameters reads the artifacts identically."""

    def __init__(
        self,
        policy: object, env: object, learning_rate: object = None,
        buffer_size: object = None, learning_starts: object = None,
        batch_size: object = None, tau: object = None,
        gamma: object = None, train_freq: object = None,
        gradient_steps: object = None,
        replay_buffer_class: object = None,
        replay_buffer_kwargs: object = None,
        optimize_memory_usage: object = None, n_steps: object = None,
        target_update_interval: object = None,
        exploration_fraction: object = None,
        exploration_initial_eps: object = None,
        exploration_final_eps: object = None,
        max_grad_norm: object = None, stats_window_size: object = None,
        tensorboard_log: object = None, policy_kwargs: object = None,
        verbose: object = None, seed: object = None,
        device: object = None, _init_setup_model: object = None,
    ) -> None:
        raise NotImplementedError('signature fixture; never constructed')


def test_committed_sb3_artifacts_evaluate_the_claim_module() -> None:
    df = load_sb3_runs(_RUNS, _DQN_SIGNATURE).with_columns(
        pl.lit('CartPole-v1').alias('env_id'),
    )
    assert df.height == 6
    # Configuration recovered from the checkpoints themselves:
    assert set(df['gamma'].to_list()) == {0.8, 0.99}
    assert df['buffer_size'].unique().to_list() == [50_000]

    leaves = sb3_config_columns(_RUNS, _DQN_SIGNATURE)
    assert 'gamma' in leaves and 'seed' in leaves
    # Runtime state never registers as configuration:
    assert 'num_timesteps' not in leaves
    assert 'exploration_rate' not in leaves

    evaluation = evaluate(higher_gamma_improves_return, df, leaves=leaves)

    assert evaluation.n_cells_in_scope == 6
    assert evaluation.extent_hash != 0
    assert evaluation.blocked_by is None
    # gamma is registered configuration, every registered leaf is
    # balanced within each seed pair, and nothing unregistered
    # rides the contrast — a clean record.
    assert evaluation.warnings == ()
    assert evaluation.verdict is Verdict.POWER_INSUFFICIENT
    assert evaluation.refutation_class is RefutationClass.UNDERPOWERED

    result = evaluation.analysis_results['paired_directional']
    assert isinstance(result, PairedDirectionalResult)
    assert result.measurable == 'return_mean'
    assert result.baseline_arm == 'baseline'
    assert result.treatment_arm == 'treatment'
    assert result.n_pairs == 3
    assert result.mean_diff == pytest.approx(-37.8, abs=0.05)
    assert result.dz == pytest.approx(-0.61, abs=0.005)
    assert result.predicted_direction == 'a_gt_b'
    assert result.alpha == 0.05
    assert result.sesoi_dz == 0.5
    assert result.minimum_pairs == 3
