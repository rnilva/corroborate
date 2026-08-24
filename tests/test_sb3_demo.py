"""End-to-end regression for the external-runs claim path.

The committed demo data is real SB3 output — the Logger's own
`progress.json`, one file per run, written by `configure(folder,
['json'])` — read through `corroborate_rl.sb3_logger`. The
configuration registry is the `config/*` keys the producer
recorded; nothing here needs stable-baselines3 installed.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from corroborate.analyses.paired.paired_directional import (
    PairedDirectionalResult,
)
from corroborate.bridge.bridge import evaluate
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate_rl.sb3_logger import load_sb3_logger_runs

from examples.sb3_demo.sb3_claim import (
    eval_reward_at_25k,
    higher_gamma_improves_return,
)


_RUNS = Path(__file__).parents[1] / 'examples' / 'sb3_demo' / 'runs'


def test_committed_sb3_logger_record_evaluates_the_claim_module() -> None:
    panel = load_sb3_logger_runs(_RUNS)
    df = panel.cells
    assert df.height == 6
    # Configuration is what the producer recorded under config/*:
    assert set(df['config/gamma'].to_list()) == {0.8, 0.99}
    assert df['config/buffer_size'].unique().to_list() == [50_000]

    # The Panel carries the registry the record declares — and
    # nothing else: record keys (rollout/*, train/*, eval/*,
    # time/*) are columns, never registry.
    leaves = panel.leaves
    assert leaves is not None
    assert 'config/gamma' in leaves and 'config/seed' in leaves
    assert 'config/env_id' in leaves
    assert not any(key.startswith(('rollout/', 'train/', 'eval/', 'time/'))
                   for key in leaves)

    # The claim's horizon column is claim-side: registered by the
    # claim module, computed from the record arrays on demand.
    panel = panel.with_measurables(('eval_reward_at_25k',))
    at_25k = panel.cells['eval_reward_at_25k'].to_list()
    assert len(at_25k) == 6
    assert all(v is not None and math.isfinite(v) for v in at_25k)

    evaluation = evaluate(higher_gamma_improves_return, panel)

    assert evaluation.n_cells_in_scope == 6
    assert evaluation.extent_hash != 0
    assert evaluation.blocked_by is None
    # config/gamma is registered configuration, every registered
    # leaf is balanced within each seed pair, and nothing
    # unregistered rides the contrast — a clean record.
    assert evaluation.warnings == ()
    assert evaluation.verdict is Verdict.POWER_INSUFFICIENT
    assert evaluation.refutation_class is RefutationClass.UNDERPOWERED

    result = evaluation.analysis_results['paired_directional']
    assert isinstance(result, PairedDirectionalResult)
    assert result.measurable == 'eval_reward_at_25k'
    assert result.baseline_arm == 'baseline'
    assert result.treatment_arm == 'treatment'
    assert result.n_pairs == 3
    assert result.mean_diff == pytest.approx(-56.27, abs=0.05)
    assert result.dz == pytest.approx(-0.81, abs=0.005)
    assert result.predicted_direction == 'a_gt_b'
    assert result.alpha == 0.05
    assert result.sesoi_dz == 0.5
    assert result.minimum_pairs == 3


def test_horizon_is_pinned_not_rebased() -> None:
    """A run never evaluated at the claim's horizon reads NaN —
    missing evidence, never silently rebased to an earlier
    evaluation point."""
    record = {
        'time/total_timesteps': [5_000, 10_000],
        'eval/mean_reward': [100.0, 120.0],
    }
    assert math.isnan(eval_reward_at_25k.fn(record))
    full = {
        'time/total_timesteps': [5_000, 25_000],
        'eval/mean_reward': [100.0, 120.0],
    }
    assert eval_reward_at_25k.fn(full) == 120.0
