"""Tests for `corroborate.rl.sweep` — RL substrate's Runner +
run_dqn_sweep convenience.

Covers:
- DQNRunner satisfies `Runner[DQNTrajectoryRecord]` Protocol
  (structural-type check via `sweep` accepting it).
- DQNRunner type-narrows env_name and seeds from grid_point.
- DQNRunner rejects unexpected grid keys (HP variation belongs
  in the Hypothesis, not in the exogenous grid).
- run_dqn_sweep iterates env_specs × hypotheses.

Tests use a SHORT-run intervention (total_steps=60, eval_every=30,
small Replay) so each test completes in seconds on CPU."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import pytest

from corroborate.core.hypothesis import Hypothesis
from corroborate.core.intervention import Intervention
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from functools import partial as _partial
from corroborate.rl.dqn.claims.optimizer import adam, warmed_update
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import get
from corroborate.rl.sweep import DQNRunner, run_dqn_sweep


_SHORT_RUN_INTERVENTION: dict[str, object] = {
    'total_steps': 60, 'eval_every': 30, 'n_episodes': 2,
    'sync_period': 10,
    'replay': Replay(capacity=200, batch_size=8),
    'optimizer': _partial(warmed_update, inner=_partial(adam), warmup_steps=10),
}


def _vanilla_h() -> Hypothesis[DQNTrajectoryRecord]:
    return Hypothesis(
        name='vanilla', intervention=dict(_SHORT_RUN_INTERVENTION),
    )


def _ddqn_h() -> Hypothesis[DQNTrajectoryRecord]:
    return Hypothesis(
        name='ddqn', intervention=dict(_SHORT_RUN_INTERVENTION),
        intervention_arms=(
            Intervention(
                slot_path='bootstrap',
                replacement=partial(
                    bootstrap, greedification=double_greedify,
                ),
            ),
        ),
    )


# ============ DQNRunner basics ============

def test_dqn_runner_returns_per_seed_runs_and_traces() -> None:
    """One call → SweepCellResult with N runs + N traces (N = len(seeds))
    and a captured ComputationGraph."""
    runner = DQNRunner({'CartPole-v1': get('CartPole-v1')})
    h = _vanilla_h()
    result = runner(h, {'env_name': 'CartPole-v1', 'seeds': (0, 1)})
    assert len(result.runs) == 2
    assert len(result.traces) == 2
    # Run/trace ids match.
    run_ids = sorted(r.id for r in result.runs)
    trace_ids = sorted(t.id for t in result.traces)
    assert run_ids == trace_ids
    # Graph captured (non-empty for a real dqn run).
    assert len(result.graph.nodes) > 0


def test_dqn_runner_rejects_non_str_env_name() -> None:
    runner = DQNRunner({'CartPole-v1': get('CartPole-v1')})
    h = _vanilla_h()
    with pytest.raises(TypeError, match='env_name'):
        runner(h, {'env_name': 123, 'seeds': (0,)})


def test_dqn_runner_rejects_non_tuple_seeds() -> None:
    runner = DQNRunner({'CartPole-v1': get('CartPole-v1')})
    h = _vanilla_h()
    with pytest.raises(TypeError, match='seeds'):
        runner(h, {'env_name': 'CartPole-v1', 'seeds': [0, 1]})


def test_dqn_runner_rejects_unexpected_grid_keys() -> None:
    """HP variation belongs in the Hypothesis. Unrecognized keys
    are loud errors, not silent passthroughs."""
    runner = DQNRunner({'CartPole-v1': get('CartPole-v1')})
    h = _vanilla_h()
    with pytest.raises(ValueError, match='unexpected grid_point keys'):
        runner(h, {
            'env_name': 'CartPole-v1', 'seeds': (0,),
            'replay.batch_size': 16,  # ← not allowed at grid level
        })


# ============ run_dqn_sweep convenience ============

def test_run_dqn_sweep_iterates_envs_and_hypotheses() -> None:
    """2 hypotheses × 1 env = 2 cell_results, each with len(seeds) runs."""
    env_specs: Mapping[str, object] = {'CartPole-v1': get('CartPole-v1')}
    result = run_dqn_sweep(
        [_vanilla_h(), _ddqn_h()],
        env_specs=env_specs,
        seeds=(0, 1),
    )
    assert len(result.cell_results) == 2
    assert len(result.all_runs) == 4  # 2 cells × 2 seeds


def test_run_dqn_sweep_no_failures_on_happy_path() -> None:
    env_specs = {'CartPole-v1': get('CartPole-v1')}
    result = run_dqn_sweep(
        [_vanilla_h()], env_specs=env_specs, seeds=(0,),
    )
    assert len(result.failures) == 0
    assert len(result.all_runs) == 1


def test_run_dqn_sweep_iterates_envs() -> None:
    """1 hypothesis × 2 envs = 2 cell_results."""
    env_specs = {
        'CartPole-v1': get('CartPole-v1'),
        'Acrobot-v1': get('Acrobot-v1'),
    }
    result = run_dqn_sweep(
        [_vanilla_h()], env_specs=env_specs, seeds=(0,),
    )
    assert len(result.cell_results) == 2
    env_names_seen = sorted({
        r.measurements['env_name'] for r in result.all_runs
    })
    assert env_names_seen == ['Acrobot-v1', 'CartPole-v1']
