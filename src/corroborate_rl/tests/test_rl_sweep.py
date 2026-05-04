"""Tests for `corroborate_rl.sweep` — RL substrate's Runner.

Covers:
- DQNRunner satisfies `Runner[DQNTrajectoryRecord]` Protocol
  (structural-type check via Runner[R] accepting it).
- DQNRunner type-narrows env_name and seeds from grid_point.
- DQNRunner rejects unexpected grid keys (HP variation belongs
  in the substrate's `base`, not in the exogenous grid).

Tests use a SHORT-run claim (total_steps=60, eval_every=30,
small Replay) so each test completes in seconds on CPU."""
from __future__ import annotations

from functools import partial

import pytest

from corroborate.core.intervention import combined_arm_key
from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
from corroborate_rl.dqn.claims.replay import Replay
from corroborate_rl.dqn.dqn import dqn
from corroborate_rl.env_catalogue import get
from corroborate_rl.sweep import DQNRunner


pytestmark = pytest.mark.slow


_SHORT_RUN_HP: dict[str, object] = {
    'total_steps': 60, 'eval_every': 30, 'n_episodes': 2,
    'sync_period': 10,
    'replay': Replay(capacity=200, batch_size=8),
    'optimizer': partial(warmed_update, inner=partial(adam), warmup_steps=10),
}


_VANILLA_CLAIM = partial(dqn, **_SHORT_RUN_HP)
_VANILLA_ARM_KEY = combined_arm_key(())


# ============ DQNRunner basics ============

def test_dqn_runner_returns_per_seed_runs_and_traces() -> None:
    """One call → SweepCellResult with N runs + N traces (N = len(seeds))
    and a captured ComputationGraph."""
    runner = DQNRunner({'CartPole-v1': get('CartPole-v1')})
    result = runner(
        _VANILLA_CLAIM, _VANILLA_ARM_KEY, (),
        {'env_name': 'CartPole-v1', 'seeds': (0, 1)},
    )
    assert len(result.runs) == 2
    assert len(result.traces) == 2
    run_ids = sorted(r.id for r in result.runs)
    trace_ids = sorted(t.id for t in result.traces)
    assert run_ids == trace_ids
    assert len(result.graph.nodes) > 0


def test_dqn_runner_rejects_non_str_env_name() -> None:
    runner = DQNRunner({'CartPole-v1': get('CartPole-v1')})
    with pytest.raises(TypeError, match='env_name'):
        runner(
            _VANILLA_CLAIM, _VANILLA_ARM_KEY, (),
            {'env_name': 123, 'seeds': (0,)},
        )


def test_dqn_runner_rejects_non_tuple_seeds() -> None:
    runner = DQNRunner({'CartPole-v1': get('CartPole-v1')})
    with pytest.raises(TypeError, match='seeds'):
        runner(
            _VANILLA_CLAIM, _VANILLA_ARM_KEY, (),
            {'env_name': 'CartPole-v1', 'seeds': [0, 1]},
        )


def test_dqn_runner_rejects_unexpected_grid_keys() -> None:
    """HP variation belongs in `base`. Unrecognized grid keys are
    loud errors, not silent passthroughs."""
    runner = DQNRunner({'CartPole-v1': get('CartPole-v1')})
    with pytest.raises(ValueError, match='unexpected grid_point keys'):
        runner(
            _VANILLA_CLAIM, _VANILLA_ARM_KEY, (),
            {
                'env_name': 'CartPole-v1', 'seeds': (0,),
                'replay.batch_size': 16,
            },
        )
