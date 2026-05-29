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


# ============ cell_idx alignment with framework cell_idx ============

def test_dqn_runner_uses_framework_cell_idx_when_provided(
    tmp_path,  # noqa: ANN001  # pytest fixture
) -> None:
    """When `__framework_cell_idx__` is injected into grid_point,
    the runner writes per-cell sidecars under that index — NOT
    under its internal `_call_count`. Critical for resume scenarios
    where some grid points are skipped (already archived) and the
    runner's call counter would otherwise diverge from the
    framework's cell_idx, causing later cells to overwrite earlier
    cells' bundles.

    Closed-form check: skip-then-train pattern — call runner with
    cell_idx=5 first, verify the bundle lands at cell005.msgpack.
    Then call again with cell_idx=6 and verify cell006.msgpack
    appears alongside (not overwriting cell005)."""
    from corroborate_rl.dqn.q_checkpoint_bundle import (
        bundle_path, load_bundle,
    )
    claim = partial(
        dqn, keep_q_checkpoint_final=True, **_SHORT_RUN_HP,
    )
    q_ckpt_dir = tmp_path / 'q_checkpoints'
    runner = DQNRunner(
        {'CartPole-v1': get('CartPole-v1')},
        q_checkpoint_dir=q_ckpt_dir,
    )
    # First call simulates the framework's resume-skipped-cells case:
    # cell_idx=5 means the framework already skipped cells 0-4.
    _ = runner(
        claim, _VANILLA_ARM_KEY, (),
        {
            'env_name': 'CartPole-v1', 'seeds': (0,),
            '__framework_cell_idx__': 5,
        },
    )
    bundle5 = load_bundle(bundle_path(q_ckpt_dir, cell_idx=5))
    assert bundle5.cell_idx == 5
    # Second call: framework's next iteration is cell_idx=6.
    # Runner's `_call_count` is now 1 (would write cell001.msgpack
    # under the legacy bug); with the fix it lands at cell006.msgpack
    # and cell005.msgpack stays untouched.
    _ = runner(
        claim, _VANILLA_ARM_KEY, (),
        {
            'env_name': 'CartPole-v1', 'seeds': (1,),
            '__framework_cell_idx__': 6,
        },
    )
    assert bundle_path(q_ckpt_dir, cell_idx=5).is_file()
    assert bundle_path(q_ckpt_dir, cell_idx=6).is_file()
    bundle6 = load_bundle(bundle_path(q_ckpt_dir, cell_idx=6))
    assert bundle6.cell_idx == 6
    # cell005 bundle stayed put — its seeds didn't get overwritten.
    bundle5_after = load_bundle(bundle_path(q_ckpt_dir, cell_idx=5))
    assert bundle5_after.seeds == (0,)


def test_dqn_runner_falls_back_to_call_count_without_framework_idx() -> None:
    """When `__framework_cell_idx__` is absent (in-process callers,
    test fixtures that bypass `run_intervention`), the runner's
    internal `_call_count` is used — preserves pre-fix behavior for
    callers that don't go through the framework's sweep loop."""
    runner = DQNRunner({'CartPole-v1': get('CartPole-v1')})
    result = runner(
        _VANILLA_CLAIM, _VANILLA_ARM_KEY, (),
        {'env_name': 'CartPole-v1', 'seeds': (0,)},
    )
    assert len(result.runs) == 1
    # Internal counter advanced; a second call would see _call_count=1.
    assert runner._call_count == 1  # noqa: SLF001  # invariant check
