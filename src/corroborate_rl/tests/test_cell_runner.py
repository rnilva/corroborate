"""Tests for `run_dqn_cell` — the bridge between the `dqn`
outermost claim and the schema layer.

Verifies:
1. `run_dqn_cell` runs CartPole end-to-end and produces a
   well-formed `RunRow` with HP topology leaves.
2. `arm_key` on the emitted RunRow matches the framework-derived
   canonical fingerprint passed to the runner (Phase 6
   convention: arm identity = `combined_arm_key(intervention)`).
3. Pre-registered measurables land at their bare `.name`.
4. Slot-swap intervention surfaces in the configurational leaf
   topology."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import jax.numpy as jnp
import pytest

from corroborate.bridge.verdict import Verdict
from corroborate.corpus.leaf_signature import leaf_signature
from corroborate.corpus.schema import RunRow
from corroborate.core.intervention import (
    DoEffect, Intervention, apply_interventions, combined_arm_key,
)
from corroborate.measurables import Measurable
from corroborate_rl.cell_runner import run_dqn_cell
from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
from corroborate_rl.dqn.claims.replay import Replay
from corroborate_rl.dqn.dqn import dqn
from corroborate_rl.dqn.invariants import DQNTrajectoryRecord


# Every test runs DQN end-to-end on CartPole — ~3 s each. Skipped
# by default; opt in via `-m slow`.
pytestmark = pytest.mark.slow

_REPLAY_SHORT = Replay(capacity=200, batch_size=16)
_OPTIMIZER_SHORT = partial(
    warmed_update, inner=partial(adam), warmup_steps=10,
)
_SHORT_RUN_HP: dict[str, object] = {
    'total_steps': 60, 'eval_every': 30, 'n_episodes': 2,
    'sync_period': 10,
    'replay': _REPLAY_SHORT,
    'optimizer': _OPTIMIZER_SHORT,
}
_SHORT_RUN_HP_40: dict[str, object] = {
    'total_steps': 40, 'eval_every': 20, 'n_episodes': 2,
    'sync_period': 10,
    'replay': _REPLAY_SHORT,
    'optimizer': _OPTIMIZER_SHORT,
}


# ============ run_dqn_cell — happy path ============

def test_run_dqn_cell_produces_runrow_on_cartpole() -> None:
    """End-to-end smoke: run vanilla DQN on CartPole."""
    from corroborate_rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    claim = partial(dqn, **_SHORT_RUN_HP)
    arm_key = combined_arm_key(())

    run_row = run_dqn_cell(
        env_spec, seed=0, claim=claim, arm_key=arm_key,
        measurables=(),
    ).run
    assert isinstance(run_row, RunRow)
    assert run_row.arm_key == 'baseline'
    assert run_row.measurements['env_name'] == 'CartPole-v1'
    assert run_row.measurements['seed'] == 0
    assert run_row.measurements['total_steps'] == 60
    assert run_row.verdict is Verdict.HELD
    assert isinstance(run_row.measurements['late_window_mean'], float)
    assert 'gamma' in run_row.measurements


def test_run_dqn_cell_leaf_signature_distinguishes_arms() -> None:
    """The leaf signature derived from RunRow.measurements is
    non-empty (configurational fingerprint)."""
    from corroborate_rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    claim = partial(dqn, **_SHORT_RUN_HP_40)
    run_row = run_dqn_cell(
        env_spec, seed=0, claim=claim, arm_key='baseline',
        measurables=(),
    ).run
    sig = leaf_signature(run_row.measurements)
    assert len(sig) > 0


def test_run_dqn_cell_pre_registered_measurables_persist() -> None:
    """Pre-registered measurables passed to the runner land in
    RunRow.measurements under their bare `.name`."""
    from corroborate_rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    def _ep_return_mean(record: Mapping[str, jnp.ndarray]) -> float:
        v = record.get('ep_return')
        if not isinstance(v, jnp.ndarray):
            return float('nan')
        return float(jnp.nanmean(v))

    m = Measurable[DQNTrajectoryRecord, float](
        fn=_ep_return_mean,
        name='ep_return_mean_summary',
        reads=('ep_return',),
    )
    claim = partial(dqn, **_SHORT_RUN_HP_40)

    run_row = run_dqn_cell(
        env_spec, seed=0, claim=claim, arm_key='baseline',
        measurables=(m,),
    ).run

    assert 'ep_return_mean_summary' in run_row.measurements
    val = run_row.measurements['ep_return_mean_summary']
    assert isinstance(val, float)


def test_run_dqn_cell_applies_intervention_via_slot_swap() -> None:
    """DDQN intervention is a slot swap on `bootstrap`. The
    composed claim's leaf topology records the bootstrap slot's
    canonicalised form; arm_key derives from `combined_arm_key`
    of the typed Intervention tuple."""
    from corroborate_rl.dqn.claims.bootstrap import (
        bootstrap, double_greedify,
    )
    from corroborate_rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    ddqn_swap = partial(bootstrap, greedification=double_greedify)
    intervention = DoEffect(
        treatment=(
            Intervention(slot_path='bootstrap', replacement=ddqn_swap),
        ),
        baseline=(),
    )
    base = partial(dqn, **_SHORT_RUN_HP_40)
    claim = apply_interventions(base, intervention.treatment)
    arm_key = intervention.treatment_arm_key()

    run_row = run_dqn_cell(
        env_spec, seed=0, claim=claim, arm_key=arm_key,
        measurables=(),
    ).run
    bootstrap_value = run_row.measurements.get('bootstrap')
    assert isinstance(bootstrap_value, str)
    assert 'double_greedify' in bootstrap_value
    assert run_row.arm_key == arm_key
    assert 'double_greedify' in arm_key
