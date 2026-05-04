"""Smoke test for the configurational-corpus path.

Verifies the end-to-end plumbing for PC-discovery node identity:
cells with the same configured composition produce stable leaf
signatures across (env, seed); different compositions produce
distinct signatures; aggregation groups RunRows correctly.

After Phase 6: arm identity flows through `arm_key` (canonical
fingerprint of the Intervention tuple); `leaf_signature` projects
the configurational fingerprint off RunRow.measurements."""
from __future__ import annotations

from functools import partial

import pytest

from corroborate.corpus.leaf_signature import leaf_signature
from corroborate.core.intervention import (
    DoEffect, Intervention, apply_interventions, combined_arm_key,
)
from corroborate_rl.cell_runner import run_dqn_cell
from corroborate_rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
from corroborate_rl.dqn.claims.replay import Replay
from corroborate_rl.dqn.dqn import dqn
from corroborate_rl.env_catalogue import get


# Whole module is end-to-end DQN smoke — multiple cells per test.
pytestmark = pytest.mark.slow


_SHORT_RUN_HP: dict[str, object] = {
    'total_steps': 40, 'eval_every': 20, 'n_episodes': 2,
    'sync_period': 10,
    'replay': Replay(capacity=200, batch_size=16),
    'optimizer': partial(warmed_update, inner=partial(adam), warmup_steps=10),
}


_BASE = partial(dqn, **_SHORT_RUN_HP)
_VANILLA_CLAIM = _BASE
_VANILLA_ARM_KEY = combined_arm_key(())


_DDQN_INTERVENTION = DoEffect(
    treatment=(
        Intervention(
            slot_path='bootstrap',
            replacement=partial(bootstrap, greedification=double_greedify),
        ),
    ),
    baseline=(),
)
_DDQN_CLAIM = apply_interventions(_BASE, _DDQN_INTERVENTION.treatment)
_DDQN_ARM_KEY = _DDQN_INTERVENTION.treatment_arm_key()


def _run_cell(env_name: str, seed: int, claim, arm_key: str):
    """Cell runner returns `CellResult(run, trace)` — these tests
    only care about the verdict-side row, so unpack `.run` here."""
    return run_dqn_cell(
        get(env_name), seed=seed, claim=claim, arm_key=arm_key,
        measurables=(),
    ).run


# ============ Stable leaf_signature across (env, seed) ============

def test_same_hypothesis_yields_stable_leaf_signature() -> None:
    """Two cells of vanilla DQN — different seeds, different envs
    — must produce the SAME `leaf_signature`. The signature
    filters out `env_name` and `seed`, so changes there don't
    perturb it."""
    run_a = _run_cell('CartPole-v1', 0, _VANILLA_CLAIM, _VANILLA_ARM_KEY)
    run_b = _run_cell('CartPole-v1', 1, _VANILLA_CLAIM, _VANILLA_ARM_KEY)
    run_c = _run_cell('Acrobot-v1', 0, _VANILLA_CLAIM, _VANILLA_ARM_KEY)

    sig_a = leaf_signature(run_a.measurements)
    sig_b = leaf_signature(run_b.measurements)
    sig_c = leaf_signature(run_c.measurements)
    assert sig_a == sig_b
    assert sig_a == sig_c


def test_different_interventions_yield_different_leaf_signatures() -> None:
    """Vanilla and DDQN canonicalise to DISTINCT leaf signatures."""
    vanilla_run = _run_cell('CartPole-v1', 0, _VANILLA_CLAIM, _VANILLA_ARM_KEY)
    ddqn_run = _run_cell('CartPole-v1', 0, _DDQN_CLAIM, _DDQN_ARM_KEY)

    assert leaf_signature(vanilla_run.measurements) != \
        leaf_signature(ddqn_run.measurements)


# ============ Corpus aggregation ============

def test_runs_group_by_intervention_via_arm_key() -> None:
    """Cells with the same arm_key are groupable. Phase 6: the
    typed `arm_key` field is the canonical group identifier;
    `leaf_signature` provides the within-arm configurational
    fingerprint (one signature per HP regime within an arm)."""
    runs = [
        _run_cell('CartPole-v1', 0, _VANILLA_CLAIM, _VANILLA_ARM_KEY),
        _run_cell('CartPole-v1', 1, _VANILLA_CLAIM, _VANILLA_ARM_KEY),
        _run_cell('CartPole-v1', 0, _DDQN_CLAIM, _DDQN_ARM_KEY),
        _run_cell('CartPole-v1', 1, _DDQN_CLAIM, _DDQN_ARM_KEY),
    ]

    by_arm_key: dict[str, list[str]] = {}
    for r in runs:
        by_arm_key.setdefault(r.arm_key, []).append(r.arm_key)

    assert len(by_arm_key) == 2  # vanilla, ddqn
    for names in by_arm_key.values():
        assert len(names) == 2  # both seeds for each arm
