"""Smoke test for the configurational-corpus path (Step 4.5).

Verifies the end-to-end plumbing for §4.3 PC-discovery node
identity: cells with the same hypothesis produce stable leaf
signatures across (env, seed); different hypotheses produce
distinct signatures; aggregation groups RunRows correctly.

This is the prerequisite for using `arm_ddqn` as a binary
intervention variable in PC graphs (PAPER_NOTES.md §4.3) — every
cell of the same intervention must canonicalise identically. The
v9-`MechanismKey` artifact has been retired in favour of
`leaf_signature(measurements)` projecting directly off the runs."""
from __future__ import annotations

from functools import partial

import pytest

from corroborate.aggregate import aggregate_runs, leaf_signature
from corroborate.hypothesis import Hypothesis
from corroborate.rl.cell_runner import run_dqn_cell
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import get

# Whole module is end-to-end DQN smoke — multiple cells per test;
# averaging ~6 s each. Skipped by default; opt in via `-m slow`.
pytestmark = pytest.mark.slow


_DDQN_BOOTSTRAP = partial(bootstrap, greedification=double_greedify)


from corroborate.rl.dqn.claims.replay import Replay

_SHORT_RUN_HP: dict[str, object] = {
    'total_steps': 40, 'eval_every': 20, 'n_episodes': 2,
    'sync_period': 10,
    'replay': Replay(capacity=200, batch_size=16),
    'optimizer': WarmedUpdate(inner=Adam(), warmup_steps=10),
}


def _make_hypothesis(
    name: str, intervention: dict[str, object],
) -> Hypothesis[DQNTrajectoryRecord]:
    return Hypothesis[DQNTrajectoryRecord](
        name=name,
        intervention={**_SHORT_RUN_HP, **intervention},
        bridges=(),
    )


def _run_cell(env_name: str, seed: int, h: Hypothesis[DQNTrajectoryRecord]):
    return run_dqn_cell(
        get(env_name), seed=seed, hypothesis=h,
        optimizer=Adam(),
    )


# ============ Stable leaf_signature across (env, seed) ============

def test_same_hypothesis_yields_stable_leaf_signature() -> None:
    """Two cells of vanilla DQN — different seeds, different envs
    — must produce the SAME `leaf_signature` (the §4.3 PC-discovery
    node-identity primitive). The signature filters out
    `env_name` and `seed`, so changes there don't perturb it."""
    vanilla = _make_hypothesis('vanilla', intervention={})

    run_a = _run_cell('CartPole-v1', seed=0, h=vanilla)
    run_b = _run_cell('CartPole-v1', seed=1, h=vanilla)
    run_c = _run_cell('Acrobot-v1', seed=0, h=vanilla)

    sig_a = leaf_signature(run_a.measurements)
    sig_b = leaf_signature(run_b.measurements)
    sig_c = leaf_signature(run_c.measurements)
    assert sig_a == sig_b
    assert sig_a == sig_c


def test_different_interventions_yield_different_leaf_signatures() -> None:
    """Vanilla and DDQN canonicalise to DISTINCT leaf signatures."""
    vanilla = _make_hypothesis('vanilla', intervention={})
    ddqn = _make_hypothesis(
        'ddqn', intervention={'bootstrap': _DDQN_BOOTSTRAP},
    )

    vanilla_run = _run_cell('CartPole-v1', seed=0, h=vanilla)
    ddqn_run = _run_cell('CartPole-v1', seed=0, h=ddqn)

    assert leaf_signature(vanilla_run.measurements) != \
        leaf_signature(ddqn_run.measurements)


# ============ Corpus aggregation ============

def test_aggregate_runs_groups_by_intervention_and_env() -> None:
    """`aggregate_runs` collapses cells into ArmRows keyed by
    (intervention_name, env_name, leaf_signature). Two seeds of
    vanilla on CartPole + two seeds of DDQN on CartPole produces
    2 ArmRows."""
    vanilla = _make_hypothesis('vanilla', intervention={})
    ddqn = _make_hypothesis(
        'ddqn', intervention={'bootstrap': _DDQN_BOOTSTRAP},
    )

    runs = [
        _run_cell('CartPole-v1', seed=0, h=vanilla),
        _run_cell('CartPole-v1', seed=1, h=vanilla),
        _run_cell('CartPole-v1', seed=0, h=ddqn),
        _run_cell('CartPole-v1', seed=1, h=ddqn),
    ]
    arms = aggregate_runs(runs)

    # Two arms: one per intervention.
    assert len(arms) == 2
    arm_names: set[str] = set()
    for arm in arms:
        v = arm.measurements['intervention_name']
        assert isinstance(v, str)
        arm_names.add(v)
    assert arm_names == {'vanilla', 'ddqn'}

    # Each arm has both seeds.
    for arm in arms:
        assert arm.measurements['n'] == 2
        assert arm.measurements['env_name'] == 'CartPole-v1'
