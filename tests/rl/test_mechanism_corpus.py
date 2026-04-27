"""Smoke test for the mechanism-marker corpus path (Step 4.5).

Verifies the end-to-end plumbing for §4.3 PC-discovery node
identity: cells with the same hypothesis produce stable
`InterventionKey`s across (env, seed); different hypotheses
produce distinct keys; aggregation groups RunRows correctly.

This is the prerequisite for using `arm_ddqn` as a binary
intervention variable in PC graphs (PAPER_NOTES.md §4.3) — every
cell of the same intervention must canonicalise identically."""
from __future__ import annotations

import optax

from corroborate.aggregate import aggregate_runs
from corroborate.hypothesis import Hypothesis, InterventionKey
from corroborate.rl.cell_runner import EvalConfig, run_dqn_cell
from corroborate.rl.dqn.claims.bootstrap import ddqn_bootstrap
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import get


def _make_hypothesis(name: str, intervention: dict[str, object]) -> Hypothesis[DQNTrajectoryRecord]:
    return Hypothesis[DQNTrajectoryRecord](
        name=name,
        intervention=intervention,
        bridges=(),
    )


# Common cell config for these smoke tests.
def _run_cell(env_name: str, seed: int, h: Hypothesis[DQNTrajectoryRecord]):
    return run_dqn_cell(
        get(env_name), seed=seed, hypothesis=h,
        total_steps=40,
        optimizer=optax.adam(1e-3),
        eval_config=EvalConfig(eval_every=20, n_episodes=2),
        warmup_steps=10, sync_period=10,
        buffer_capacity=200, batch_size=16,
    )


# ============ Stable InterventionKey across (env, seed) ============

def test_same_hypothesis_yields_stable_intervention_key() -> None:
    """Two cells of vanilla DQN — different seeds, different envs
    — must produce the SAME `InterventionKey` (the §4.3 PC-
    discovery node-identity primitive)."""
    vanilla = _make_hypothesis('vanilla', intervention={})

    run_a, _ = _run_cell('CartPole-v1', seed=0, h=vanilla)
    run_b, _ = _run_cell('CartPole-v1', seed=1, h=vanilla)
    run_c, _ = _run_cell('Acrobot-v1', seed=0, h=vanilla)

    key_a = run_a.mechanism_key.intervention_only()
    key_b = run_b.mechanism_key.intervention_only()
    key_c = run_c.mechanism_key.intervention_only()
    assert key_a == key_b
    assert key_a == key_c
    assert isinstance(key_a, InterventionKey)


def test_different_interventions_yield_different_keys() -> None:
    """Vanilla and DDQN must canonicalise to DISTINCT
    InterventionKeys — otherwise PC-discovery couldn't tell
    them apart."""
    vanilla = _make_hypothesis('vanilla', intervention={})
    ddqn = _make_hypothesis(
        'ddqn', intervention={'bootstrap': ddqn_bootstrap},
    )

    vanilla_run, _ = _run_cell('CartPole-v1', seed=0, h=vanilla)
    ddqn_run, _ = _run_cell('CartPole-v1', seed=0, h=ddqn)

    assert vanilla_run.mechanism_key.intervention_only() != \
        ddqn_run.mechanism_key.intervention_only()


# ============ Corpus aggregation by mechanism_key ============

def test_aggregate_runs_groups_by_intervention_and_env() -> None:
    """`aggregate_runs` collapses cells into ArmRows keyed by
    (intervention_name, env_name, mechanism_key). Two seeds of
    vanilla on CartPole + two seeds of DDQN on CartPole produces
    2 ArmRows."""
    vanilla = _make_hypothesis('vanilla', intervention={})
    ddqn = _make_hypothesis(
        'ddqn', intervention={'bootstrap': ddqn_bootstrap},
    )

    runs = [
        _run_cell('CartPole-v1', seed=0, h=vanilla)[0],
        _run_cell('CartPole-v1', seed=1, h=vanilla)[0],
        _run_cell('CartPole-v1', seed=0, h=ddqn)[0],
        _run_cell('CartPole-v1', seed=1, h=ddqn)[0],
    ]
    arms = aggregate_runs(runs)

    # Two arms: one per intervention.
    assert len(arms) == 2
    arm_names = {a.intervention_name for a in arms}
    assert arm_names == {'vanilla', 'ddqn'}

    # Each arm has both seeds.
    for arm in arms:
        assert arm.n == 2
        assert set(arm.seeds) == {0, 1}
        assert arm.env_name == 'CartPole-v1'

    # The two arms have distinct InterventionKeys.
    arm_keys = {a.mechanism_key.intervention_only() for a in arms}
    assert len(arm_keys) == 2
