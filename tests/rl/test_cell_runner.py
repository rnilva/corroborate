"""Tests for `run_dqn_cell` — the bridge between the DQN
substrate and the schema layer.

Verifies:
1. `EvalConfig.n_evals` constructs eval-loop schedules that
   align cleanly with `total_steps`.
2. `run_dqn_cell` runs CartPole end-to-end and produces a
   well-formed `RunRow` + `EvalTrajectoryRecord`.
3. RunRow's `mechanism_key` matches the hypothesis's.
4. RunRow's `facts` includes both bridge and invariant
   classifications, derived from `stats['kind']`.
5. INVARIANT_VIOLATION on any fact propagates to the run-level
   verdict (axiom 18 precedence)."""
from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import optax

from corroborate.bridge import BridgeResult, bridge
from corroborate.hypothesis import Hypothesis
from corroborate.invariant import at_most
from corroborate.rl.cell_runner import EvalConfig, run_dqn_cell
from corroborate.rl.dqn.claims.q_network import mlp_q
from corroborate.rl.dqn.invariants import (
    DQNTrajectoryRecord,
    fqi_decay_gap,
)
from corroborate.schema import RunRow
from corroborate.verdict import Verdict


# ============ EvalConfig ============

def test_eval_config_n_evals_factory_evenly_spaces() -> None:
    cfg = EvalConfig.n_evals(total_steps=100, n_evals=5)
    assert cfg.eval_every == 20
    assert cfg.n_episodes == 20  # default


def test_eval_config_n_evals_custom_n_episodes() -> None:
    cfg = EvalConfig.n_evals(total_steps=200, n_evals=4, n_episodes=10)
    assert cfg.eval_every == 50
    assert cfg.n_episodes == 10


def test_eval_config_n_evals_rejects_n_evals_larger_than_steps() -> None:
    try:
        EvalConfig.n_evals(total_steps=10, n_evals=20)
        raise AssertionError('expected ValueError')
    except ValueError:
        pass


# ============ run_dqn_cell — happy path ============

def test_run_dqn_cell_produces_runrow_on_cartpole() -> None:
    """End-to-end smoke: run vanilla DQN on CartPole for 60
    steps with one eval burst at step 30 and another at 60."""
    from corroborate.rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    h = Hypothesis[DQNTrajectoryRecord](
        name='vanilla',
        intervention={},  # no slot swaps — vanilla DQN
        bridges=(),
        predicted_direction=None,
    )

    run_row, eval_record = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
        total_steps=60,
        optimizer=optax.adam(1e-3),
        eval_config=EvalConfig(eval_every=30, n_episodes=2),
        warmup_steps=10, sync_period=10,
        buffer_capacity=200, batch_size=16,
    )
    # RunRow shape.
    assert isinstance(run_row, RunRow)
    assert run_row.intervention_name == 'vanilla'
    assert run_row.env_name == 'CartPole-v1'
    assert run_row.seed == 0
    assert run_row.total_steps == 60
    # Empty bridges → no facts → POWER_INSUFFICIENT.
    assert run_row.facts == ()
    assert run_row.verdict is Verdict.POWER_INSUFFICIENT
    assert isinstance(run_row.primary_outcome_summary, float)

    # Eval record shape.
    assert eval_record['predicted_q_at_start'].shape == (2, 2)
    assert eval_record['mc_return'].shape == (2, 2)
    assert eval_record['eval_step_index'].shape == (2,)


def test_run_dqn_cell_mechanism_key_matches_hypothesis() -> None:
    """The RunRow's mechanism_key is the hypothesis's verbatim —
    intervention_signature, bridge_names, direction all match."""
    from corroborate.rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    @bridge(targets=('ep_return',))
    def some_bridge(record: Mapping[str, jnp.ndarray]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='ok', stats={},
            name='some_bridge', targets=('ep_return',),
        )

    h = Hypothesis[DQNTrajectoryRecord](
        name='ddqn',
        intervention={},
        bridges=(some_bridge,),
        predicted_direction='a_gt_b',
    )

    run_row, _eval = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
        total_steps=40,
        optimizer=optax.adam(1e-3),
        eval_config=EvalConfig(eval_every=20, n_episodes=2),
        warmup_steps=10, sync_period=10,
        buffer_capacity=200, batch_size=16,
    )

    # mechanism_key carries the hypothesis's exact identity.
    assert run_row.mechanism_key == h.mechanism_key
    assert run_row.mechanism_key.direction == 'a_gt_b'
    assert 'some_bridge' in run_row.mechanism_key.bridge_names


# ============ Bridge → FactRow conversion ============

def test_run_dqn_cell_classifies_invariant_facts() -> None:
    """A bridge created via `at_most(...)` has `stats['kind']=
    'tautological'` → FactRow.kind='invariant'. A plain bridge
    has FactRow.kind='bridge'."""
    from corroborate.rl.dqn.claims.target_sync import periodic_copy
    from corroborate.rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    @bridge(targets=('ep_return',))
    def plain_bridge(record: Mapping[str, jnp.ndarray]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='ok', stats={},
            name='plain_bridge', targets=('ep_return',),
        )

    invariant_bridge = at_most(
        fqi_decay_gap(sync_period=10, gamma=0.99),
        threshold=10.0,
        of_claim=periodic_copy,
    )

    h = Hypothesis[DQNTrajectoryRecord](
        name='mixed',
        intervention={},
        bridges=(plain_bridge, invariant_bridge),
    )

    run_row, _ = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
        total_steps=40,
        optimizer=optax.adam(1e-3),
        eval_config=EvalConfig(eval_every=20, n_episodes=2),
        warmup_steps=10, sync_period=10,
        buffer_capacity=200, batch_size=16,
    )

    kinds = {f.name: f.kind for f in run_row.facts}
    assert kinds.get('plain_bridge') == 'bridge'
    # at_most's name uses 'fqi_decay_gap' from the measurable
    inv_facts = [f for f in run_row.facts if f.kind == 'invariant']
    assert len(inv_facts) == 1
    assert 'fqi_decay_gap' in inv_facts[0].name


def test_run_dqn_cell_invariant_violation_dominates_verdict() -> None:
    """Axiom 18: INVARIANT_VIOLATION preempts NO_EFFECT/HELD at
    the cell verdict layer."""
    from corroborate.rl.dqn.claims.target_sync import periodic_copy
    from corroborate.rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    # A bridge that always returns INVARIANT_VIOLATION via at_most
    # with an impossibly-tight threshold.
    impossible = at_most(
        fqi_decay_gap(sync_period=10, gamma=0.99),
        threshold=-1.0,  # gap is non-negative; can never be ≤ -1
        of_claim=periodic_copy,
    )

    @bridge(targets=('ep_return',))
    def held_bridge(record: Mapping[str, jnp.ndarray]) -> BridgeResult:
        del record
        return BridgeResult(
            verdict=Verdict.HELD, reason='ok', stats={},
            name='held_bridge', targets=('ep_return',),
        )

    h = Hypothesis[DQNTrajectoryRecord](
        name='mixed',
        intervention={},
        bridges=(held_bridge, impossible),
    )

    run_row, _ = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
        total_steps=40,
        optimizer=optax.adam(1e-3),
        eval_config=EvalConfig(eval_every=20, n_episodes=2),
        warmup_steps=10, sync_period=10,
        buffer_capacity=200, batch_size=16,
    )
    # held_bridge=HELD; impossible=INVARIANT_VIOLATION → cell-
    # verdict is INVARIANT_VIOLATION.
    assert run_row.verdict is Verdict.INVARIANT_VIOLATION


# ============ DDQN intervention via slot swap ============

# ============ Eval-bridge wiring ============

def test_run_dqn_cell_runs_eval_bridges_against_eval_record() -> None:
    """Hypothesis.eval_bridges target the eval record (predicted
    Q vs MC return); cell runner threads them through trace.eval
    and produces FactRows alongside train-bridge facts."""
    from corroborate.invariant import at_most
    from corroborate.rl.dqn.claims.bootstrap import vanilla_bootstrap
    from corroborate.rl.dqn.invariants import jensen_overestimation_gap
    from corroborate.rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    jensen_scope = at_most(
        jensen_overestimation_gap(),
        threshold=1e9,  # generous; expected to HELD on a smoke run
        of_claim=vanilla_bootstrap,
    )

    h: Hypothesis[DQNTrajectoryRecord, Mapping[str, object]] = Hypothesis(
        name='vanilla_with_jensen_scope',
        intervention={},
        bridges=(),
        eval_bridges=(jensen_scope,),
        predicted_direction=None,
    )

    run_row, _eval = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
        total_steps=40,
        optimizer=optax.adam(1e-3),
        eval_config=EvalConfig(eval_every=20, n_episodes=2),
        warmup_steps=10, sync_period=10,
        buffer_capacity=200, batch_size=16,
    )
    # Exactly one fact (the jensen scope).
    assert len(run_row.facts) == 1
    fact = run_row.facts[0]
    assert fact.kind == 'invariant'
    assert 'jensen_overestimation_gap' in fact.name
    # The eval-bridge name appears in mechanism_key.bridge_names.
    assert any(
        'jensen_overestimation_gap' in n
        for n in run_row.mechanism_key.bridge_names
    )


def test_run_dqn_cell_applies_intervention_via_slot_swap() -> None:
    """DDQN intervention is `intervention={'bootstrap':
    ddqn_bootstrap}`. The cell runner must apply this through
    `partial(dqn_step, **intervention)`."""
    from corroborate.rl.dqn.claims.bootstrap import ddqn_bootstrap
    from corroborate.rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    h = Hypothesis[DQNTrajectoryRecord](
        name='ddqn',
        intervention={'bootstrap': ddqn_bootstrap},
        bridges=(),
    )

    run_row, _ = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
        total_steps=40,
        optimizer=optax.adam(1e-3),
        eval_config=EvalConfig(eval_every=20, n_episodes=2),
        q_network=mlp_q,
        warmup_steps=10, sync_period=10,
        buffer_capacity=200, batch_size=16,
    )
    # Intervention identity is preserved on RunRow.
    sig = dict(run_row.mechanism_key.intervention_signature)
    assert 'bootstrap' in sig
    assert 'ddqn_bootstrap' in sig['bootstrap']
