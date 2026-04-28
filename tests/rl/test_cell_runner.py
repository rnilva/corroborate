"""Tests for `run_dqn_cell` — the bridge between the `dqn`
outermost claim and the schema layer.

Verifies:
1. `run_dqn_cell` runs CartPole end-to-end and produces a
   well-formed `RunRow` whose record carries both training fields
   (per-step) and eval fields (per-burst).
2. RunRow's `mechanism_key` matches the hypothesis's.
3. RunRow's `facts` includes both bridge and invariant
   classifications, derived from `stats['kind']`.
4. INVARIANT_VIOLATION on any fact propagates to the run-level
   verdict (axiom 18 precedence)."""
from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import optax

from corroborate.bridge import BridgeResult, bridge
from corroborate.hypothesis import Hypothesis
from corroborate.invariant import at_most
from corroborate.rl.cell_runner import run_dqn_cell
from corroborate.rl.dqn.invariants import (
    DQNTrajectoryRecord,
    fqi_decay_gap,
)
from corroborate.schema import RunRow
from corroborate.verdict import Verdict


# Compact HP bundle reused across cell-runner tests. Authors spread
# these into `intervention` as flat kwargs; cell runner forwards
# `**intervention` into `partial(dqn, ...)` so the intervention's
# shape mirrors `dqn`'s signature.
_SHORT_RUN_HP: dict[str, object] = {
    'total_steps': 60, 'eval_every': 30, 'n_episodes': 2,
    'warmup_steps': 10, 'sync_period': 10,
    'buffer_capacity': 200, 'batch_size': 16,
}
_SHORT_RUN_HP_40: dict[str, object] = {
    'total_steps': 40, 'eval_every': 20, 'n_episodes': 2,
    'warmup_steps': 10, 'sync_period': 10,
    'buffer_capacity': 200, 'batch_size': 16,
}


# ============ run_dqn_cell — happy path ============

def test_run_dqn_cell_produces_runrow_on_cartpole() -> None:
    """End-to-end smoke: run vanilla DQN on CartPole for 60
    steps with one eval burst at step 30 and another at 60."""
    from corroborate.rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    h = Hypothesis[DQNTrajectoryRecord](
        name='vanilla',
        intervention={**_SHORT_RUN_HP},  # HPs only, no slot swaps
        bridges=(),
        predicted_direction=None,
    )

    run_row = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
        optimizer=optax.adam(1e-3),
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
    # Record keys include both training fields and eval fields
    # (cell runner produces ONE merged record).
    assert 'max_q' in run_row.record_keys
    assert 'predicted_q_at_start' in run_row.record_keys
    assert 'mc_return' in run_row.record_keys
    assert 'eval_step_index' in run_row.record_keys


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
        intervention={**_SHORT_RUN_HP_40},
        bridges=(some_bridge,),
        predicted_direction='a_gt_b',
    )

    run_row = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
        optimizer=optax.adam(1e-3),
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
        intervention={**_SHORT_RUN_HP_40},
        bridges=(plain_bridge, invariant_bridge),
    )

    run_row = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
        optimizer=optax.adam(1e-3),
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
        intervention={**_SHORT_RUN_HP_40},
        bridges=(held_bridge, impossible),
    )

    run_row = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
        optimizer=optax.adam(1e-3),
    )
    # held_bridge=HELD; impossible=INVARIANT_VIOLATION → cell-
    # verdict is INVARIANT_VIOLATION.
    assert run_row.verdict is Verdict.INVARIANT_VIOLATION


# ============ Bridges over the merged record ============

def test_run_dqn_cell_runs_bridges_against_merged_record() -> None:
    """Hypothesis.bridges target whichever record keys they care
    about. `jensen_overestimation_gap` reads `predicted_q_at_start`
    + `mc_return` (eval-burst-shaped fields); the merged record
    carries them alongside per-step training fields. No
    `eval_bridges` distinction in framework code — the cell
    runner produces ONE record dict, bridges pick keys."""
    from corroborate.invariant import at_most
    from corroborate.rl.dqn.claims.bootstrap import bootstrap
    from corroborate.rl.dqn.invariants import jensen_overestimation_gap
    from corroborate.rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    jensen_scope = at_most(
        jensen_overestimation_gap(),
        threshold=1e9,  # generous; expected to HELD on a smoke run
        of_claim=bootstrap,
    )

    h: Hypothesis[DQNTrajectoryRecord] = Hypothesis(
        name='vanilla_with_jensen_scope',
        intervention={**_SHORT_RUN_HP_40},
        bridges=(jensen_scope,),
        predicted_direction=None,
    )

    run_row = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
        optimizer=optax.adam(1e-3),
    )
    assert len(run_row.facts) == 1
    fact = run_row.facts[0]
    assert fact.kind == 'invariant'
    assert 'jensen_overestimation_gap' in fact.name
    # The bridge name appears in mechanism_key.bridge_names —
    # no '@E:' prefix nor any train/eval distinction.
    assert any(
        'jensen_overestimation_gap' in n
        for n in run_row.mechanism_key.bridge_names
    )


def test_intervention_overrides_dont_leak_default_invariants() -> None:
    """The cell runner uses `partial(dqn, **intervention)` to
    compute the EFFECTIVE composition. `collect_invariants` walks
    the partial-aware tree, so an invariant attached to
    `max_greedify` does NOT fire when the intervention swaps the
    greedification slot to `double_greedify` (DDQN)."""
    from functools import partial
    from corroborate.invariant import attach_invariant, at_most
    from corroborate.rl.dqn.claims.bootstrap import (
        bootstrap,
        double_greedify,
        max_greedify,
    )
    from corroborate.rl.dqn.invariants import jensen_overestimation_gap
    from corroborate.rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    # Attach an invariant only to max_greedify (vanilla's value
    # computation). DDQN swaps to double_greedify, so this
    # invariant must NOT fire under DDQN intervention.
    only_vanilla = at_most(
        jensen_overestimation_gap(),
        threshold=1e9,
        of_claim=max_greedify,
    )
    attach_invariant(only_vanilla, to=max_greedify)

    try:
        h = Hypothesis[DQNTrajectoryRecord](
            name='ddqn_no_leak',
            intervention={
                **_SHORT_RUN_HP_40,
                'bootstrap': partial(
                    bootstrap, greedification=double_greedify,
                ),
            },
            bridges=(),
        )
        run_row = run_dqn_cell(
            env_spec, seed=0, hypothesis=h,
            optimizer=optax.adam(1e-3),
        )
        names = [f.name for f in run_row.facts]
        assert not any('jensen_overestimation_gap' in n for n in names), (
            f'max_greedify-only invariant leaked into a DDQN run; '
            f'got {names}'
        )
    finally:
        from corroborate.invariant import detach_invariant
        detach_invariant(only_vanilla, from_claim=max_greedify)


def test_composition_discovered_invariants_fire() -> None:
    """When a substrate author attaches an invariant to a claim
    (`@invariant(of=...)` or `attach_invariant`), the cell runner
    auto-discovers it via composition-tree walk and fires it
    against the per-cell record — without the hypothesis having to
    list it in `bridges`."""
    from corroborate.bridge import BridgeResult
    from corroborate.invariant import attach_invariant, at_most
    from corroborate.rl.dqn.claims.bootstrap import bootstrap
    from corroborate.rl.dqn.invariants import jensen_overestimation_gap
    from corroborate.rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    # Substrate-attached: build a tautological bridge and attach
    # to a default sub-claim of dqn (`bootstrap` is dqn's default,
    # with default `greedification=max_greedify` = vanilla DQN).
    auto_bridge = at_most(
        jensen_overestimation_gap(),
        threshold=1e9,  # generous; expected HELD on smoke
        of_claim=bootstrap,
    )
    attach_invariant(auto_bridge, to=bootstrap)

    try:
        h: Hypothesis[DQNTrajectoryRecord] = Hypothesis(
            name='vanilla_no_explicit_bridges',
            intervention={**_SHORT_RUN_HP_40},
            bridges=(),  # author declares NO bridges
            predicted_direction=None,
        )

        run_row = run_dqn_cell(
            env_spec, seed=0, hypothesis=h,
            optimizer=optax.adam(1e-3),
        )

        # Composition-discovery should have surfaced the
        # vanilla_bootstrap-attached invariant.
        names = [f.name for f in run_row.facts]
        assert any('jensen_overestimation_gap' in n for n in names), (
            f'expected auto-discovered jensen invariant in facts, got {names}'
        )
    finally:
        from corroborate.invariant import detach_invariant
        detach_invariant(auto_bridge, from_claim=bootstrap)


def test_run_dqn_cell_applies_intervention_via_slot_swap() -> None:
    """DDQN intervention is `intervention={'bootstrap':
    partial(bootstrap, greedification=double_greedify)}`. The
    cell runner spreads `**intervention` into `partial(dqn, ...)`."""
    from functools import partial
    from corroborate.rl.dqn.claims.bootstrap import (
        bootstrap, double_greedify,
    )
    from corroborate.rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    h = Hypothesis[DQNTrajectoryRecord](
        name='ddqn',
        intervention={
            **_SHORT_RUN_HP_40,
            'bootstrap': partial(bootstrap, greedification=double_greedify),
        },
        bridges=(),
    )

    run_row = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
        optimizer=optax.adam(1e-3),
    )
    # Intervention identity is preserved on RunRow — the partial
    # canonicalises with the wrapped claim's name + baked kwargs.
    sig = dict(run_row.mechanism_key.intervention_signature)
    assert 'bootstrap' in sig
    assert 'double_greedify' in sig['bootstrap']
