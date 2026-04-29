"""Tests for `run_dqn_cell` — the bridge between the `dqn`
outermost claim and the schema layer.

Verifies:
1. `run_dqn_cell` runs CartPole end-to-end and produces a
   well-formed `RunRow` whose measurements carry HP topology
   leaves + bridge result paths.
2. RunRow's measurements identify the hypothesis's intervention
   (via `intervention_name` and the HP-subset of measurements).
3. RunRow's measurements include both bridge and invariant
   classifications (`bridge.<name>.*` vs. `invariant.<name>.*`).
4. INVARIANT_VIOLATION on any bridge propagates to the run-level
   verdict (axiom 18 precedence)."""
from __future__ import annotations

from collections.abc import Mapping

import jax.numpy as jnp
import pytest

from corroborate.aggregate import leaf_signature
from corroborate.bridge import BridgeResult, bridge
from corroborate.hypothesis import Hypothesis
from corroborate.invariant import at_most
from corroborate.rl.cell_runner import run_dqn_cell
from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate.rl.dqn.invariants import (
    DQNTrajectoryRecord,
    fqi_decay_gap,
)
from corroborate.schema import RunRow
from corroborate.verdict import Verdict


# Compact HP bundle reused across cell-runner tests. Authors spread
# these into `intervention` as flat kwargs; cell runner forwards
# `**intervention` into `partial(dqn, ...)` so the intervention's
# shape mirrors `dqn`'s signature. Module-owned HPs (buffer
# capacity, batch size) live on a `Replay` instance under the
# `replay` key.
from corroborate.rl.dqn.claims.replay import Replay  # noqa: E402

# Every test runs DQN end-to-end on CartPole — ~3 s each. Skipped
# by default; opt in via `-m slow` (or `-m ''` for the full suite).
pytestmark = pytest.mark.slow

_REPLAY_SHORT = Replay(capacity=200, batch_size=16)
_OPTIMIZER_SHORT = WarmedUpdate(inner=Adam(), warmup_steps=10)
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


def _bridge_names_in(run: RunRow) -> set[str]:
    """Collect the set of bridge/invariant names from a RunRow's
    measurements — the verdict-bearing keys are
    `bridge.<name>.verdict` / `invariant.<name>.verdict`."""
    out: set[str] = set()
    for k in run.measurements:
        if k.startswith('bridge.') and k.endswith('.verdict'):
            out.add(k.removeprefix('bridge.').removesuffix('.verdict'))
        elif k.startswith('invariant.') and k.endswith('.verdict'):
            out.add(k.removeprefix('invariant.').removesuffix('.verdict'))
    return out


def _has_invariant(run: RunRow, substring: str) -> bool:
    """True iff some `invariant.<name>.verdict` key in the row's
    measurements has `substring` in `<name>`."""
    return any(
        k.startswith('invariant.') and k.endswith('.verdict')
        and substring in k.removeprefix('invariant.').removesuffix('.verdict')
        for k in run.measurements
    )


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
        
    ).run
    # RunRow shape.
    assert isinstance(run_row, RunRow)
    assert run_row.measurements['intervention_name'] == 'vanilla'
    assert run_row.measurements['env_name'] == 'CartPole-v1'
    assert run_row.measurements['seed'] == 0
    assert run_row.measurements['total_steps'] == 60
    # Empty bridges → no bridge measurements → POWER_INSUFFICIENT.
    assert _bridge_names_in(run_row) == set()
    assert run_row.verdict is Verdict.POWER_INSUFFICIENT
    # Outcome reduction landed.
    assert isinstance(run_row.measurements['outcome.late_window_mean'], float)
    # Leaf topology paths populated.
    assert 'gamma' in run_row.measurements


def test_run_dqn_cell_leaf_signature_matches_hypothesis() -> None:
    """The leaf signature derived from the RunRow's measurements
    distinguishes hypotheses by their intervention overrides."""
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
        
    ).run

    # Bridge result surfaces in measurements with bridge name.
    assert 'bridge.some_bridge.verdict' in run_row.measurements
    # Leaf signature is non-empty (configurational fingerprint).
    sig = leaf_signature(run_row.measurements)
    assert len(sig) > 0


# ============ Bridge → measurements conversion ============

def test_run_dqn_cell_classifies_invariant_facts() -> None:
    """A bridge created via `at_most(...)` has `stats['kind']=
    'tautological'` → measurements appear under
    `invariant.<name>.*`. A plain bridge has `bridge.<name>.*`."""
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
        
    ).run

    # Plain bridge under 'bridge.' prefix.
    assert 'bridge.plain_bridge.verdict' in run_row.measurements
    # Invariant under 'invariant.' prefix; name contains the
    # measurable's name (fqi_decay_gap).
    assert _has_invariant(run_row, 'fqi_decay_gap')


def test_run_dqn_cell_invariant_violation_dominates_verdict() -> None:
    """Axiom 18: INVARIANT_VIOLATION preempts NO_EFFECT/HELD at
    the cell verdict layer."""
    from corroborate.rl.dqn.claims.target_sync import periodic_copy
    from corroborate.rl.env_catalogue import get
    env_spec = get('CartPole-v1')

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
        
    ).run
    assert run_row.verdict is Verdict.INVARIANT_VIOLATION


# ============ Bridges over the merged record ============

def test_run_dqn_cell_runs_bridges_against_merged_record() -> None:
    """Hypothesis.bridges target whichever record keys they care
    about. `jensen_overestimation_gap` reads `predicted_q_at_start`
    + `mc_return` (eval-burst-shaped fields); the merged record
    carries them alongside per-step training fields."""
    from corroborate.invariant import at_most
    from corroborate.rl.dqn.claims.bootstrap import bootstrap
    from corroborate.rl.dqn.invariants import jensen_overestimation_gap
    from corroborate.rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    jensen_scope = at_most(
        jensen_overestimation_gap(),
        threshold=1e9,
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
        
    ).run
    # Invariant verdict surfaced under 'invariant.' prefix.
    assert _has_invariant(run_row, 'jensen_overestimation_gap')


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
            
        ).run
        assert not _has_invariant(run_row, 'jensen_overestimation_gap'), (
            f'max_greedify-only invariant leaked into a DDQN run; '
            f'got measurements {sorted(run_row.measurements)}'
        )
    finally:
        from corroborate.invariant import detach_invariant
        detach_invariant(only_vanilla, from_claim=max_greedify)


def test_composition_discovered_invariants_fire() -> None:
    """When a substrate author attaches an invariant to a claim,
    the cell runner auto-discovers it via composition-tree walk
    and fires it against the per-cell record — without the
    hypothesis having to list it in `bridges`."""
    from corroborate.invariant import attach_invariant, at_most
    from corroborate.rl.dqn.claims.bootstrap import bootstrap
    from corroborate.rl.dqn.invariants import jensen_overestimation_gap
    from corroborate.rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    auto_bridge = at_most(
        jensen_overestimation_gap(),
        threshold=1e9,
        of_claim=bootstrap,
    )
    attach_invariant(auto_bridge, to=bootstrap)

    try:
        h: Hypothesis[DQNTrajectoryRecord] = Hypothesis(
            name='vanilla_no_explicit_bridges',
            intervention={**_SHORT_RUN_HP_40},
            bridges=(),
            predicted_direction=None,
        )

        run_row = run_dqn_cell(
            env_spec, seed=0, hypothesis=h,
            
        ).run

        assert _has_invariant(run_row, 'jensen_overestimation_gap'), (
            f'expected auto-discovered jensen invariant; got '
            f'{sorted(run_row.measurements)}'
        )
    finally:
        from corroborate.invariant import detach_invariant
        detach_invariant(auto_bridge, from_claim=bootstrap)


def test_run_dqn_cell_applies_intervention_via_slot_swap() -> None:
    """DDQN intervention is `intervention={'bootstrap':
    partial(bootstrap, greedification=double_greedify)}`. The
    HP-subset of measurements records the bootstrap slot's
    canonicalised form."""
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
        
    ).run
    # The bootstrap HP topology path carries the canonicalised
    # form of the partial — `double_greedify` appears in it.
    bootstrap_value = run_row.measurements.get('bootstrap')
    assert isinstance(bootstrap_value, str)
    assert 'double_greedify' in bootstrap_value
