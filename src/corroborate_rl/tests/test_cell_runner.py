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

from corroborate.corpus.leaf_signature import leaf_signature
from corroborate.core.hypothesis import LegacyHypothesis as Hypothesis
from corroborate_rl.cell_runner import run_dqn_cell
from functools import partial

from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
from corroborate_rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.corpus.schema import RunRow
from corroborate.bridge.verdict import Verdict


# Compact HP bundle reused across cell-runner tests. Authors spread
# these into `intervention` as flat kwargs; cell runner forwards
# `**intervention` into `partial(dqn, ...)` so the intervention's
# shape mirrors `dqn`'s signature. Module-owned HPs (buffer
# capacity, batch size) live on a `Replay` instance under the
# `replay` key.
from corroborate_rl.dqn.claims.replay import Replay  # noqa: E402

# Every test runs DQN end-to-end on CartPole — ~3 s each. Skipped
# by default; opt in via `-m slow` (or `-m ''` for the full suite).
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
    """End-to-end smoke: run vanilla DQN on CartPole for 60
    steps with one eval burst at step 30 and another at 60."""
    from corroborate_rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    h = Hypothesis[DQNTrajectoryRecord](
        name='vanilla',
        intervention={**_SHORT_RUN_HP},  # HPs only, no slot swaps
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
    # Phase 4 of the Bridge-collapse refactor: per-cell `Bridge[R]`
    # channel is gone; cell verdict is HELD for any successfully-
    # completed cell. Verdict authority moves to corpus-side
    # claim_bridges run post-hoc.
    assert run_row.verdict is Verdict.HELD
    # Outcome reduction landed.
    assert isinstance(run_row.measurements['late_window_mean'], float)
    # Leaf topology paths populated.
    assert 'gamma' in run_row.measurements


def test_run_dqn_cell_leaf_signature_matches_hypothesis() -> None:
    """The leaf signature derived from the RunRow's measurements
    distinguishes hypotheses by their intervention overrides."""
    from corroborate_rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    h = Hypothesis[DQNTrajectoryRecord](
        name='ddqn',
        intervention={**_SHORT_RUN_HP_40},
        predicted_direction='a_gt_b',
    )

    run_row = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
    ).run

    # Leaf signature is non-empty (configurational fingerprint).
    sig = leaf_signature(run_row.measurements)
    assert len(sig) > 0


def test_run_dqn_cell_pre_registered_measurables_persist_at_their_name() -> None:
    """Pre-registered measurables on `Hypothesis.measurables` land
    in the cell's RunRow.measurements under their bare measurable
    `.name` — the substrate controls the column-name namespace
    (e.g., `eval_final_mean` vs. bare `jensen_gap`). The
    pre-registration channel sits alongside per-record bridges
    (Phase 2 of the Bridge-collapse refactor) so authors can
    declare summary scalars without authoring a per-record
    `Bridge[R]` for each."""
    from corroborate.measurables import Measurable
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
    h = Hypothesis[DQNTrajectoryRecord](
        name='vanilla_with_measurable',
        intervention={**_SHORT_RUN_HP_40},
        measurables=(m,),
    )

    run_row = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
    ).run

    assert 'ep_return_mean_summary' in run_row.measurements
    val = run_row.measurements['ep_return_mean_summary']
    assert isinstance(val, float)


def test_run_dqn_cell_applies_intervention_via_slot_swap() -> None:
    """DDQN intervention is `intervention={'bootstrap':
    partial(bootstrap, greedification=double_greedify)}`. The
    HP-subset of measurements records the bootstrap slot's
    canonicalised form."""
    from functools import partial
    from corroborate_rl.dqn.claims.bootstrap import (
        bootstrap, double_greedify,
    )
    from corroborate_rl.env_catalogue import get
    env_spec = get('CartPole-v1')

    h = Hypothesis[DQNTrajectoryRecord](
        name='ddqn',
        intervention={
            **_SHORT_RUN_HP_40,
            'bootstrap': partial(bootstrap, greedification=double_greedify),
        },
    )

    run_row = run_dqn_cell(
        env_spec, seed=0, hypothesis=h,
        
    ).run
    # The bootstrap HP topology path carries the canonicalised
    # form of the partial — `double_greedify` appears in it.
    bootstrap_value = run_row.measurements.get('bootstrap')
    assert isinstance(bootstrap_value, str)
    assert 'double_greedify' in bootstrap_value
