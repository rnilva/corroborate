"""Smoke test — `paired_step` (deep van Hasselt 2010, `DDQN-indp`)
runs on CartPole and trains two coupled learners over one shared
rollout + replay buffer.

Verifies the Phase-2 host (`dqn_paired.py`):
- `paired_step` runs end-to-end via `python_loop` with the same
  `(state, idx) -> (state, record)` shape as `dqn_step`.
- A (the acting unit) trains: its online params move from init.
- B (the non-acting co-learner) ALSO trains and stays independent:
  B's online params move AND differ from A's — confirming the
  second learner shares the buffer but learns its own estimator.
- The emitted record is A's per-step dict (same keys as dqn_step).
"""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import gymnax
import jax
import jax.numpy as jnp
import pytest
from gymnax import EnvParams

from corroborate_rl.dqn.claims.bootstrap import (
    bootstrap as bootstrap_claim,
    double_greedify,
)
from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
from corroborate_rl.dqn.claims.replay import Replay
from corroborate_rl.dqn.claims.q_network import MLP
from corroborate_rl.dqn.dqn_paired import (
    init_paired_state,
    paired_dqn,
    paired_step,
)
from corroborate_rl.loop import python_loop

if TYPE_CHECKING:
    from gymnax import Env


# `_make_env` is the shared CartPole builder (see tests/dqn/_helpers).
from ._helpers import make_cartpole_env as _make_env  # noqa: E402


def _max_abs_diff(a: dict[str, jax.Array], b: dict[str, jax.Array]) -> float:
    """Largest absolute element-wise difference across two param
    pytrees (flat dicts of arrays)."""
    leaves = [
        jnp.max(jnp.abs(a[k] - b[k])) for k in a
    ]
    return float(jnp.max(jnp.stack(leaves)))


@pytest.mark.slow
def test_paired_step_trains_two_independent_learners_on_cartpole() -> None:
    env, env_params, obs_shape, n_actions = _make_env()
    optimizer = warmed_update(inner=partial(adam), warmup_steps=10)
    replay = Replay(capacity=200, batch_size=16)

    init = init_paired_state(
        env=env, env_params=env_params,
        obs_shape=obs_shape, n_actions=n_actions,
        rng_key=jax.random.PRNGKey(0), optimizer=optimizer,
        replay=replay,
    )

    # B starts independent of A (distinct init keys).
    assert _max_abs_diff(init.a.online_params, init.b_online) > 0.0

    step_fn = partial(
        paired_step,
        env=env, env_params=env_params, n_actions=n_actions,
        optimizer=optimizer, sync_period=10, replay=replay,
        # Standard double-Q selector; the cross-evaluation is the
        # evaluator_params injection inside paired_step, NOT a
        # greedification marker.
        bootstrap=partial(bootstrap_claim, greedification=double_greedify),
    )

    final_state, record = python_loop(step_fn, init, length=50)

    # Shared agent advanced 50 steps.
    assert int(final_state.a.step) == 50

    # A trained: online params moved from init.
    assert _max_abs_diff(final_state.a.online_params, init.a.online_params) > 0.0
    # B trained: online params moved from init.
    assert _max_abs_diff(final_state.b_online, init.b_online) > 0.0
    # B stays an independent estimator: its online net differs from
    # A's after 50 steps of independent minibatches over the shared
    # buffer.
    assert _max_abs_diff(final_state.a.online_params, final_state.b_online) > 0.0

    # Record is A's per-step diagnostic dict (same surface as dqn_step).
    for key in ('reward', 'done', 'loss', 'td_error', 'max_q'):
        assert key in record
        assert record[key].shape[0] == 50


@pytest.mark.slow
def test_paired_dqn_full_run_produces_dqn_shaped_record() -> None:
    """The outer `paired_dqn` claim runs a full train+eval loop via
    the generic `train_with_eval` driver and returns a record with
    the SAME surface as `dqn` — training fields shape (total_steps,)
    and eval-burst fields shape (n_super_steps, n_episodes)."""
    env, env_params, obs_shape, n_actions = _make_env()
    total_steps, eval_every, n_episodes = 20, 10, 2

    record = paired_dqn.fn(
        env_name='CartPole-v1',
        env=env, env_params=env_params,
        obs_shape=obs_shape, n_actions=n_actions,
        q_network=MLP(hidden=(32,)),
        replay=Replay(capacity=200, batch_size=16),
        sync_period=10,
        total_steps=total_steps, eval_every=eval_every,
        n_episodes=n_episodes, eval_episode_cap=50,
    )

    # Training fields: (total_steps,).
    for key in ('reward', 'loss', 'td_error'):
        assert key in record
        assert record[key].shape[0] == total_steps

    # Eval-burst fields: (n_super_steps, n_episodes).
    n_super_steps = total_steps // eval_every
    assert record['predicted_q_at_start'].shape == (n_super_steps, n_episodes)
    assert record['mc_return'].shape == (n_super_steps, n_episodes)
    # Eval step indices land at the end of each super-step.
    assert list(record['eval_step_index']) == [10, 20]


@pytest.mark.slow
def test_paired_dqn_runs_through_runner_cell() -> None:
    """Integration: the substrate runner's cell path (`run_dqn_cell`,
    same entry the sweep dispatches through) invokes `paired_dqn`
    exactly as it does `dqn` — the Exogenous cell kwargs + `seed`,
    no `init_override` — builds the trajectory record, walks the
    bound leaves for the fingerprint, and records program identity
    on the typed `RunRow.program` column. End-to-end proof that the
    base-program swap is call-compatible: `paired_dqn`'s signature
    accepts every kwarg the runner injects, and its record is
    `dqn`-shaped so the leaf projection succeeds."""
    from corroborate_rl.cell_runner import run_dqn_cell
    from corroborate_rl.env_catalogue import get

    env_spec = get('CartPole-v1')
    claim = partial(
        paired_dqn,
        total_steps=20, eval_every=10, n_episodes=2, sync_period=10,
        replay=Replay(capacity=200, batch_size=16),
        q_network=MLP(hidden=(8,)),
    )

    cell = run_dqn_cell(
        env_spec, seed=0, claim=claim,
        arm_key='baseline', measurables=(), cell_idx=0,
    )

    # Program identity is the typed RunRow.program column — derived
    # by the runner from the ROOT claim that actually ran (via
    # signature.root_claim_name on the composed partial), NOT a
    # passed-in string and NOT smuggled into arm_key. arm_key stays
    # the pure intervention fingerprint ('baseline' for the empty arm).
    assert cell.run.program == 'paired_dqn'
    assert cell.run.arm_key == 'baseline'
    # walk_paths surfaced the bound leaves (gamma + replay.batch_size
    # are leaves on paired_dqn, mirroring dqn) — the leaf fingerprint
    # projection succeeded on partial(paired_dqn).
    assert 'gamma' in cell.run.measurements
    assert cell.run.measurements['replay.batch_size'] == 16
