"""End-to-end smoke for the raw-trace store against a real DQN run.

Builds `TraceRow`s from a tiny CartPole DQN run, persists to
parquet, reads back, and verifies:

1. HP paths from `walk_paths(walk(configured), regime='hp')`
   appear as scalar leaves at dotted topology paths
   (`gamma`, `optimizer.inner.lr`, `replay.batch_size`).
2. Per-step trajectories from the configured-claim's record dict
   appear as `list[float]` leaves at flat author-chosen keys
   (`reward`, `loss`, `td_error`, ...).
3. The two namespaces (dotted HP paths vs. flat trajectory keys)
   coexist in one parquet without collision.
4. Round-trip through parquet preserves both shapes.

This is the storage-layer smoke for Step 1. Step 2 will integrate
this into `cell_runner.run_dqn_arm` and link it to RunRow.id."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import gymnax
import jax
import jax.numpy as jnp
import pytest

from corroborate.hypothesis import _canonical_str
from corroborate.persistence import read_tracerows, write_tracerows
from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate.rl.dqn.dqn import default_state_hash, dqn
from corroborate.rl.env_catalogue import HasN, HasShape
from corroborate.schema import TraceLeaf, TraceRow
from corroborate.signature import KwargInfo, walk, walk_paths

# Real DQN rollouts on CartPole — both tests run training to
# completion; ~6 s each. Skipped by default; opt in via `-m slow`.
pytestmark = pytest.mark.slow


def _hp_scalar(value: object) -> str | int | float | bool:
    """Coerce an HP value to a scalar trace leaf. Primitives pass
    through; structured values (Modules, partials, FnClaims)
    canonicalise to string."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return _canonical_str(value)


def _hp_leaves(configured: object) -> dict[str, str | int | float | bool]:
    """Topology walk → dotted-path HP leaves dict. Each KwargInfo's
    default contributes one leaf at its dotted path."""
    paths: dict[str, KwargInfo] = walk_paths(walk(configured), regime='hp')
    return {path: _hp_scalar(kw.default) for path, kw in paths.items()}


def _trajectory_leaves(
    record: Mapping[str, jax.Array],
) -> dict[str, list[float]]:
    """1-D record entries → `list[float]` trace leaves keyed by
    the author-chosen record key. Non-1-D entries are skipped
    (v0 trace store covers scalars + 1-D only)."""
    out: dict[str, list[float]] = {}
    for key, arr in record.items():
        if arr.ndim == 1:
            out[key] = [float(v) for v in arr]
    return out


def test_trace_row_round_trip_for_real_dqn_run(tmp_path: Path) -> None:
    env, env_params = gymnax.make('CartPole-v1')
    obs_space = env.observation_space(env_params)
    act_space = env.action_space(env_params)
    assert isinstance(obs_space, HasShape)
    assert isinstance(act_space, HasN)
    obs_dim = int(obs_space.shape[0])
    n_actions = int(act_space.n)

    # Configure dqn with a non-default optimizer to verify HPs are
    # captured at depth (optimizer.inner.lr).
    configured = partial(
        dqn,
        env=env, env_params=env_params,
        obs_dim=obs_dim, n_actions=n_actions,
        eval_episode_cap=200,
        state_hash=default_state_hash,
        total_steps=200, eval_every=100, n_episodes=2,
        optimizer=WarmedUpdate(inner=Adam(lr=2e-3), warmup_steps=50),
    )

    record = configured(rng_key=jax.random.PRNGKey(0))

    # Step 1: HP leaves from the topology walk.
    hp_leaves = _hp_leaves(configured)
    assert hp_leaves['gamma'] == 0.99
    # WarmedUpdate.inner.Adam.lr — captured at the deep dotted path.
    assert hp_leaves['optimizer.inner.lr'] == 2e-3
    assert hp_leaves['optimizer.warmup_steps'] == 50

    # Step 2: trajectory leaves from the record dict.
    traj_leaves = _trajectory_leaves(record)
    # Per-step training fields are 1-D shape (total_steps,).
    assert 'reward' in traj_leaves
    assert 'loss' in traj_leaves
    assert len(traj_leaves['reward']) == 200
    assert len(traj_leaves['loss']) == 200

    # Step 3: assemble + persist.
    leaves: dict[str, TraceLeaf] = {**hp_leaves, **traj_leaves}
    row = TraceRow(
        id='cartpole-smoke',
        cycle_id=None,
        timestamp=datetime.now(UTC).isoformat(timespec='seconds'),
        leaves=leaves,
    )
    path = tmp_path / 'traces.parquet'
    write_tracerows([row], path)

    # Step 4: round-trip preserves both kinds of leaves.
    [restored] = read_tracerows(path)
    assert restored.id == 'cartpole-smoke'
    assert restored.leaves['gamma'] == 0.99
    assert restored.leaves['optimizer.inner.lr'] == 2e-3
    # Trajectory survives as list[float] via the persistence layer.
    reward_traj = restored.leaves['reward']
    assert isinstance(reward_traj, list)
    assert len(reward_traj) == 200


def test_hp_and_trajectory_namespaces_do_not_collide(tmp_path: Path) -> None:
    """HPs use dotted topology paths (`replay.batch_size`),
    trajectories use flat author-chosen keys (`reward`). The
    cell layer relies on this naming-convention split — no
    framework-level collision check is needed."""
    env, env_params = gymnax.make('CartPole-v1')
    obs_space = env.observation_space(env_params)
    act_space = env.action_space(env_params)
    assert isinstance(obs_space, HasShape)
    assert isinstance(act_space, HasN)
    obs_dim = int(obs_space.shape[0])
    n_actions = int(act_space.n)

    configured = partial(
        dqn,
        env=env, env_params=env_params,
        obs_dim=obs_dim, n_actions=n_actions,
        total_steps=100, eval_every=100, n_episodes=1,
    )

    hp_leaves = _hp_leaves(configured)
    record = configured(rng_key=jax.random.PRNGKey(0))
    traj_leaves = _trajectory_leaves(record)

    # All HP keys contain a '.' iff they live below the top level
    # (top-level kwargs of dqn appear flat, e.g. 'gamma').
    # Trajectory keys are always flat (no '.').
    nested_hp_keys = [k for k in hp_leaves if '.' in k]
    flat_traj_keys = [k for k in traj_leaves if '.' not in k]
    assert nested_hp_keys, 'expected at least one nested HP path'
    assert flat_traj_keys, 'expected at least one flat trajectory key'

    overlap = set(hp_leaves) & set(traj_leaves)
    assert not overlap, f'HP and trajectory keys collide: {overlap}'
