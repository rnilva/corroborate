"""End-to-end smoke for the raw-trace store against a real DQN run.

Builds `TraceRow`s from a tiny CartPole DQN run, persists to
parquet, reads back, and verifies:

1. Configurational leaves from `walk_paths(walk(configured),
   regime='leaf')` appear as scalar entries at dotted topology
   paths (`gamma`, `optimizer.inner.lr`, `replay.batch_size`).
2. Per-step trajectories from the configured-claim's record dict
   appear as `list[float]` leaves at flat author-chosen keys
   (`reward`, `loss`, `td_error`, ...).
3. The two namespaces (dotted leaf paths vs. flat trajectory keys)
   coexist in one parquet without collision.
4. Round-trip through parquet preserves both shapes."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

import gymnax
import jax
import pytest

from corroborate._internals.canonical import canonical_str
from corroborate.corpus.persistence import read_tracerows, write_tracerows
from functools import partial as _partial

from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
from corroborate_rl.dqn.dqn import default_state_hash, dqn
from corroborate_rl.env_catalogue import HasN, HasShape
from corroborate.corpus.schema import TraceLeaf, TraceRow
from corroborate.core.signature import KwargInfo, walk, walk_paths

# Real DQN rollouts on CartPole — both tests run training to
# completion; ~6 s each. Skipped by default; opt in via `-m slow`.
pytestmark = pytest.mark.slow


def _leaf_scalar(value: object) -> str | int | float | bool:
    """Coerce a leaf value to a scalar trace entry. Primitives pass
    through; structured values (Modules, partials, FnClaims)
    canonicalise to string."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float, str)):
        return value
    return canonical_str(value)


def _leaf_values(configured: object) -> dict[str, str | int | float | bool]:
    """Topology walk → dotted-path leaf values. Each KwargInfo's
    default contributes one entry at its dotted path."""
    paths: dict[str, KwargInfo] = walk_paths(walk(configured), regime='leaf')
    return {path: _leaf_scalar(kw.default) for path, kw in paths.items()}


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
    obs_shape = tuple(int(d) for d in obs_space.shape)
    n_actions = int(act_space.n)

    # Configure dqn with a non-default optimizer to verify HPs are
    # captured at depth (optimizer.inner.lr).
    configured = partial(
        dqn,
        env=env, env_params=env_params,
        obs_shape=obs_shape, n_actions=n_actions,
        eval_episode_cap=200,
        state_hash=default_state_hash,
        total_steps=200, eval_every=100, n_episodes=2,
        optimizer=_partial(
            warmed_update, inner=_partial(adam, lr=2e-3), warmup_steps=50,
        ),
    )

    record = configured(rng_key=jax.random.PRNGKey(0))

    # Step 1: leaf values from the topology walk.
    leaf_values = _leaf_values(configured)
    assert leaf_values['gamma'] == 0.99
    # warmed_update.inner.adam.lr — captured at the deep dotted path.
    assert leaf_values['optimizer.inner.lr'] == 2e-3
    assert leaf_values['optimizer.warmup_steps'] == 50

    # Step 2: trajectory leaves from the record dict.
    traj_leaves = _trajectory_leaves(record)
    # Per-step training fields are 1-D shape (total_steps,).
    assert 'reward' in traj_leaves
    assert 'loss' in traj_leaves
    assert len(traj_leaves['reward']) == 200
    assert len(traj_leaves['loss']) == 200

    # Step 3: assemble + persist.
    leaves: dict[str, TraceLeaf] = {**leaf_values, **traj_leaves}
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


def test_leaf_and_trajectory_namespaces_do_not_collide(tmp_path: Path) -> None:
    """Configurational leaves use dotted topology paths
    (`replay.batch_size`); trajectories use flat author-chosen
    keys (`reward`). The cell layer relies on this naming-
    convention split — no framework-level collision check is
    needed."""
    env, env_params = gymnax.make('CartPole-v1')
    obs_space = env.observation_space(env_params)
    act_space = env.action_space(env_params)
    assert isinstance(obs_space, HasShape)
    assert isinstance(act_space, HasN)
    obs_shape = tuple(int(d) for d in obs_space.shape)
    n_actions = int(act_space.n)

    configured = partial(
        dqn,
        env=env, env_params=env_params,
        obs_shape=obs_shape, n_actions=n_actions,
        total_steps=100, eval_every=100, n_episodes=1,
    )

    leaf_values = _leaf_values(configured)
    record = configured(rng_key=jax.random.PRNGKey(0))
    traj_leaves = _trajectory_leaves(record)

    # Nested leaf keys contain a '.' iff they live below the top
    # level (top-level kwargs of dqn appear flat, e.g. 'gamma').
    # Trajectory keys are always flat (no '.').
    nested_leaf_keys = [k for k in leaf_values if '.' in k]
    flat_traj_keys = [k for k in traj_leaves if '.' not in k]
    assert nested_leaf_keys, 'expected at least one nested leaf path'
    assert flat_traj_keys, 'expected at least one flat trajectory key'

    overlap = set(leaf_values) & set(traj_leaves)
    assert not overlap, f'leaf and trajectory keys collide: {overlap}'
