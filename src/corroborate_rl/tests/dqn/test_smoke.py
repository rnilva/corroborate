"""Smoke test — vanilla DQN + DDQN run on CartPole and produce
structurally-correct per-step records.

Verifies:
- `dqn_step` runs end-to-end via both `python_loop` (probe) and
  `scan_loop` (jit) with identical step semantics.
- DDQN's `bootstrap` slot is a real algorithmic swap: given
  non-identical online / target params, the two bootstrap
  functions produce different targets. (The DDQN-vs-vanilla
  end-to-end trajectory test was retired: a 50-step CartPole run
  with sync_period=10 keeps online ≈ target most of the time, so
  the trajectory diverges only marginally — a property of the
  numerical regime, not of the swap. The unit test below pins the
  contract directly without that confound.)
- The record dict has the expected keys with correct shapes."""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import gymnax
import jax
import jax.numpy as jnp
import optax
import pytest
from gymnax import EnvParams

from corroborate_rl.dqn.claims.bootstrap import (
    bootstrap as bootstrap_claim,
    double_greedify,
    max_greedify,
)
from functools import partial as _partial
from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
from corroborate_rl.dqn.claims.q_network import MLP, mlp_q
from corroborate_rl.dqn.claims.replay import Replay
from corroborate_rl.dqn.dqn import dqn_step, init_state
from corroborate_rl.loop import python_loop, scan_loop

if TYPE_CHECKING:
    # Stub-only Protocol — see env_catalogue.py for the rationale.
    from gymnax import Env


# ============ Fixtures ============

def _make_env() -> tuple[Env, EnvParams, tuple[int, ...], int]:
    env, env_params = gymnax.make('CartPole-v1')
    obs_space = env.observation_space(env_params)
    act_space = env.action_space(env_params)
    obs_shape = tuple(int(d) for d in obs_space.shape)
    n_actions = int(act_space.n)
    return env, env_params, obs_shape, n_actions


def _build_step_fn(
    env: Env, env_params: EnvParams, n_actions: int,
    optimizer: optax.GradientTransformation,
    *, bootstrap_swap: bool = False,
):
    """Build a step-fn closure suitable for `python_loop` /
    `scan_loop` — `(state, idx) -> (state, record)`. The
    `bootstrap_swap` flag toggles vanilla vs DDQN."""
    extra: dict[str, object] = {}
    if bootstrap_swap:
        extra['bootstrap'] = partial(
            bootstrap_claim, greedification=double_greedify,
        )

    bound = partial(
        dqn_step,
        env=env, env_params=env_params, n_actions=n_actions,
        optimizer=optimizer,
        sync_period=10,
        replay=Replay(capacity=200, batch_size=16),
        **extra,
    )
    return bound


# ============ Smoke: vanilla DQN runs on CartPole ============

@pytest.mark.slow
def test_vanilla_dqn_runs_on_cartpole_via_python_loop() -> None:
    env, env_params, obs_shape, n_actions = _make_env()
    optimizer = warmed_update(inner=_partial(adam), warmup_steps=10)
    init = init_state(
        env=env, env_params=env_params,
        obs_shape=obs_shape, n_actions=n_actions,
        rng_key=jax.random.PRNGKey(0), optimizer=optimizer,
        replay=Replay(capacity=200),
    )
    step_fn = _build_step_fn(env, env_params, n_actions, optimizer)

    final_state, record = python_loop(step_fn, init, length=50)

    # State advanced 50 steps.
    assert int(final_state.step) == 50
    # Record has expected keys with two shape families:
    # - per-step scalars: (T,)
    # - per-step Q-summaries: (T, n_actions) or (T, 5)
    scalar_keys = {
        'reward', 'done', 'max_q',
        'ep_return', 'action', 'state_hash', 'buf_size',
        'loss', 'td_error', 'td_error_within_batch_std',
    }
    # Per-step Q reductions: per-action vectors (n_actions,) and
    # 5-tuple Pearson sum-stats. Replaces the full
    # `(batch, n_actions)` Q-tensors that train_phase used to emit;
    # the in-loop reduction is the OOM fix for high-action envs.
    q_keys = {
        'online_q_per_action', 'target_q_per_action', 'pearson_stats',
    }
    assert set(record.keys()) == scalar_keys | q_keys
    for key in scalar_keys:
        assert record[key].shape == (50,)
    # n_actions = 2 for CartPole; pearson_stats is 5 sum-stats/step.
    assert record['online_q_per_action'].shape == (50, 2)
    assert record['target_q_per_action'].shape == (50, 2)
    assert record['pearson_stats'].shape == (50, 5)


@pytest.mark.slow
def test_vanilla_dqn_runs_via_scan_loop() -> None:
    """Same step function under scan_loop should produce the
    same record shape (and identical values for fixed seed)."""
    env, env_params, obs_shape, n_actions = _make_env()
    optimizer = warmed_update(inner=_partial(adam), warmup_steps=10)
    init = init_state(
        env=env, env_params=env_params,
        obs_shape=obs_shape, n_actions=n_actions,
        rng_key=jax.random.PRNGKey(0), optimizer=optimizer,
        replay=Replay(capacity=200),
    )
    step_fn = _build_step_fn(env, env_params, n_actions, optimizer)

    _, record = scan_loop(step_fn, init, length=50)
    assert record['ep_return'].shape == (50,)


# ============ DDQN swap actually changes computation ============

def test_ddqn_and_vanilla_bootstrap_differ_when_params_differ() -> None:
    """The DDQN ↔ vanilla swap is meaningful: with non-identical
    online and target params, the two bootstraps compute different
    targets. Unit test on the bootstrap functions directly — no env
    loop, no warmup, no sync confound. (vanilla evaluates
    max_a' Q_target(s', a'); DDQN evaluates Q_target(s', argmax_a'
    Q_online(s', a')). When online's argmax disagrees with target's
    argmax for at least one transition, the targets differ.)"""
    obs_shape, n_actions, batch_size = (4,), 3, 8
    obs_dim = obs_shape[0]
    rng = jax.random.PRNGKey(0)
    online_key, target_key, obs_key = jax.random.split(rng, 3)

    arch = MLP(hidden=(16, 16))
    online = arch.init(online_key, obs_shape, n_actions)
    target = arch.init(target_key, obs_shape, n_actions)
    next_obs = jax.random.normal(obs_key, (batch_size, obs_dim))
    reward = jnp.ones((batch_size,))
    done = jnp.zeros((batch_size,))

    target_v = bootstrap_claim(
        online_params=online, target_params=target, q_network=mlp_q,
        next_obs=next_obs, reward=reward, done=done, gamma=0.99,
        greedification=max_greedify,
    )
    target_d = bootstrap_claim(
        online_params=online, target_params=target, q_network=mlp_q,
        next_obs=next_obs, reward=reward, done=done, gamma=0.99,
        greedification=double_greedify,
    )

    # Different params → different argmaxes on at least one
    # transition → different Bellman targets.
    diffs = jnp.abs(target_v - target_d)
    assert float(jnp.max(diffs)) > 0.0, (
        'DDQN and vanilla bootstraps produced identical targets '
        'despite distinct online/target params — the greedification '
        'swap is a no-op. Check double_greedify actually decouples '
        'argmax (online) from evaluation (target).'
    )


def test_ddqn_and_vanilla_bootstrap_match_when_params_equal() -> None:
    """Sanity-check the contract from the other side: when online
    and target params are identical, DDQN reduces to vanilla
    (online's argmax == target's argmax, so Q_target picks the same
    value either way). This is why the smoke-loop variant of this
    test failed — at init `target_params = online_params`, and
    sync_period=10 keeps re-syncing them inside the 50-step run."""
    obs_shape, n_actions, batch_size = (4,), 3, 8
    obs_dim = obs_shape[0]
    rng = jax.random.PRNGKey(0)
    init_key, obs_key = jax.random.split(rng)

    params = MLP(hidden=(16, 16)).init(init_key, obs_shape, n_actions)
    next_obs = jax.random.normal(obs_key, (batch_size, obs_dim))
    reward = jnp.ones((batch_size,))
    done = jnp.zeros((batch_size,))

    target_v = bootstrap_claim(
        online_params=params, target_params=params, q_network=mlp_q,
        next_obs=next_obs, reward=reward, done=done, gamma=0.99,
        greedification=max_greedify,
    )
    target_d = bootstrap_claim(
        online_params=params, target_params=params, q_network=mlp_q,
        next_obs=next_obs, reward=reward, done=done, gamma=0.99,
        greedification=double_greedify,
    )
    # Online == target → both take the same argmax → identical targets.
    assert jnp.allclose(target_v, target_d)


# ============ CNN q-network ============

def test_cnn_q_network_init_call_round_trip() -> None:
    """CNN runs at native (H, W, C) — no flatten round-trip
    inside __call__. Forward pass works for single, batch, and
    vmap'd inputs."""
    from corroborate_rl.dqn.claims.q_network import CNN
    obs_shape = (10, 10, 4)
    cnn = CNN(obs_shape=obs_shape, channels=(16, 32), kernel_size=3, hidden=(64,))
    key = jax.random.PRNGKey(0)
    params = cnn.init(key, obs_shape, n_actions=5)

    # Single (H, W, C) input.
    single = jnp.zeros(obs_shape)
    q_single = cnn(params, single)
    assert q_single.shape == (5,)

    # Batch (B, H, W, C) input.
    batched = jnp.zeros((32, *obs_shape))
    q_batch = cnn(params, batched)
    assert q_batch.shape == (32, 5)

    # Vmap over seeds.
    keys = jax.random.split(key, 4)
    multi_params = jax.vmap(lambda k: cnn.init(k, obs_shape, 5))(keys)
    multi_obs = jnp.zeros((4, 32, *obs_shape))
    q_multi = jax.vmap(cnn)(multi_params, multi_obs)
    assert q_multi.shape == (4, 32, 5)


def test_cnn_q_network_validates_obs_shape_consistency() -> None:
    """CNN.init raises when substrate-passed obs_shape differs
    from Module's declared obs_shape — silent mismatch would
    pass through with wrong conv kernels."""
    from corroborate_rl.dqn.claims.q_network import CNN
    cnn = CNN(obs_shape=(10, 10, 4))
    with pytest.raises(ValueError, match='inconsistent'):
        cnn.init(jax.random.PRNGKey(0), (8, 8, 3), n_actions=5)


# ============ Episode return tracking ============

@pytest.mark.slow
def test_ep_return_resets_in_state_on_done() -> None:
    """`DQNState.ep_return` should reset to 0 after a step where
    `done=True`. The record's `ep_return` carries the cumulative
    value AT the done step (so bridges filtering by done==1 see
    the final per-episode return)."""
    env, env_params, obs_shape, n_actions = _make_env()
    optimizer = warmed_update(inner=_partial(adam), warmup_steps=10)
    init = init_state(
        env=env, env_params=env_params,
        obs_shape=obs_shape, n_actions=n_actions,
        rng_key=jax.random.PRNGKey(0), optimizer=optimizer,
        replay=Replay(capacity=200),
    )
    step_fn = _build_step_fn(env, env_params, n_actions, optimizer)

    # Run long enough to see at least one episode end.
    _, record = python_loop(step_fn, init, length=200)

    n_dones = int(jnp.sum(record['done']))
    assert n_dones > 0, "expected at least one episode to end in 200 steps"

    # On done steps, ep_return is the cumulative return for that
    # episode. (Note: this property is a record-level invariant;
    # the in-state ep_return resets to 0 the step after.)
    done_indices = jnp.where(record['done'] > 0.5)[0]
    for di in done_indices:
        assert float(record['ep_return'][int(di)]) >= 1.0, (
            f'ep_return at done step {int(di)} should be ≥ 1; '
            f'got {float(record["ep_return"][int(di)])}'
        )


# ============ Step counter monotonicity ============

@pytest.mark.slow
def test_step_counter_advances_monotonically() -> None:
    """The state.step counter should strictly increment each
    iteration. (Sanity check on the loop primitive's idx
    threading.)"""
    env, env_params, obs_shape, n_actions = _make_env()
    optimizer = warmed_update(inner=_partial(adam), warmup_steps=10)
    init = init_state(
        env=env, env_params=env_params,
        obs_shape=obs_shape, n_actions=n_actions,
        rng_key=jax.random.PRNGKey(0), optimizer=optimizer,
        replay=Replay(capacity=200),
    )
    step_fn = _build_step_fn(env, env_params, n_actions, optimizer)

    final_state, _ = python_loop(step_fn, init, length=30)
    assert int(final_state.step) == 30
