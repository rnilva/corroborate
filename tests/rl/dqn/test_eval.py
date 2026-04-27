"""Tests for eval-loop infrastructure — `eval_episode`,
`eval_burst`, `train_with_eval`.

Verifies:
1. `eval_episode` returns finite predicted Q + MC return on a real
   CartPole rollout.
2. `eval_burst` stacks K episodes correctly.
3. `train_with_eval` produces a `ComposedTrace` with both
   training and eval data, correctly shaped.
4. The composed trace's eval data lets `jensen_overestimation_gap`
   compute a meaningful number (next phase, but smoke the data
   plumbing here)."""
from __future__ import annotations

from functools import partial

import gymnax
import jax
import jax.numpy as jnp
import optax

from corroborate.rl.dqn.claims.q_network import mlp_q
from corroborate.rl.dqn.dqn import dqn_step, init_state
from corroborate.rl.dqn.eval import (
    ComposedTrace,
    EvalBurstOut,
    eval_burst,
    eval_episode,
    train_with_eval,
)
from corroborate.rl.dqn.state import DQNState
from corroborate.rl.env_catalogue import GymnaxEnvLike, HasN, HasShape


def _make_env() -> tuple[GymnaxEnvLike, object, int, int]:
    env, env_params = gymnax.make('CartPole-v1')
    obs_space = env.observation_space(env_params)
    act_space = env.action_space(env_params)
    assert isinstance(obs_space, HasShape)
    assert isinstance(act_space, HasN)
    obs_dim = int(obs_space.shape[0])
    n_actions = int(act_space.n)
    return env, env_params, obs_dim, n_actions


# ============ eval_episode ============

def test_eval_episode_returns_finite_scalars() -> None:
    env, env_params, obs_dim, n_actions = _make_env()
    optimizer = optax.adam(1e-3)
    state = init_state(
        env=env, env_params=env_params,
        obs_dim=obs_dim, n_actions=n_actions,
        seed=0, optimizer=optimizer, buffer_capacity=200,
    )

    out = eval_episode(
        online_params=state.online_params,
        env=env, env_params=env_params,
        rng_key=jax.random.PRNGKey(42),
        q_network=mlp_q,
        gamma=0.99,
        episode_cap=200,
    )
    # All three fields are scalars.
    assert out.predicted_q_at_start.shape == ()
    assert out.mc_return.shape == ()
    assert out.episode_length.shape == ()
    # Episode ran some non-zero number of steps.
    assert int(out.episode_length) > 0
    # Predicted Q is a finite number (not nan/inf).
    assert jnp.isfinite(out.predicted_q_at_start)


def test_eval_episode_mc_return_matches_episode_length_for_cartpole() -> None:
    """CartPole gives reward=1 every step until done. With
    γ=1.0, the MC return equals the episode length."""
    env, env_params, obs_dim, n_actions = _make_env()
    optimizer = optax.adam(1e-3)
    state = init_state(
        env=env, env_params=env_params,
        obs_dim=obs_dim, n_actions=n_actions,
        seed=0, optimizer=optimizer, buffer_capacity=200,
    )

    out = eval_episode(
        online_params=state.online_params,
        env=env, env_params=env_params,
        rng_key=jax.random.PRNGKey(42),
        q_network=mlp_q,
        gamma=1.0,  # no discount → return = sum of rewards = length
        episode_cap=200,
    )
    assert int(out.mc_return) == int(out.episode_length)


# ============ eval_burst ============

def test_eval_burst_stacks_k_episodes() -> None:
    env, env_params, obs_dim, n_actions = _make_env()
    optimizer = optax.adam(1e-3)
    state = init_state(
        env=env, env_params=env_params,
        obs_dim=obs_dim, n_actions=n_actions,
        seed=0, optimizer=optimizer, buffer_capacity=200,
    )

    burst = eval_burst(
        online_params=state.online_params,
        env=env, env_params=env_params,
        rng_key=jax.random.PRNGKey(42),
        q_network=mlp_q,
        gamma=0.99,
        episode_cap=200,
        n_episodes=4,
    )
    # Three (4,)-shaped arrays.
    assert burst.predicted_q_at_start.shape == (4,)
    assert burst.mc_return.shape == (4,)
    assert burst.episode_length.shape == (4,)


def test_eval_burst_episodes_are_distinct() -> None:
    """Different seeds → different episode lengths (CartPole's
    end-time depends on initial state and action sequence)."""
    env, env_params, obs_dim, n_actions = _make_env()
    optimizer = optax.adam(1e-3)
    state = init_state(
        env=env, env_params=env_params,
        obs_dim=obs_dim, n_actions=n_actions,
        seed=0, optimizer=optimizer, buffer_capacity=200,
    )

    burst = eval_burst(
        online_params=state.online_params,
        env=env, env_params=env_params,
        rng_key=jax.random.PRNGKey(0),
        q_network=mlp_q,
        gamma=1.0,
        episode_cap=200,
        n_episodes=8,
    )
    # At least some episodes should have different lengths
    # (with random init params and 8 distinct seeds).
    lengths = burst.episode_length
    assert int(jnp.max(lengths)) > int(jnp.min(lengths))


# ============ train_with_eval ============

def test_train_with_eval_produces_composed_trace() -> None:
    env, env_params, obs_dim, n_actions = _make_env()
    optimizer = optax.adam(1e-3)
    state = init_state(
        env=env, env_params=env_params,
        obs_dim=obs_dim, n_actions=n_actions,
        seed=0, optimizer=optimizer, buffer_capacity=200,
    )
    step_fn = partial(
        dqn_step,
        env=env, env_params=env_params, n_actions=n_actions,
        optimizer=optimizer,
        warmup_steps=10, sync_period=10,
        buffer_capacity=200, batch_size=16,
    )

    def eval_fn(s: DQNState, idx: jax.Array) -> EvalBurstOut:
        return eval_burst(
            online_params=s.online_params,
            env=env, env_params=env_params,
            rng_key=jax.random.fold_in(jax.random.PRNGKey(99), idx),
            q_network=mlp_q, gamma=0.99,
            episode_cap=100, n_episodes=2,
        )

    _final_state, trace = train_with_eval(
        step_fn, state, total_steps=40,
        eval_fn=eval_fn, eval_every=20,
    )

    assert isinstance(trace, ComposedTrace)
    # Train trace is flat (40,) per field.
    assert trace.train['epsilon'].shape == (40,)
    assert trace.train['loss'].shape == (40,)
    # Eval trace stacks (n_bursts=2, K=2).
    assert trace.eval['predicted_q_at_start'].shape == (2, 2)
    assert trace.eval['mc_return'].shape == (2, 2)
    # eval_step_index records which training step each burst ran at.
    assert trace.eval['eval_step_index'].shape == (2,)
    assert int(trace.eval['eval_step_index'][0]) == 20
    assert int(trace.eval['eval_step_index'][1]) == 40


def test_train_with_eval_rejects_misaligned_steps() -> None:
    env, env_params, obs_dim, n_actions = _make_env()
    optimizer = optax.adam(1e-3)
    state = init_state(
        env=env, env_params=env_params,
        obs_dim=obs_dim, n_actions=n_actions,
        seed=0, optimizer=optimizer, buffer_capacity=200,
    )
    step_fn = partial(
        dqn_step,
        env=env, env_params=env_params, n_actions=n_actions,
        optimizer=optimizer,
        warmup_steps=10, sync_period=10,
        buffer_capacity=200, batch_size=16,
    )

    def dummy_eval(s: DQNState, idx: jax.Array) -> EvalBurstOut:
        del s, idx
        raise AssertionError('should not be reached')

    try:
        train_with_eval(
            step_fn, state, total_steps=37,  # not a multiple of 20
            eval_fn=dummy_eval, eval_every=20,
        )
        raise AssertionError('expected ValueError for misaligned steps')
    except ValueError as e:
        assert 'total_steps' in str(e)
        assert 'eval_every' in str(e)
