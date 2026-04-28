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

import gymnax
import jax
import jax.numpy as jnp
import optax
import pytest

from corroborate.rl.dqn.claims.q_network import mlp_q
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.dqn import init_state
from corroborate.rl.dqn.eval import (
    eval_burst,
    eval_episode,
)
from corroborate.rl.env_catalogue import GymnaxEnvLike, HasN, HasShape

# Eval primitives all run real CartPole rollouts under jit/vmap;
# ~1.5 s each. Skipped by default; opt in via `-m slow`.
pytestmark = pytest.mark.slow


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
        rng_key=jax.random.PRNGKey(0), optimizer=optimizer,
        replay=Replay(capacity=200),
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
        rng_key=jax.random.PRNGKey(0), optimizer=optimizer,
        replay=Replay(capacity=200),
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
        rng_key=jax.random.PRNGKey(0), optimizer=optimizer,
        replay=Replay(capacity=200),
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
        rng_key=jax.random.PRNGKey(0), optimizer=optimizer,
        replay=Replay(capacity=200),
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


# `train_with_eval` retired — its responsibility (init + nested
# scan + record assembly) moved into the `dqn` outermost claim in
# `dqn.py`. End-to-end coverage of that composition lives in
# `tests/rl/test_cell_runner.py` (which calls `dqn` via the cell
# runner). This file tests the still-standalone eval primitives
# (`eval_episode`, `eval_burst`).
