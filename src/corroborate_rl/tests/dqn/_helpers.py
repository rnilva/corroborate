"""Shared fixtures/helpers for the dqn test cohort.

`make_cartpole_env` was copy-pasted verbatim into test_smoke.py,
test_eval.py, and test_paired.py; hoisted here so the env-construction
contract (obs/act-space → obs_shape/n_actions coercion) lives in one
place and can't drift across the three call sites."""
from __future__ import annotations

from typing import TYPE_CHECKING

import gymnax
from gymnax import EnvParams

if TYPE_CHECKING:
    from gymnax import Env


def make_cartpole_env() -> tuple[Env, EnvParams, tuple[int, ...], int]:
    """Build CartPole-v1 + its obs_shape / n_actions, the canonical
    tiny env for dqn/paired smoke + eval tests."""
    env, env_params = gymnax.make('CartPole-v1')
    obs_space = env.observation_space(env_params)
    act_space = env.action_space(env_params)
    obs_shape = tuple(int(d) for d in obs_space.shape)
    n_actions = int(act_space.n)
    return env, env_params, obs_shape, n_actions
