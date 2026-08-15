"""Env-boundary helper for the rollout phase.

The implementation's rollout body should read like Mnih 2015 Algorithm 1:
score the current obs, pick an action, step the env, store the
transition. Pulling truncation extraction + type narrowing into a
boundary helper keeps `rollout_phase` paper-shaped — the
`isinstance(info_obj, jax.Array)` runtime narrowing for the
heterogeneous `info: dict[str, object]` lives HERE, not in the
per-step algorithm body.

The invariant `truncated=1 ⇒ done=1` is enforced **at the wrapper
boundary** (each wrapper that publishes `info['truncated']` masks
with `jnp.where(done, raw, 0)` at the point of emission). This
helper therefore does NOT re-mask — by the time `step_env` returns,
every concrete wrapper already guarantees the implication. See
`env_catalogue.py::EpisodeLengthCappedEnv._classify_truncated`,
`pgx_adapter.PgxEnv.step_env`, `jumanji_adapter.JumanjiEnv._classify_done`,
and `lunar_lander_jax.LunarLanderEnv.step_env` for the
wrapper-level mask form.

Envs without truncation (vanilla gymnax CartPole, MountainCar, …)
don't publish the key; the default-branch returns
`jnp.zeros_like(done)` so the implementation sees a uniform float32
scalar regardless of env."""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    # Stub-only Protocol — see env_catalogue.py for the rationale.
    from gymnax import Env, EnvParams, EnvState


def env_step_typed(
    env: Env,
    rng: jax.Array,
    state: EnvState,
    action: jax.Array,
    params: EnvParams,
) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, jax.Array]:
    """Call `env.step_env` and surface `truncated` as a typed
    sibling of `done` — flattens the heterogeneous-info contract
    that mid-substrate JAX code shouldn't have to narrow over.

    Returns `(next_obs, next_state, reward, done, truncated)`, all
    `jax.Array`. `truncated` is float32 in {0.0, 1.0}; envs that
    don't publish the key fall back to `jnp.zeros_like(done)` so
    downstream code never needs a None check.

    The `isinstance(_, jax.Array)` narrowing is the price of
    keeping `info: dict[str, object]` as the wrapper Protocol's
    contract (heterogeneous values — wrappers add arbitrary
    diagnostic keys); it runs once at trace time, not under scan.
    """
    next_obs, next_state, reward, done, info = env.step_env(
        rng, state, action, params,
    )
    truncated_obj = info.get('truncated')
    if isinstance(truncated_obj, jax.Array):
        truncated = truncated_obj.astype(jnp.float32)
    else:
        truncated = jnp.zeros_like(done, dtype=jnp.float32)
    return next_obs, next_state, reward, done, truncated
