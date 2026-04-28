"""Eval-loop infrastructure — greedy rollouts for the
Hasselt-style overestimation-bias measurement.

The Jensen overestimation gap (Hasselt 2010, 2016) requires
empirical Q̂ at start states vs. Monte-Carlo return ground truth.
That can't be derived from training-loop data alone; it needs
periodic greedy rollouts during training, with predicted-Q
recorded at start and discounted MC return computed from the
realised reward sequence.

Three pieces:

1. `eval_episode` — single greedy rollout from a reset state.
2. `eval_burst` — K greedy rollouts via vmap over fresh seeds.
3. `train_with_eval` — nested `scan_loop` driver: outer over
   super-steps (one eval burst at the end of each), inner over
   training steps. Returns the merged record dict — training
   fields shape `(total_steps, ...)` + eval fields shape
   `(n_bursts, K, ...)` + `eval_step_index`.

`train_with_eval` is the loop-orchestration primitive — separate
from `dqn` itself so the algorithm composition stays paper-prose.
Same backbone could drive a different RL algorithm with eval
bursts (PPO, SAC, etc.)."""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from corroborate.claim import claim
from corroborate.loop import scan_loop
from corroborate.rl.dqn.state import DQNState
from corroborate.rl.dqn.types import QFunction, StepRecord
from corroborate.rl.env_catalogue import GymnaxEnvLike


# ============ Eval per-episode and per-burst record shapes ============

class EvalEpisodeOut(NamedTuple):
    """One eval episode's per-burst record."""
    predicted_q_at_start: jax.Array   # () — max_a Q_online(s_0, a)
    mc_return: jax.Array              # () — Σ γ^t r_t over the episode
    episode_length: jax.Array         # () int32


class EvalBurstOut(NamedTuple):
    """K stacked eval episodes."""
    predicted_q_at_start: jax.Array   # (K,)
    mc_return: jax.Array              # (K,)
    episode_length: jax.Array         # (K,) int32


# ============ Single greedy episode ============

@claim
def eval_episode(
    *,
    online_params: dict[str, jax.Array],
    env: GymnaxEnvLike,
    env_params: object,
    rng_key: jax.Array,
    q_network: QFunction,
    gamma: float,
    episode_cap: int,
) -> EvalEpisodeOut:
    """One greedy rollout from a fresh env reset.

    Records `predicted_q_at_start = max_a Q_online(s_0, a)` —
    the agent's value prediction at the episode start — and
    accumulates the discounted MC return as the actual outcome.
    The Hasselt-style gap is `predicted - actual` (positive ⇒
    overestimation, the Jensen-bias signature)."""
    reset_key, run_key = jax.random.split(rng_key)
    obs_0, env_state_0 = env.reset(reset_key, env_params)
    # Flatten env-side multi-dim obs to match the flat-MLP shape.
    obs_0 = obs_0.reshape(-1)

    q_at_start = q_network(online_params, obs_0)
    predicted_q_at_start = jnp.max(q_at_start)

    class Carry(NamedTuple):
        obs: jax.Array
        env_state: object
        done: jax.Array
        rng: jax.Array
        cumulative_return: jax.Array
        steps: jax.Array

    init_carry = Carry(
        obs=obs_0,
        env_state=env_state_0,
        done=jnp.bool_(False),
        rng=run_key,
        cumulative_return=jnp.float32(0.0),
        steps=jnp.int32(0),
    )

    def step(carry: Carry, _idx: jax.Array) -> tuple[Carry, None]:
        q_values = q_network(online_params, carry.obs)
        action = jnp.argmax(q_values).astype(jnp.int32)

        env_key, next_rng = jax.random.split(carry.rng)
        next_obs, next_env_state, reward, done, _info = env.step(
            env_key, carry.env_state, action, env_params,
        )
        # Flatten multi-dim obs to match the flat-MLP shape.
        next_obs = next_obs.reshape(carry.obs.shape)

        already_done = carry.done
        active = jnp.logical_not(already_done)
        discount = jnp.power(gamma, carry.steps.astype(jnp.float32))
        new_cumulative = carry.cumulative_return + jnp.where(
            active, reward * discount, 0.0,
        )
        new_steps = carry.steps + jnp.where(active, jnp.int32(1), jnp.int32(0))
        new_done = jnp.logical_or(already_done, done.astype(jnp.bool_))

        return (
            Carry(
                obs=next_obs,
                env_state=next_env_state,
                done=new_done,
                rng=next_rng,
                cumulative_return=new_cumulative,
                steps=new_steps,
            ),
            None,
        )

    final_carry, _ = jax.lax.scan(step, init_carry, jnp.arange(episode_cap))

    return EvalEpisodeOut(
        predicted_q_at_start=predicted_q_at_start,
        mc_return=final_carry.cumulative_return,
        episode_length=final_carry.steps,
    )


# ============ K-episode burst via vmap ============

def eval_burst(
    *,
    online_params: dict[str, jax.Array],
    env: GymnaxEnvLike,
    env_params: object,
    rng_key: jax.Array,
    q_network: QFunction,
    gamma: float,
    episode_cap: int,
    n_episodes: int,
) -> EvalBurstOut:
    """Run `n_episodes` greedy rollouts in parallel (vmap over
    seeds). Returns `EvalBurstOut` with stacked `(n_episodes,)`
    arrays."""
    keys = jax.random.split(rng_key, n_episodes)

    def one(key: jax.Array) -> EvalEpisodeOut:
        return eval_episode(
            online_params=online_params,
            env=env, env_params=env_params,
            rng_key=key,
            q_network=q_network,
            gamma=gamma,
            episode_cap=episode_cap,
        )

    stacked = jax.vmap(one)(keys)
    return EvalBurstOut(
        predicted_q_at_start=stacked.predicted_q_at_start,
        mc_return=stacked.mc_return,
        episode_length=stacked.episode_length,
    )


# ============ train_with_eval — nested scan driver ============

def train_with_eval(
    *,
    step_fn: Callable[[DQNState, jax.Array], tuple[DQNState, StepRecord]],
    eval_fn: Callable[[DQNState, jax.Array], EvalBurstOut],
    init_state: DQNState,
    total_steps: int,
    eval_every: int,
) -> dict[str, jax.Array]:
    """Run `step_fn` for `total_steps` with an `eval_fn` burst at
    the end of every `eval_every` chunk. Returns the merged
    record dict.

    Outer scan over `total_steps // eval_every` super-steps; inner
    scan over `eval_every` training steps. The outer scan's per-
    super-step output is `(train_chunk, eval_burst)`. After the
    full run, train chunks reshape from `(n_super_steps,
    eval_every, ...)` → `(total_steps, ...)`; eval burst fields
    stack as `(n_super_steps, K, ...)`.

    Decoupled from `dqn` itself so the algorithm composition stays
    paper-prose. The same driver can power any RL algorithm with
    a step+eval shape (PPO, SAC, distributional Q)."""
    n_super_steps = total_steps // eval_every

    def super_step(
        s: DQNState, super_idx: jax.Array,
    ) -> tuple[DQNState, tuple[StepRecord, EvalBurstOut]]:
        s, train_chunk = scan_loop(step_fn, s, eval_every)
        burst = eval_fn(s, super_idx)
        return s, (train_chunk, burst)

    _final, (train_chunks, eval_bursts) = scan_loop(
        super_step, init_state, n_super_steps,
    )

    def _flatten(x: jax.Array) -> jax.Array:
        # Each leaf: (n_super_steps, eval_every, *original) →
        # (total_steps, *original).
        return x.reshape(total_steps, *x.shape[2:])

    train_trace: StepRecord = jax.tree.map(_flatten, train_chunks)
    eval_step_indices = (
        jnp.arange(n_super_steps, dtype=jnp.int32) + 1
    ) * eval_every

    return {
        **train_trace,
        'predicted_q_at_start': eval_bursts.predicted_q_at_start,
        'mc_return': eval_bursts.mc_return,
        'episode_length': eval_bursts.episode_length,
        'eval_step_index': eval_step_indices,
    }
