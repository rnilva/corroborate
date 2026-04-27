"""Eval-loop infrastructure — greedy rollouts for the
Hasselt-style overestimation-bias measurement.

The Jensen overestimation gap (Hasselt 2010, 2016) requires
empirical Q̂ at start states vs. Monte-Carlo return ground truth.
That can't be derived from training-loop data alone; it needs
periodic greedy rollouts during training, with predicted-Q
recorded at start and discounted MC return computed from the
realised reward sequence.

Three pieces in this module:

1. `eval_episode` (`@claim`) — single greedy rollout from a
   reset state; returns `(predicted_q_at_start, mc_return,
   episode_length)`.
2. `eval_burst` — K greedy rollouts via vmap over fresh seeds;
   stacks per-episode results into `(K,)`-shaped arrays.
3. `train_with_eval` — nested `jax.lax.scan`. Outer scan iterates
   `n_bursts = total_steps // eval_every` super-steps; each
   super-step body runs an inner scan over `eval_every` training
   steps, then one `eval_burst`. Both scans jit-trace their
   bodies; the function itself is NOT `@jax.jit`-decorated, so
   the top-level Python call re-traces each invocation. Callers
   that want the full single-compile boundary can wrap with
   `jax.jit(train_with_eval, static_argnums=(2,), static_argnames=
   ('eval_every',))` — the `total_steps` and `eval_every` are
   structural, not traced. Returns `(state, record)` where
   `record` is a single dict mixing training fields (shape
   `(total_steps, ...)`) and eval fields (shape `(n_bursts, K,
   ...)`). The author's bridges read whichever keys they care
   about.

Eval IS part of training — they're aspects of one experiment
run. The merged-dict return shape reflects that: no separate
"eval record" stream, no `bridges_e`, no train/eval distinction
in framework code. The cell runner produces one record; bridges
target arbitrary keys regardless of which sub-process produced
them."""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from corroborate.claim import claim
from corroborate.rl.dqn.state import DQNState
from corroborate.rl.dqn.types import QNetwork, StepRecord
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
    q_network: QNetwork,
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
    q_network: QNetwork,
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


# ============ train_with_eval — single nested scan ============

def train_with_eval(
    step_fn: Callable[[DQNState, jax.Array], tuple[DQNState, StepRecord]],
    init: DQNState,
    total_steps: int,
    *,
    eval_fn: Callable[[DQNState, jax.Array], EvalBurstOut],
    eval_every: int,
) -> tuple[DQNState, dict[str, jax.Array]]:
    """Single nested `jax.lax.scan`: outer over `n_bursts =
    total_steps // eval_every` super-steps; each super-step body
    runs an inner scan over `eval_every` training steps then one
    `eval_burst`. Single jit-compile boundary (no Python overhead
    between super-steps, no recompile per chunk).

    Returns `(state, record)` where `record` is a single dict
    mixing:
      - training fields, shape `(total_steps, ...)` per field
      - eval fields, shape `(n_bursts, K, ...)` per field
      - `eval_step_index`, shape `(n_bursts,)` int32

    The eval-side keys (`predicted_q_at_start`, `mc_return`,
    `episode_length`, `eval_step_index`) are disjoint from the
    training keys by RL-substrate convention — bridges read
    whichever they need without naming collision.

    `eval_fn` is the caller's closure around `eval_burst`,
    capturing env, q_network, gamma, etc. Single-jit means the
    closure must be jit-compatible (no Python-level branching on
    traced values).

    Raises `ValueError` if `total_steps` isn't a multiple of
    `eval_every` (no partial trailing chunk)."""
    if total_steps % eval_every != 0:
        raise ValueError(
            f'total_steps ({total_steps}) must be a multiple of '
            f'eval_every ({eval_every}); got remainder '
            f'{total_steps % eval_every}',
        )
    if total_steps < eval_every:
        raise ValueError(
            f'total_steps ({total_steps}) must be ≥ eval_every '
            f'({eval_every}) — at least one super-step is required.',
        )

    n_super_steps = total_steps // eval_every

    def super_step(
        state: DQNState, super_idx: jax.Array,
    ) -> tuple[DQNState, tuple[StepRecord, EvalBurstOut]]:
        # Inner scan over eval_every training steps. Build global
        # step indices so step_fn sees absolute step number even
        # though we're in a chunked outer loop.
        offset = super_idx * eval_every
        inner_indices = offset + jnp.arange(eval_every, dtype=jnp.int32)
        state, train_chunk = jax.lax.scan(step_fn, state, inner_indices)
        # Eval burst at the end of this chunk.
        burst = eval_fn(state, super_idx)
        return state, (train_chunk, burst)

    # int32 across both nesting levels — uniform dtype keeps the
    # `super_idx * eval_every + jnp.arange(eval_every)` arithmetic
    # in a single integer regime (no silent uint32→int64 promotion
    # under x64-enabled jax, no implicit downcast under x64-disabled).
    super_indices = jnp.arange(n_super_steps, dtype=jnp.int32)
    state, (train_chunks, eval_bursts) = jax.lax.scan(
        super_step, init, super_indices,
    )

    # train_chunks: pytree where each leaf has shape
    #   (n_super_steps, eval_every, *original_shape).
    # Reshape to (total_steps, *original_shape).
    def _flatten_chunks(x: jax.Array) -> jax.Array:
        return x.reshape(total_steps, *x.shape[2:])

    train_trace = jax.tree.map(_flatten_chunks, train_chunks)

    # eval_bursts: NamedTuple of (n_super_steps, K, ...) arrays.
    # Compute eval_step_index as super_idx-aware boundaries.
    eval_step_indices = (jnp.arange(n_super_steps) + 1) * eval_every

    record: dict[str, jax.Array] = {**train_trace}
    record['predicted_q_at_start'] = eval_bursts.predicted_q_at_start
    record['mc_return'] = eval_bursts.mc_return
    record['episode_length'] = eval_bursts.episode_length
    record['eval_step_index'] = eval_step_indices.astype(jnp.int32)

    return state, record
