"""Eval-loop infrastructure — greedy rollouts for the
Hasselt-style overestimation-bias measurement.

The Jensen overestimation gap (Hasselt 2010, 2016) requires
empirical Q̂ at start states vs. Monte-Carlo return ground truth.
That can't be derived from training-loop data; it needs a
*separate* eval pass — periodic greedy rollouts from fresh env
resets, with predicted-Q recorded at start and discounted MC
return computed from the realised reward sequence.

Three pieces in this module:

1. `eval_episode` (`@claim`) — single greedy rollout from a
   reset state; returns `(predicted_q_at_start, mc_return,
   episode_length)`.
2. `eval_burst` — K greedy rollouts via vmap over fresh seeds;
   stacks per-episode results.
3. `train_with_eval` — Python outer loop running
   `total_steps // eval_every` super-steps. Each super-step runs
   `eval_every` training steps via the inner `Loop[C, T]` (jit-
   compiled), then one `eval_burst`. Returns
   `(final_state, ComposedTrace)` where `ComposedTrace.train` is
   the flat training trace `(total_steps,)`-shaped and
   `ComposedTrace.eval` is the burst-stacked eval trace
   `(n_bursts, K)`-shaped.

The eval pass is a separate measurement, NOT part of `dqn_step`.
This keeps the training step's slot-Protocol surface clean (no
new slots for eval-frequency) and lets eval be opt-in: callers
who only need training data use `scan_loop(dqn_step, ...)`;
callers who want overestimation gap data use `train_with_eval`."""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import jax
import jax.numpy as jnp

from corroborate.claim import claim
from corroborate.rl.dqn.state import DQNState
from corroborate.rl.dqn.types import QNetwork, StepRecord
from corroborate.rl.loop import Loop, scan_loop


# ============ Eval record shapes ============

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


type EvalTrajectoryRecord = dict[str, jax.Array]
"""After all bursts stack: each value has shape `(n_bursts, K)`
(per-episode quantities) or `(n_bursts,)` (per-burst metadata
like `eval_step_index`). Keyed by the same field names as
`EvalBurstOut` plus `eval_step_index`."""


class ComposedTrace(NamedTuple):
    """Compose training and eval traces under one return value.

    `train` is the same shape `dqn_step` produces — flat
    `(total_steps, ...)` per field. `eval` is burst-stacked:
    each field shaped `(n_bursts, K, ...)` for per-episode
    quantities, plus `eval_step_index: (n_bursts,)` recording at
    which training step each burst fired."""
    train: StepRecord
    eval: EvalTrajectoryRecord


# ============ Single greedy episode ============

@claim
def eval_episode(
    *,
    online_params: dict[str, jax.Array],
    env: object,
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
    obs_0, env_state_0 = env.reset(reset_key, env_params)  # type: ignore[attr-defined]

    # Predicted Q at start.
    q_at_start = q_network(online_params, obs_0)
    predicted_q_at_start = jnp.max(q_at_start)

    # Greedy rollout via lax.scan with manual termination.
    class Carry(NamedTuple):
        obs: jax.Array
        env_state: object
        done: jax.Array            # (1,) bool — once True, freeze
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
        # Greedy action under online net.
        q_values = q_network(online_params, carry.obs)
        action = jnp.argmax(q_values).astype(jnp.int32)

        env_key, next_rng = jax.random.split(carry.rng)
        next_obs, next_env_state, reward, done, _info = env.step(  # type: ignore[attr-defined]
            env_key, carry.env_state, action, env_params,
        )

        # Mask post-done updates: once done, freeze return / state.
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
    env: object,
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


# ============ train_with_eval — the composed loop ============

def train_with_eval(
    step_fn: Callable[[DQNState, jax.Array], tuple[DQNState, StepRecord]],
    init: DQNState,
    total_steps: int,
    *,
    eval_fn: Callable[[DQNState, jax.Array], EvalBurstOut],
    eval_every: int,
    inner_loop: Loop[DQNState, StepRecord] = scan_loop,
) -> tuple[DQNState, ComposedTrace]:
    """Python outer loop; each super-step runs `eval_every`
    training steps via `inner_loop` (default `scan_loop` —
    jit-compiled) then one `eval_burst` via `eval_fn`.

    `eval_fn` is the closure the caller built around `eval_burst`
    — it captures `env`, `env_params`, `q_network`, `gamma`,
    `episode_cap`, `n_episodes`. The framework intentionally
    doesn't bundle these into the `train_with_eval` signature
    because they're orthogonal to the loop's shape; the closure
    pattern keeps the loop primitive minimal.

    Returns:
        - final state after `total_steps` of training
        - `ComposedTrace(train, eval)`: train arrays are flat
          `(total_steps, ...)`; eval arrays are `(n_bursts, K)`
          stacks plus `eval_step_index: (n_bursts,)`.

    Raises `ValueError` if `total_steps` isn't a multiple of
    `eval_every` (avoids partial chunks at the tail)."""
    if total_steps % eval_every != 0:
        raise ValueError(
            f'total_steps ({total_steps}) must be a multiple of '
            f'eval_every ({eval_every}); got remainder '
            f'{total_steps % eval_every}',
        )

    n_super_steps = total_steps // eval_every
    state = init
    train_chunks: list[StepRecord] = []
    eval_bursts: list[EvalBurstOut] = []
    eval_step_indices: list[int] = []

    for super_idx in range(n_super_steps):
        # Inner loop: eval_every training steps (jit if scan_loop).
        state, chunk = inner_loop(step_fn, state, eval_every)
        train_chunks.append(chunk)

        # Eval burst at the END of this chunk (training has
        # advanced eval_every steps from previous burst).
        eval_step_at = (super_idx + 1) * eval_every
        eval_step_indices.append(eval_step_at)
        burst = eval_fn(state, jnp.uint32(super_idx))
        eval_bursts.append(burst)

    # Concatenate training chunks → (total_steps, ...) per field.
    def concat_along_zero(*arrays: jax.Array) -> jax.Array:
        return jnp.concatenate(arrays, axis=0)

    train_trace: StepRecord = jax.tree.map(concat_along_zero, *train_chunks)

    # Stack eval bursts → (n_super_steps, K, ...) per field.
    def stack_along_zero(*arrays: jax.Array) -> jax.Array:
        return jnp.stack(arrays, axis=0)

    stacked_predicted = stack_along_zero(*[b.predicted_q_at_start for b in eval_bursts])
    stacked_returns = stack_along_zero(*[b.mc_return for b in eval_bursts])
    stacked_lengths = stack_along_zero(*[b.episode_length for b in eval_bursts])

    eval_trace: EvalTrajectoryRecord = {
        'predicted_q_at_start': stacked_predicted,
        'mc_return': stacked_returns,
        'episode_length': stacked_lengths,
        'eval_step_index': jnp.asarray(eval_step_indices, dtype=jnp.int32),
    }

    return state, ComposedTrace(train=train_trace, eval=eval_trace)
