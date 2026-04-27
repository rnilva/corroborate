"""DQN phases — rollout / train / sync as separately-typed
functions composing slots.

Each phase advances `DQNState` by one piece of the algorithm:

- `rollout_phase`: select action, step env, store transition.
- `train_phase`: sample batch, compute TD-error, gradient step.
- `sync_phase`: target-network update.

Phases call slots through the Protocols in `types.py` — no
`jnp.argmax` / `jnp.max` / `jnp.dot` inline. All such primitives
live inside the `@claim`'d implementations under `claims/`. The
theory layer (`dqn.py`) composes phases; this layer composes
slots."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax

from corroborate.claim import claim
from corroborate.rl.dqn.claims import buffer_add, buffer_sample
from corroborate.rl.dqn.state import DQNState
from corroborate.rl.dqn.types import (
    ActionSelect,
    Bootstrap,
    EpsilonSchedule,
    LossFn,
    Params,
    QNetwork,
    TargetSync,
)


# ============ Per-phase output records ============

class RolloutOut(NamedTuple):
    """Diagnostic record from `rollout_phase`. The next-state's
    fields go on `DQNState`; this carries per-step scalars the
    record/bridges read.

    `action` exposes the actually-taken integer action so the
    Watkins-coverage invariant can verify the policy explored
    the action space."""
    epsilon: jax.Array
    reward: jax.Array
    done: jax.Array
    max_q: jax.Array      # max(Q(s, ·)) at action selection — overestimation diagnostic
    ep_return: jax.Array  # cumulative within current episode (reset on done in state)
    action: jax.Array     # int32 — the action ε-greedy returned this step


class TrainOut(NamedTuple):
    """Diagnostic record from `train_phase`.

    `online_argmax` and `target_argmax` are computed by the
    independence probe (see `_argmax_probe`) and let the
    Hasselt-independence invariant (`online_target_disagreement`)
    measure how often the two networks pick different actions.
    They are the same shape `(batch,)` so disagreement-rate is a
    plain element-wise mean.

    `sample_indices` carries the indices `buffer_sample` drew this
    step; the Lin-coverage invariant (`buffer_coverage`) reads
    these to verify the replay isn't sampling the same handful
    of transitions throughout training."""
    loss: jax.Array            # mean of per-sample losses
    td_error: jax.Array        # mean |predicted - target|, abs
    online_argmax: jax.Array   # (batch,) int32 — argmax_a Q_online(next_obs)
    target_argmax: jax.Array   # (batch,) int32 — argmax_a Q_target(next_obs)
    sample_indices: jax.Array  # (batch,) int32 — indices buffer_sample drew


def _argmax_probe(
    state: DQNState,
    q_network: QNetwork,
    next_obs_b: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Compute `argmax_a Q_online` and `argmax_a Q_target` on the
    same batch — the data the Hasselt-independence invariant
    needs to test the DDQN-decoupling assumption.

    Independent of which `bootstrap` slot is used. Cheap (two
    forward passes on the already-sampled batch); always-on so
    the invariant has data even when vanilla bootstrap is
    selected (the disagreement-rate is meaningful for both)."""
    online_q = q_network(state.online_params, next_obs_b)
    target_q = q_network(state.target_params, next_obs_b)
    return (
        jnp.argmax(online_q, axis=-1).astype(jnp.int32),
        jnp.argmax(target_q, axis=-1).astype(jnp.int32),
    )


# ============ Rollout ============

def rollout_phase(
    state: DQNState,
    *,
    env: object,
    env_params: object,
    n_actions: int,
    capacity: int,
    q_network: QNetwork,
    action_select: ActionSelect,
    eps_schedule: EpsilonSchedule,
) -> tuple[DQNState, RolloutOut]:
    """One step of acting in the env.

    Reads `q_network` to score the current observation, calls
    `action_select` (with `eps_schedule(step)` for ε), steps the
    env, appends the transition to the replay buffer."""
    # Q-values at current obs — single observation, not batched.
    q_values = q_network(state.online_params, state.obs)
    epsilon = eps_schedule(state.step)

    select_key, env_key, next_rng_key = jax.random.split(state.rng_key, 3)
    action = action_select(q_values, select_key, epsilon, n_actions)

    # gymnax's env.step returns (next_obs, env_state, reward, done, info)
    next_obs, next_env_state, reward, done, _info = env.step(  # type: ignore[attr-defined]
        env_key, state.env_state, action, env_params,
    )

    # Append to FIFO buffer.
    new_buf = buffer_add(
        state=state, capacity=capacity,
        obs=state.obs, action=action,
        reward=reward, next_obs=next_obs, done=done,
    )
    buf_obs, buf_action, buf_reward, buf_next_obs, buf_done, buf_size = new_buf

    # Episode return: accumulate this step's reward into the
    # running tally. The state's tally resets to 0 on done so the
    # next step starts a fresh episode; the record's `ep_return`
    # carries the cumulative value AT this step (so bridges
    # filtering on done==1 see the final per-episode return).
    cumulative = state.ep_return + reward
    next_ep_return = jnp.where(done, jnp.float32(0.0), cumulative)

    new_state = state._replace(
        buf_obs=buf_obs, buf_action=buf_action, buf_reward=buf_reward,
        buf_next_obs=buf_next_obs, buf_done=buf_done, buf_size=buf_size,
        env_state=next_env_state,
        obs=next_obs,
        rng_key=next_rng_key,
        ep_return=next_ep_return,
    )
    out = RolloutOut(
        epsilon=epsilon,
        reward=reward,
        done=done.astype(jnp.float32),
        max_q=jnp.max(q_values),
        ep_return=cumulative,
        action=action.astype(jnp.int32),
    )
    return new_state, out


# ============ Train ============

def train_phase(
    state: DQNState,
    *,
    q_network: QNetwork,
    bootstrap: Bootstrap,
    loss_fn: LossFn,
    optimizer: optax.GradientTransformation,
    gamma: float,
    batch_size: int,
    capacity: int,
    warmup_steps: int,
) -> tuple[DQNState, TrainOut]:
    """One gradient step on a batch sampled from the buffer.

    Skipped (no-op) until `state.step >= warmup_steps`; before
    warmup the buffer doesn't have enough transitions to train
    meaningfully. After warmup, samples uniformly from the
    populated portion."""
    sample_key, next_rng_key = jax.random.split(state.rng_key)

    obs_b, action_b, reward_b, next_obs_b, done_b, sample_indices = buffer_sample(
        state=state, rng_key=sample_key,
        batch_size=batch_size, capacity=capacity,
    )

    # Compute target via bootstrap slot — DDQN swaps live here.
    target_b = bootstrap(
        online_params=state.online_params,
        target_params=state.target_params,
        q_network=q_network,
        next_obs=next_obs_b, reward=reward_b, done=done_b,
        gamma=gamma,
    )

    # Always-on probe for the Hasselt-independence invariant. Two
    # forward passes; same batch as bootstrap so the disagreement
    # rate is faithful to the actual training data the swap
    # operates on. Independent of which bootstrap is selected.
    online_argmax, target_argmax = _argmax_probe(state, q_network, next_obs_b)

    def compute_loss(params: Params) -> tuple[jax.Array, jax.Array]:
        # Predicted Q for the action actually taken in each transition.
        q_b = q_network(params, obs_b)               # (batch, n_actions)
        predicted = jnp.take_along_axis(
            q_b, action_b[..., None], axis=-1,
        ).squeeze(-1)                                 # (batch,)
        per_sample = loss_fn(predicted, target_b)     # (batch,)
        return per_sample.mean(), jnp.abs(predicted - target_b).mean()

    (loss, td_error), grads = jax.value_and_grad(
        compute_loss, has_aux=True,
    )(state.online_params)

    updates, new_opt_state = optimizer.update(grads, state.opt_state, state.online_params)
    new_online = optax.apply_updates(state.online_params, updates)

    # Skip the gradient step before warmup — buffer is too small.
    skip = state.step < warmup_steps

    def select_param(new: jax.Array, old: jax.Array) -> jax.Array:
        return jnp.where(skip, old, new)

    final_online: Params = jax.tree.map(
        select_param, new_online, state.online_params,
    )

    new_state = state._replace(
        online_params=final_online,
        opt_state=new_opt_state,
        rng_key=next_rng_key,
    )
    out = TrainOut(
        loss=jnp.where(skip, jnp.float32(0.0), loss),
        td_error=jnp.where(skip, jnp.float32(0.0), td_error),
        online_argmax=online_argmax,
        target_argmax=target_argmax,
        sample_indices=sample_indices,
    )
    return new_state, out


# ============ Sync ============

def sync_phase(
    state: DQNState,
    *,
    target_sync: TargetSync,
    sync_period: int,
) -> DQNState:
    """Apply the target-network update rule. v0's `periodic_copy`
    triggers only every `sync_period` steps; the slot makes the
    cadence + rule pluggable (Polyak averaging is a future swap)."""
    new_target = target_sync(
        online_params=state.online_params,
        target_params=state.target_params,
        step=state.step,
        sync_period=sync_period,
    )
    return state._replace(target_params=new_target)


# Tag the phase functions as Claims via the @claim decorator. Done
# below rather than as a stacked decorator so each phase reads as
# a plain function in its definition (no decorator stack noise);
# the @claim re-bind makes them framework-introspectable.
rollout_phase = claim(rollout_phase)
train_phase = claim(train_phase)
sync_phase = claim(sync_phase)
