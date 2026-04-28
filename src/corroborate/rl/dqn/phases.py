"""DQN phases — rollout / train / sync as `@claim`'d functions
returning their diagnostic dicts directly.

Each phase advances `DQNState` by one piece of the algorithm AND
emits a dict of per-step diagnostics. The phase claim's output
IS the measurable surface — `dqn_step` composes phases via
`{**rollout, **train}`, no hand-aggregated `_build_record`.

Phases:

- `rollout_phase`: select action, step env, store transition.
  Emits `reward, done, max_q, ep_return, action, state_hash,
  buf_size`.
- `train_phase`: sample batch, compute TD-error, gradient step.
  Emits `loss, td_error, online_q_values, target_q_values,
  sample_indices`.
- `sync_phase`: target-network update. Emits no diagnostic
  (state-only).

Phases call slots through Protocols in `types.py` — no
`jnp.argmax` / `jnp.max` / `jnp.dot` inline. JAX primitives live
inside the `@claim`'d implementations under `claims/`. The theory
layer (`dqn.py`) composes phases; this layer composes slots."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from corroborate.claim import claim
from corroborate.rl.dqn.claims.replay import Replay, Transition
from corroborate.rl.dqn.state import DQNState
from corroborate.rl.dqn.types import (
    ActionSelect,
    Bootstrap,
    LossFn,
    Params,
    QFunction,
    TargetSync,
)
from corroborate.rl.env_catalogue import GymnaxEnvLike, StateHash


# ============ Rollout ============

@claim
def rollout_phase(
    state: DQNState,
    *,
    env: GymnaxEnvLike,
    env_params: object,
    n_actions: int,
    replay: Replay,
    q_network: QFunction,
    action_select: ActionSelect,
    state_hash: StateHash,
) -> tuple[DQNState, dict[str, jax.Array]]:
    """One step of acting in the env.

    Reads `q_network` to score the current observation, calls
    `action_select(q_values, key, step, n_actions)`, steps the
    env, appends the transition to `replay`.

    Returns `(new_state, diagnostic_dict)` — the dict's keys
    (`reward, done, max_q, ep_return, action, state_hash,
    buf_size`) are the measurable signals bridges target."""
    # Q-values at current obs — single observation, not batched.
    q_values = q_network(state.online_params, state.obs)

    select_key, env_key, next_rng_key = jax.random.split(state.rng_key, 3)
    action = action_select(q_values, select_key, state.step, n_actions)

    # gymnax's env.step returns (next_obs, env_state, reward, done, info)
    next_obs, next_env_state, reward, done, _info = env.step(
        env_key, state.env_state, action.astype(jnp.int32), env_params,
    )
    # Flatten multi-dim obs to match the flat-MLP's input shape
    # (state.obs is already flat from init_state).
    next_obs = next_obs.reshape(state.obs.shape)

    new_replay = replay.add(state.replay, Transition(
        obs=state.obs, action=action,
        reward=reward, next_obs=next_obs, done=done,
    ))

    # Episode return: accumulate this step's reward into the
    # running tally. The state's tally resets to 0 on done so the
    # next step starts a fresh episode; the record's `ep_return`
    # carries the cumulative value AT this step (so bridges
    # filtering on done==1 see the final per-episode return).
    cumulative = state.ep_return + reward
    next_ep_return = jnp.where(done, jnp.float32(0.0), cumulative)

    new_state = state._replace(
        replay=new_replay,
        env_state=next_env_state,
        obs=next_obs,
        rng_key=next_rng_key,
        ep_return=next_ep_return,
    )
    # State-hash logged at action-selection time (the state
    # observed when the action was chosen, not after the env step).
    obs_hash = state_hash(state.obs).astype(jnp.int32)

    diagnostics: dict[str, jax.Array] = {
        'reward': reward,
        'done': done.astype(jnp.float32),
        'max_q': jnp.max(q_values),
        'ep_return': cumulative,
        'action': action.astype(jnp.int32),
        'state_hash': obs_hash,
        'buf_size': new_replay.size.astype(jnp.int32),
    }
    return new_state, diagnostics


# ============ Train ============

@claim
def train_phase(
    state: DQNState,
    *,
    q_network: QFunction,
    bootstrap: Bootstrap,
    loss_fn: LossFn,
    optimizer: optax.GradientTransformation,
    gamma: float,
    replay: Replay,
) -> tuple[DQNState, dict[str, jax.Array]]:
    """One gradient step: sample batch → bootstrap target →
    compute loss → apply update.

    Reads paper-honestly. Buffer warmup (skipping params updates
    until enough transitions are stored) lives on the optimizer
    via `WarmedUpdate(inner=..., warmup_steps=...)` — not in this
    phase. Authors who don't want warmup pass an unwrapped
    `Adam()` / `RMSProp()` directly.

    Returns `(new_state, diagnostic_dict)`. Dict keys: `loss,
    td_error, online_q_values, target_q_values, sample_indices`.
    Q-vectors are full `(batch, n_actions)` — bridges that need
    argmaxes derive post-hoc."""
    sample_key, next_rng_key = jax.random.split(state.rng_key)

    batch = replay.sample_batch(state.replay, sample_key)

    # Always-on probe for derived Q measurables. Compute sufficient
    # statistics in-loop and DO NOT propagate the full
    # `(batch, n_actions)` tensors — those stacked over scan
    # `(super_steps, seeds, train_steps, batch, n_actions)` OOM the
    # device for high-action envs (observed at MNISTBandit /
    # BernoulliBandit / MinAtar with n_actions ≥ 6 — autotune asks
    # for ~4GB on a transpose fusion). The Pearson-r measurable
    # (hasselt_covariance_gap) reads the per-step (mean, mean_sq,
    # cross_mean) sum-stats below and aggregates post-hoc — same
    # information without the materialised tensors.
    online_q_full = q_network(state.online_params, batch.next_obs)
    target_q_full = q_network(state.target_params, batch.next_obs)
    # Per-step Q reductions for the measurable layer (q_mean, q_max,
    # q_std, q_gap-via-(top-second)).
    online_q_per_action = online_q_full.mean(axis=0)  # avg over batch
    target_q_per_action = target_q_full.mean(axis=0)
    # Pearson sum-stats: enough to compute correlation post-hoc
    # without storing the full tensors.
    on_flat = online_q_full.reshape(-1)
    tg_flat = target_q_full.reshape(-1)
    pearson_stats = jnp.stack([
        on_flat.mean(),
        tg_flat.mean(),
        (on_flat ** 2).mean(),
        (tg_flat ** 2).mean(),
        (on_flat * tg_flat).mean(),
    ])  # shape (5,) per step

    def compute_loss(params: Params) -> tuple[jax.Array, jax.Array]:
        q_b = q_network(params, batch.obs)            # (batch, n_actions)
        predicted = jnp.take_along_axis(
            q_b, batch.action[..., None], axis=-1,
        ).squeeze(-1)                                  # (batch,)
        # Target computed *inside* the loss closure so
        # `gradient_rule` (semi_gradient vs full_gradient) actually
        # controls cotangent flow. With target hoisted outside, it
        # becomes a constant under value_and_grad and the
        # stop_gradient is theatre.
        target = bootstrap(
            online_params=params,
            target_params=state.target_params,
            q_network=q_network,
            next_obs=batch.next_obs, reward=batch.reward, done=batch.done,
            gamma=gamma,
        )
        per_sample = loss_fn(predicted, target)        # (batch,)
        return per_sample.mean(), jnp.abs(predicted - target).mean()

    (loss, td_error), grads = jax.value_and_grad(
        compute_loss, has_aux=True,
    )(state.online_params)

    updates, new_opt_state = optimizer.update(
        grads, state.opt_state, state.online_params,
    )
    new_online = optax.apply_updates(state.online_params, updates)

    new_state = state._replace(
        online_params=new_online,
        opt_state=new_opt_state,
        rng_key=next_rng_key,
    )
    diagnostics: dict[str, jax.Array] = {
        'loss': loss,
        'td_error': td_error,
        # Pre-reduced Q-summaries: per-step (n_actions,) vectors
        # instead of the full (batch, n_actions). Shrinks the
        # per-step trace ~64× without losing the action-axis
        # structure bridges might want.
        'online_q_per_action': online_q_per_action,
        'target_q_per_action': target_q_per_action,
        # Pearson sufficient-stats: 5 scalars/step. The
        # `hasselt_covariance_gap` measurable aggregates these
        # post-hoc to recover the population-level Pearson r over
        # all (s', a) pairs across training.
        'pearson_stats': pearson_stats,
        'sample_indices': batch.indices,
    }
    return new_state, diagnostics


# ============ Sync ============

@claim
def sync_phase(
    state: DQNState,
    *,
    target_sync: TargetSync,
    sync_period: int,
) -> DQNState:
    """Apply the target-network update rule. v0's `periodic_copy`
    triggers only every `sync_period` steps; the slot makes the
    cadence + rule pluggable (Polyak averaging is a future swap).

    Emits no diagnostic — sync is state-only. If a future
    target_sync rule wants to expose a "did_sync" signal, this
    phase would adopt the same `(state, dict)` return shape."""
    new_target = target_sync(
        online_params=state.online_params,
        target_params=state.target_params,
        step=state.step,
        sync_period=sync_period,
    )
    return state._replace(target_params=new_target)
