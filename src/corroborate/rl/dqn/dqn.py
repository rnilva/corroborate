"""DQN — Mnih 2015 Algorithm 1 in paper-prose form.

`dqn_step` reads top-to-bottom: rollout → train → sync → record.
Each phase is a typed slot composition; each slot is a Protocol
contract from `types.py`; each slot's default implementation is
under `claims/`. JAX primitives (`jnp.dot`, `jnp.argmax`, etc.)
are NOT in this file — they live inside the `@claim`'d
implementations where they're a replaceable unit.

Intervention: `partial(dqn_step, bootstrap=ddqn_bootstrap)` swaps
the bootstrap slot wholesale. Hypothesis.intervention's mapping
is exactly the kwargs `dqn_step` accepts."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from corroborate.claim import claim
from corroborate.rl.dqn.claims import (
    epsilon_greedy,
    init_mlp,
    linear_epsilon,
    mlp_q,
    periodic_copy,
    squared_error,
    vanilla_bootstrap,
)
from corroborate.rl.dqn.claims.replay import buffer_init
from corroborate.rl.dqn.phases import (
    RolloutOut,
    TrainOut,
    rollout_phase,
    sync_phase,
    train_phase,
)
from corroborate.rl.dqn.state import DQNState
from corroborate.rl.dqn.types import (
    ActionSelect,
    Bootstrap,
    EpsilonSchedule,
    LossFn,
    QNetwork,
    StepRecord,
    TargetSync,
)
from corroborate.rl.env_catalogue import StateHash


def default_state_hash(obs: jax.Array) -> jax.Array:
    """Default `StateHash`: returns 0 for any obs. Sentinel
    wired when no env-specific state_hash is provided (image
    envs, or experiments not consuming the (s, a)-coverage gap).
    The gap measurable detects this via the env spec, not via
    record inspection — see `state_action_coverage_gap` in
    `invariants.py`."""
    del obs
    return jnp.int32(0)


def init_state(
    *,
    env: object,
    env_params: object,
    obs_dim: int,
    n_actions: int,
    seed: int,
    hidden: tuple[int, ...] = (64, 64),
    buffer_capacity: int = 10_000,
    optimizer: optax.GradientTransformation,
) -> DQNState:
    """Build initial DQNState for a single env instance.

    Allocates parameter sets (online + target identical at t=0),
    initial optimizer state, FIFO replay buffer, env state via
    gymnax. RNG is split off the seed."""
    rng = jax.random.PRNGKey(seed)
    init_key, env_key, run_key = jax.random.split(rng, 3)
    online = init_mlp(init_key, obs_dim, n_actions, hidden=hidden)
    opt_state = optimizer.init(online)
    obs, env_state = env.reset(env_key, env_params)  # type: ignore[attr-defined]
    buf_obs, buf_action, buf_reward, buf_next_obs, buf_done, buf_size = (
        buffer_init(buffer_capacity, obs_dim)
    )
    return DQNState(
        online_params=online,
        target_params=online,
        opt_state=opt_state,
        buf_obs=buf_obs,
        buf_action=buf_action,
        buf_reward=buf_reward,
        buf_next_obs=buf_next_obs,
        buf_done=buf_done,
        buf_size=buf_size,
        env_state=env_state,
        obs=obs,
        step=jnp.int32(0),
        rng_key=run_key,
        ep_return=jnp.float32(0.0),
    )


@claim
def dqn_step(
    state: DQNState,
    idx: jax.Array,
    *,
    # Exogenous (env + numerical config; not slots)
    env: object,
    env_params: object,
    n_actions: int,
    optimizer: optax.GradientTransformation,
    state_hash: StateHash = default_state_hash,
    gamma: float = 0.99,
    batch_size: int = 64,
    buffer_capacity: int = 10_000,
    warmup_steps: int = 1_000,
    sync_period: int = 100,
    # Slots (each conforms to a Protocol in `types.py`)
    q_network: QNetwork = mlp_q,
    action_select: ActionSelect = epsilon_greedy,
    eps_schedule: EpsilonSchedule = linear_epsilon,
    bootstrap: Bootstrap = vanilla_bootstrap,
    loss_fn: LossFn = squared_error,
    target_sync: TargetSync = periodic_copy,
) -> tuple[DQNState, StepRecord]:
    """One DQN step: rollout → train → sync → record.

    Reads top-to-bottom like Mnih 2015 Algorithm 1:

    1. Rollout: with ε-greedy on Q(s, ·), step the env, store
       transition.
    2. Train: sample a batch from replay, compute the bootstrap
       target via the `bootstrap` slot (vanilla / DDQN), gradient
       step the online network.
    3. Sync: update target network per the `target_sync` slot.

    DDQN intervention: `partial(dqn_step, bootstrap=ddqn_bootstrap)`.
    The slot-Protocol contract ensures the alternative has a
    matching signature; pyright catches mismatches at the swap
    site."""
    del idx  # `step` is on `state`; idx is the loop's bookkeeping arg

    # --- Rollout: act in env, store transition --------------------
    state, rollout_out = rollout_phase(
        state,
        env=env, env_params=env_params, n_actions=n_actions,
        capacity=buffer_capacity,
        q_network=q_network,
        action_select=action_select,
        eps_schedule=eps_schedule,
        state_hash=state_hash,
    )

    # --- Train: sample batch, bootstrap target, gradient step ----
    state, train_out = train_phase(
        state,
        q_network=q_network, bootstrap=bootstrap, loss_fn=loss_fn,
        optimizer=optimizer, gamma=gamma,
        batch_size=batch_size, capacity=buffer_capacity,
        warmup_steps=warmup_steps,
    )

    # --- Sync: target network update -----------------------------
    state = sync_phase(state, target_sync=target_sync, sync_period=sync_period)

    # --- Step counter advance (do this last so phases see the
    #     pre-advance step for warmup / sync gating) -------------
    state = state._replace(step=state.step + 1)

    record = _build_record(rollout_out, train_out)
    return state, record


def _build_record(
    rollout: RolloutOut, train: TrainOut,
) -> StepRecord:
    """Assemble the per-step record from phase outputs.

    Keys are semantic-role names bridges target by paper-prose
    (`max_q`, `loss`, `ep_return`, etc.). Adding a new diagnostic
    is one line in a phase + one line here.

    The Q-value fields (`online_q_values`, `target_q_values`)
    carry shape `(batch, n_actions)` per step; stacking adds a
    leading T → `(T, batch, n_actions)`. `sample_indices` is
    `(batch,)` per step → `(T, batch)`. Reductions flatten or
    fold the appropriate axes at consumption time; the record
    deliberately stores raw values, not pre-reductions."""
    return {
        'epsilon': rollout.epsilon,
        'reward': rollout.reward,
        'done': rollout.done,
        'max_q': rollout.max_q,
        'ep_return': rollout.ep_return,
        'action': rollout.action,
        'state_hash': rollout.state_hash,
        'loss': train.loss,
        'td_error': train.td_error,
        'online_q_values': train.online_q_values,
        'target_q_values': train.target_q_values,
        'sample_indices': train.sample_indices,
    }
