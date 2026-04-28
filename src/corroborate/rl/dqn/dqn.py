"""DQN — Mnih 2015 Algorithm 1 in paper-prose form.

Two claims live in this file:

- `dqn_step` — one training step (rollout → train → sync →
  record). Each phase is a typed slot composition.
- `dqn` — the OUTERMOST claim. The full training+eval run as one
  composition: init_state → nested scan over training+eval bursts
  → assembled record. Hypothesis intervention names `dqn`'s
  kwargs; the cell runner is a thin harness that vmaps `dqn` over
  seeds.

Exogenous kwargs (env, env_params, obs/action dims, eval episode
cap, state_hash, rng_key) carry `Annotated[T, Exogenous]`
markers — these are what we generalize *over*, not intervene on.
Everything else is HP and interventionable by default; authors
who want to hide an HP from intervention bake it in via
`functools.partial`."""
from __future__ import annotations

from functools import partial
from typing import Annotated

import jax
import jax.numpy as jnp
import optax

from corroborate.claim import claim
from corroborate.loop import scan_loop
from corroborate.rl.dqn.claims import (
    bootstrap as default_bootstrap,
    epsilon_greedy,
    mlp_q,
    periodic_copy,
    squared_error,
)
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.eval import EvalBurstOut, eval_burst
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
    LossFn,
    QFunction,
    StepRecord,
    TargetSync,
)
from corroborate.rl.env_catalogue import GymnaxEnvLike, StateHash
from corroborate.signature import Exogenous


def default_state_hash(obs: jax.Array) -> jax.Array:
    """Default `StateHash`: returns 0 for any obs. Sentinel
    wired when no env-specific state_hash is provided (image
    envs, or experiments not consuming the (s, a)-coverage gap).
    The gap measurable detects this via the env spec, not via
    record inspection — see `state_action_coverage_gap` in
    `invariants.py`."""
    del obs
    return jnp.int32(0)


def init_state_from_key(
    *,
    env: GymnaxEnvLike,
    env_params: object,
    obs_dim: int,
    n_actions: int,
    rng_key: jax.Array,
    optimizer: optax.GradientTransformation,
    q_network: QFunction = mlp_q,
    replay: Replay = Replay(),
) -> DQNState:
    """Build initial DQNState from a `jax.random.PRNGKey` directly.

    `q_network` and `replay` are Modules: each owns its own
    architecture / capacity HPs as fields, allocates its own
    sub-state via `init`, and is opaque to dqn beyond the
    Module-level operations. dqn doesn't see `buffer_capacity` or
    `batch_size` — those live on `Replay`.

    Vmap-friendly: under `jax.vmap` over a batched key array, this
    function produces a batched DQNState (each leaf has a leading
    seed-axis). `init_state` is the seed-int convenience wrapper
    for non-vmapped callers."""
    init_key, env_key, run_key = jax.random.split(rng_key, 3)
    online = q_network.init(init_key, obs_dim, n_actions)
    opt_state = optimizer.init(online)
    obs, env_state = env.reset(env_key, env_params)
    # Flatten env-side multi-dim obs (e.g. Catch's (5, 5) grid,
    # DeepSea's (8, 8)) to a 1D vector matching the flat MLP's
    # input shape. Conv-based q_networks would NOT use this
    # flattening; they'd pair with a different state-init path.
    obs = obs.reshape(obs_dim)
    return DQNState(
        online_params=online,
        target_params=online,
        opt_state=opt_state,
        replay=replay.init(obs_dim),
        env_state=env_state,
        obs=obs,
        step=jnp.int32(0),
        rng_key=run_key,
        ep_return=jnp.float32(0.0),
    )


def init_state(
    *,
    env: GymnaxEnvLike,
    env_params: object,
    obs_dim: int,
    n_actions: int,
    seed: int,
    optimizer: optax.GradientTransformation,
    q_network: QFunction = mlp_q,
    replay: Replay = Replay(),
) -> DQNState:
    """Build initial DQNState for a single env instance.

    Allocates parameter sets (online + target identical at t=0),
    initial optimizer state, FIFO replay buffer, env state via
    gymnax. RNG is split off the seed.

    Single-seed convenience wrapper; for vmap-over-seeds use
    `init_state_from_key` directly with a batched PRNGKey."""
    return init_state_from_key(
        env=env, env_params=env_params,
        obs_dim=obs_dim, n_actions=n_actions,
        rng_key=jax.random.PRNGKey(seed),
        optimizer=optimizer,
        q_network=q_network,
        replay=replay,
    )


@claim
def dqn_step(
    state: DQNState,
    idx: jax.Array,
    *,
    # Exogenous (env + numerical config; not slots)
    env: GymnaxEnvLike,
    env_params: object,
    n_actions: int,
    optimizer: optax.GradientTransformation,
    state_hash: StateHash = default_state_hash,
    # HPs (paper-honest where present in math; engineering otherwise)
    gamma: float = 0.99,
    warmup_steps: int = 1_000,
    sync_period: int = 100,
    # Slot Claims (each satisfies a Protocol in `types.py`)
    q_network: QFunction = mlp_q,
    action_select: ActionSelect = epsilon_greedy,
    bootstrap: Bootstrap = default_bootstrap,
    loss_fn: LossFn = squared_error,
    target_sync: TargetSync = periodic_copy,
    replay: Replay = Replay(),
) -> tuple[DQNState, StepRecord]:
    """One DQN step: rollout → train → sync → record.

    Reads top-to-bottom like Mnih 2015 Algorithm 1:

    1. Rollout: with ε-greedy on Q(s, ·), step the env, store
       transition.
    2. Train: sample a batch from replay, compute the bootstrap
       target via the `bootstrap` slot (vanilla / DDQN), gradient
       step the online network.
    3. Sync: update target network per the `target_sync` slot.

    Construction-time HPs travel with their owning Module:
    `replay.capacity`, `replay.batch_size`, `MLP.hidden`,
    `EpsilonGreedy.schedule`. dqn-level HPs (`gamma`,
    `warmup_steps`, `sync_period`) are paper-honest cross-cutting
    parameters that don't belong inside any single Module.

    DDQN intervention: `partial(dqn_step, bootstrap=partial(
    bootstrap, greedification=double_greedify))`. Schedule swap:
    `partial(dqn_step, action_select=replace(EpsilonGreedy(),
    schedule=other_schedule))`. Capacity swap: `partial(dqn_step,
    replay=replace(Replay(), capacity=50_000))`."""
    del idx  # `step` is on `state`; idx is the loop's bookkeeping arg

    # --- Rollout: act in env, store transition --------------------
    state, rollout_out = rollout_phase(
        state,
        env=env, env_params=env_params, n_actions=n_actions,
        replay=replay,
        q_network=q_network,
        action_select=action_select,
        state_hash=state_hash,
    )

    # --- Train: sample batch, bootstrap target, gradient step ----
    state, train_out = train_phase(
        state,
        q_network=q_network, bootstrap=bootstrap, loss_fn=loss_fn,
        optimizer=optimizer, gamma=gamma,
        replay=replay,
        warmup_steps=warmup_steps,
    )

    # --- Sync: target network update -----------------------------
    state = sync_phase(
        state, target_sync=target_sync, sync_period=sync_period,
    )

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
        'reward': rollout.reward,
        'done': rollout.done,
        'max_q': rollout.max_q,
        'ep_return': rollout.ep_return,
        'action': rollout.action,
        'state_hash': rollout.state_hash,
        'buf_size': rollout.buf_size,
        'loss': train.loss,
        'td_error': train.td_error,
        'online_q_values': train.online_q_values,
        'target_q_values': train.target_q_values,
        'sample_indices': train.sample_indices,
    }


# ============ dqn — outermost claim (full run) ============

@claim
def dqn(
    *,
    # Exogenous: per-cell conditions; we generalize *over* these.
    rng_key: Annotated[jax.Array, Exogenous],
    env: Annotated[GymnaxEnvLike, Exogenous],
    env_params: Annotated[object, Exogenous],
    obs_dim: Annotated[int, Exogenous],
    n_actions: Annotated[int, Exogenous],
    eval_episode_cap: Annotated[int, Exogenous] = 500,
    state_hash: Annotated[StateHash, Exogenous] = default_state_hash,
    # HPs (paper-honest where part of the math; engineering otherwise)
    optimizer: optax.GradientTransformation = optax.adam(1e-3),
    gamma: float = 0.99,
    warmup_steps: int = 1_000,
    sync_period: int = 100,
    total_steps: int = 50_000,
    eval_every: int = 5_000,
    n_episodes: int = 20,
    # Slot Claims (each satisfies a Protocol in `types.py`)
    q_network: QFunction = mlp_q,
    action_select: ActionSelect = epsilon_greedy,
    bootstrap: Bootstrap = default_bootstrap,
    loss_fn: LossFn = squared_error,
    target_sync: TargetSync = periodic_copy,
    replay: Replay = Replay(),
) -> dict[str, jax.Array]:
    """Full DQN training+eval run as one claim.

    Composition reads top-to-bottom:

    1. Split `rng_key` into init / run keys.
    2. `init_state_from_key(...)` — allocate params, opt state,
       replay buffer, env state.
    3. Nested scan: outer `scan_loop` over `total_steps //
       eval_every` super-steps; each super-step body runs an
       inner `scan_loop` over `eval_every` training steps then
       one `eval_burst`.
    4. Assemble the merged record dict — training fields shape
       `(total_steps, ...)` + eval fields shape `(n_bursts, K,
       ...)` + `eval_step_index` shape `(n_bursts,)`.

    HPs are flat top-level kwargs (gamma, batch_size, ...) —
    intervention names them directly. Architecture HPs (e.g.
    `MLP.hidden`) live on the slot Module that owns them, NOT
    here. Construction-time bake-ins (`partial(linear_epsilon,
    anneal_steps=...)`) flow through `_canonical_str`'s
    partial-canonicalisation cleanly, so substrate authors who
    bake-in are honest about WHAT was set.

    Eval IS part of training; bridges read whichever record keys
    they target without train/eval distinction.

    Hypothesis intervention names this claim's kwargs directly:
    `intervention={'bootstrap': partial(bootstrap, greedification=
    double_greedify), 'gamma': 0.95}` is just
    `partial(dqn, **intervention)`. No broadcast, no flatten, no
    validation — pyright catches signature mismatches at the swap
    site; `Annotated[..., Exogenous]` markers tell the framework
    which kwargs are NOT intervention surface.

    Raises `ValueError` if `total_steps` isn't a multiple of
    `eval_every`."""
    if total_steps % eval_every != 0:
        raise ValueError(
            f'total_steps ({total_steps}) must be a multiple of '
            f'eval_every ({eval_every}); got remainder '
            f'{total_steps % eval_every}',
        )
    if total_steps < eval_every:
        raise ValueError(
            f'total_steps ({total_steps}) must be ≥ eval_every '
            f'({eval_every}) — at least one super-step required.',
        )

    init_key, run_key = jax.random.split(rng_key, 2)
    state = init_state_from_key(
        env=env, env_params=env_params,
        obs_dim=obs_dim, n_actions=n_actions,
        rng_key=init_key, optimizer=optimizer,
        q_network=q_network,
        replay=replay,
    )

    step_fn = partial(
        dqn_step,
        env=env, env_params=env_params, n_actions=n_actions,
        optimizer=optimizer, state_hash=state_hash,
        gamma=gamma,
        warmup_steps=warmup_steps,
        sync_period=sync_period,
        q_network=q_network, action_select=action_select,
        bootstrap=bootstrap,
        loss_fn=loss_fn, target_sync=target_sync,
        replay=replay,
    )

    n_super_steps = total_steps // eval_every

    def super_step(
        s: DQNState, super_idx: jax.Array,
    ) -> tuple[DQNState, tuple[StepRecord, EvalBurstOut]]:
        # Inner scan: `eval_every` training steps. dqn_step
        # discards the index (its step counter lives on `state`),
        # so relative indices from scan_loop are fine.
        s, train_chunk = scan_loop(step_fn, s, eval_every)
        burst = eval_burst(
            online_params=s.online_params,
            env=env, env_params=env_params,
            rng_key=jax.random.fold_in(run_key, super_idx),
            q_network=q_network,
            gamma=gamma,
            episode_cap=eval_episode_cap,
            n_episodes=n_episodes,
        )
        return s, (train_chunk, burst)

    _final_state, (train_chunks, eval_bursts) = scan_loop(
        super_step, state, n_super_steps,
    )

    # train_chunks: pytree where each leaf has shape
    #   (n_super_steps, eval_every, *original_shape).
    # Reshape to (total_steps, *original_shape).
    def _flatten_chunks(x: jax.Array) -> jax.Array:
        return x.reshape(total_steps, *x.shape[2:])

    train_trace: StepRecord = jax.tree.map(  # pyright: ignore[reportAny]
        _flatten_chunks, train_chunks,
    )

    eval_step_indices = (
        jnp.arange(n_super_steps, dtype=jnp.int32) + 1
    ) * eval_every

    record: dict[str, jax.Array] = {**train_trace}
    record['predicted_q_at_start'] = eval_bursts.predicted_q_at_start
    record['mc_return'] = eval_bursts.mc_return
    record['episode_length'] = eval_bursts.episode_length
    record['eval_step_index'] = eval_step_indices

    return record
