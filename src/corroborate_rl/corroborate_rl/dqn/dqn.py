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
from typing import TYPE_CHECKING, Annotated

import jax
import jax.numpy as jnp
import optax
from gymnax import EnvParams

from corroborate import claim
from corroborate_rl.dqn.claims import (
    bootstrap as default_bootstrap,
    epsilon_greedy,
    mlp_q,
    periodic_copy,
    squared_error,
)
from corroborate_rl.dqn.claims.action_select import ActionSelect
from corroborate_rl.dqn.claims.optimizer import (
    OptimizerFactory,
    default_optimizer,
)
from corroborate_rl.dqn.claims.q_network import Params, QFunction
from corroborate_rl.dqn.claims.replay import Replay, init_pending_n_step
from corroborate_rl.dqn.eval import EvalBurstOut, eval_burst, train_with_eval
from corroborate_rl.dqn.phases import (
    rollout_phase,
    sync_phase,
    train_phase,
)
from corroborate_rl.dqn.state import DQNState
from corroborate_rl.dqn.types import (
    Bootstrap,
    LossFn,
    StepRecord,
    TargetSync,
)
from corroborate_rl.env_catalogue import EnvWrapper, StateHash
from corroborate.core.signature import Exogenous

if TYPE_CHECKING:
    # Stub-only Protocol — gymnax doesn't export `Env` at runtime
    # (the real class is `gymnax.environments.environment.Environment`).
    # `from __future__ import annotations` stringifies all annotations
    # below, so this TYPE_CHECKING import is sufficient for pyright.
    from gymnax import Env


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
    env: Env,
    env_params: EnvParams,
    obs_shape: tuple[int, ...],
    n_actions: int,
    rng_key: jax.Array,
    optimizer: optax.GradientTransformation,
    q_network: QFunction = mlp_q,
    replay: Replay = Replay(),
    state_hash_cardinality: int = 1,
    init_online_params: Params | None = None,
) -> DQNState:
    """Build initial DQNState from a `jax.random.PRNGKey` directly.

    `q_network` and `replay` are config bundles: each owns its
    own architecture / capacity HPs as fields, allocates its own
    sub-state via `init`, and is opaque to dqn beyond the
    bundle-level operations. dqn doesn't see `buffer_capacity` or
    `batch_size` — those live on `Replay`.

    Vmap-friendly: under `jax.vmap` over a batched key array, this
    function produces a batched DQNState (each leaf has a leading
    seed-axis). For non-vmap callers, pass
    `rng_key=jax.random.PRNGKey(seed)` directly."""
    init_key, env_key, run_key = jax.random.split(rng_key, 3)
    if init_online_params is not None:
        online = init_online_params
    else:
        online = q_network.init(init_key, obs_shape, n_actions)
    opt_state = optimizer.init(online)
    obs, env_state = env.reset(env_key, env_params)
    # Substrate stores obs at native shape — q_network handles the
    # input shape (MLP flattens trailing dims internally; CNN reads
    # the spatial structure directly). Replay stores at native
    # shape too, so the rank flowing through training is consistent.
    return DQNState(
        online_params=online,
        target_params=online,
        opt_state=opt_state,
        replay=replay.init(obs_shape),
        pending_n_step=init_pending_n_step(obs_shape),
        env_state=env_state,
        obs=obs,
        step=jnp.int32(0),
        rng_key=run_key,
        ep_return=jnp.float32(0.0),
        state_hash_count=jnp.zeros(
            (state_hash_cardinality,), dtype=jnp.int32,
        ),
    )


@claim
def dqn_step(
    state: DQNState,
    idx: jax.Array,
    *,
    env: Env,
    env_params: EnvParams,
    n_actions: int,
    optimizer: optax.GradientTransformation,
    state_hash: StateHash = default_state_hash,
    gamma: float = 0.99,
    sync_period: int = 100,
    n_step: int = 1,
    q_network: QFunction = mlp_q,
    action_select: ActionSelect = epsilon_greedy,
    replay: Replay = Replay(),
    bootstrap: Bootstrap = default_bootstrap,
    loss_fn: LossFn = squared_error,
    target_sync: TargetSync = periodic_copy,
    count_weight_alpha: float = 0.0,
) -> tuple[DQNState, StepRecord]:
    """One DQN step: rollout → train → sync → record.

    Reads top-to-bottom like Mnih 2015 Algorithm 1:

    1. Rollout: ε-greedy on Q(s, ·), step the env, store transition.
    2. Train: sample batch from replay, bootstrap target, gradient
       step the online network.
    3. Sync: update target network.

    Construction-time HPs travel with their owning component:
    `replay.capacity`, `replay.batch_size`, `MLP.hidden` live on
    config bundles (`Replay`, `MLP`); `epsilon_greedy.schedule`,
    `warmed_update.warmup_steps`, `adam.lr` live on `@claim`
    factory signatures, baked at composition time via `partial`.
    The only top-level HPs are `gamma` (Bellman γ) and
    `sync_period` (target-sync cadence) — paper-honest cross-
    cutting parameters that don't belong inside any single
    component.

    Author interventions read uniformly — name the kwarg, supply
    the alternative:

        # DDQN: swap the greedification slot inside bootstrap
        partial(dqn_step, bootstrap=partial(
            bootstrap, greedification=double_greedify))
        # Schedule swap: bake schedule into action_select
        partial(dqn_step, action_select=partial(
            epsilon_greedy, schedule=other_schedule))
        # Capacity swap: replace a config-bundle field
        partial(dqn_step, replay=replace(Replay(), capacity=50_000))
        # Q-network swap: another config bundle
        partial(dqn_step, q_network=MLP(hidden=(128,)))
        # Optimizer swap: factory partial (this kwarg is on `dqn`,
        # threaded through `dqn_step` as the raw optax handle —
        # see `dqn`'s docstring)
        partial(dqn_step, optimizer=partial(rmsprop, lr=2.5e-4))"""
    del idx  # `step` is on `state`; idx is the loop's bookkeeping arg

    state, rollout = rollout_phase(
        state,
        env=env, env_params=env_params, n_actions=n_actions,
        replay=replay,
        q_network=q_network,
        action_select=action_select,
        state_hash=state_hash,
        n_step=n_step, gamma=gamma,
    )

    state, train = train_phase(
        state,
        q_network=q_network, bootstrap=bootstrap, loss_fn=loss_fn,
        optimizer=optimizer, gamma=gamma, n_step=n_step,
        replay=replay, state_hash=state_hash,
        count_weight_alpha=count_weight_alpha,
    )

    state = sync_phase(
        state, target_sync=target_sync, sync_period=sync_period,
    )

    # Step counter advance (last so phases see the pre-advance
    # step for sync gating).
    state = state._replace(step=state.step + 1)

    # Per-step record = union of phase-emitted diagnostic dicts.
    return state, {**rollout, **train}


# ============ dqn — outermost claim (full run) ============

@claim
def dqn(
    *,
    # Per-cell author primitives — what the experimenter generalizes
    # OVER across cells. `env_name`/`seed`/`wrappers` are author-set
    # at design time but vary per cell (the grid-loop dimension);
    # `env`/`env_params`/`n_actions`/etc. are framework-derived from
    # them by the cell_runner. All seven are `Annotated[..., Exogenous]`
    # — the framework's "we generalize over this, not intervene on
    # it" marker. The endogeneity gate consumes leaf ∪ exogenous as
    # the substrate's author-primitive set
    # (cf. ENDOGENEITY_TOPOLOGY.md).
    env_name: Annotated[str, Exogenous],
    seed: Annotated[int, Exogenous] = 0,
    wrappers: Annotated[tuple[EnvWrapper, ...], Exogenous] = (),
    env: Annotated[Env, Exogenous],
    env_params: Annotated[EnvParams, Exogenous],
    obs_shape: Annotated[tuple[int, ...], Exogenous],
    n_actions: Annotated[int, Exogenous],
    eval_episode_cap: Annotated[int, Exogenous] = 500,
    state_hash: Annotated[StateHash, Exogenous] = default_state_hash,
    state_hash_cardinality: Annotated[int, Exogenous] = 1,
    # Q-network checkpoint persistence flags. Exogenous because
    # they're framework-side bookkeeping (whether to snapshot
    # params for post-hoc Q-evaluation analyses) — NOT theoretical
    # content, so they don't change the leaf-signature fingerprint
    # that groups cells into arms. The cell runner intercepts the
    # sentinel-prefixed checkpoint keys these enable.
    keep_q_checkpoint_final: Annotated[bool, Exogenous] = False,
    keep_q_checkpoint_per_burst: Annotated[bool, Exogenous] = False,
    # Optional initialization-from-checkpoint override. When set,
    # replaces the freshly-initialized online_params (and target_params
    # via copy) inside init_state. Used for "continue training from a
    # saved policy" interventions. Exogenous because it's per-cell
    # data, not a theoretical knob.
    init_online_params: Annotated[Params | None, Exogenous] = None,
    # Cross-cutting HPs (no single Module owns these).
    gamma: float = 0.99,
    sync_period: int = 100,
    n_step: int = 1,
    total_steps: int = 50_000,
    eval_every: int = 5_000,
    n_episodes: int = 20,
    # Slot Claims.
    q_network: QFunction = mlp_q,
    action_select: ActionSelect = epsilon_greedy,
    replay: Replay = Replay(),
    bootstrap: Bootstrap = default_bootstrap,
    loss_fn: LossFn = squared_error,
    target_sync: TargetSync = periodic_copy,
    optimizer: OptimizerFactory = default_optimizer,
    count_weight_alpha: float = 0.0,
) -> dict[str, jax.Array]:
    """Full DQN training+eval run as one claim.

    Composition:

    1. `init_state(rng_key, ...)` — allocate params, opt
       state, replay buffer, env state.
    2. `train_with_eval(step_fn, eval_fn, ...)` — nested scan
       driver: outer over super-steps (one eval burst at the end
       of each), inner over training steps. Returns the merged
       record dict.

    Eval IS part of training; bridges read whichever record keys
    they target without train/eval distinction. Adding a
    diagnostic = adding a key in the owning phase's dict
    (rollout / train / eval).

    Each kwarg of `dqn` is a Claim-graph slot the substrate may
    intervene on. `Annotated[..., Exogenous]` markers declare which
    kwargs are NOT intervention surface (env, rng_key, etc.);
    everything else is interventionable.

    **Intervention cookbook.** Wrap each substitution in
    `Intervention(slot_path=..., replacement=...)` and assemble
    `DoEffect(arms=tuple-of-tuples)`. Five canonical
    patterns for the `replacement`:

        # 1. Cross-cutting scalar HP — pass the value directly.
        Intervention(slot_path='gamma', replacement=0.95)

        # 2. Config bundle — instantiate with the new field.
        # `MLP`, `CNN`, `Replay` are frozen dataclasses; pass an
        # instance. `replace(MLP(), hidden=...)` is equivalent.
        Intervention(slot_path='q_network',
                     replacement=MLP(hidden=(128,)))
        Intervention(slot_path='replay',
                     replacement=Replay(capacity=50_000))

        # 3. Free Claim with sub-slot — bake the sub-slot via
        # `partial`. The walker recurses into the partial and
        # surfaces the sub-leaf at composition time.
        Intervention(slot_path='bootstrap', replacement=partial(
            bootstrap, greedification=double_greedify))
        Intervention(slot_path='action_select', replacement=partial(
            epsilon_greedy, schedule=cosine_epsilon))

        # 4. Free Claim factory — `partial(factory, **leaves)`.
        # Same pattern as (3); the factory returns a runtime
        # handle (e.g. `optax.GradientTransformation`).
        Intervention(slot_path='optimizer',
                     replacement=partial(rmsprop, lr=2.5e-4))

        # 5. Wholesale Free Claim swap — supply a different
        # `@claim` function. Loss / target_sync / etc. are pure
        # Free Claims; just hand over a sibling.
        Intervention(slot_path='loss_fn', replacement=huber_loss)
        Intervention(slot_path='target_sync',
                     replacement=polyak_average)

    Combine freely. The DDQN-vs-vanilla contrast at γ=0.95:

        DoEffect(arms=(
            (
                Intervention(slot_path='gamma', replacement=0.95),
            ),
            (
                Intervention(slot_path='bootstrap', replacement=partial(
                    bootstrap, greedification=double_greedify)),
                Intervention(slot_path='gamma', replacement=0.95),
            ),
        ))

    Raises `ValueError` if `total_steps` isn't a multiple of
    `eval_every`."""
    # `env_name` and `wrappers` are structural markers — author
    # primitives the cell_runner consumed Python-side to build
    # `env` and `env_params`. They appear in `walk_paths(dqn).leaves`
    # so the framework's endogeneity gate can classify them
    # (cf. ENDOGENEITY_TOPOLOGY.md); the dqn body itself only sees
    # the resolved env.
    del env_name, wrappers
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

    # Build the optax handle once. The Module-shaped
    # `OptimizerFactory` (Adam / RMSProp / WarmedUpdate)
    # canonicalises cleanly in mechanism_key; the raw
    # GradientTransformation is what dqn_step needs internally.
    optax_handle = optimizer()

    # Derive rng_key from seed JAX-side so vmap-over-seeds threads
    # cleanly through PRNGKey. Replaces the previous
    # `Annotated[jax.Array, Exogenous] rng_key` kwarg — the runner
    # used to vmap over pre-built rng_keys; now seeds are the
    # vmap dimension and PRNGKey moves into the trace.
    rng_key = jax.random.PRNGKey(seed)
    init_key, run_key = jax.random.split(rng_key, 2)
    state = init_state(
        env=env, env_params=env_params,
        obs_shape=obs_shape, n_actions=n_actions,
        rng_key=init_key, optimizer=optax_handle,
        q_network=q_network,
        replay=replay,
        state_hash_cardinality=state_hash_cardinality,
        init_online_params=init_online_params,
    )

    step_fn = partial(
        dqn_step,
        env=env, env_params=env_params, n_actions=n_actions,
        optimizer=optax_handle, state_hash=state_hash,
        gamma=gamma, sync_period=sync_period, n_step=n_step,
        q_network=q_network, action_select=action_select,
        replay=replay,
        bootstrap=bootstrap,
        loss_fn=loss_fn, target_sync=target_sync,
        count_weight_alpha=count_weight_alpha,
    )

    def eval_fn(s: DQNState, super_idx: jax.Array) -> EvalBurstOut:
        return eval_burst(
            online_params=s.online_params,
            env=env, env_params=env_params,
            rng_key=jax.random.fold_in(run_key, super_idx),
            q_network=q_network, gamma=gamma,
            episode_cap=eval_episode_cap, n_episodes=n_episodes,
        )

    return train_with_eval(
        step_fn=step_fn, eval_fn=eval_fn,
        init_state=state,
        total_steps=total_steps, eval_every=eval_every,
        keep_q_checkpoint_final=keep_q_checkpoint_final,
        keep_q_checkpoint_per_burst=keep_q_checkpoint_per_burst,
    )
