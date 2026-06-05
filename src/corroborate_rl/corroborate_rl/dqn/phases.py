"""DQN phases — rollout / train / sync as `@claim`'d functions
returning their diagnostic dicts directly.

Each phase advances `DQNState` by one piece of the algorithm AND
emits a dict of per-step diagnostics. The phase claim's output
IS the measurable surface — `dqn_step` composes phases via
`{**rollout, **train}`, no hand-aggregated `_build_record`.

Phases:

- `rollout_phase`: select action, step env, store transition.
  Emits `reward, done, max_q, ep_return, action, state_hash_per_step,
  buf_size`.
- `train_phase`: sample batch, compute TD-error, gradient step.
  Emits `loss, td_error, online_q_per_action, target_q_per_action,
  pearson_stats`.
- `sync_phase`: target-network update. Emits no diagnostic
  (state-only).

Phases call slots through Protocols in `types.py` — no
`jnp.argmax` / `jnp.max` / `jnp.dot` inline. JAX primitives live
inside the `@claim`'d implementations under `claims/`. The theory
layer (`dqn.py`) composes phases; this layer composes slots."""
from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import optax
from gymnax import EnvParams

from corroborate import claim
from corroborate_rl.dqn._step_boundary import env_step_typed
from corroborate_rl.dqn.claims.replay import (
    Replay, Transition, n_step_return,
)
from corroborate_rl.dqn.state import DQNState
from corroborate_rl.dqn.claims.action_select import ActionSelect
from corroborate_rl.dqn.claims.q_network import Params, QFunction
from corroborate_rl.dqn.types import (
    Bootstrap,
    LossFn,
    TargetSync,
)
from corroborate_rl.env_catalogue import StateHash


# ============ Symmetric Double Q-learning — the evaluator edge ======
#
# Faithful van Hasselt 2010 lives in the `paired_dqn` PROGRAM
# (dqn_paired.py), NOT in a `dqn`-step flag. The single principled
# surface `train_phase` exposes for it is `evaluator_params`: A SELECTS
# with its own online argmax and is EVALUATED by whatever target net is
# passed there —
#   y_A = r + γ·(1−term)·Q_{eval}(s', argmax_a Q_A(s', a)).
# Default `None` → the unit's own A⁻ (vanilla / DDQN-2016). The paired
# program supplies the partner's time-delayed target B⁻, which IS the
# DDQN-indp intervention: hold selector + acting + target stationarity
# fixed, swap only WHICH target evaluates. B's params arrive from the
# caller's state → constant under A's `value_and_grad` (no cross-
# gradient leak). The two coupled learners are composed in
# `paired_step`, which drives `train_phase` once per net.


def _zero_state_hash(obs: jax.Array) -> jax.Array:
    """Local single-bucket fallback (mirrors dqn.default_state_hash —
    keeping a local copy to avoid the phases→dqn circular import).
    Used as default for `train_phase.state_hash` so count-weighted
    interventions are opt-in; the caller wires the env's real hash."""
    return jnp.zeros((), dtype=jnp.int32)


# ============ Gradient-probe runtime flag ============
#
# Module-level flag controlling whether `train_phase` runs the
# Jacobian-based intra/inter-state α probes. Default True
# (backwards compat). Set False to skip them — the probes scale
# O(n_actions × n_params) per training step and become the
# dominant compute on |A|≥12 envs.
#
# `yaml_sweep.dispatch_sweep` mutates this from the YAML field
# `gradient_probes: bool` before the sweep loop runs. Mutating
# AFTER the sweep starts is racy (the train_phase is jit-compiled
# on first call); set BEFORE any cell runs.
#
# Schema-stable: even when disabled, train_phase emits the
# diagnostic keys as NaN so the persisted parquet schema is
# invariant. Downstream `q_action_grad_overlap_late` /
# `q_inter_state_grad_overlap_late` measurables NaN-propagate;
# bridges using them as `is_finite()` scope predicates drop
# disabled-probe cells cleanly.
_GRADIENT_PROBES_ENABLED: bool = True

if TYPE_CHECKING:
    # Stub-only Protocol — see env_catalogue.py for the rationale.
    from gymnax import Env


# ============ Rollout ============

@claim
def rollout_phase(
    state: DQNState,
    *,
    env: Env,
    env_params: EnvParams,
    n_actions: int,
    replay: Replay,
    q_network: QFunction,
    action_select: ActionSelect,
    state_hash: StateHash,
    n_step: int = 1,
    gamma: float = 0.99,
) -> tuple[DQNState, dict[str, jax.Array]]:
    """One step of acting in the env.

    Reads `q_network` to score the current observation, calls
    `action_select(q_values, key, step, n_actions)`, steps the
    env, folds the raw transition into the n-step pending window
    via the `n_step_return` Free Claim, and appends the
    aggregated transition to `replay` when the window emits.

    For `n_step=1` the window emits every step (mask=1), so
    `replay.add` runs every step exactly as in plain DQN. For
    `n_step>1`, `replay.add` runs gated by the emit mask — adds
    are no-op when the window hasn't yet filled.

    **Pardo 2018 / Sutton-Barto §6.6 truncation-aware path.** This
    phase calls `env.step_env` (via `env_step_typed`, the no-auto-
    reset primitive) instead of `env.step` so the `next_obs` /
    `next_env_state` stored in the replay are the PHYSICAL
    continuation of the trajectory — not the fresh-episode initial
    state gymnax's auto-resetting `step` would `lax.select` in on
    `done`. At a truncated transition (artificial time-limit
    cutoff), the Bellman target then bootstraps against
    `v(s_pre_reset)` rather than `v(s_reset_initial)`, which is
    the load-bearing correctness property — without it, the
    half-fixed refactor still teaches the agent that "the world
    resets to obs[0]" at the cap. After storing the transition,
    the rollout calls `env.reset_env` when `done` to obtain a
    fresh `(obs, state)` for the next iteration's `state.obs` /
    `state.env_state` — `jax.tree.map` + `lax.select` interleaves
    the reset, JIT-compatible.

    Returns `(new_state, diagnostic_dict)` — the dict's keys
    (`reward, done, max_q, ep_return, action, state_hash_per_step,
    buf_size`) are the measurable signals bridges target. Reward
    and done in the diagnostic dict are the RAW per-step values
    from this rollout step, not the n-step aggregates (those are
    in the buffer, not in per-step traces)."""
    # Q-values at current obs — single observation, not batched.
    q_values = q_network(state.online_params, state.obs)

    select_key, env_key, reset_key, next_rng_key = jax.random.split(
        state.rng_key, 4,
    )
    action = action_select(q_values, select_key, state.step, n_actions)

    # `env_step_typed` wraps `env.step_env` and surfaces
    # `truncated` as a typed sibling of `done` — extracting the
    # heterogeneous-info key narrowing into the env-boundary
    # helper so this body reads paper-shaped. The invariant
    # `truncated=1 ⇒ done=1` is enforced at each wrapper boundary
    # (see `_step_boundary.py` for the contract).
    (
        next_obs_pre, next_env_state_pre, reward, done, truncated,
    ) = env_step_typed(
        env, env_key, state.env_state, action.astype(jnp.int32),
        env_params,
    )
    # state.obs is at native shape from init_state; reshape is a
    # no-op for well-shaped envs and a defensive guard.
    next_obs_pre = next_obs_pre.reshape(state.obs.shape)

    raw_transition = Transition(
        obs=state.obs, action=action,
        reward=reward, next_obs=next_obs_pre, done=done,
        truncated=truncated,
    )
    new_pending, emitted, should_emit = n_step_return(
        pending=state.pending_n_step,
        transition=raw_transition,
        n_step=n_step, gamma=gamma,
    )
    new_replay = replay.add(state.replay, emitted, mask=should_emit)

    # Manual auto-reset. `env.step_env` returned the PRE-reset
    # `(next_obs, next_state)`; for the next rollout iteration we
    # need a fresh episode when `done` fires. `env.reset_env`
    # produces `(reset_obs, reset_state)`; `jax.tree.map` blends
    # under `done`, mirroring gymnax base `Environment.step`'s
    # behaviour but cleanly separated from the replay write path.
    reset_obs, reset_env_state = env.reset_env(reset_key, env_params)
    reset_obs = reset_obs.reshape(state.obs.shape)
    done_bool = done.astype(jnp.bool_)
    next_env_state = jax.tree.map(
        lambda r, p: jnp.where(done_bool, r, p),
        reset_env_state, next_env_state_pre,
    )
    next_obs = jnp.where(done_bool, reset_obs, next_obs_pre)

    # Episode return: accumulate this step's reward into the
    # running tally. The state's tally resets to 0 on done so the
    # next step starts a fresh episode; the record's `ep_return`
    # carries the cumulative value AT this step (so bridges
    # filtering on done==1 see the final per-episode return).
    cumulative = state.ep_return + reward
    next_ep_return = jnp.where(done, jnp.float32(0.0), cumulative)

    # State-hash logged at action-selection time (the state
    # observed when the action was chosen, not after the env step).
    obs_hash = state_hash(state.obs).astype(jnp.int32)

    # Increment per-state visit counter. JAX-friendly scatter via
    # `at[].add` — for state_hash_cardinality=1 (envs without a
    # registered hash) the array is a single bucket and the counter
    # is effectively a global step count (harmless: count-weighted
    # loss reduces to a uniform weight).
    new_state_hash_count = state.state_hash_count.at[obs_hash].add(1)

    new_state = state._replace(
        replay=new_replay,
        pending_n_step=new_pending,
        env_state=next_env_state,
        obs=next_obs,
        rng_key=next_rng_key,
        ep_return=next_ep_return,
        state_hash_count=new_state_hash_count,
    )

    diagnostics: dict[str, jax.Array] = {
        'reward': reward,
        'done': done.astype(jnp.float32),
        # Per-step truncation flag (artificial time-limit cutoff,
        # distinct from genuine env terminal). Defaults to 0.0 for
        # envs that don't publish `info['truncated']`. Downstream
        # measurables that need to discriminate "the trajectory
        # continued physically; only the experiment chose to stop"
        # consume this column; the corrected `bootstrap_fraction`
        # measurable reads `terminated = done * (1 - truncated)`.
        # The `truncated=1 ⇒ done=1` invariant is structurally
        # guaranteed at the wrapper boundary (each wrapper that
        # publishes `info['truncated']` masks with done at the
        # point of emission) — no in-rollout assertion needed.
        'truncated': truncated,
        'max_q': jnp.max(q_values),
        'ep_return': cumulative,
        'action': action.astype(jnp.int32),
        # Per-step state bucket from the env-registered state_hash
        # callable (`bucket_hash` for vector envs, `image_downsample_hash`
        # for MinAtar). Named with the `_per_step` suffix to avoid
        # colliding with the `state_hash` LEAF (config-level callable
        # identifier in runs.parquet) at join time. Pre-2026-05-16
        # this key was just `state_hash` — see sanitizer for legacy
        # trace.parquets.
        'state_hash_per_step': obs_hash,
        # Populated-slot count for the diagnostic. `state.size` is a
        # monotonic add-counter (post FIFO-fix) that grows past
        # `capacity`; clip here so `buf_size` keeps the historical
        # "how many slots are populated" meaning that
        # `fill_ratio_late = mean(buf_size / capacity)` reads as a
        # 0..1 coverage scalar.
        'buf_size': jnp.minimum(
            new_replay.size, replay.capacity,
        ).astype(jnp.int32),
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
    n_step: int,
    replay: Replay,
    state_hash: StateHash = _zero_state_hash,
    count_weight_alpha: float = 0.0,
    evaluator_params: Params | None = None,
    probes: bool | None = None,
) -> tuple[DQNState, dict[str, jax.Array]]:
    """One gradient step: sample batch → bootstrap target →
    compute loss → apply update.

    Reads paper-honestly. Buffer warmup (skipping params updates
    until enough transitions are stored) lives on the optimizer
    via `WarmedUpdate(inner=..., warmup_steps=...)` — not in this
    phase. Authors who don't want warmup pass an unwrapped
    `Adam()` / `RMSProp()` directly.

    Returns `(new_state, diagnostic_dict)`. Dict keys: `loss,
    td_error, online_q_per_action, target_q_per_action,
    pearson_stats`. Q-vectors are pre-reduced per-action means
    (batch axis collapsed); bridges that want full distributional
    information consume `pearson_stats` (5 sufficient statistics
    per step) instead."""
    # A's minibatch key + the carried key for the next step.
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

    # ============ Cross-action bootstrap rate ============
    # Per-step fraction over the batch where the bootstrap target's
    # argmax (a' = argmax_a' Q_online(s', a')) differs from the
    # action actually taken at s (a = batch.action). When this rate
    # is high, the TD update is CROSS-ACTION: Q(s, a) is pulled
    # toward Q(s', a') with a' ≠ a — exactly where DDQN's argmax-
    # target decoupling has leverage. When the rate is near zero,
    # bootstrap stays within-action and DDQN's decorrelation has no
    # work to do.
    online_argmax_at_sp = jnp.argmax(online_q_full, axis=-1)  # (batch,)
    bootstrap_action_mismatch = jnp.mean(
        (online_argmax_at_sp != batch.action).astype(jnp.float32)
    )
    # ============ end cross-action bootstrap rate ============

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

    # A's bootstrap EVALUATOR target net. Faithful van Hasselt 2010 —
    # A SELECTS with its OWN online argmax and is EVALUATED by a target
    # net:
    #   y_A = r + γ·Q_{eval}(s', argmax_a Q_A(s', a)).
    # The SELECTOR stays `params` (A's online — the var being
    # differentiated); only the EVALUATOR target varies. The
    # evaluator-edge hook (the principled single-edge surface): when
    # `evaluator_params` is supplied, it IS the bootstrap evaluator —
    # the net whose Q scores the selected action. Default None → the
    # unit's own target net (`state.target_params`), the historical
    # path (vanilla / DDQN-2016 both evaluate against A⁻). `paired_step`
    # (dqn_paired.py) supplies the independent partner's time-delayed
    # target B⁻ here, which is the whole DDQN-indp intervention: hold
    # selector + acting + target stationarity fixed, swap only WHICH
    # target evaluates. B⁻ comes from `state` (the partner's), so it's
    # a constant under value_and_grad — no gradient flows into B from
    # A's loss.
    eval_target_params = (
        evaluator_params
        if evaluator_params is not None
        else state.target_params
    )

    def compute_loss(
        params: Params,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array]]:
        q_b = q_network(params, batch.obs)            # (batch, n_actions)
        predicted = jnp.take_along_axis(
            q_b, batch.action[..., None], axis=-1,
        ).squeeze(-1)                                  # (batch,)
        # Target computed *inside* the loss closure so
        # `gradient_rule` (semi_gradient vs full_gradient) actually
        # controls cotangent flow. With target hoisted outside, it
        # becomes a constant under value_and_grad and the
        # stop_gradient is theatre.
        # Bootstrap takes γⁿ as its `gamma` (the discount on
        # v(s')); n_step is consumed at the dqn-level only.
        bootstrap_gamma = float(gamma) ** int(n_step)
        target = bootstrap(
            # SELECTOR: A's own online net (the var being
            # differentiated) — A selects its own argmax in BOTH its
            # target and behavior → coherent policy. Vanilla ignores
            # this; DDQN-family reads it.
            online_params=params,
            # EVALUATOR: the unit's own A⁻ by default, or the
            # partner's B⁻ when `evaluator_params` is supplied.
            target_params=eval_target_params,
            q_network=q_network,
            next_obs=batch.next_obs, reward=batch.reward, done=batch.done,
            truncated=batch.truncated,
            gamma=bootstrap_gamma,
        )
        per_sample = loss_fn(predicted, target)        # (batch,)
        abs_td = jnp.abs(predicted - target)           # (batch,)
        # Count-weighted loss aggregation (loop-mediation
        # falsification): if α > 0, weight each sample's loss by
        # 1 / (1 + count[state_hash(next_obs)])^α — downweights
        # over-visited next-states so Q trains uniformly across
        # state coverage. α=0 (default) is the standard uniform
        # mean. Re-normalised so loss magnitude is comparable to
        # the uniform-weight baseline; weights are stop-gradient
        # so they don't enter the parameter gradient.
        # state_hash takes a single obs (shape obs_shape) and
        # returns a scalar; vmap over the batch axis.
        next_hash = jax.vmap(state_hash)(batch.next_obs).astype(jnp.int32)
        counts = state.state_hash_count[next_hash]
        weights = 1.0 / jnp.power(1.0 + counts.astype(jnp.float32),
                                   count_weight_alpha)
        weights = jax.lax.stop_gradient(weights)
        weighted_loss = (
            (weights * per_sample).sum() / (weights.sum() + 1e-12)
        )
        # Aux tuple: (batch-mean |TD|, batch-std |TD|). The latter
        # captures *training-signal heterogeneity* per step — high
        # std = the gradient is averaging diverse transitions; low
        # std = the batch is dominated by similar samples (small
        # replay or correlated transitions).
        return weighted_loss, (abs_td.mean(), abs_td.std())

    (loss, (td_error, td_error_within_batch_std)), grads = jax.value_and_grad(
        compute_loss, has_aux=True,
    )(state.online_params)

    # ============ Gradient-overlap probes (intra-α + inter-α) ============
    #
    # Gated by `probes` when the caller passes it explicitly, else by
    # the module flag `_GRADIENT_PROBES_ENABLED` (set via YAML
    # `gradient_probes: false`, which yaml_sweep mutates at sweep
    # launch). The per-call override exists for the paired program:
    # the co-learner B's diagnostic dict is DISCARDED, so B's
    # train_phase passes `probes=False` to skip ~3 jacrev passes per
    # step that nothing ever reads — independent of whether the sweep
    # wants A's probes. Probes scale O(n_actions × n_params) per
    # training step — ~2× cell time on |A|=12 envs.
    #
    # Schema-stable: even when disabled, emits diagnostic keys as
    # NaN so downstream parquet shape is invariant. Measurables
    # `q_action_grad_overlap_late` / `q_inter_state_grad_overlap_late`
    # NaN-propagate; bridges with `is_finite()` scope filters drop
    # disabled-probe cells cleanly.
    run_probes = _GRADIENT_PROBES_ENABLED if probes is None else probes
    if not run_probes:
        q_action_grad_overlap = jnp.asarray(jnp.nan, dtype=jnp.float32)
        q_inter_state_grad_overlap = jnp.asarray(jnp.nan, dtype=jnp.float32)
        q_inter_state_grad_overlap_random = jnp.asarray(jnp.nan, dtype=jnp.float32)
    else:
        # `findings_fa_depth_within_env`: gradient overlap between
        # action heads when the FA's trunk is updated.
        #   - Linear FA `Q(s,a) = W_a · obs + b_a`: rows of W are
        #     independent across actions → α = 0 by construction.
        #   - Shared-trunk MLP: trunk gradients flow into all action
        #     heads → α > 0; specifically, α ≈ (heads · heads^T) /
        #     ||heads||² for the trunk-output layer.
        #   - Tabular Q: per-(s,a) entry is independent → α = 0.
        probe_obs = batch.obs[0]  # single state per training step

        def _q_for_grad(p: Params) -> jax.Array:
            return q_network(p, probe_obs)  # shape (n_actions,)

        J_pytree = jax.jacrev(_q_for_grad)(state.online_params)
        J_leaves = jax.tree_util.tree_leaves(J_pytree)
        J_flat = jnp.concatenate(
            [x.reshape(x.shape[0], -1) for x in J_leaves], axis=1,
        )  # (n_actions, n_params_total)
        norms = jnp.sqrt(jnp.sum(J_flat ** 2, axis=1) + 1e-12)
        J_unit = J_flat / norms[:, None]
        gram = J_unit @ J_unit.T
        n_a = gram.shape[0]
        mask = 1.0 - jnp.eye(n_a)
        off_diag_count = jnp.maximum(jnp.sum(mask), 1.0)
        q_action_grad_overlap = jnp.sum(gram * mask) / off_diag_count

        # Inter-state α: cosine of ∂Q(s, a)/∂θ vs ∂Q(s', a)/∂θ at
        # paired (batch.obs[0], batch.next_obs[0]) for the SAME action,
        # averaged across actions. Closed-form: linear FA gives
        # cos(obs, obs') (env-dynamics-driven); deep MLP gradient
        # depends on trunk + ReLU gating.
        probe_obs_sp = batch.next_obs[0]

        def _q_at_sp(p: Params) -> jax.Array:
            return q_network(p, probe_obs_sp)

        J_sp_pytree = jax.jacrev(_q_at_sp)(state.online_params)
        J_sp_leaves = jax.tree_util.tree_leaves(J_sp_pytree)
        J_sp_flat = jnp.concatenate(
            [x.reshape(x.shape[0], -1) for x in J_sp_leaves], axis=1,
        )
        norms_sp = jnp.sqrt(jnp.sum(J_sp_flat ** 2, axis=1) + 1e-12)
        J_sp_unit = J_sp_flat / norms_sp[:, None]
        per_action_overlap = jnp.sum(J_unit * J_sp_unit, axis=1)
        q_inter_state_grad_overlap = per_action_overlap.mean()

        # "Lag-k" baseline: cosine of ∂Q(s, a)/∂θ vs ∂Q(s_random, a)/∂θ
        # at (batch.obs[0], batch.obs[-1]) — two states sampled from
        # uniform-random replay positions, generally from different
        # trajectories. Continuous-state envs (LL, MC) saturate the
        # lag-1 overlap near 1 because consecutive observations differ
        # by infinitesimal continuous deltas; pairing across the batch
        # gives a "global Q smoothness" baseline.
        #
        # The discriminative signal is the DIFFERENCE
        # `(lag-1 overlap) − (random-pair overlap)`: at discrete envs,
        # lag-1 > random-pair (trajectory-adjacency confers extra
        # smoothness); at continuous envs, both saturate near 1 and
        # the difference drops to 0. Authoring a measurable + bridge
        # to consume this is the next-step for the cross-env
        # smoothness sign-alignment claim.
        probe_obs_random = batch.obs[-1]

        def _q_at_random(p: Params) -> jax.Array:
            return q_network(p, probe_obs_random)

        J_rand_pytree = jax.jacrev(_q_at_random)(state.online_params)
        J_rand_leaves = jax.tree_util.tree_leaves(J_rand_pytree)
        J_rand_flat = jnp.concatenate(
            [x.reshape(x.shape[0], -1) for x in J_rand_leaves], axis=1,
        )
        norms_rand = jnp.sqrt(jnp.sum(J_rand_flat ** 2, axis=1) + 1e-12)
        J_rand_unit = J_rand_flat / norms_rand[:, None]
        per_action_overlap_rand = jnp.sum(J_unit * J_rand_unit, axis=1)
        q_inter_state_grad_overlap_random = per_action_overlap_rand.mean()
    # ============ end gradient-overlap probes ============

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
        # Within-batch std of |TD-error|. A measurement of training-
        # signal heterogeneity — independent of the HP capacity by
        # construction (depends on which transitions ended up in the
        # batch, not just on buffer size).
        'td_error_within_batch_std': td_error_within_batch_std,
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
        # FA-coherence intra-state α probe (theoretical Hasselt-
        # bound assumption: iid action noise). Single-state Jacobian
        # cosine overlap across action heads. See block above and
        # `findings_fa_depth_within_env` for theory ↔ measurement.
        'q_action_grad_overlap_per_step': q_action_grad_overlap,
        # FA-coherence INTER-state α probe (theory's axis i). Cosine
        # overlap of ∂Q(s, a)/∂θ vs ∂Q(s', a)/∂θ at paired
        # (s, s') = (batch.obs[0], batch.next_obs[0]). The proper
        # spatial-smoothness measurement — independent of env-state-
        # trajectory smoothness (which confounds the eval-trajectory
        # autocorr `q_trajectory_autocorr_late`).
        'q_inter_state_grad_overlap_per_step': q_inter_state_grad_overlap,
        # "Lag-k" baseline: same Jacobian overlap but at a
        # random-batch-partner pair (batch.obs[0], batch.obs[-1]).
        # Used to normalize the adjacent-pair measure at continuous-
        # state envs where lag-1 saturates near 1. See block above.
        'q_inter_state_grad_overlap_random_per_step': q_inter_state_grad_overlap_random,
        # Cross-action bootstrap rate: per-step fraction over batch
        # where argmax_a' Q_online(s', a') ≠ action_taken_at_s.
        # When high, the TD bootstrap pulls Q(s, a) toward
        # Q(s', a') for a' ≠ a — the regime where DDQN's argmax-
        # target decoupling has maximum leverage. When low, the
        # bootstrap is within-action and DDQN's mechanism is dormant.
        'bootstrap_action_mismatch_per_step': bootstrap_action_mismatch,
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
