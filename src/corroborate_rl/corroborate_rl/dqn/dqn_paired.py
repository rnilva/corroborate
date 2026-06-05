"""Deep van Hasselt 2010 — the symmetric two-learner host (`DDQN-indp`).

This is the **paired** program from `ESTIMATOR_REFACTOR.md` §4: a
single-edge causal probe hosted as two coupled DQN learners that
share one rollout and one replay buffer. It is **not** a slot swap
on `dqn` (it threads two `(online, target)` pairs, which one `dqn`
cannot), and it is **not** a from-scratch reimplementation — `dqn`
is the single ground, used twice, and every per-step operation
reuses the existing `rollout_phase` / `train_phase` / `sync_phase`
`@claim`s verbatim (D1).

The structure (one acting agent A + a non-acting co-learner B):

    A's target:  y_A = r + γ·(1−term)·Q_{B⁻}(s', argmax_a Q_A(s', a))
    B's target:  y_B = r + γ·(1−term)·Q_{A⁻}(s', argmax_a Q_B(s', a))

A selects with its own online net and is **evaluated by B's
time-delayed target B⁻** (and symmetrically for B). That cross is
the whole intervention: relative to DDQN-2016 (where A is evaluated
by its *own* target A⁻), only the *identity* of the evaluator net
moves — own → independent — while selector, acting policy, and the
"evaluator is a frozen target" structure are held fixed. The edge
is single (see §4a). The cross flows entirely through the
`evaluator_params` hook on `train_phase`; no marker, no slot-key,
no greedification rename.

**Shared agent (load-bearing).** A is the only net that acts; there
is one replay buffer; B is a non-acting learner drawing independent
minibatches from the *same* buffer (independence is in the sample +
init + optimizer, not a separate buffer — DDQL's default
approximation; cf. §4e). `PairedDQNState.a` is a real `DQNState`
carrying A's nets AND the shared agent, so eval / checkpoints reuse
it unchanged; B is just three extra param/opt fields.

**Evaluator is `B⁻`, not live B (§4a / D5).** `paired_step` feeds
`b_target` (B's periodic-copy target) as A's evaluator, never
`b_online`. That holds the target-net stationarity constant so the
edge stays single; wiring `b_online` here would silently become the
two-edge confound (independence *and* stationarity) and break the
reproduction."""
from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING, Annotated, NamedTuple

import jax
import optax
from gymnax import EnvParams

from corroborate import claim
from corroborate.core.signature import Exogenous
from corroborate_rl.dqn.claims.bootstrap import double_greedify
from corroborate_rl.dqn.claims.optimizer import (
    OptimizerFactory,
    default_optimizer,
)
from corroborate_rl.dqn.claims.q_network import Params, QFunction, mlp_q
from corroborate_rl.dqn.claims.replay import Replay
from corroborate_rl.dqn.claims import (
    bootstrap as default_bootstrap,
    epsilon_greedy,
    periodic_copy,
    squared_error,
)
from corroborate_rl.dqn.claims.action_select import ActionSelect
from corroborate_rl.dqn.dqn import default_state_hash, init_state
from corroborate_rl.dqn.eval import run_with_eval, validate_eval_schedule
from corroborate_rl.dqn.phases import (
    rollout_phase,
    sync_phase,
    train_phase,
)
from corroborate_rl.dqn.state import DQNState
from corroborate_rl.dqn.types import (
    Bootstrap,
    LossFn,
    OptState,
    StepRecord,
    TargetSync,
)
from corroborate_rl.env_catalogue import EnvWrapper, StateHash

if TYPE_CHECKING:
    from gymnax import Env


# Default bootstrap for the paired program: standard double-Q
# selection (online selects its own argmax). The cross-evaluation
# is NOT a greedification choice here — it is the `evaluator_params`
# injection inside `paired_step`, so the greedification stays the
# ordinary `double_greedify`. (A bare `default_bootstrap` would
# default to `max_greedify`, i.e. vanilla selection — wrong for a
# double-Q program.)
_DOUBLE_BOOTSTRAP: Bootstrap = partial(
    default_bootstrap, greedification=double_greedify,
)


# Independent-partner RNG salt. Folded into A's post-rollout key to
# derive B's minibatch key — uncorrelated with A's sample draw while
# leaving A's RNG stream untouched. (The exact constant is free: the
# paired program carries its own arm_key, and B-independence only
# requires the draws be uncorrelated, not bit-matched to any prior
# crude implementation — see §4e.)
_PARTNER_SALT: int = 0xB2010


class PairedDQNState(NamedTuple):
    """Symmetric-2010 carry: a full DQN agent A + co-learner B.

    `a` is a real `DQNState` — it carries A's online/target/opt
    nets AND the shared agent (replay, env, obs, rng, step,
    ep_return, state_hash_count). Eval and checkpoint code read
    `a.online_params` exactly as for single-net `dqn`; A is the
    acting/measured unit.

    `b_*` are B's three net fields only — B has no agent of its
    own (it never acts; it learns from A's shared replay)."""

    a: DQNState
    b_online: Params
    b_target: Params
    b_opt: OptState

    @property
    def online_params(self) -> Params:
        """A's online net — the acting/eval surface. Lets paired
        state reuse `eval_burst` / checkpoint code that reads
        `state.online_params` without knowing it is paired."""
        return self.a.online_params

    @property
    def target_params(self) -> Params:
        """A's target net — checkpoint-surface parity with
        `DQNState.target_params`."""
        return self.a.target_params


def init_paired_state(
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
) -> PairedDQNState:
    """Build the initial paired state: A via the standard
    `init_state` (so the shared agent is allocated exactly once),
    plus an independently-initialised co-learner B (distinct key,
    own optimizer state; same architecture as A).

    B is genuinely independent — its own init key and optimizer
    state — but trained on the shared buffer (allocated inside
    A's `init_state`)."""
    a_key, b_key = jax.random.split(rng_key)
    a = init_state(
        env=env, env_params=env_params,
        obs_shape=obs_shape, n_actions=n_actions,
        rng_key=a_key, optimizer=optimizer,
        q_network=q_network, replay=replay,
        state_hash_cardinality=state_hash_cardinality,
    )
    b_online = q_network.init(b_key, obs_shape, n_actions)
    b_opt = optimizer.init(b_online)
    return PairedDQNState(
        a=a, b_online=b_online, b_target=b_online, b_opt=b_opt,
    )


def paired_step(
    state: PairedDQNState,
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
    # Double-Q selection is structural to the paired program — a
    # bare `default_bootstrap` (max_greedify) would make A's selector
    # vanilla, silently running a NON-DDQN-indp update under the
    # paired label. Default to the double form so a standalone
    # `paired_step` call (not just via `paired_dqn`) is correct.
    bootstrap: Bootstrap = _DOUBLE_BOOTSTRAP,
    loss_fn: LossFn = squared_error,
    target_sync: TargetSync = periodic_copy,
    count_weight_alpha: float = 0.0,
) -> tuple[PairedDQNState, StepRecord]:
    """One symmetric-2010 step over the paired state.

    Same shape as `dqn_step` — `(state, idx) -> (state, record)` —
    so `train_with_eval` / loop backends drive it identically. The
    returned record is A's per-step diagnostic dict (A is the
    measured unit; B's diagnostics are not emitted, matching the
    crude side-net's schema-invariance).

    Sequence (every phase is the existing single-net `@claim`,
    A directly and B through a temporary `DQNState` view that
    shares A's agent sub-state):

    1. **Rollout** — A acts, advancing the SHARED agent (replay,
       env, obs, rng, ep_return, counts). B never acts.
    2. **Train A** — `evaluator_params = B's target` → A is scored
       by B⁻ (the cross). Selector is A's own online (via the
       `double_greedify` the caller binds). Samples shared replay.
    3. **Train B** — a view with B's online/opt + an independent
       minibatch RNG, `evaluator_params = A's target` (A⁻, scored
       by A's still-unsynced target). Samples the SAME replay.
    4. **Sync A**, then **Sync B** — both targets periodic-copy
       their own online on the same cadence (`state.step`).
    5. Advance the shared step counter (last, so all phases saw
       the pre-advance step — matching `dqn_step`)."""
    del idx  # `step` lives on `state.a`; idx is loop bookkeeping.
    a = state.a

    # 1. Rollout — A acts into the shared replay/agent.
    a, rollout = rollout_phase(
        a,
        env=env, env_params=env_params, n_actions=n_actions,
        replay=replay, q_network=q_network,
        action_select=action_select, state_hash=state_hash,
        n_step=n_step, gamma=gamma,
    )

    # 2. Train A — evaluated by B's time-delayed target B⁻ (the
    # cross). `a.target_params` (A⁻) is unchanged by rollout, so it
    # is still A's start-of-step target for B to read in step 3.
    a, train = train_phase(
        a,
        q_network=q_network, bootstrap=bootstrap, loss_fn=loss_fn,
        optimizer=optimizer, gamma=gamma, n_step=n_step,
        replay=replay, state_hash=state_hash,
        count_weight_alpha=count_weight_alpha,
        evaluator_params=state.b_target,
    )

    # 3. Train B — own online/opt, independent minibatch, scored by
    # A's (still-unsynced) target A⁻. Shares A's replay + counts via
    # the view; only B's online/opt are kept from the result.
    b_view = a._replace(
        online_params=state.b_online,
        opt_state=state.b_opt,
        rng_key=jax.random.fold_in(a.rng_key, _PARTNER_SALT),
    )
    # `count_weight_alpha=0.0` (NOT the arm's α): the count-weighting
    # rule downweights loss by visit counts, but `b_view` carries A's
    # `state_hash_count` (B never acts → it has no histogram of its
    # own). Forwarding the arm's α would weight B's loss by A's
    # exploration distribution, coupling the "independent" estimator
    # to A and diverging from the crude path (whose B used a plain
    # unweighted mean). B trains unweighted; α remains an A-only
    # intervention axis.
    # `probes=False`: B's diagnostic dict (`_b_train`) is discarded —
    # only B's online params + opt-state are threaded forward. Skip
    # the ~3 jacrev gradient-overlap passes B would otherwise compute
    # every step (nothing reads them), independent of the sweep-wide
    # `gradient_probes` flag that still governs A's probes.
    b_view, _b_train = train_phase(
        b_view,
        q_network=q_network, bootstrap=bootstrap, loss_fn=loss_fn,
        optimizer=optimizer, gamma=gamma, n_step=n_step,
        replay=replay, state_hash=state_hash,
        count_weight_alpha=0.0,
        evaluator_params=a.target_params,
        probes=False,
    )
    new_b_online = b_view.online_params
    new_b_opt = b_view.opt_state

    # 4a. Sync A's target ← A's online.
    a = sync_phase(
        a, target_sync=target_sync, sync_period=sync_period,
    )
    # 4b. Sync B's target ← B's online, same cadence (reads
    # `a.step`, still pre-advance). View carries B's new online +
    # B's old target; only B's new target is kept.
    b_sync_view = a._replace(
        online_params=new_b_online, target_params=state.b_target,
    )
    b_sync_view = sync_phase(
        b_sync_view, target_sync=target_sync, sync_period=sync_period,
    )
    new_b_target = b_sync_view.target_params

    # 5. Advance the shared step counter (last, mirroring dqn_step).
    a = a._replace(step=a.step + 1)

    new_state = PairedDQNState(
        a=a, b_online=new_b_online,
        b_target=new_b_target, b_opt=new_b_opt,
    )
    return new_state, {**rollout, **train}


# ============ paired_dqn — outermost claim (full DDQN-indp run) ============

@claim
def paired_dqn(
    *,
    # Exogenous author primitives — mirror `dqn` (generalised over,
    # not intervened on). The paired program is a DISTINCT claim from
    # `dqn` (its own arm_key); it is the honest host for deep van
    # Hasselt 2010 / DN-DDQL_DoubleDQN, not a slot config of `dqn`.
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
    keep_q_checkpoint_final: Annotated[bool, Exogenous] = False,
    keep_q_checkpoint_per_burst: Annotated[bool, Exogenous] = False,
    # Cross-cutting HPs.
    gamma: float = 0.99,
    sync_period: int = 100,
    n_step: int = 1,
    total_steps: int = 50_000,
    eval_every: int = 5_000,
    n_episodes: int = 20,
    # Slot Claims. `bootstrap` defaults to double-Q selection; the
    # cross-evaluation is structural to `paired_step`, not a slot.
    q_network: QFunction = mlp_q,
    action_select: ActionSelect = epsilon_greedy,
    replay: Replay = Replay(),
    bootstrap: Bootstrap = _DOUBLE_BOOTSTRAP,
    loss_fn: LossFn = squared_error,
    target_sync: TargetSync = periodic_copy,
    optimizer: OptimizerFactory = default_optimizer,
    count_weight_alpha: float = 0.0,
) -> dict[str, jax.Array]:
    """Full deep van Hasselt 2010 (`DDQN-indp`) training+eval run.

    Mirrors `dqn`'s composition — `init_paired_state` →
    `train_with_eval(paired_step, eval_fn, ...)` — but threads
    `PairedDQNState`: two coupled learners (A acting, B co-learning)
    over one shared rollout + replay buffer, reusing the generic
    driver and the phase claims (D1). Eval reads A's online net (A
    is the acting/measured unit); the per-step + per-burst record
    surface is identical to `dqn`'s, so every measurable and bridge
    consumes it unchanged.

    Checkpoint-resume (`init_override`) is intentionally NOT exposed
    yet — the paired program has a four-net state; resume support is
    a follow-on if needed. Raises `ValueError` if `total_steps`
    isn't a positive multiple of `eval_every` (same contract as
    `dqn`)."""
    del env_name, wrappers  # structural markers consumed by the runner
    validate_eval_schedule(total_steps, eval_every)

    optax_handle = optimizer()
    rng_key = jax.random.PRNGKey(seed)
    init_key, run_key = jax.random.split(rng_key, 2)
    state = init_paired_state(
        env=env, env_params=env_params,
        obs_shape=obs_shape, n_actions=n_actions,
        rng_key=init_key, optimizer=optax_handle,
        q_network=q_network, replay=replay,
        state_hash_cardinality=state_hash_cardinality,
    )

    step_fn = partial(
        paired_step,
        env=env, env_params=env_params, n_actions=n_actions,
        optimizer=optax_handle, state_hash=state_hash,
        gamma=gamma, sync_period=sync_period, n_step=n_step,
        q_network=q_network, action_select=action_select,
        replay=replay, bootstrap=bootstrap,
        loss_fn=loss_fn, target_sync=target_sync,
        count_weight_alpha=count_weight_alpha,
    )

    # `run_with_eval` evals `state.online_params` — the
    # `PairedDQNState.online_params` @property delegates to A (the
    # acting/measured unit), so eval reads A's net exactly as the
    # crude path did.
    return run_with_eval(
        init_state=state, step_fn=step_fn, run_key=run_key,
        env=env, env_params=env_params, q_network=q_network,
        gamma=gamma, eval_episode_cap=eval_episode_cap,
        n_episodes=n_episodes,
        total_steps=total_steps, eval_every=eval_every,
        keep_q_checkpoint_final=keep_q_checkpoint_final,
        keep_q_checkpoint_per_burst=keep_q_checkpoint_per_burst,
    )
