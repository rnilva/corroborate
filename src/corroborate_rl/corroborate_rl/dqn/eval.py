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
from typing import TYPE_CHECKING, NamedTuple, cast

import jax
import jax.numpy as jnp
from gymnax import EnvParams, EnvState

from corroborate import claim
from corroborate.core.loop import Loop, iterate
from corroborate_rl.loop import scan_loop
from corroborate_rl.dqn.state import DQNState
from corroborate_rl.dqn.claims.q_network import QFunction
from corroborate_rl.dqn.q_checkpoint import checkpoint_key
from corroborate_rl.dqn.types import StepRecord

if TYPE_CHECKING:
    # Stub-only Protocol — see env_catalogue.py for the rationale.
    from gymnax import Env


# ============ Eval per-episode and per-burst record shapes ============

class EvalEpisodeOut(NamedTuple):
    """One eval episode's per-burst record."""
    predicted_q_at_start: jax.Array   # () — max_a Q_online(s_0, a)
    mc_return: jax.Array              # () — Σ γ^t r_t over the episode
    episode_length: jax.Array         # () int32
    # Per-state probes for chain-traced cumulative bias measurement.
    # `predicted_q_per_step[t]` = max_a Q_online(s_t, a) at each
    # visited state. `mc_return_from_step[t]` = Σ_{k≥t} γ^(k−t) r_k —
    # the discounted realised return from state t onward (the
    # "remaining-chain MC ground truth" at that state). Both are
    # `(episode_cap,)` with `active_per_step[t] = 1` while the
    # episode is still running.
    predicted_q_per_step: jax.Array   # (episode_cap,)
    mc_return_from_step: jax.Array    # (episode_cap,)
    active_per_step: jax.Array        # (episode_cap,) float32 (0/1)


class EvalBurstOut(NamedTuple):
    """K stacked eval episodes."""
    predicted_q_at_start: jax.Array   # (K,)
    mc_return: jax.Array              # (K,)
    episode_length: jax.Array         # (K,) int32
    predicted_q_per_step: jax.Array   # (K, episode_cap)
    mc_return_from_step: jax.Array    # (K, episode_cap)
    active_per_step: jax.Array        # (K, episode_cap)


# ============ Single greedy episode ============

@claim
def eval_episode(
    *,
    online_params: dict[str, jax.Array],
    env: Env,
    env_params: EnvParams,
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
    # Substrate keeps obs at native shape; q_network handles its own
    # input shape (MLP flattens trailing dims internally; CNN reads
    # spatial structure directly).

    q_at_start = q_network(online_params, obs_0)
    predicted_q_at_start = jnp.max(q_at_start)

    class Carry(NamedTuple):
        obs: jax.Array
        env_state: EnvState
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

    def step(
        carry: Carry, _idx: jax.Array,
    ) -> tuple[Carry, tuple[jax.Array, jax.Array, jax.Array]]:
        q_values = q_network(online_params, carry.obs)
        action = jnp.argmax(q_values).astype(jnp.int32)
        # Per-step probes: capture max-Q at *current* state and the
        # immediate reward / active-indicator for backward MC accumulation.
        q_max_at_state = jnp.max(q_values)

        env_key, next_rng = jax.random.split(carry.rng)
        next_obs, next_env_state, reward, done, _info = env.step(
            env_key, carry.env_state, action, env_params,
        )
        # carry.obs is at native shape; reshape is a no-op for
        # well-shaped envs and a defensive guard against gymnax
        # quirks that emit a different flat shape on step.
        next_obs = next_obs.reshape(carry.obs.shape)

        already_done = carry.done
        active = jnp.logical_not(already_done)
        active_f = active.astype(jnp.float32)
        discount = jnp.power(gamma, carry.steps.astype(jnp.float32))
        new_cumulative = carry.cumulative_return + jnp.where(
            active, reward * discount, 0.0,
        )
        new_steps = carry.steps + jnp.where(active, jnp.int32(1), jnp.int32(0))
        new_done = jnp.logical_or(already_done, done.astype(jnp.bool_))

        # Per-step output: (q_at_state, reward_if_active, active_indicator).
        # Inactive (post-done) steps contribute 0 reward and 0 active
        # mass — the backward MC sum will drop them, and analysis
        # measurables filter on `active_per_step==1`.
        per_step = (
            q_max_at_state,
            jnp.where(active, reward.astype(jnp.float32), jnp.float32(0.0)),
            active_f,
        )
        return (
            Carry(
                obs=next_obs,
                env_state=next_env_state,
                done=new_done,
                rng=next_rng,
                cumulative_return=new_cumulative,
                steps=new_steps,
            ),
            per_step,
        )

    final_carry, (q_per_step, reward_per_step, active_per_step) = jax.lax.scan(
        step, init_carry, jnp.arange(episode_cap),
    )

    # Backward discounted-return scan: mc_return_from_step[t] =
    # Σ_{k≥t} γ^(k−t) r_k. Implements the recurrence
    # mc[t] = active[t] · (r[t] + γ · mc[t+1]). Inactive steps return 0,
    # which is correct under our convention that post-done steps don't
    # contribute. Use lax.scan with reverse=True.
    def backward(
        carry: jax.Array, x: tuple[jax.Array, jax.Array],
    ) -> tuple[jax.Array, jax.Array]:
        r, a = x
        new_carry = a * (r + jnp.float32(gamma) * carry)
        return new_carry, new_carry

    _, mc_return_from_step = jax.lax.scan(
        backward, jnp.float32(0.0), (reward_per_step, active_per_step),
        reverse=True,
    )

    return EvalEpisodeOut(
        predicted_q_at_start=predicted_q_at_start,
        mc_return=final_carry.cumulative_return,
        episode_length=final_carry.steps,
        predicted_q_per_step=q_per_step,
        mc_return_from_step=mc_return_from_step,
        active_per_step=active_per_step,
    )


# ============ K-episode burst via vmap ============

def eval_burst(
    *,
    online_params: dict[str, jax.Array],
    env: Env,
    env_params: EnvParams,
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
        predicted_q_per_step=stacked.predicted_q_per_step,
        mc_return_from_step=stacked.mc_return_from_step,
        active_per_step=stacked.active_per_step,
    )


# ============ train_with_eval — nested scan driver ============

def train_with_eval(
    *,
    step_fn: Callable[[DQNState, jax.Array], tuple[DQNState, StepRecord]],
    eval_fn: Callable[[DQNState, jax.Array], EvalBurstOut],
    init_state: DQNState,
    total_steps: int,
    eval_every: int,
    loop: Loop[
        DQNState,
        tuple[
            StepRecord, EvalBurstOut,
            dict[str, jax.Array], dict[str, jax.Array],
        ],
        jax.Array,
    ] = scan_loop,
    keep_q_checkpoint_final: bool = False,
    keep_q_checkpoint_per_burst: bool = False,
) -> dict[str, jax.Array]:
    """Run `step_fn` for `total_steps` with an `eval_fn` burst at
    the end of every `eval_every` chunk. Returns the merged
    record dict.

    Outer loop over `total_steps // eval_every` super-steps; inner
    loop over `eval_every` training steps. The outer loop's per-
    super-step output is `(train_chunk, eval_burst)`. After the
    full run, train chunks reshape from `(n_super_steps,
    eval_every, ...)` → `(total_steps, ...)`; eval burst fields
    stack as `(n_super_steps, K, ...)`.

    `loop` is the iteration backend (default `scan_loop` for
    JIT-fast production). Pass `python_loop` (the rl-flavored
    variant — same `jax.Array` step idx as `scan_loop`) for probe
    runs that need exhaustive `@claim` records under
    `trace_context()`.

    The `loop` parameter's outer `T` is `tuple[StepRecord,
    EvalBurstOut]` — the per-super-step aggregate. The inner
    `iterate` call (over training steps) reuses the same Loop
    backend with a different `T` binding (`StepRecord` only); the
    Protocol's parametric polymorphism makes that re-binding
    static.

    Decoupled from `dqn` itself so the algorithm composition stays
    paper-prose. The same driver can power any RL algorithm with
    a step+eval shape (PPO, SAC, distributional Q).

    `keep_q_checkpoint_final` / `keep_q_checkpoint_per_burst` are
    persistence-only flags: when enabled, the corresponding param
    snapshots are emitted under sentinel-prefixed keys
    (`__q_checkpoint__<arm>__<role>__<param_key>`) in the returned
    record. The cell runner intercepts these keys, writes msgpack
    files, and filters them from the trace columns — measurables
    ignore unknown keys, so the sentinel-prefixed entries are
    transparent to the existing analysis pipeline. See
    `q_checkpoint.py` for the key-convention details."""
    n_super_steps = total_steps // eval_every

    # Inner loop: re-bind the backend's `T` to `StepRecord` (the
    # training step's per-step output). `Loop` is genuinely
    # parametric in T at the value level (`scan_loop[C, T]` is
    # generic; the same instance type-checks at any T binding), but
    # Python's type system can't express a callable that re-binds
    # T per call site without higher-kinded polymorphism.
    # `cast` is the documented escape hatch — same Loop instance,
    # different T binding for the inner-vs-outer call.
    inner_loop = cast('Loop[DQNState, StepRecord, jax.Array]', loop)

    # Decide ONCE outside the scan whether per-burst snapshots are
    # captured: the scan body's pytree shape must be static across
    # super-steps. The capture itself is cheap (params are small —
    # ~25 KB MLP / ~80 KB CNN — and stay device-resident; the
    # outer scan stacks them to `(n_super_steps, *param_shape)`).
    # The gating at this level decides whether to allocate the
    # stacked-checkpoint slot in scan output at all.
    capture_per_burst = keep_q_checkpoint_per_burst

    # super_step's per-step output is a 4-tuple:
    #   (train_chunk: StepRecord,
    #    burst: EvalBurstOut,
    #    online_snap: dict[str, jax.Array],  # per_burst online params
    #    target_snap: dict[str, jax.Array])  # per_burst target params
    # When per-burst capture is OFF the two snap dicts are EMPTY —
    # the scan's pytree shape stays uniform across the on/off
    # branches without forcing the framework's JIT cache to re-trace.

    def super_step(
        s: DQNState, super_idx: jax.Array,
    ) -> tuple[
        DQNState,
        tuple[
            StepRecord, EvalBurstOut,
            dict[str, jax.Array], dict[str, jax.Array],
        ],
    ]:
        s, train_chunk_obj = iterate(
            step=step_fn, init=s, length=eval_every, backend=inner_loop,
        )
        # `iterate`'s return is `tuple[C, object]` — aggregation
        # polymorphism at the Protocol seam. The runtime invariant
        # under `scan_loop`/rl-`python_loop` is that the aggregated
        # half is a `StepRecord` pytree (each leaf stacked to leading
        # `(eval_every, ...)`). Cast at the use site.
        train_chunk = cast(StepRecord, train_chunk_obj)
        burst = eval_fn(s, super_idx)
        # Snapshot the post-burst online + target params for the
        # per-burst checkpoint stack. When the feature is OFF, emit
        # empty dicts so the scan output is structurally uniform
        # (no JAX retrace on the off-path).
        if capture_per_burst:
            online_snap: dict[str, jax.Array] = dict(s.online_params)
            target_snap: dict[str, jax.Array] = dict(s.target_params)
        else:
            online_snap = {}
            target_snap = {}
        return s, (train_chunk, burst, online_snap, target_snap)

    final_state, super_aggregated_obj = iterate(
        step=super_step, init=init_state, length=n_super_steps,
        backend=loop,
    )
    # Same cast pattern at the outer scope: scan_loop / rl-python_loop
    # stack super_step's output, producing
    # `tuple[StepRecord (n_super_steps-stacked),
    #        EvalBurstOut (n_super_steps-stacked),
    #        per_burst online dict — leaves (n_super_steps, *p),
    #        per_burst target dict — same shape]`.
    train_chunks, eval_bursts, per_burst_online, per_burst_target = cast(
        tuple[
            StepRecord, EvalBurstOut,
            dict[str, jax.Array], dict[str, jax.Array],
        ],
        super_aggregated_obj,
    )

    def _flatten(x: jax.Array) -> jax.Array:
        # Each leaf: (n_super_steps, eval_every, *original) →
        # (total_steps, *original).
        return x.reshape(total_steps, *x.shape[2:])

    train_trace: StepRecord = jax.tree.map(_flatten, train_chunks)
    eval_step_indices = (
        jnp.arange(n_super_steps, dtype=jnp.int32) + 1
    ) * eval_every

    record: dict[str, jax.Array] = {
        **train_trace,
        'predicted_q_at_start': eval_bursts.predicted_q_at_start,
        'mc_return': eval_bursts.mc_return,
        'episode_length': eval_bursts.episode_length,
        'predicted_q_per_step': eval_bursts.predicted_q_per_step,
        'mc_return_from_step': eval_bursts.mc_return_from_step,
        'active_per_step': eval_bursts.active_per_step,
        'eval_step_index': eval_step_indices,
    }

    # Sentinel-prefixed checkpoint payloads. Per-burst arrays carry
    # the leading `(n_super_steps, *param_shape)` axis; the final
    # snapshot is a single `(*param_shape)` array taken from the
    # post-scan state. The cell runner partitions on
    # `CHECKPOINT_KEY_PREFIX` to extract them.
    if keep_q_checkpoint_per_burst:
        for pk, arr in per_burst_online.items():
            record[checkpoint_key('online', 'per_burst', pk)] = arr
        for pk, arr in per_burst_target.items():
            record[checkpoint_key('target', 'per_burst', pk)] = arr
    if keep_q_checkpoint_final:
        for pk, arr in final_state.online_params.items():
            record[checkpoint_key('online', 'final', pk)] = arr
        for pk, arr in final_state.target_params.items():
            record[checkpoint_key('target', 'final', pk)] = arr

    return record
