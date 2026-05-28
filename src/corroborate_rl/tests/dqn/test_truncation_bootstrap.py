"""Closed-form Bellman-target tests under the truncation flag.

Sutton-Barto §6.6 / Gymnasium-API distinction: `done` is the
env-reset signal (any reason the episode ended); `truncated=1`
flags the subset of dones that were artificial time-limit cutoffs
(experiment chose to stop, trajectory continues physically) vs
`truncated=0` genuine terminal absorbing states. The bootstrap
target consumes `terminated = done * (1 - truncated)` so the
discount mask continues for truncations, zeros for terminations.

These tests pin the contract by constructing scalar transitions
with hand-controlled Q(s', a) values, then asserting the
bootstrap claim returns the closed-form target. The Q-net is a
single linear layer with weights set so that for the test obs the
output's `max` equals a controlled scalar — closed-form, no
sampling-distribution slack.

Replay round-trip and n-step accumulation are covered as well —
the truncation column has to survive the buffer write/sample
cycle for end-to-end correctness.

The Pardo 2018 fix (rollout uses `step_env` + manual reset to
capture pre-reset `next_obs` in the replay) is exercised by the
`test_rollout_stores_pre_reset_next_obs_on_truncation` test
against a synthetic step-counter env."""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from gymnax import EnvParams

from corroborate_rl.dqn.claims.bootstrap import (
    bootstrap as bootstrap_claim,
    max_greedify,
)
from corroborate_rl.dqn.claims.q_network import Params, QFunction
from corroborate_rl.dqn.claims.replay import (
    Replay,
    Transition,
    init_pending_n_step,
    n_step_return,
)


def _constant_q_network(value: float, n_actions: int = 2) -> QFunction:
    """Build a typed `QFunction` whose forward returns the constant
    vector `[value, value, ..., value]` of length `n_actions`,
    independent of params or obs. Closed-form control: `max_a Q(s',
    a) == value` for any `next_obs`.

    The framework's `QFunction` Protocol requires `(params, obs) ->
    jax.Array`. Implemented as a plain function (no params used)
    that returns a vector at the action axis with the same leading
    batch shape as `obs` (so batched / single-obs inputs both
    work)."""
    def fn(params: Params, obs: jax.Array) -> jax.Array:
        del params
        # Batch shape = obs.shape minus the trailing obs_dim. For a
        # single obs (1-D), the result is `(n_actions,)`; for
        # batched obs `(B, obs_dim)` the result is `(B, n_actions)`.
        if obs.ndim == 1:
            return jnp.full((n_actions,), value, dtype=jnp.float32)
        batch_shape = obs.shape[:-1]
        return jnp.full((*batch_shape, n_actions), value, dtype=jnp.float32)
    return fn


def test_bellman_target_terminal_zeroes_bootstrap() -> None:
    """`r=1, gamma=0.9, Q(s')=10, done=1, truncated=0` → target=1.
    True termination zeroes the discount: trajectory ended, no
    future."""
    q_network = _constant_q_network(value=10.0)
    # Single-cell batch — scalar batch dim 1 so all per-sample
    # ops stay vector-shaped.
    next_obs = jnp.zeros((1, 4), dtype=jnp.float32)
    reward = jnp.asarray([1.0], dtype=jnp.float32)
    done = jnp.asarray([1.0], dtype=jnp.float32)
    truncated = jnp.asarray([0.0], dtype=jnp.float32)
    target = bootstrap_claim(
        online_params={}, target_params={}, q_network=q_network,
        next_obs=next_obs, reward=reward, done=done,
        truncated=truncated, gamma=0.9,
        greedification=max_greedify,
    )
    # terminated = done * (1 - truncated) = 1 * 1 = 1
    # target = r + γ · (1 − 1) · v(s') = 1
    assert jnp.allclose(target, jnp.asarray([1.0], dtype=jnp.float32)), (
        f'terminal target expected 1.0, got {float(target[0])}'
    )


def test_bellman_target_truncated_continues_bootstrap() -> None:
    """`r=1, gamma=0.9, Q(s')=10, done=1, truncated=1` → target=10.
    Truncation (artificial time-limit cutoff) continues the
    bootstrap as if the trajectory had not ended."""
    q_network = _constant_q_network(value=10.0)
    next_obs = jnp.zeros((1, 4), dtype=jnp.float32)
    reward = jnp.asarray([1.0], dtype=jnp.float32)
    done = jnp.asarray([1.0], dtype=jnp.float32)
    truncated = jnp.asarray([1.0], dtype=jnp.float32)
    target = bootstrap_claim(
        online_params={}, target_params={}, q_network=q_network,
        next_obs=next_obs, reward=reward, done=done,
        truncated=truncated, gamma=0.9,
        greedification=max_greedify,
    )
    # terminated = 1 * (1 - 1) = 0
    # target = 1 + 0.9 · (1 − 0) · 10 = 10
    assert jnp.allclose(target, jnp.asarray([10.0], dtype=jnp.float32)), (
        f'truncated target expected 10.0, got {float(target[0])}'
    )


def test_bellman_target_nonterminal_full_bootstrap() -> None:
    """`r=1, gamma=0.9, Q(s')=10, done=0, truncated=0` → target=10.
    Standard mid-episode step: bootstrap fully active."""
    q_network = _constant_q_network(value=10.0)
    next_obs = jnp.zeros((1, 4), dtype=jnp.float32)
    reward = jnp.asarray([1.0], dtype=jnp.float32)
    done = jnp.asarray([0.0], dtype=jnp.float32)
    truncated = jnp.asarray([0.0], dtype=jnp.float32)
    target = bootstrap_claim(
        online_params={}, target_params={}, q_network=q_network,
        next_obs=next_obs, reward=reward, done=done,
        truncated=truncated, gamma=0.9,
        greedification=max_greedify,
    )
    # terminated = 0 * (1 - 0) = 0
    # target = 1 + 0.9 · 1 · 10 = 10
    assert jnp.allclose(target, jnp.asarray([10.0], dtype=jnp.float32)), (
        f'nonterminal target expected 10.0, got {float(target[0])}'
    )


def test_bellman_target_per_sample_mask() -> None:
    """Batch with mixed flags: each sample's target obeys its own
    (done, truncated) pair independently. Asserts per-row formula
    in a single call (catches reductions across the batch axis
    that would short-circuit the mask)."""
    q_network = _constant_q_network(value=10.0)
    next_obs = jnp.zeros((3, 4), dtype=jnp.float32)
    reward = jnp.asarray([1.0, 1.0, 1.0], dtype=jnp.float32)
    done = jnp.asarray([1.0, 1.0, 0.0], dtype=jnp.float32)
    truncated = jnp.asarray([0.0, 1.0, 0.0], dtype=jnp.float32)
    target = bootstrap_claim(
        online_params={}, target_params={}, q_network=q_network,
        next_obs=next_obs, reward=reward, done=done,
        truncated=truncated, gamma=0.9,
        greedification=max_greedify,
    )
    # row 0: terminated=1 → 1
    # row 1: terminated=0 (truncated) → 10
    # row 2: terminated=0 (mid-episode) → 10
    expected = jnp.asarray([1.0, 10.0, 10.0], dtype=jnp.float32)
    assert jnp.allclose(target, expected), (
        f'per-sample mask broken: expected {expected.tolist()}, '
        f'got {target.tolist()}'
    )


def test_replay_round_trip_preserves_truncated() -> None:
    """`Replay.add(...)` writes truncated; `uniform_sample(...)`
    reads it back. Closed-form: a transition added with
    `truncated=1.0` round-trips to a `Batch.truncated[i] == 1.0`."""
    replay = Replay(capacity=4, batch_size=4)
    state = replay.init(obs_shape=(1,))
    # Add one transition with truncated=1, one with truncated=0.
    txn_trunc = Transition(
        obs=jnp.float32([1.0]),
        action=jnp.int32(0),
        reward=jnp.float32(1.0),
        next_obs=jnp.float32([2.0]),
        done=jnp.float32(1.0),
        truncated=jnp.float32(1.0),
    )
    txn_term = Transition(
        obs=jnp.float32([3.0]),
        action=jnp.int32(0),
        reward=jnp.float32(1.0),
        next_obs=jnp.float32([4.0]),
        done=jnp.float32(1.0),
        truncated=jnp.float32(0.0),
    )
    state = replay.add(state, txn_trunc)
    state = replay.add(state, txn_term)
    # The capacity-4 buffer has two slots populated; valid_size = 2.
    # uniform_sample draws 4 from [0, 2); both indices present in
    # the sample by birthday-style guarantee under 4 draws.
    batch = replay.sample_batch(state, jax.random.PRNGKey(0))
    # For every drawn index i, batch.truncated[i] matches the
    # buffered value at that slot. We assert the per-slot
    # round-trip directly from the replay state.
    assert float(state.truncated[0]) == 1.0, (
        f'slot 0 truncated expected 1.0, got {float(state.truncated[0])}'
    )
    assert float(state.truncated[1]) == 0.0, (
        f'slot 1 truncated expected 0.0, got {float(state.truncated[1])}'
    )
    # Batch must carry the field per-row.
    assert batch.truncated.shape == (4,), (
        f'Batch.truncated shape expected (4,), got {batch.truncated.shape}'
    )
    # Each batch row's truncated value matches the buffer entry at
    # batch.indices[row].
    expected_per_row = state.truncated[batch.indices]
    assert jnp.allclose(batch.truncated, expected_per_row), (
        f'Batch.truncated diverged from buffer round-trip: '
        f'batch={batch.truncated.tolist()} vs '
        f'expected={expected_per_row.tolist()}'
    )


def test_n_step_return_propagates_truncated_within_window() -> None:
    """n_step=3 window: a mid-window truncated step propagates the
    flag through to the emitted aggregated transition. acc_truncated
    is the OR-aggregate (via max) — at most one done/truncated step
    per window (window emits on done), so `max` matches the OR
    semantic exactly."""
    pending = init_pending_n_step(obs_shape=(1,))
    txn1 = Transition(
        obs=jnp.float32([1.0]), action=jnp.int32(0),
        reward=jnp.float32(1.0), next_obs=jnp.float32([2.0]),
        done=jnp.float32(0.0), truncated=jnp.float32(0.0),
    )
    txn2 = Transition(
        obs=jnp.float32([2.0]), action=jnp.int32(1),
        reward=jnp.float32(2.0), next_obs=jnp.float32([3.0]),
        done=jnp.float32(1.0), truncated=jnp.float32(1.0),
    )
    pending, emitted_1, should_emit_1 = n_step_return(
        pending=pending, transition=txn1, n_step=3, gamma=0.9,
    )
    # First step: window has 1 of 3, no terminal → no emit.
    assert float(should_emit_1) == 0.0
    pending, emitted_2, should_emit_2 = n_step_return(
        pending=pending, transition=txn2, n_step=3, gamma=0.9,
    )
    # Second step: terminal mid-window → emit early.
    assert float(should_emit_2) == 1.0
    # The emitted transition carries acc_done=1 AND acc_truncated=1.
    assert float(emitted_2.done) == 1.0
    assert float(emitted_2.truncated) == 1.0, (
        'n-step emit lost the truncated flag from a mid-window '
        'truncation; bootstrap will incorrectly zero its target'
    )
    # Closed-form n-step return: r0 + γ r1 = 1 + 0.9·2 = 2.8.
    assert jnp.allclose(emitted_2.reward, jnp.float32(2.8)), (
        f'n-step accumulated reward off: expected 2.8, '
        f'got {float(emitted_2.reward)}'
    )


# ============ Pardo 2018 — pre-reset obs in replay (#1) ============

class _StepCounterEnvState(NamedTuple):
    """Synthetic env state carrying a single int32 step counter.
    `time` field satisfies gymnax's base `EnvState` interface so
    the rollout phase can read it without special-casing."""
    time: jax.Array  # () int32


def _make_step_counter_env(cap: int = 5) -> object:
    """Build a tiny synthetic env that auto-resets at step `cap`.
    The obs IS the step counter: `obs[0] = state.time` (float).

    Pre-reset path (`step_env`):
      `obs[0] = state.time + 1` (the count of steps taken so far).
      `done = (state.time + 1 >= cap)`.

    Auto-reset path (`step`):
      Wraps `step_env` then `lax.select`s in `reset_env`'s obs when
      done.

    The env's reset returns `obs=[0.0]`, distinguishable from any
    `state.time + 1 >= 1` pre-reset obs — the test's discriminator
    for "replay stored pre-reset" vs "replay stored auto-reset"."""
    cap_local = cap

    class StepCounterEnv:
        def reset(
            self, rng: jax.Array, params: EnvParams,
        ) -> tuple[jax.Array, _StepCounterEnvState]:
            del rng, params
            return jnp.float32([0.0]), _StepCounterEnvState(
                time=jnp.int32(0),
            )

        def reset_env(
            self, rng: jax.Array, params: EnvParams,
        ) -> tuple[jax.Array, _StepCounterEnvState]:
            return self.reset(rng, params)

        def step_env(
            self,
            rng: jax.Array,
            state: _StepCounterEnvState,
            action: jax.Array,
            params: EnvParams,
        ) -> tuple[
            jax.Array,
            _StepCounterEnvState,
            jax.Array,
            jax.Array,
            dict[str, object],
        ]:
            del rng, action, params
            new_time = state.time + jnp.int32(1)
            next_state = _StepCounterEnvState(time=new_time)
            next_obs = new_time.astype(jnp.float32).reshape((1,))
            done = (new_time >= jnp.int32(cap_local)).astype(jnp.bool_)
            # Mark the cap-triggered done as truncated (artificial
            # time-limit cutoff). Natural terminations are
            # truncated=0 by construction here.
            truncated = done.astype(jnp.float32)
            info: dict[str, object] = {'truncated': truncated}
            return next_obs, next_state, jnp.float32(1.0), done, info

        def step(
            self,
            rng: jax.Array,
            state: _StepCounterEnvState,
            action: jax.Array,
            params: EnvParams,
        ) -> tuple[
            jax.Array,
            _StepCounterEnvState,
            jax.Array,
            jax.Array,
            dict[str, object],
        ]:
            next_obs_pre, next_state_pre, reward, done, info = self.step_env(
                rng, state, action, params,
            )
            reset_obs, reset_state = self.reset_env(rng, params)
            final_state = jax.tree.map(
                lambda r, n: jnp.where(done, r, n), reset_state, next_state_pre,
            )
            final_obs = jnp.where(done, reset_obs, next_obs_pre)
            return final_obs, final_state, reward, done, info

        def observation_space(self, params: EnvParams):  # type: ignore[no-untyped-def]
            del params
            from gymnax.environments import spaces
            return spaces.Box(
                low=0.0, high=float(cap_local), shape=(1,), dtype=jnp.float32,
            )

        def action_space(self, params: EnvParams):  # type: ignore[no-untyped-def]
            del params
            from gymnax.environments import spaces
            return spaces.Discrete(num_categories=2)

    return StepCounterEnv()


def test_rollout_stores_pre_reset_next_obs_on_truncation() -> None:
    """Pardo 2018 fix: when the env auto-resets on cap (truncation),
    the replay's `next_obs` must be the PRE-RESET continuation
    state, not the post-reset initial obs. The Bellman target then
    bootstraps against `v(s_pre_reset)` — without this, the agent
    learns "the world resets to obs[0] at the cap" instead of the
    physical continuation.

    Closed-form test: a synthetic step-counter env where
    `obs[0] = state.time`. At cap=5, the truncated transition's
    `state.obs = [4.0]`, `next_obs_pre_reset = [5.0]`,
    `next_obs_post_reset = [0.0]`. The replay's stored next_obs
    must be `[5.0]`, NOT `[0.0]`."""
    from corroborate_rl.dqn.dqn import init_state
    from corroborate_rl.dqn.phases import rollout_phase
    from corroborate_rl.dqn.claims.action_select import (
        epsilon_greedy, linear_epsilon,
    )
    from corroborate_rl.dqn.claims.q_network import mlp_q, MLP
    from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
    from functools import partial

    env = _make_step_counter_env(cap=5)
    rng = jax.random.PRNGKey(7)
    obs_shape = (1,)
    n_actions = 2

    arch = MLP(hidden=(8,))
    arch.init(rng, obs_shape, n_actions)

    # State machine — random policy is fine (the env ignores the
    # action). Replay capacity 16 / batch 1; we only need to
    # observe the cap step.
    replay = Replay(capacity=16, batch_size=1)
    init = init_state(
        env=env,  # type: ignore[arg-type]  # synthetic struct conforms to Env Protocol
        env_params=EnvParams(),  # type: ignore[call-arg]  # gymnax base EnvParams; default field is enough
        obs_shape=obs_shape, n_actions=n_actions,
        rng_key=rng, optimizer=warmed_update(inner=partial(adam), warmup_steps=10),
        replay=replay,
    )

    # Step 5 times — episode caps at step 5 (4→5 transition).
    state = init
    step_fn = partial(
        rollout_phase,
        env=env, env_params=EnvParams(),  # type: ignore[arg-type,call-arg]
        n_actions=n_actions, replay=replay,
        q_network=mlp_q,
        action_select=partial(
            epsilon_greedy,
            # ε=0 throughout — deterministic argmax. Action doesn't
            # affect the step-counter env's dynamics anyway.
            schedule=partial(
                linear_epsilon, eps_init=0.0, eps_final=0.0,
                anneal_steps=1,
            ),
        ),
        state_hash=lambda obs: jnp.int32(0),  # type: ignore[arg-type,return-value]
        n_step=1, gamma=0.99,
    )
    for _ in range(5):
        state, _ = step_fn(state)

    # After 5 steps, the buffer has 5 transitions. The cap-step
    # transition has `obs=[4.0]` and pre-reset `next_obs=[5.0]`.
    buf_obs = state.replay.obs  # (capacity, 1)
    buf_next_obs = state.replay.next_obs  # (capacity, 1)
    buf_done = state.replay.done  # (capacity,)
    buf_truncated = state.replay.truncated  # (capacity,)

    # Slot 4 (the 5th transition) is the cap-step.
    cap_obs = float(buf_obs[4, 0])
    cap_next_obs = float(buf_next_obs[4, 0])
    cap_done = float(buf_done[4])
    cap_truncated = float(buf_truncated[4])

    # The 5th action was taken from `state.time=4` → obs=[4.0].
    assert cap_obs == 4.0, f'cap-step obs expected 4.0, got {cap_obs}'
    # Cap fired: done=1, truncated=1.
    assert cap_done == 1.0
    assert cap_truncated == 1.0, (
        f'cap-step truncated expected 1.0, got {cap_truncated}'
    )
    # THE LOAD-BEARING ASSERTION: pre-reset next_obs is [5.0], not [0.0].
    # If the bug were still present (env.step used instead of
    # env.step_env), the next_obs would be the reset initial [0.0]
    # and the bootstrap target would mask against v(s=0) instead of
    # v(s=5) — a different MDP than the physical continuation.
    assert cap_next_obs == 5.0, (
        f'Pardo 2018 bug: replay stored post-reset next_obs={cap_next_obs} '
        f'instead of pre-reset value 5.0. The Bellman target would now '
        f'bootstrap against v(s_reset_initial=0) — teaching the agent the '
        f'wrong MDP.'
    )


def test_rollout_state_obs_resets_after_truncation() -> None:
    """Companion to `test_rollout_stores_pre_reset_next_obs_on_truncation`:
    the rollout's `state.obs` (for the NEXT step) MUST be the
    reset initial obs after a `done`. Replay sees the pre-reset
    `next_obs`; the state's next iteration starts fresh. Without
    this, training would run on a permanently-terminal state."""
    from corroborate_rl.dqn.dqn import init_state
    from corroborate_rl.dqn.phases import rollout_phase
    from corroborate_rl.dqn.claims.action_select import (
        epsilon_greedy, linear_epsilon,
    )
    from corroborate_rl.dqn.claims.q_network import mlp_q, MLP
    from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
    from functools import partial

    env = _make_step_counter_env(cap=3)
    rng = jax.random.PRNGKey(11)
    obs_shape = (1,)
    n_actions = 2

    arch = MLP(hidden=(4,))
    arch.init(rng, obs_shape, n_actions)

    replay = Replay(capacity=8, batch_size=1)
    init = init_state(
        env=env,  # type: ignore[arg-type]
        env_params=EnvParams(),  # type: ignore[call-arg]
        obs_shape=obs_shape, n_actions=n_actions,
        rng_key=rng, optimizer=warmed_update(inner=partial(adam), warmup_steps=2),
        replay=replay,
    )

    step_fn = partial(
        rollout_phase,
        env=env, env_params=EnvParams(),  # type: ignore[arg-type,call-arg]
        n_actions=n_actions, replay=replay,
        q_network=mlp_q,
        action_select=partial(
            epsilon_greedy,
            schedule=partial(
                linear_epsilon, eps_init=0.0, eps_final=0.0,
                anneal_steps=1,
            ),
        ),
        state_hash=lambda obs: jnp.int32(0),  # type: ignore[arg-type,return-value]
        n_step=1, gamma=0.99,
    )
    # Step through full episode + one step into fresh episode.
    state = init
    for _ in range(4):
        state, _ = step_fn(state)
    # After the cap (step 3 of 1st ep) the state.obs was reset to
    # [0.0]; step 4 brought us to [1.0] in the 2nd episode.
    next_obs_after_reset = float(state.obs[0])
    assert next_obs_after_reset == 1.0, (
        f'After truncation + 1 more step, state.obs expected [1.0] '
        f'(reset to [0.0] then advanced to [1.0]); got {next_obs_after_reset}. '
        f'state.obs did not reset on done — agent is still in the '
        f'terminal state.'
    )


