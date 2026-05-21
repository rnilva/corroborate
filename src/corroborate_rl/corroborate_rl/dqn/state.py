"""DQN state — single threaded structure read by `dqn_step`.

A `NamedTuple` so it threads through `jax.lax.scan` cleanly. The
state carries everything `dqn_step` needs across iterations: the
two parameter sets (online + target), optimizer state, replay
sub-state, env state, current observation, step counter, RNG, and
running episode return.

Sub-component states are bundled where their owning Module is a
typed component:

- `replay: ReplayState` — bundled because `Replay` is a Module
  with `init/add/sample` methods. Authors swapping for
  PrioritisedReplay define their own ReplayState shape.

Other state — `online_params`, `target_params`, `opt_state`,
`env_state`, `obs`, `step`, `rng_key`, `ep_return` — stays flat
for now (Phase 4 will bundle params+opt_state into a `Learner`
sub-state when that lands)."""
from __future__ import annotations

from typing import NamedTuple

import jax

from corroborate_rl.dqn.claims.replay import PendingNStepState, ReplayState
from corroborate_rl.dqn.types import EnvState, OptState, Params


class DQNState(NamedTuple):
    """Per-step DQN state. Threaded through `jax.lax.scan`.

    `env_state` and `opt_state` are typed `object` (via
    `types.EnvState` / `types.OptState`) — the framework can't
    constrain third-party (gymnax / optax) pytree shapes. Bridge
    bodies that consume these narrow at use site."""

    # Parameter sets — both same pytree shape (MLP weights/biases).
    online_params: Params
    target_params: Params

    # optax optimizer state — opaque to the framework.
    opt_state: OptState

    # Replay sub-state — owned by the `Replay` Module's
    # `init/add/sample_batch` methods. dqn threads it but doesn't
    # introspect; alternative replay implementations define their
    # own ReplayState shape.
    replay: ReplayState

    # In-progress n-step pending window. Owned by the
    # `n_step_return` Free Claim called from `rollout_phase`.
    # For n_step=1 this is a no-op state (count goes 0→1 every
    # step and resets on emit), for n_step>1 it holds the running
    # Σ γᵏ rₖ aggregate over the window.
    pending_n_step: PendingNStepState

    # Env state pytree (gymnax-specific) and current observation.
    env_state: EnvState
    obs: jax.Array             # (obs_dim,)

    # Bookkeeping.
    step: jax.Array            # () int32 — total steps elapsed
    rng_key: jax.Array         # PRNGKey

    # Running per-episode return; resets on `done`.
    ep_return: jax.Array       # () float32

    # Per-state-hash visit count, indexed by the env's registered
    # `state_hash(obs) → int`. Incremented at action-selection time
    # in rollout_phase. Used by count-weighted-loss interventions
    # (loop-mediation falsification): downweight per-sample loss
    # at over-visited states to test if uniform-state-coverage
    # training reproduces DDQN's benefit. Zero-init; size =
    # env_spec.state_hash_cardinality.
    state_hash_count: jax.Array   # (cardinality,) int32
