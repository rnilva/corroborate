"""Typed override schemas for the DDQN substrate — closed
TypedDicts that pyright narrows against at intervention-construction
sites.

corroborate's intervention surface is `dict[str, object]` passed
to `Hypothesis(intervention=...)`, then spread into `partial(dqn,
**intervention)`. The dict's untyped — typos and wrong-type values
slip through to runtime.

These TypedDicts add compile-time discipline:

    intervention: DQNOverrides = {
        'gamma': 0.95,
        'replay': Replay(capacity=50_000),
        'bootstrap': partial(bootstrap, greedification=double_greedify),
    }
    h = Hypothesis(name='ddqn', intervention=intervention, ...)

Pyright narrows each key against `DQNOverrides`; with `closed=True`
unknown keys are hard errors and value types are checked against
the TypedDict declaration. `total=False` means each key is
optional.

Note: this is a pure-typing layer. The runtime behaviour of
`partial(dqn, **intervention)` is unchanged — no `bind` indirection
(v10's `bind` was retired in corroborate via subtraction; the
walker / `canonical_str` handle `partial` directly).

Module Claims (`MLP`, `EpsilonGreedy`, `Replay`, optimizer
subclasses) are typed as their concrete dataclass classes — the
override surface is the constructor / `replace()`. Free-function
Claims (`bootstrap`, `periodic_copy`, `linear_epsilon`,
`squared_error`) accept `Callable` for full swap or a `partial`
binding."""
from __future__ import annotations

from collections.abc import Callable

from typing_extensions import TypedDict

from corroborate.rl.dqn.claims.action_select import EpsilonGreedy
from corroborate.rl.dqn.claims.q_network import MLP
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.claims.optimizer import OptimizerFactory


# Optimizer factories — any callable returning an optax handle.
# `OptimizerFactory` Protocol covers `partial(adam, lr=...)`,
# `partial(warmed_update, inner=..., warmup_steps=...)`, etc.
# Authors swap via a different partial composition.
type OptimizerOverride = OptimizerFactory


class DQNOverrides(TypedDict, total=False, closed=True):
    """Closed TypedDict for `dqn`'s intervention surface.

    Listed kwargs match the slot + cross-cutting kwargs of `dqn`
    in `corroborate.rl.dqn.dqn`. Exogenous kwargs (`rng_key`,
    `env`, `env_params`, `obs_dim`, `n_actions`,
    `eval_episode_cap`, `state_hash`) are NOT intervention surface
    and are not included.

    Use:

        from corroborate.rl.dqn.overrides import DQNOverrides

        intervention: DQNOverrides = {
            'gamma': 0.95,
            'replay': Replay(capacity=50_000),
        }
        h = Hypothesis(name='lower_gamma', intervention=intervention, ...)

    Pyright narrows each key against the declared types; unknown
    keys raise at typecheck time."""

    # Cross-cutting scalar HPs.
    gamma: float
    sync_period: int
    total_steps: int
    eval_every: int
    n_episodes: int

    # Slot Claims — Module Claims as concrete dataclass instances;
    # free-function Claims as Callable (full swap or partial).
    q_network: MLP | Callable[..., object]
    action_select: EpsilonGreedy | Callable[..., object]
    replay: Replay
    bootstrap: Callable[..., object]
    loss_fn: Callable[..., object]
    target_sync: Callable[..., object]
    optimizer: OptimizerOverride
