"""Optimizer factories — `@claim`-wrapped optax handles.

Adam vs RMSProp is a famous DRL swap: Mnih 2015's original Atari
DQN used RMSProp; later work (Hessel 2018 Rainbow, etc.) defaulted
to Adam. The swap measurably changes results on hard-exploration
envs and the literature inconsistently specifies which was used —
making the *optimizer* a first-class intervention surface, not an
engineering detail.

Optax's `optax.adam(lr=...)` is itself a factory returning a
`GradientTransformation` (a NamedTuple of `(init, update)`
closures). Two problems for `mechanism_key` stability if the
substrate consumed it directly:

1. `repr(optax.adam(0.001))` includes memory addresses of internal
   functions — process-unstable canonical strings.
2. The construction-time leaves (lr, b1, b2, eps) aren't visible
   at the optax-handle level; intervention can't address them.

This module's solution: `@claim`-wrapped factory functions
(`adam`, `rmsprop`, `sgd`, `warmed_update`). Substrate composes
via `partial(adam, lr=2e-3)` — the walker descends into the
partial's `.keywords` AND the wrapped function's signature
defaults, surfacing every leaf at composition time.
`canonical_str` produces stable strings like
`partial(Claim:adam;lr=0.002)` regardless of process address.

dqn calls `optimizer()` once at the top of a run to build the
optax handle; `dqn_step` still takes the raw
`GradientTransformation` internally (the `.init` / `.update`
interface JAX traces over).

Author intervention:

    intervention={'optimizer': partial(rmsprop, lr=2.5e-4)}    # Mnih-style
    intervention={'optimizer': partial(adam, lr=1e-4)}        # post-Hessel default
    intervention={'optimizer': partial(sgd, lr=0.1, momentum=0.9)}

Module-Claim form (`Adam(ClaimBase)`, etc.) was retired —
`@claim`-wrapped factories cover the same ground without
duplicating optax's natural shape. See FUTURE_WORKS.md
"Module Claims → pure-functional"."""
from __future__ import annotations

from functools import partial
from typing import Protocol

import jax
import jax.numpy as jnp
import optax

from corroborate.claim import claim


class OptimizerFactory(Protocol):
    """Constructs an `optax.GradientTransformation`. Substrate
    authors compose via `partial(adam, lr=...)` etc. — the partial
    is itself a callable matching this Protocol; calling it builds
    the optax `(init, update)` pair.

    dqn calls `optimizer()` once at the top of a run to build the
    optax handle; `train_phase` consumes the raw handle's
    `.init` / `.update` interface JAX traces over."""
    def __call__(self) -> optax.GradientTransformation: ...


@claim
def adam(
    *,
    lr: float = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> optax.GradientTransformation:
    """Adam optimizer (Kingma & Ba 2015). Default lr=1e-3 matches
    optax's default; b1/b2/eps are the canonical Adam leaves.

    `weight_decay > 0` switches to AdamW (Loshchilov & Hutter 2019)
    — decoupled weight decay applied after the Adam update. Used
    as an explicit L2 regularizer to test whether γ-as-implicit-
    regularizer confounds chain-depth-amplifier interpretations at
    high γ. Default 0 preserves the historical Adam behavior."""
    if weight_decay > 0.0:
        # AdamW = scale_by_adam → add_decayed_weights → scale_by_lr.
        return optax.chain(
            optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
            optax.add_decayed_weights(weight_decay),
            optax.scale_by_learning_rate(lr),
        )
    return optax.adam(learning_rate=lr, b1=b1, b2=b2, eps=eps)


@claim
def rmsprop(
    *,
    lr: float = 2.5e-4,
    decay: float = 0.95,
    eps: float = 1e-2,
) -> optax.GradientTransformation:
    """RMSProp (Tieleman & Hinton 2012). Mnih 2015's original
    Atari DQN choice. `decay` is the running-mean decay; `eps`
    the numerical stabiliser inside the update rule."""
    return optax.rmsprop(learning_rate=lr, decay=decay, eps=eps)


@claim
def sgd(
    *,
    lr: float = 1e-3,
    momentum: float = 0.0,
) -> optax.GradientTransformation:
    """Plain SGD with optional momentum. Rarely used in DQN
    practice but included as the trivial baseline + ablation."""
    return optax.sgd(
        learning_rate=lr,
        momentum=momentum if momentum > 0.0 else None,
    )


@claim
def warmed_update(
    *,
    inner: 'OptimizerFactory',
    warmup_steps: int = 1_000,
) -> optax.GradientTransformation:
    """Optimizer wrapper that zeros parameter updates during the
    first `warmup_steps` calls — bridges the buffer-warmup gap
    where sampled gradients are uninformative (replay buffer is
    near-empty, batches are degenerate).

    Compositionally: `chain(inner(), scale_by_schedule(0 while
    count < warmup_steps else 1))`. The inner optimizer's internal
    state (e.g. Adam's moment estimates) still advances during
    warmup; only the parameter deltas are zeroed. Same end behavior
    as the historical hand-coded skip in `train_phase`, but encoded
    as a property of the optimizer.

    `warmup_steps` is engineering, not paper-honest theory; making
    it a wrapper-claim rather than a top-level dqn HP keeps the
    paper-prose composition (`rollout → train → sync`) clean. The
    update mechanism owns its own warmup.

    `inner` is itself an `OptimizerFactory` — typically
    `partial(adam, lr=...)` or another @claim'd factory. The walker
    recurses through nested partials, so leaves like
    `inner.keywords.lr` surface at composition time."""

    def schedule(count: jax.Array) -> jax.Array:
        return jnp.where(count < warmup_steps, 0.0, 1.0)

    return optax.chain(inner(), optax.scale_by_schedule(schedule))


# Default optimizer for `dqn`'s signature default. Matches the
# historical `WarmedUpdate(inner=Adam())` baseline — Adam at
# Kingma-Ba defaults, 1000-step warmup. Authors override per-arm
# via `intervention={'optimizer': partial(adam, lr=2e-3)}`.
default_optimizer: OptimizerFactory = partial(
    warmed_update, inner=partial(adam),
)
