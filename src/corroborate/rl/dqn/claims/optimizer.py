"""Optimizer factories — Module-wrapped optax handles.

Adam vs RMSProp is a famous DRL swap: Mnih 2015's original Atari
DQN used RMSProp; later work (Hessel 2018 Rainbow, etc.) defaulted
to Adam. The swap measurably changes results on hard-exploration
envs and the literature inconsistently specifies which was used —
making the *optimizer* a first-class intervention surface, not an
engineering detail.

Optax's `optax.adam(lr=...)` returns a `NamedTuple` of closures.
Two problems for `mechanism_key` stability:

1. `repr(optax.adam(0.001))` includes memory addresses of internal
   functions — process-unstable canonical strings.
2. The construction-time HPs (lr, b1, b2, eps) aren't visible at
   the optax-handle level; intervention can't address them.

This module's solution: thin frozen-dataclass factory Modules
(`Adam`, `RMSProp`, `SGD`) that own optimizer HPs as typed fields.
`__call__()` returns a fresh optax handle. `mechanism_key` sees
`Module:Adam(b1=0.9,b2=0.999,eps=1e-8,lr=0.001)` — process-stable
+ HP-introspectable.

dqn calls `optimizer()` once at the top of a run to build the
handle; dqn_step still takes the raw `optax.GradientTransformation`
internally (the .init/.update interface JAX traces over).

Author intervention:

    intervention={'optimizer': RMSProp(lr=2.5e-4)}  # Mnih-style
    intervention={'optimizer': Adam(lr=1e-4)}      # post-Hessel default
    intervention={'optimizer': SGD(lr=0.1, momentum=0.9)}"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import optax

from corroborate.claim import ClaimBase, record_call
from corroborate.rl.dqn.types import OptimizerFactory


@dataclass(frozen=True, slots=True)
class Adam(ClaimBase):
    """Adam optimizer (Kingma & Ba 2015). Default lr=1e-3 matches
    optax's default; b1/b2/eps are the canonical Adam HPs."""
    lr: float = 1e-3
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8

    def __call__(self) -> optax.GradientTransformation:
        result = optax.adam(
            learning_rate=self.lr, b1=self.b1, b2=self.b2, eps=self.eps,
        )
        record_call(self, (), {}, result)
        return result


@dataclass(frozen=True, slots=True)
class RMSProp(ClaimBase):
    """RMSProp (Tieleman & Hinton 2012). Mnih 2015's original
    Atari DQN choice. `decay` is the running-mean decay; `eps`
    the numerical stabiliser inside the update rule."""
    lr: float = 2.5e-4
    decay: float = 0.95
    eps: float = 1e-2

    def __call__(self) -> optax.GradientTransformation:
        result = optax.rmsprop(
            learning_rate=self.lr, decay=self.decay, eps=self.eps,
        )
        record_call(self, (), {}, result)
        return result


@dataclass(frozen=True, slots=True)
class SGD(ClaimBase):
    """Plain SGD with optional momentum. Rarely used in DQN
    practice but included as the trivial baseline + ablation."""
    lr: float = 1e-3
    momentum: float = 0.0

    def __call__(self) -> optax.GradientTransformation:
        result = optax.sgd(
            learning_rate=self.lr,
            momentum=self.momentum if self.momentum > 0.0 else None,
        )
        record_call(self, (), {}, result)
        return result


@dataclass(frozen=True, slots=True)
class WarmedUpdate(ClaimBase):
    """Optimizer wrapper that zeros parameter updates during the
    first `warmup_steps` calls — bridges the buffer-warmup gap
    where sampled gradients are uninformative (replay buffer is
    near-empty, batches are degenerate).

    Compositionally: `chain(inner_factory(), scale_by_schedule(0
    while count < warmup_steps else 1))`. The inner optimizer's
    internal state (e.g. Adam's moment estimates) still advances
    during warmup; only the parameter deltas are zeroed. Same end
    behavior as the historical hand-coded skip in `train_phase`,
    but encoded as a property of the optimizer.

    `warmup_steps` is engineering, not paper-honest theory; making
    it a wrapper-Module rather than a top-level dqn HP keeps the
    paper-prose composition (`rollout → train → sync`) clean. The
    update mechanism owns its own warmup."""
    inner: 'OptimizerFactory' = field(default_factory=lambda: Adam())
    warmup_steps: int = 1_000

    def __call__(self) -> optax.GradientTransformation:
        warmup = self.warmup_steps

        def schedule(count: jax.Array) -> jax.Array:
            return jnp.where(count < warmup, 0.0, 1.0)

        result = optax.chain(
            self.inner(),
            optax.scale_by_schedule(schedule),
        )
        record_call(self, (), {}, result)
        return result
