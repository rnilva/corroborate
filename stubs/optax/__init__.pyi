"""Minimal optax stubs — typed facade at the library boundary.

Optax's real `Params` / `Updates` / `OptState` aliases collapse to a
ForwardRef'd `ArrayTree` union that pyright cannot narrow without
casts at every call site. Rather than scatter casts through every
DQN claim, we narrow at the boundary: a generic `_T` flows the
caller's declared pytree type through `update` / `apply_updates`
without losing it.

Mirrors poc_v9's `poc_v8/stubs/optax/__init__.pyi`. Scope: only
the surface that `corroborate.rl.dqn` actually touches. Expand
when a new optax function is reached for."""
from __future__ import annotations

from typing import Protocol, TypeAlias, TypeVar, runtime_checkable


_T = TypeVar('_T')


# Public name aliases. Real optax exposes these as ForwardRef'd
# unions over `chex.ArrayTree`; the framework holds them as opaque
# pytrees and never inspects them, so `object` is honest. Real
# narrowing for params / updates flows through the generic `_T`
# on `apply_updates` / `GradientTransformation.update`.
Params: TypeAlias = object
Updates: TypeAlias = object
OptState: TypeAlias = object


class EmptyState:
    """Opaque OptState used by stateless transforms."""
    ...


@runtime_checkable
class GradientTransformation(Protocol):
    """Pair of (init, update) functions transforming gradients.

    Generic in tree type `T`: `update` round-trips the caller's
    pytree type, `init` produces an opaque OptState."""
    def init(self, params: _T, /) -> object: ...
    def update(
        self, updates: _T, state: object, params: _T | None = ..., /,
    ) -> tuple[_T, object]: ...


def apply_updates(params: _T, updates: _T) -> _T: ...


# Common optimizer constructors used by claims.
def adam(
    learning_rate: float,
    b1: float = ...,
    b2: float = ...,
    eps: float = ...,
    eps_root: float = ...,
    mu_dtype: object = ...,
    *,
    nesterov: bool = ...,
) -> GradientTransformation: ...


def sgd(
    learning_rate: float,
    momentum: float | None = ...,
    nesterov: bool = ...,
    accumulator_dtype: object = ...,
) -> GradientTransformation: ...


def rmsprop(
    learning_rate: float,
    decay: float = ...,
    eps: float = ...,
    initial_scale: float = ...,
    centered: bool = ...,
    momentum: float | None = ...,
    nesterov: bool = ...,
) -> GradientTransformation: ...


# Composition + scheduling primitives. `chain` glues
# transformations sequentially; `scale_by_schedule` multiplies
# update magnitudes by a step-indexed scalar (used by
# `WarmedUpdate` to zero deltas during the warmup window).
def chain(*args: GradientTransformation) -> GradientTransformation: ...


from collections.abc import Callable as _Callable

import jax as _jax


def scale_by_schedule(
    step_size_fn: _Callable[[_jax.Array], _jax.Array],
) -> GradientTransformation: ...


# AdamW components — `optax.adamw` decomposes as
# `chain(scale_by_adam, add_decayed_weights, scale_by_learning_rate)`.
# Exposed as separate primitives for callers that need explicit
# composition (e.g. matching weight-decay placement to a custom
# schedule).
def scale_by_adam(
    b1: float = ...,
    b2: float = ...,
    eps: float = ...,
    eps_root: float = ...,
    mu_dtype: object = ...,
    *,
    nesterov: bool = ...,
) -> GradientTransformation: ...


def add_decayed_weights(
    weight_decay: float = ...,
    mask: object = ...,
) -> GradientTransformation: ...


def scale_by_learning_rate(
    learning_rate: float,
    *,
    flip_sign: bool = ...,
) -> GradientTransformation: ...
