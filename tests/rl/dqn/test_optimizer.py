"""Tests for optimizer factories — Adam / RMSProp / SGD / WarmedUpdate.

WarmedUpdate's contract: parameter updates are zero during the
first `warmup_steps` calls, then identical to the inner optimizer's
updates afterwards. Inner state still advances through warmup."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from corroborate.rl.dqn.claims.optimizer import (
    Adam,
    RMSProp,
    SGD,
    WarmedUpdate,
)


def _grads_like(params: jax.Array) -> jax.Array:
    """Constant ones gradient — easy to reason about."""
    return jnp.ones_like(params)


def test_adam_returns_optax_handle() -> None:
    handle = Adam(lr=1e-3)()
    assert isinstance(handle, optax.GradientTransformation)


def test_rmsprop_returns_optax_handle() -> None:
    handle = RMSProp(lr=2.5e-4)()
    assert isinstance(handle, optax.GradientTransformation)


def test_sgd_returns_optax_handle() -> None:
    handle = SGD(lr=0.1)()
    assert isinstance(handle, optax.GradientTransformation)


def test_warmed_update_zeros_updates_during_warmup() -> None:
    """During the first `warmup_steps` calls, the wrapped
    optimizer must produce zero parameter deltas regardless of
    gradient magnitude. After warmup, deltas match Adam-style
    nonzero updates."""
    warmup = 5
    factory = WarmedUpdate(inner=Adam(lr=1e-2), warmup_steps=warmup)
    handle = factory()

    params = jnp.zeros((4,))
    state = handle.init(params)
    grads = _grads_like(params)

    # Inside warmup window: every update should be (close to) zero.
    for _ in range(warmup):
        updates, state = handle.update(grads, state, params)
        assert jnp.allclose(updates, 0.0), (
            f'expected zero updates during warmup; got {updates}'
        )

    # First post-warmup call: scale flips to 1, Adam delta is nonzero.
    updates, state = handle.update(grads, state, params)
    assert not jnp.allclose(updates, 0.0), (
        f'expected nonzero updates after warmup; got {updates}'
    )


def test_warmed_update_inner_state_advances_during_warmup() -> None:
    """Adam's moment estimates must keep updating through warmup
    so the *first* post-warmup step uses correctly-warmed moments
    (not fresh-init moments). Compared by running two factories:
    one with warmup, one without. The post-warmup updates should
    match the no-warmup updates at the same call index, since the
    inner optimizer has seen the same gradient sequence."""
    warmup = 3
    warmed = WarmedUpdate(inner=Adam(lr=1e-2), warmup_steps=warmup)()
    plain = Adam(lr=1e-2)()

    params = jnp.zeros((4,))
    grads = _grads_like(params)

    s_warmed = warmed.init(params)
    s_plain = plain.init(params)

    # Run both for `warmup` steps.
    for _ in range(warmup):
        _, s_warmed = warmed.update(grads, s_warmed, params)
        _, s_plain = plain.update(grads, s_plain, params)

    # Step `warmup + 1`: warmed should now produce the same delta
    # as plain at the same step index.
    u_warmed, _ = warmed.update(grads, s_warmed, params)
    u_plain, _ = plain.update(grads, s_plain, params)

    assert jnp.allclose(u_warmed, u_plain), (
        f'inner state divergence: warmed={u_warmed}, plain={u_plain}'
    )
