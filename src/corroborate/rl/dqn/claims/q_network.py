"""Q-function — forward Claim + config-bundle.

Two surfaces in this file:

- `mlp_forward` / `cnn_forward` — `@claim`'d free functions. The
  forward pass IS the framework Claim — universal approximation
  (Hornik 1989, Cybenko 1989) realises here on each call.
  `FnClaim.__call__` records the invocation into the active
  trace.
- `MLP` / `CNN` — frozen-dataclass **config bundles** (not Claims
  themselves). They carry construction-time architecture leaves
  (`hidden`, `obs_shape`, ...) so the walker surfaces them at
  composition time, and provide `init` (parameter allocation
  mechanics, no theorem) + `__call__` (delegates to the forward
  Claim). `MLP(hidden=(128,))` is the configured Q-function;
  intervention swaps it whole or via `replace(MLP(), hidden=...)`.

`init` is **mechanics** — parameter allocation has no theorem,
no `record_call`, just bookkeeping that materialises the
parameter pytree once per cell. Hornik's claim attaches to the
forward pass, hence `mlp_forward` is the Claim.

Substrate flattens obs to a 1D vector for replay storage; CNN
reshapes back to its declared `obs_shape` inside `cnn_forward`.
The framework's `init_state` and `Replay` layers stay obs-flat;
only CNN sees the spatial structure.

dqn doesn't know what's INSIDE the params PyTree — `Params` is
opaque. Tabular Q (params = lookup table), linear Q (params =
weight matrix), MLP Q (params = list of layer dicts), conv Q,
etc. all conform to the same `QFunction` Protocol. The
parameterisation is genuinely Q's business.

**Theorem reference (on `_forward`, not on `_init`).** Universal
approximation (Hornik 1989, Cybenko 1989): a sufficiently wide
MLP can approximate any continuous function on a compact set.
The function class *contains* Q*; gradient descent under
bootstrap is *not* guaranteed to find it (deadly triad — off-
policy + bootstrap + FA can diverge per Tsitsiklis & Van Roy
1997). Banach-contraction-rate gap (Bertsekas-Tsitsiklis §6.3)
is the principled measurement; deferred — needs Q-snapshot
probe (see FUTURE_WORKS.md)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import jax
import jax.numpy as jnp

from corroborate.claim import claim


type Params = dict[str, jax.Array]
"""Opaque PyTree of Q-function parameters. Each Q-function
implementation defines its own internal layout (e.g. MLP uses
`w0`, `b0`, `w1`, `b1`, ...); dqn only threads the pytree
through, never inspects it."""


# ============ Protocol — Q-function structural contract ============

@runtime_checkable
class QFunction(Protocol):
    """Q-function bundle — `init` (mechanics) + `__call__`
    (forward Claim delegation).

    Concrete impls (`MLP`, `CNN`) are frozen-dataclass config
    bundles — NOT Claims themselves; their `__call__` delegates
    to a free-function `@claim` (e.g. `mlp_forward`) that records
    itself into the trace. The bundle's purpose is to carry
    construction-time architecture leaves (`MLP.hidden`,
    `CNN.obs_shape`, ...) so the walker surfaces them at
    composition time.

    `init(rng, obs_shape, n_actions) -> params` is **mechanics** —
    parameter allocation has no theorem; the framework doesn't
    record it. It's part of the Protocol because dqn calls it
    once at cell-init to materialise the parameter pytree.

    `obs_shape` is the env's full observation shape — typically
    `(d,)` for vector envs (CartPole, Acrobot, ...) and
    `(H, W, C)` / similar for image-shaped envs (MinAtar,
    procgen). Q-network impls flatten or convolve internally
    based on the shape they receive.

    `Params` is opaque to dqn; tabular, linear, and MLP
    Q-functions each define their own internal layout. The
    bundle's `init` returns it and `__call__` consumes it."""
    def init(
        self,
        rng_key: jax.Array,
        obs_shape: tuple[int, ...],
        n_actions: int,
    ) -> Params: ...
    def __call__(self, params: Params, obs: jax.Array) -> jax.Array: ...


# ============ MLP — init mechanics + forward Claim + bundle ============


def mlp_init(
    rng_key: jax.Array,
    obs_shape: tuple[int, ...],
    n_actions: int,
    *,
    hidden: tuple[int, ...],
) -> Params:
    """Initialise MLP parameter pytree. Layer-by-layer uniform
    init in `[-sqrt(1/fan_in), sqrt(1/fan_in)]`; biases zero.

    Mechanics: parameter allocation has no theorem; no
    `record_call`. `obs_shape` may be multi-dim (e.g.
    `(10, 10, 4)` for MinAtar); the first weight matrix's input
    dim is `prod(obs_shape)` — `mlp_forward` flattens at call
    time."""
    in_dim = 1
    for d in obs_shape:
        in_dim *= d
    params: Params = {}
    keys = jax.random.split(rng_key, len(hidden) + 1)
    for i, h in enumerate(hidden):
        bound = jnp.sqrt(1.0 / in_dim)
        params[f'w{i}'] = jax.random.uniform(
            keys[i], (in_dim, h), minval=-bound, maxval=bound,
        )
        params[f'b{i}'] = jnp.zeros((h,))
        in_dim = h
    bound = jnp.sqrt(1.0 / in_dim)
    params[f'w{len(hidden)}'] = jax.random.uniform(
        keys[-1], (in_dim, n_actions), minval=-bound, maxval=bound,
    )
    params[f'b{len(hidden)}'] = jnp.zeros((n_actions,))
    return params


@claim
def mlp_forward(params: Params, obs: jax.Array) -> jax.Array:
    """ReLU MLP forward pass — universal approximation Claim
    (Hornik 1989, Cybenko 1989).

    Returns Q-values: `(n_actions,)` for a single obs or
    `(batch, n_actions)` for a batch.

    Accepts multi-dim `obs` (e.g. (..., H, W, C)) by flattening
    trailing axes greedily until their product matches the
    first weight matrix's input dim. Substrate stores obs at
    native shape; the forward fn collapses for the dot product
    here.

    Generic over architecture: layer count is read from the
    params dict size (`len(params) // 2`), so the same forward
    fn handles any `hidden` configuration. `MLP.hidden` lives
    on the bundle; the forward only sees the materialised
    params."""
    n_layers = len(params) // 2
    in_dim = int(params['w0'].shape[0])
    cum = 1
    k = 0
    for axis in reversed(obs.shape):
        k += 1
        cum *= int(axis)
        if cum == in_dim:
            break
    if cum != in_dim:
        raise ValueError(
            f'mlp_forward: cannot match obs.shape={obs.shape} '
            f'trailing dims to in_dim={in_dim}',
        )
    x = obs.reshape((*obs.shape[:obs.ndim - k], in_dim))
    for i in range(n_layers - 1):
        x = jnp.dot(x, params[f'w{i}']) + params[f'b{i}']
        x = jnp.maximum(x, 0.0)
    result: jax.Array = (
        jnp.dot(x, params[f'w{n_layers - 1}'])
        + params[f'b{n_layers - 1}']
    )
    return result


@dataclass(frozen=True, slots=True)
class MLP:
    """Two-hidden-layer ReLU MLP Q-function bundle.

    Frozen-dataclass **config bundle** (NOT a Claim) —
    construction-time leaf `hidden` is observed at composition
    time; `init` is mechanics; `__call__` delegates to
    `mlp_forward` (the Hornik 1989 Free Claim, which records
    itself).

    Authors override via `replace(MLP(), hidden=(128,))` or
    pass a different bundle wholesale via intervention
    (`SpectralNormMLP`, `Tabular`, etc. — all matching the
    `QFunction` Protocol structurally)."""
    hidden: tuple[int, ...] = (64, 64)

    def init(
        self,
        rng_key: jax.Array,
        obs_shape: tuple[int, ...],
        n_actions: int,
    ) -> Params:
        return mlp_init(
            rng_key, obs_shape, n_actions, hidden=self.hidden,
        )

    def __call__(self, params: Params, obs: jax.Array) -> jax.Array:
        return mlp_forward(params, obs)


# Default Q-function — re-exported under the historical name so
# call sites that imported `mlp_q` continue to read paper-honestly
# (the symbol is the configured bundle).
mlp_q = MLP()


# ============ CNN — init mechanics + forward Claim + bundle ============


def cnn_init(
    rng_key: jax.Array,
    obs_shape: tuple[int, ...],
    n_actions: int,
    *,
    cnn_obs_shape: tuple[int, ...],
    channels: tuple[int, ...],
    kernel_size: int,
    hidden: tuple[int, ...],
) -> Params:
    """Allocate conv kernels + dense head parameter pytree.

    Mechanics: He init for conv kernels (`bound = sqrt(2/fan_in)`),
    Glorot uniform for dense weights. Biases zero.

    `obs_shape` (the substrate-passed env obs shape) must match
    the bundle's `cnn_obs_shape` field — loud failure on mismatch
    since the conv kernels' first-layer in-channels are sized at
    construction time."""
    if obs_shape != cnn_obs_shape:
        raise ValueError(
            f'cnn_init: substrate obs_shape={obs_shape} '
            f'inconsistent with bundle obs_shape={cnn_obs_shape}',
        )
    if len(cnn_obs_shape) != 3:
        raise ValueError(
            f'cnn_init: obs_shape must be (H, W, C); '
            f'got {cnn_obs_shape}',
        )
    h_in, w_in, c_in = cnn_obs_shape
    n_keys = len(channels) + len(hidden) + 1
    keys = jax.random.split(rng_key, n_keys)
    params: Params = {}

    c_prev = c_in
    for i, c_out in enumerate(channels):
        fan_in = kernel_size * kernel_size * c_prev
        bound = jnp.sqrt(2.0 / fan_in)
        params[f'kw{i}'] = jax.random.uniform(
            keys[i],
            (kernel_size, kernel_size, c_prev, c_out),
            minval=-bound, maxval=bound,
        )
        params[f'kb{i}'] = jnp.zeros((c_out,))
        c_prev = c_out

    flat_dim = h_in * w_in * c_prev
    in_dim = flat_dim
    for i, h_dense in enumerate(hidden):
        bound = jnp.sqrt(1.0 / in_dim)
        params[f'dw{i}'] = jax.random.uniform(
            keys[len(channels) + i],
            (in_dim, h_dense),
            minval=-bound, maxval=bound,
        )
        params[f'db{i}'] = jnp.zeros((h_dense,))
        in_dim = h_dense

    bound = jnp.sqrt(1.0 / in_dim)
    params[f'dw{len(hidden)}'] = jax.random.uniform(
        keys[-1], (in_dim, n_actions),
        minval=-bound, maxval=bound,
    )
    params[f'db{len(hidden)}'] = jnp.zeros((n_actions,))
    return params


@claim
def cnn_forward(
    params: Params,
    obs: jax.Array,
    *,
    obs_shape: tuple[int, ...],
) -> jax.Array:
    """Conv stack → flatten → dense head → Q-values
    (..., n_actions). Universal approximation Claim with
    inductive bias for translation equivariance + locality
    (LeCun 1989).

    `obs_shape` is the (H, W, C) shape declared on the bundle —
    needed at forward time to verify the obs trailing dims and
    fold leading dims (single, batch, vmap'd seeds) into a
    single batch axis at conv time. The number of conv layers
    is inferred from the params dict (`kw{i}` / `kb{i}` keys);
    same for dense layers (`dw{i}` / `db{i}`)."""
    n_obs_dims = len(obs_shape)
    if obs.shape[-n_obs_dims:] != obs_shape:
        raise ValueError(
            f'cnn_forward: obs trailing shape '
            f'{obs.shape[-n_obs_dims:]} != obs_shape '
            f'{obs_shape}',
        )
    leading = obs.shape[:-n_obs_dims]
    if not leading:
        x = obs[None]
    else:
        collapsed = 1
        for d in leading:
            collapsed *= d
        x = obs.reshape((collapsed, *obs_shape))

    n_conv = sum(1 for k in params if k.startswith('kw'))
    for i in range(n_conv):
        x = jax.lax.conv_general_dilated(
            x, params[f'kw{i}'],
            window_strides=(1, 1), padding='SAME',
            dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
        )
        x = x + params[f'kb{i}']
        x = jnp.maximum(x, 0.0)

    x = x.reshape((x.shape[0], -1))

    n_dense_total = sum(1 for k in params if k.startswith('dw'))
    n_dense_hidden = n_dense_total - 1
    for i in range(n_dense_hidden):
        x = jnp.dot(x, params[f'dw{i}']) + params[f'db{i}']
        x = jnp.maximum(x, 0.0)
    x = (
        jnp.dot(x, params[f'dw{n_dense_hidden}'])
        + params[f'db{n_dense_hidden}']
    )

    if not leading:
        return x[0]
    return x.reshape((*leading, x.shape[-1]))


@dataclass(frozen=True, slots=True)
class CNN:
    """Convolutional Q-function bundle for image obs.

    Frozen-dataclass config bundle — construction-time leaves
    (`obs_shape`, `channels`, `kernel_size`, `hidden`) are
    observed at composition time; `init` is mechanics;
    `__call__` delegates to `cnn_forward` (the LeCun-style
    Free Claim, recorded into the trace).

    `obs_shape` must match the env's observation shape (H, W, C).
    Conv stack uses SAME padding (H, W preserved) + ReLU;
    flatten + dense head emits Q-values."""
    obs_shape: tuple[int, ...] = (10, 10, 4)
    channels: tuple[int, ...] = (16, 32)
    kernel_size: int = 3
    hidden: tuple[int, ...] = (128,)

    def init(
        self,
        rng_key: jax.Array,
        obs_shape: tuple[int, ...],
        n_actions: int,
    ) -> Params:
        return cnn_init(
            rng_key, obs_shape, n_actions,
            cnn_obs_shape=self.obs_shape,
            channels=self.channels,
            kernel_size=self.kernel_size,
            hidden=self.hidden,
        )

    def __call__(self, params: Params, obs: jax.Array) -> jax.Array:
        return cnn_forward(params, obs, obs_shape=self.obs_shape)
