"""Q-function — Module Claim with `__call__` = forward pass.

`MLP` is a Module Claim. Its `__call__` IS the theoretical
Claim — universal approximation (Hornik 1989, Cybenko 1989) —
recorded via `record_call`. `init` is parameter-allocation
**mechanics** — no theorem, no record_call, just bookkeeping
that materialises the parameter pytree once per cell.

`CNN` is a sibling Module Claim for image-shaped observations.
Same theorem reference at the function-class level; the
inductive bias for translation equivariance + locality is what
makes CNNs sample-efficient on pixel-grid envs (LeCun 1989).
Substrate flattens obs to a 1D vector for replay storage; CNN
reshapes back to its declared `obs_shape` inside `__call__`.
The framework's `init_state` and `Replay` layers stay obs-flat;
only CNN sees the spatial structure.

Architecture leaves (`hidden`, etc.) live as frozen-dataclass
fields so they travel with the Module. `MLP(hidden=(128,))` is
the configured Q-function; intervention swaps it whole. The
walker surfaces `q_network.hidden` as a topology leaf.

dqn doesn't know what's INSIDE the params PyTree — `Params` is
opaque. Tabular Q (params = lookup table), linear Q (params =
weight matrix), MLP Q (params = list of layer dicts), conv Q,
etc. all conform to the same `QFunction` Protocol. The
parameterisation is genuinely Q's business.

**Theorem reference (on `__call__`, not on `init`).** Universal
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

import jax
import jax.numpy as jnp

from corroborate.claim import ClaimBase, record_call


type Params = dict[str, jax.Array]
"""Opaque PyTree of Q-function parameters. Each Q-function
implementation defines its own internal layout (e.g. MLP uses
`w0`, `b0`, `w1`, `b1`, ...); dqn only threads the pytree
through, never inspects it."""


@dataclass(frozen=True, slots=True)
class MLP(ClaimBase):
    """Two-hidden-layer ReLU MLP Q-function — Module Claim.

    Inherits `ClaimBase` for typed `name` + `invariants` access.
    `__call__` IS the framework Claim (universal approximation,
    Hornik 1989) — invokes `record_call` to participate in the
    trace. `init` is mechanics (parameter allocation); no
    `record_call` because there's no theorem to attach to it.

    Construction-time leaf: `hidden`. Architecture-defining; lives
    on the Module itself. Authors override via
    `replace(MLP(), hidden=(128,))` or pass a different Module
    (`SpectralNormMLP`, `Tabular`, etc.) wholesale via
    intervention."""
    hidden: tuple[int, ...] = (64, 64)

    def init(
        self,
        rng_key: jax.Array,
        obs_shape: tuple[int, ...],
        n_actions: int,
    ) -> Params:
        """Initialise parameter pytree. Layer-by-layer uniform
        init in `[-sqrt(1/fan_in), sqrt(1/fan_in)]`; biases zero.

        `obs_shape` may be multi-dim (e.g. `(10, 10, 4)` for
        MinAtar); MLP flattens internally at call time. The first
        weight matrix's input dim is `prod(obs_shape)`."""
        in_dim = 1
        for d in obs_shape:
            in_dim *= d
        params: Params = {}
        keys = jax.random.split(rng_key, len(self.hidden) + 1)
        for i, h in enumerate(self.hidden):
            bound = jnp.sqrt(1.0 / in_dim)
            params[f'w{i}'] = jax.random.uniform(
                keys[i], (in_dim, h), minval=-bound, maxval=bound,
            )
            params[f'b{i}'] = jnp.zeros((h,))
            in_dim = h
        bound = jnp.sqrt(1.0 / in_dim)
        params[f'w{len(self.hidden)}'] = jax.random.uniform(
            keys[-1], (in_dim, n_actions), minval=-bound, maxval=bound,
        )
        params[f'b{len(self.hidden)}'] = jnp.zeros((n_actions,))
        return params

    def __call__(self, params: Params, obs: jax.Array) -> jax.Array:
        """Forward: ReLU MLP over `obs`. Returns Q-values:
        `(n_actions,)` for a single obs or `(batch, n_actions)`
        for a batch.

        Accepts multi-dim `obs` (e.g. (..., H, W, C)) by flattening
        trailing axes greedily until their product matches the
        first weight matrix's input dim. Substrate stores obs at
        native shape; MLP collapses for the dot product here."""
        n_layers = len(params) // 2
        in_dim = int(params['w0'].shape[0])
        # Greedy: smallest k such that prod(obs.shape[-k:]) == in_dim.
        cum = 1
        k = 0
        for axis in reversed(obs.shape):
            k += 1
            cum *= int(axis)
            if cum == in_dim:
                break
        if cum != in_dim:
            raise ValueError(
                f'MLP.__call__: cannot match obs.shape={obs.shape} '
                f'trailing dims to in_dim={in_dim}',
            )
        x = obs.reshape((*obs.shape[:obs.ndim - k], in_dim))
        for i in range(n_layers - 1):
            x = jnp.dot(x, params[f'w{i}']) + params[f'b{i}']
            x = jnp.maximum(x, 0.0)  # ReLU
        result: jax.Array = (
            jnp.dot(x, params[f'w{n_layers - 1}'])
            + params[f'b{n_layers - 1}']
        )
        record_call(self, (params, obs), {}, result)
        return result


# Default Q-function — re-exported under the historical name so
# call sites that imported `mlp_q` continue to read paper-honestly
# (the symbol is now the configured Module, not a forward fn).
mlp_q = MLP()


@dataclass(frozen=True, slots=True)
class CNN(ClaimBase):
    """Convolutional Q-function — Module Claim for image obs.

    Substrate flattens obs to 1D for replay storage; CNN reshapes
    to its declared `obs_shape` (H, W, C) inside `__call__` to
    recover spatial structure. Conv stack uses SAME padding (H, W
    preserved) + ReLU; flatten + dense head emits Q-values.

    Construction-time leaves: `obs_shape` (must match the env's
    obs.shape), `channels` (per-conv-layer output channels),
    `kernel_size`, `hidden` (dense head widths). The walker
    surfaces all four as topology leaves under
    `q_network.{obs_shape, channels, kernel_size, hidden}`."""
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
        """Allocate conv kernels + dense head parameter pytree.

        He init for conv kernels (`bound = sqrt(2 / fan_in)`),
        Glorot uniform for dense weights. Biases zero.

        `obs_shape` (the substrate-passed env obs shape) must
        match the Module's `obs_shape` field — loud failure on
        mismatch since the conv kernels' first-layer in-channels
        are sized at construction time."""
        if obs_shape != self.obs_shape:
            raise ValueError(
                f'CNN.init: substrate obs_shape={obs_shape} '
                f'inconsistent with Module obs_shape={self.obs_shape}',
            )
        if len(self.obs_shape) != 3:
            raise ValueError(
                f'CNN.init: obs_shape must be (H, W, C); '
                f'got {self.obs_shape}',
            )
        h_in, w_in, c_in = self.obs_shape
        n_keys = len(self.channels) + len(self.hidden) + 1
        keys = jax.random.split(rng_key, n_keys)
        params: Params = {}

        c_prev = c_in
        for i, c_out in enumerate(self.channels):
            fan_in = self.kernel_size * self.kernel_size * c_prev
            bound = jnp.sqrt(2.0 / fan_in)
            params[f'kw{i}'] = jax.random.uniform(
                keys[i],
                (self.kernel_size, self.kernel_size, c_prev, c_out),
                minval=-bound, maxval=bound,
            )
            params[f'kb{i}'] = jnp.zeros((c_out,))
            c_prev = c_out

        flat_dim = h_in * w_in * c_prev
        in_dim = flat_dim
        for i, h_dense in enumerate(self.hidden):
            bound = jnp.sqrt(1.0 / in_dim)
            params[f'dw{i}'] = jax.random.uniform(
                keys[len(self.channels) + i],
                (in_dim, h_dense),
                minval=-bound, maxval=bound,
            )
            params[f'db{i}'] = jnp.zeros((h_dense,))
            in_dim = h_dense

        bound = jnp.sqrt(1.0 / in_dim)
        params[f'dw{len(self.hidden)}'] = jax.random.uniform(
            keys[-1], (in_dim, n_actions),
            minval=-bound, maxval=bound,
        )
        params[f'db{len(self.hidden)}'] = jnp.zeros((n_actions,))
        return params

    def __call__(self, params: Params, obs: jax.Array) -> jax.Array:
        """Forward: obs at native (..., H, W, C) → conv stack →
        flatten → dense head → Q-values (..., n_actions).

        Substrate stores obs at native shape; CNN consumes
        directly. Leading dims (single, batch, vmap'd seeds) are
        folded into a single batch axis at conv time and unfolded
        after."""
        n_obs_dims = len(self.obs_shape)
        if obs.shape[-n_obs_dims:] != self.obs_shape:
            raise ValueError(
                f'CNN.__call__: obs trailing shape '
                f'{obs.shape[-n_obs_dims:]} != obs_shape '
                f'{self.obs_shape}',
            )
        leading = obs.shape[:-n_obs_dims]
        if not leading:
            x = obs[None]
        else:
            collapsed = 1
            for d in leading:
                collapsed *= d
            x = obs.reshape((collapsed, *self.obs_shape))

        n_conv = len(self.channels)
        for i in range(n_conv):
            x = jax.lax.conv_general_dilated(
                x, params[f'kw{i}'],
                window_strides=(1, 1), padding='SAME',
                dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
            )
            x = x + params[f'kb{i}']
            x = jnp.maximum(x, 0.0)

        x = x.reshape((x.shape[0], -1))

        n_dense = len(self.hidden)
        for i in range(n_dense):
            x = jnp.dot(x, params[f'dw{i}']) + params[f'db{i}']
            x = jnp.maximum(x, 0.0)
        x = (
            jnp.dot(x, params[f'dw{n_dense}'])
            + params[f'db{n_dense}']
        )

        if not leading:
            result = x[0]
        else:
            result = x.reshape((*leading, x.shape[-1]))
        record_call(self, (params, obs), {}, result)
        return result
