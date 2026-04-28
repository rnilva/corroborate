"""Q-function — frozen-dataclass functor.

A Q-function is two paired operations:

- `init(rng, obs_dim, n_actions) -> params` — allocate parameters.
- `__call__(params, obs) -> q_values` — forward pass.

Bundling them in one frozen-dataclass functor makes architecture
HPs travel with the function. `MLP(hidden=(128,))` is the
configured Q-function; intervention swaps it whole.

dqn doesn't know what's INSIDE the params PyTree — `Params` is
opaque. Tabular Q (params = lookup table), linear Q (params =
weight matrix), MLP Q (params = list of layer dicts), conv Q,
etc. all conform to the same `QFunction` Protocol. The
parameterisation is genuinely Q's business.

**Theorem reference.** Universal approximation (Hornik 1989,
Cybenko 1989): a sufficiently wide MLP can approximate any
continuous function on a compact set. The function class
*contains* Q*; gradient descent under bootstrap is *not*
guaranteed to find it (deadly triad — off-policy + bootstrap +
FA can diverge per Tsitsiklis & Van Roy 1997). Banach-
contraction-rate gap (Bertsekas-Tsitsiklis §6.3) is the
principled measurement; deferred — needs Q-snapshot probe
(see FUTURE_WORKS.md)."""
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
    """Two-hidden-layer ReLU MLP Q-function.

    Inherits `ClaimBase` for typed `name` property + `invariants`
    ClassVar. Authors apply `@dataclass(frozen=True, slots=True)`
    themselves; framework doesn't mutate the class.

    Construction-time HP: `hidden`. Architecture-defining; lives
    on the functor itself. Authors override via
    `replace(MLP(), hidden=(128,))` or by passing a different
    architecture functor (`SpectralNormMLP`, `Tabular`, etc.).

    `__call__` invokes `record_call` to participate in the trace
    explicitly — no decorator-injected wrapping."""
    hidden: tuple[int, ...] = (64, 64)

    def init(
        self,
        rng_key: jax.Array,
        obs_dim: int,
        n_actions: int,
    ) -> Params:
        """Initialise parameter pytree. Layer-by-layer uniform
        init in `[-sqrt(1/fan_in), sqrt(1/fan_in)]`; biases zero."""
        params: Params = {}
        in_dim = obs_dim
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
        for a batch."""
        n_layers = len(params) // 2
        x = obs
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
