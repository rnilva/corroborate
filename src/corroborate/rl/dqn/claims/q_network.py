"""Q-network — parameter init + forward pass.

The Q-network is a function `(params, obs) -> Q-values`. v0
ships a two-hidden-layer ReLU MLP; alternative architectures
(dueling, layer-norm, residual, etc.) are future @claim files
in this module.

**Theorem reference.** Universal approximation (Hornik 1989,
Cybenko 1989): a sufficiently wide MLP can approximate any
continuous function on a compact set. The function class
*contains* Q*; gradient descent under bootstrap is *not*
guaranteed to find it (deadly triad — off-policy + bootstrap +
FA can diverge per Tsitsiklis & Van Roy 1997). The
`q_bounded` invariant (`invariants.py`) is the empirical
divergence detector."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from corroborate.claim import claim


@claim
def init_mlp(
    rng_key: jax.Array,
    obs_dim: int,
    n_actions: int,
    hidden: tuple[int, ...] = (64, 64),
) -> dict[str, jax.Array]:
    """MLP parameter init. Returns a dict of weights+biases keyed
    by layer index (`w0`, `b0`, `w1`, `b1`, ...)."""
    params: dict[str, jax.Array] = {}
    in_dim = obs_dim
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
def mlp_q(params: dict[str, jax.Array], obs: jax.Array) -> jax.Array:
    """ReLU MLP forward pass. Returns Q-values: `(n_actions,)` for
    a single obs or `(batch, n_actions)` for a batch."""
    n_layers = len(params) // 2
    x = obs
    for i in range(n_layers - 1):
        x = jnp.dot(x, params[f'w{i}']) + params[f'b{i}']
        x = jnp.maximum(x, 0.0)  # ReLU
    return jnp.dot(x, params[f'w{n_layers - 1}']) + params[f'b{n_layers - 1}']
