"""RL-flavored Loop backends — `scan_loop` (jit) and `python_loop`
(eager, jax-typed). Both satisfy framework's
`Loop[C, T, jax.Array]` Protocol.

A `Loop[C, T, jax.Array]` runs a step
`(C, step_idx: jax.Array) -> (new_C, per_step_T)` for `length`
iterations and returns `(final_C, stacked_T)` — `stacked_T` is a
pytree of `(length, ...)` arrays, the raw trace of the run.

Two implementations:

- `scan_loop` — `jax.lax.scan` backend. JIT-compilable, fast.
  Python side-effects (and so contextvars / `trace_context`) do
  NOT fire inside `jax.lax.scan` because jit elides them; one fire
  during the abstract-trace pass is what the structural graph
  extractor needs. Use for production training where the per-step
  record is the only observable.
- `python_loop` — pure Python `for` backend, jax-typed step idx
  (so the SAME step function works under both backends). Eager,
  slow. Contextvars FIRE at every iteration, so `trace_context`
  captures every `@claim` invocation across every step. Use for
  probes, structural analysis, and the functional-claim graph
  derivation (PAPER_NOTES.md §1.1 graph (a)).

The step function is identical under both backends — authors
write the theory once; callers pick the loop at call-time based
on what they need (speed vs observability).

The `Loop` Protocol itself lives in `corroborate.loop` (framework-
shared). This module previously declared a redundant local copy
that drifted on the `Idx` axis; the framework Protocol is now
parameterized over `Idx` and these impls bind it to `jax.Array`."""
from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp


def scan_loop[C, T](
    step: Callable[[C, jax.Array], tuple[C, T]],
    init: C,
    length: int,
) -> tuple[C, T]:
    """`jax.lax.scan` backend — production training.

    Structurally satisfies `corroborate.loop.Loop[C, T, jax.Array]`.
    Returns `tuple[C, T]` with each leaf of T stacked to leading
    `(length, ...)` axis."""
    return jax.lax.scan(step, init, jnp.arange(length), length=length)


def python_loop[C, T](
    step: Callable[[C, jax.Array], tuple[C, T]],
    init: C,
    length: int,
) -> tuple[C, T]:
    """Pure Python `for` backend — probe / trace-observable runs.

    Output pytree is stacked to match scan's shape so downstream
    consumers (bridges, measurables) are backend-agnostic.

    Structurally satisfies `corroborate.loop.Loop[C, T, jax.Array]`."""
    carry = init
    outs: list[T] = []
    for i in range(length):
        carry, out = step(carry, jnp.asarray(i))
        outs.append(out)

    def stack(*xs: jax.Array) -> jax.Array:
        return jnp.stack(xs)

    # `jax.tree.map`'s return is `Any` because pytree heterogeneity
    # is genuine polymorphism the type system can't capture without
    # dependent types — that's the "polymorphism truly requires"
    # carve-out. The explicit `: T` annotation pins the framework-
    # contracted return shape for the caller; pyright trusts it.
    # (The rl/ executionEnvironment relaxes reportAny, so no inline
    # ignore is required.)
    stacked: T = jax.tree.map(stack, *outs)
    return carry, stacked
