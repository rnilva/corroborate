"""Loop primitive — Protocol abstracting the iteration backend.

A `Loop[C, T]` runs a step `(C, step_idx) -> (new_C, per_step_T)`
for `length` iterations and returns `(final_C, stacked_T)` —
`stacked_T` is a pytree of `(length, ...)` arrays, the raw trace
of the run.

Two implementations:

- `scan_loop` — `jax.lax.scan` backend. JIT-compilable, fast.
  Python side-effects (and so contextvars / `TraceContext`) do
  NOT fire inside `jax.lax.scan` because jit elides them. Use
  for production training where the per-step record is the only
  observable.
- `python_loop` — pure Python `for` backend. Eager, slow.
  Contextvars FIRE at every iteration, so `TraceContext` captures
  every `@claim` invocation across every step. Use for probes,
  structural analysis, and the functional-claim graph derivation
  (PAPER_NOTES.md §1.1 graph (a)).

The step function is identical under both backends — authors
write the theory once; callers pick the loop at call-time based
on what they need (speed vs observability). The Protocol IS the
contract; alternative backends conform structurally.

Ported from poc_v10's `loop.py` with the same shape; the two-
backend pattern is what makes paper-honest derivation of the
functional-claim graph compatible with jit-fast training."""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import jax
import jax.numpy as jnp


class Loop[C, T](Protocol):
    """Iteration-backend contract. Implementations are
    `scan_loop` (jit) and `python_loop` (eager)."""
    def __call__(
        self,
        step: Callable[[C, jax.Array], tuple[C, T]],
        init: C,
        length: int,
    ) -> tuple[C, T]: ...


def scan_loop[C, T](
    step: Callable[[C, jax.Array], tuple[C, T]],
    init: C,
    length: int,
) -> tuple[C, T]:
    """`jax.lax.scan` backend — production training."""
    return jax.lax.scan(step, init, jnp.arange(length), length=length)


def python_loop[C, T](
    step: Callable[[C, jax.Array], tuple[C, T]],
    init: C,
    length: int,
) -> tuple[C, T]:
    """Pure Python `for` backend — probe / trace-observable runs.

    Output pytree is stacked to match scan's shape so downstream
    consumers (bridges, measurables) are backend-agnostic."""
    carry = init
    outs: list[T] = []
    for i in range(length):
        carry, out = step(carry, jnp.asarray(i))
        outs.append(out)

    def stack(*xs: jax.Array) -> jax.Array:
        return jnp.stack(xs)

    stacked = jax.tree.map(stack, *outs)
    return carry, stacked
