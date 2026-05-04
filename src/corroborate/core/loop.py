"""Substrate-agnostic loop primitive — `Loop` Protocol + Python
for-loop impl + `iterate` claim wrapper.

A `Loop[C, T, Idx]` is the iteration-backend contract: run
`step(state, idx)` for `length` iterations, return
`(final_state, aggregated_outputs)`. `Idx` is parameterized so
substrates pick the form that fits their backend:

- `int` for substrate-agnostic Python (`python_loop` here).
- `jax.Array` for jax-flavored backends (`corroborate.rl.loop
  .scan_loop` and the rl-flavored `python_loop`).

The aggregation contract (list, stacked pytree, generator, ...)
is the impl's choice — the Protocol's return is annotated loosely
(`tuple[C, object]`) because aggregation polymorphism is intrinsic.
Callers pick the impl whose aggregation matches their consumer.

**Why the parametric `Idx`**: jax's `lax.scan` produces array
indices (its scan body is traced; `int(...)` on a tracer is a
type error inside jit). Substrate-agnostic Python loops produce
plain `int`. The two can't share a step-fn signature without the
parameter; without it, either the framework primitive depends on
jax (wrong) or the rl substrate has its own duplicated `Loop`
Protocol that drifts (the original sin this redesign closes).

**Trace-context behaviour.** Under an active `trace_context()`:
- `python_loop` (this module) fires `@claim` records every
  iteration — exhaustive trace.
- `scan_loop` fires records once during JAX's abstract-trace
  pass, sufficient for static graph capture
  (`build_computation_graph`); compiled invocations don't re-fire
  contextvars. Correct for the typical graph-extraction use case
  (the graph is structurally constant across iterations).

No framework-level dispatcher is needed: `record_call` already
handles jit/scan/vmap correctly (see `claim.py:82-110`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from corroborate.core.claim import claim


class Loop[C, T, Idx](Protocol):
    """Iteration-backend contract.

    `step` takes a state and an idx (of type `Idx`), returns
    `(new_state, per_step_T)`. The `Loop` runs this for `length`
    iterations and returns `(final_state, aggregated_T)`. The
    aggregation strategy (list, stacked pytree, generator, ...) is
    the impl's choice — different impls have different
    `aggregated_T` shapes; the Protocol's return is `tuple[C,
    object]` to admit them all.

    Implementations:
    - `python_loop` (this module): `Loop[C, T, int]`, returns
      `tuple[C, list[T]]`.
    - `corroborate.rl.loop.scan_loop`: `Loop[C, T, jax.Array]`,
      returns `tuple[C, T]` with each T-leaf stacked to leading
      `(length, ...)` axis.
    - `corroborate.rl.loop.python_loop`: `Loop[C, T, jax.Array]`,
      returns `tuple[C, T]` (jax-stacked to match scan's shape).

    All structurally satisfy this Protocol with their respective
    `Idx` binding."""
    def __call__(
        self,
        step: Callable[[C, Idx], tuple[C, T]],
        init: C,
        length: int,
    ) -> tuple[C, object]: ...


def python_loop[C, T](
    step: Callable[[C, int], tuple[C, T]],
    init: C,
    length: int,
) -> tuple[C, list[T]]:
    """Pure Python `for`-loop. Substrate-agnostic — no jax dep.
    Per-step outputs collected as `list[T]`; the substrate
    stacks if it wants array-typed aggregation.

    Under `trace_context()`, every iteration fires `@claim`
    records — exhaustive trace coverage. Use for substrates
    without a fast backend, or for probe runs where every
    iteration's call sequence matters.

    Structurally satisfies `Loop[C, T, int]`."""
    state = init
    outs: list[T] = []
    for i in range(length):
        state, out = step(state, i)
        outs.append(out)
    return state, outs


@claim
def iterate[C, T, Idx](
    *,
    step: Callable[[C, Idx], tuple[C, T]],
    init: C,
    length: int,
    backend: Loop[C, T, Idx],
) -> tuple[C, object]:
    """Iteration as a typed claim. Thin wrapper over the
    `Loop[C, T, Idx]` Protocol that records the loop boundary as
    a single `@claim` call.

    Why this exists: per-iteration `@claim` records get
    deduplicated by `build_computation_graph` (the same
    `(reader, source)` tuple appears N times but is collapsed
    to one edge), so the recurrence — what the loop adds beyond
    its body — is invisible to the structural graph extractor.
    Wrapping the iteration in this claim surfaces (`step`, `init`,
    `length`, `backend`) as typed inputs of a single recorded
    call. The result `final_state` then has tracked identity for
    downstream consumers (outcome aggregators, eval drivers).

    The trace edges become:
      step's leaves (γ, env_attrs, hp_*) → step → iterate →
        final_state → outcome_aggregator
    where the integration over `length` iterations is captured
    structurally by the iterate node, not erased by dedup. Any
    substrate composing with this primitive gets the loop axis
    in its claim graph for free — no per-substrate ad-hoc node
    needed.

    Returns whatever `backend` returns (`tuple[final_state,
    aggregated]`). Authors who only need the structural
    boundary record can ignore the aggregated half. The aggregated
    half is typed `object` because the Protocol admits multiple
    aggregation contracts (list, stacked tree, ...); narrow at the
    use site after picking a specific backend."""
    return backend(step, init, length)
