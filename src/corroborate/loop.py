"""Substrate-agnostic loop primitive — `Loop` Protocol + Python
for-loop impl.

A `Loop[C, T]` is the iteration-backend contract: run
`step(state, idx)` for `length` iterations, return
`(final_state, aggregated_outputs)`. Two reference impls
coexist; both structurally satisfy the Protocol:

- `python_loop` (this module) — pure Python `for`-loop, no jax
  dep. Aggregates per-step `T` as `list[T]`. Substrate-agnostic.
- `corroborate.rl.loop.scan_loop` — `jax.lax.scan` backend.
  Aggregates with `jnp.stack` so each leaf gains a leading
  `(length, ...)` axis. Production-grade for JAX-shaped state.

The two impls have *different* aggregation contracts (list vs
stacked tree), which is why the Protocol's return is annotated
loosely (`tuple[C, object]`). Callers pick the impl whose
aggregation matches their consumer; the Protocol just says
"this thing iterates."

Substrate authors who don't have a fast/jit path use
`python_loop` directly. RL substrate authors thread `scan_loop`
through their `train_with_eval`-shaped consumers via a
`loop: Loop = scan_loop` kwarg.

**Trace-context behaviour.** Under an active `trace_context()`:
- `python_loop` fires `@claim` records every iteration —
  exhaustive trace.
- `scan_loop` fires records once during JAX's abstract-trace
  pass, which is sufficient for static graph capture
  (`build_computation_graph`); subsequent compiled invocations
  don't re-fire contextvars. This is *correct* for the typical
  graph-extraction use case (the graph is structurally constant
  across iterations).

No framework-level dispatcher is needed: `record_call` already
handles jit/scan/vmap correctly (see `claim.py:82-110`)."""
from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

from corroborate.claim import claim


class Loop[C, T](Protocol):
    """Iteration-backend contract.

    `step` takes a state and an idx, returns
    `(new_state, per_step_T)`. The `Loop` runs this for
    `length` iterations and returns
    `(final_state, aggregated_T)`. The aggregation strategy
    (list, stacked pytree, generator, ...) is the impl's choice
    — different impls have different `aggregated_T` shapes.

    Implementations:
    - `python_loop` returns `tuple[C, list[T]]`.
    - `corroborate.rl.loop.scan_loop` returns `tuple[C, T]`
      with each T-leaf stacked to leading `(length, ...)` axis.

    Both structurally satisfy this Protocol. Callers pick by
    aggregation shape; the Protocol is the seam, not the
    aggregator."""
    def __call__(
        self,
        step: Callable[..., tuple[C, T]],
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
    iteration's call sequence matters."""
    state = init
    outs: list[T] = []
    for i in range(length):
        state, out = step(state, i)
        outs.append(out)
    return state, outs


@claim
def iterate[C, T](
    *,
    step: Callable[[C, int], tuple[C, T]],
    init: C,
    length: int,
    backend: Loop[C, T],
) -> tuple[C, object]:
    """Iteration as a typed claim. Thin wrapper over the
    `Loop[C, T]` Protocol that records the loop boundary as a
    single `@claim` call.

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
    boundary record can ignore the aggregated half."""
    return backend(step, init, length)
