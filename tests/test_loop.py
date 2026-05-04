"""Tests for the framework `Loop` Protocol + `python_loop` impl
(`src/corroborate/core/loop.py`).

Verifies:

1. Both `python_loop` (substrate-agnostic) and rl-substrate
   `scan_loop` / `python_loop` structurally satisfy the
   framework `Loop[C, T]` Protocol.
2. `python_loop` produces correct (final_state, list_of_T)
   output for a hand-checked tiny step function.
3. Under an active `trace_context()`, `@claim`-decorated step
   bodies fire records on every iteration (the eager-trace
   guarantee that `python_loop` provides for substrates without
   a fast/jit backend).

The substrate-side conformance cases (scan_loop, rl python_loop,
cell-runner graph capture) live in
`src/corroborate_rl/tests/test_loop_protocol_conformance.py`."""
from __future__ import annotations

from corroborate.core.claim import claim, trace_context
from corroborate.core.loop import Loop, python_loop


def test_python_loop_basic_for_loop_semantics() -> None:
    """`python_loop(step, init, n)` runs step `n` times,
    threading state, returning (final_state, list_of_T)."""
    def step(s: int, i: int) -> tuple[int, int]:
        return s + i + 1, s

    final, outs = python_loop(step, init=0, length=4)
    # i=0: step(0, 0) -> (1, 0)
    # i=1: step(1, 1) -> (3, 1)
    # i=2: step(3, 2) -> (6, 3)
    # i=3: step(6, 3) -> (10, 6)
    assert final == 10
    assert outs == [0, 1, 3, 6]


def test_python_loop_zero_length_returns_init_and_empty_list() -> None:
    """`length=0` → state is the initial; outputs is the empty
    list. No iterations executed."""
    def step(s: int, _i: int) -> tuple[int, str]:
        return s + 1, 'should-not-fire'

    final, outs = python_loop(step, init=42, length=0)
    assert final == 42
    assert outs == []


def test_python_loop_passes_step_index_to_step_fn() -> None:
    """The second positional arg to `step` is the iteration
    index; `python_loop` passes 0..length-1 in order."""
    received_indices: list[int] = []

    def step(s: int, i: int) -> tuple[int, int]:
        received_indices.append(i)
        return s, s

    _final, _outs = python_loop(step, init=0, length=5)
    assert received_indices == [0, 1, 2, 3, 4]


def test_python_loop_satisfies_loop_protocol() -> None:
    """Static + structural check: `python_loop` is assignable to
    a `Loop[int, int, int]`-typed slot — `Idx=int` because this is
    the substrate-agnostic Python backend. The Protocol is
    satisfied structurally — no explicit subclassing."""
    holder: Loop[int, int, int] = python_loop  # pyright/static check
    # Runtime structural sanity: holder is callable with the
    # expected signature.
    final, outs = holder(lambda s, _i: (s + 1, s), 0, 3)
    assert final == 3
    assert outs == [0, 1, 2]


def test_python_loop_fires_at_claim_records_under_trace_context() -> None:
    """Every iteration's `@claim`-decorated call appends a
    record to the active trace. This is what the substrate-
    agnostic `python_loop` guarantees for graph-extraction
    workloads — fast/jit loops only fire once during abstract
    trace, but `python_loop` fires every iteration."""
    @claim
    def inc(x: int) -> int:
        return x + 1

    def step(s: int, _i: int) -> tuple[int, int]:
        return inc(s), s

    with trace_context() as records:
        final, outs = python_loop(step, init=0, length=3)

    assert final == 3
    assert outs == [0, 1, 2]
    # Three iterations, each calls `inc` once → 3 records.
    inc_records = [r for r in records if r.claim.name == 'inc']
    assert len(inc_records) == 3
