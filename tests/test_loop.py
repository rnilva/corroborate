"""Tests for `Loop` Protocol + scan_loop / python_loop backends.

Verifies the two-backend invariant: same `step` function, same
inputs, same output shape and values across `scan_loop` and
`python_loop`. The framework's paper-honest derivation depends
on this — `python_loop` runs with `TraceContext` capturing every
@claim call; `scan_loop` runs the same theory at jit speed."""
from __future__ import annotations

import jax
import jax.numpy as jnp

from corroborate.loop import python_loop, scan_loop


def test_scan_and_python_loops_produce_same_output() -> None:
    """Identity property: same step + same init + same length →
    same final carry + same stacked outputs across backends."""
    def step(c: jax.Array, i: jax.Array) -> tuple[jax.Array, jax.Array]:
        # Carry counts up; output is i squared.
        return c + 1, i ** 2

    init = jnp.int32(0)
    length = 10

    final_scan, out_scan = scan_loop(step, init, length)
    final_python, out_python = python_loop(step, init, length)

    assert int(final_scan) == int(final_python)
    assert jnp.array_equal(out_scan, out_python)
    assert int(final_scan) == length


def test_python_loop_stacks_to_match_scan_shape() -> None:
    """python_loop's output pytree shape matches scan_loop's so
    downstream consumers (bridges, measurables) don't need to
    branch on backend."""
    def step(c: jax.Array, i: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        # Step output is a dict — pytree shape that scan_loop also stacks.
        return c + i, {'cumsum': c + i, 'i_doubled': 2 * i}

    init = jnp.int32(0)
    length = 5

    _, out_scan = scan_loop(step, init, length)
    _, out_python = python_loop(step, init, length)

    assert set(out_scan.keys()) == set(out_python.keys())
    for key in out_scan:
        assert out_scan[key].shape == out_python[key].shape
        assert jnp.array_equal(out_scan[key], out_python[key])


def test_python_loop_fires_python_side_effects() -> None:
    """`python_loop` runs each step in a Python frame, so
    side-effects (counters, prints, contextvars) fire per step.
    `scan_loop` would elide these inside jit — that's the whole
    reason for the two-backend split."""
    counter: list[int] = []

    def step(c: jax.Array, i: jax.Array) -> tuple[jax.Array, jax.Array]:
        counter.append(int(i))
        return c + 1, i

    init = jnp.int32(0)
    _, _ = python_loop(step, init, 4)
    assert counter == [0, 1, 2, 3]
