"""Tests for the framework `Loop` Protocol + `python_loop` impl
(`src/corroborate/loop.py`).

Three things to verify:

1. Both `python_loop` (substrate-agnostic) and rl-substrate
   `scan_loop` / `python_loop` structurally satisfy the
   framework `Loop[C, T]` Protocol.
2. `python_loop` produces correct (final_state, list_of_T)
   output for a hand-checked tiny step function.
3. Under an active `trace_context()`, `@claim`-decorated step
   bodies fire records on every iteration (the eager-trace
   guarantee that `python_loop` provides for substrates without
   a fast/jit backend)."""
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


def test_rl_scan_loop_satisfies_loop_protocol() -> None:
    """The rl-substrate's `scan_loop` is `Loop[C, T, jax.Array]` —
    the `Idx` parameter binds to `jax.Array` because scan's body
    is traced (jit elides Python int conversion)."""
    import jax

    from corroborate_rl.loop import scan_loop

    holder: Loop[object, object, jax.Array] = scan_loop
    assert holder is scan_loop


def test_rl_python_loop_satisfies_loop_protocol() -> None:
    """The rl-substrate's `python_loop` (with jax stacking)
    likewise satisfies `Loop[C, T, jax.Array]` — same step-fn
    signature as `scan_loop` so authors write the theory once."""
    import jax

    from corroborate_rl.loop import python_loop as rl_python_loop

    holder: Loop[object, object, jax.Array] = rl_python_loop
    assert holder is rl_python_loop


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


def test_graph_capture_on_run_dqn_arm_with_real_run() -> None:
    """Integration test: run a tiny dqn arm under `trace_context`
    via the cell-runner; verify `arm.graph` is a populated
    `ComputationGraph` with the expected DQN claim hierarchy."""
    import os
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

    from corroborate.graph import Graph
    from corroborate.core.hypothesis import Hypothesis
    from corroborate_rl.cell_runner import run_dqn_arm
    from corroborate_rl.env_catalogue import get

    intervention = {
        'total_steps': 100, 'eval_every': 50, 'n_episodes': 2,
        'gamma': 0.99, 'sync_period': 25,
    }
    h = Hypothesis(
        name='vanilla', intervention=intervention, predicted_direction=None,
    )
    arm = run_dqn_arm(get('CartPole-v1'), (0,), hypothesis=h)

    assert isinstance(arm.graph, Graph)
    assert len(arm.graph.nodes) > 0
    assert len(arm.graph.edges) > 0
    # The DQN claim hierarchy must include rollout / train / sync
    # phases + the dqn outermost claim itself. Check by claim name.
    node_names = set(arm.graph.nodes)
    assert 'dqn' in node_names
    assert 'dqn_step' in node_names
    assert 'rollout_phase' in node_names
    assert 'train_phase' in node_names
    assert 'sync_phase' in node_names
