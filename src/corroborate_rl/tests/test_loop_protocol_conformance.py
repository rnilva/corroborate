"""Substrate-side `Loop` Protocol conformance + cell-runner graph
capture.

The substrate-agnostic `Loop` Protocol and the `python_loop`
behavior are tested in the framework `tests/test_loop.py`; the
cases here verify the JAX-backed implementations
(`corroborate_rl.loop.scan_loop` / `python_loop`) and the
cell-runner integration that yields a populated
`ComputationGraph`."""
from __future__ import annotations

from corroborate.core.loop import Loop


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


def test_graph_capture_on_run_dqn_arm_with_real_run() -> None:
    """Integration test: run a tiny dqn arm under `trace_context`
    via the cell-runner; verify `arm.graph` is a populated
    `ComputationGraph` with the expected DQN claim hierarchy."""
    import os
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')

    from functools import partial

    from corroborate.graph import Graph
    from corroborate_rl.cell_runner import run_dqn_arm
    from corroborate_rl.dqn.dqn import dqn
    from corroborate_rl.env_catalogue import get

    claim = partial(
        dqn,
        total_steps=100, eval_every=50, n_episodes=2,
        gamma=0.99, sync_period=25,
    )
    arm = run_dqn_arm(
        get('CartPole-v1'), (0,), claim=claim, arm_key='baseline',
        measurables=(),
    )

    assert isinstance(arm.graph, Graph)
    assert len(arm.graph.nodes) > 0
    assert len(arm.graph.edges) > 0
    node_names = set(arm.graph.nodes)
    assert 'dqn' in node_names
    assert 'dqn_step' in node_names
    assert 'rollout_phase' in node_names
    assert 'train_phase' in node_names
    assert 'sync_phase' in node_names
