"""Smoke: auto-induced computation graphs for vanilla DQN vs DDQN.

Calls the OUTERMOST `dqn` claim (the full training+eval run, with
nested scan over training+eval bursts) on a degenerate tiny
CartPole config. With `record_call` recording under
jit/scan/vmap tracers (v10 parity), every `@claim` that fires
inside the scan's tracing pass shows up in the trace — so the
graph captures the entire mechanism, not just the top-level
phase claims.

Output: the auto-induced computation graph for vanilla DQN, the
graph for DDQN (intervention: `bootstrap=partial(bootstrap,
greedification=double_greedify)`), and the structural diff. The
DDQN intervention surfaces at both the node level
(`max_greedify` ↔ `double_greedify`) and the edge level (every
edge consuming the swapped slot).

Run: `uv run python experiments/smoke_dqn_graphs.py`."""
from __future__ import annotations

# CPU-only: smoke is structural tracing, not training compute.
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from functools import partial

import gymnax
import jax
import jax.numpy as jnp

from corroborate.claim import trace_context
from corroborate.computation_graph import (
    ComputationGraph,
    build_computation_graph,
    signature,
)
from corroborate.rl.dqn.claims.bootstrap import (
    bootstrap,
    double_greedify,
)
from corroborate.rl.dqn.dqn import dqn


def _trace_full_dqn(
    intervention: dict[str, object] | None = None,
) -> ComputationGraph:
    """Run the FULL `dqn` (the outermost claim — nested scan over
    training+eval bursts) under `trace_context()` and build the
    resulting computation graph.

    Tiny config so the smoke runs fast on CPU (one super-step,
    one eval burst, one episode). Every `@claim` that fires inside
    the scan's tracing pass shows up in the trace because
    `record_call` records under jit/scan/vmap tracers (v10
    parity).

    `intervention` is merged into dqn kwargs — same shape as
    `cell_runner`'s composition via `partial(dqn, **intervention)`.
    """
    env, env_params = gymnax.make('CartPole-v1')
    eff_kwargs: dict[str, object] = {
        'rng_key': jax.random.PRNGKey(0),
        'env': env, 'env_params': env_params,
        'obs_dim': 4, 'n_actions': 2,
        'eval_episode_cap': 8,
        # Tiny budget — just enough to populate the scan.
        'total_steps': 4, 'eval_every': 4, 'n_episodes': 1,
        # Default optimizer / replay / etc. are the v0 vanilla DQN.
        **(intervention or {}),
    }
    with trace_context() as records:
        _ = dqn(**eff_kwargs)  # pyright: ignore[reportArgumentType]
    return build_computation_graph(records)


def _print_graph(name: str, g: ComputationGraph) -> None:
    print(f'\n  --- {name} ---')
    print(g.to_tree())
    nodes = sorted(g.nodes)
    print(f'  ({len(nodes)} nodes, {len(g.edges)} edges)')
    print(f'  nodes: {nodes}')


def _print_diff(label_a: str, label_b: str, g_a: ComputationGraph,
                g_b: ComputationGraph) -> None:
    diff = g_a.diff(g_b)
    print(f'\n  --- diff ({label_a} vs {label_b}) ---')
    if diff.is_empty():
        print('  (graphs are structurally identical)')
        return
    if diff.nodes_only_in_self:
        print(f'  nodes only in {label_a}: '
              f'{sorted(diff.nodes_only_in_self)}')
    if diff.nodes_only_in_other:
        print(f'  nodes only in {label_b}: '
              f'{sorted(diff.nodes_only_in_other)}')
    if diff.edges_only_in_self:
        print(f'  edges only in {label_a}:')
        for e in diff.edges_only_in_self:
            print(f'    {e.source} → {e.target} [{e.metadata}]')
    if diff.edges_only_in_other:
        print(f'  edges only in {label_b}:')
        for e in diff.edges_only_in_other:
            print(f'    {e.source} → {e.target} [{e.metadata}]')


def main() -> None:
    print('=' * 72)
    print('Full `dqn` claim — outermost training+eval run, traced.')
    print('=' * 72)

    g_vanilla_step = _trace_full_dqn()
    ddqn_intervention: dict[str, object] = {
        'bootstrap': partial(bootstrap, greedification=double_greedify),
    }
    g_ddqn_step = _trace_full_dqn(ddqn_intervention)

    _print_graph('VANILLA dqn', g_vanilla_step)
    _print_graph('DDQN dqn   ', g_ddqn_step)
    _print_diff('vanilla', 'ddqn', g_vanilla_step, g_ddqn_step)

    sig_v = signature(g_vanilla_step)
    sig_d = signature(g_ddqn_step)
    print(f'\n  signatures equal? = {sig_v == sig_d}')
    assert sig_v != sig_d, 'expected DDQN slot swap to differ'
    print('  ✓ slot-swap intervention is structurally distinct.')


if __name__ == '__main__':
    main()
