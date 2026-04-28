"""Smoke: print the auto-induced computation graphs for vanilla
DQN and DDQN, demonstrate the structural difference.

The cleanest minimal trace target is `bootstrap` — it composes
`greedification` (the DDQN axis: `max_greedify` vs
`double_greedify`) and `gradient_rule`. Calling `bootstrap` once
under `trace_context()` with concrete (non-traced) arrays exposes
the call graph without needing a full training loop.

This smoke does:

1. Build synthetic concrete inputs (small jax.Arrays — no jit, no
   scan, no vmap so `record_call` actually fires).
2. Trace `bootstrap(...)` → vanilla DQN: `max_greedify` slot.
3. Trace `partial(bootstrap, greedification=double_greedify)(...)`
   → DDQN: `double_greedify` slot.
4. Build computation graphs for both.
5. Print each graph and the diff.
6. Print signatures and assert they differ.

Demonstrates: a slot-swap intervention (vanilla → DDQN) IS a
structural change at the auto-induced graph level. A pure HP
tweak would not be.

Run: `uv run python experiments/smoke_dqn_graphs.py`."""
from __future__ import annotations

# Smoke runs on CPU — only structural tracing is exercised; no
# training compute, no jit. Set BEFORE any jax import.
import os

os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from functools import partial

import jax
import jax.numpy as jnp

from corroborate.claim import trace_context
from corroborate.computation_graph import (
    build_computation_graph,
    signature,
)
from corroborate.rl.dqn.claims.bootstrap import (
    bootstrap,
    double_greedify,
)
from corroborate.rl.dqn.claims.q_network import MLP


def _synthetic_inputs() -> dict[str, object]:
    """Concrete (non-traced) inputs for one bootstrap call.

    Tiny shapes: 1 sample, 2-D obs, 3 actions. Keep memory minimal
    — sweep workers may be holding most of the GPU."""
    rng = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(rng)
    q_network = MLP(hidden=(8,))
    online_params = q_network.init(k1, obs_dim=2, n_actions=3)
    target_params = q_network.init(k2, obs_dim=2, n_actions=3)
    next_obs = jnp.array([[0.5, -0.3]])
    reward = jnp.array([1.0])
    done = jnp.array([0.0])
    return {
        'online_params': online_params,
        'target_params': target_params,
        'q_network': q_network,
        'next_obs': next_obs,
        'reward': reward,
        'done': done,
        'gamma': 0.99,
    }


def _print_graph(name: str, g: object) -> None:
    print(f'\n  --- {name} graph ---')
    if hasattr(g, 'to_tree'):
        print(g.to_tree())  # pyright: ignore[reportAny]
    if hasattr(g, 'edges'):
        print(f'  ({len(g.edges)} edges)')  # pyright: ignore[reportAny]


def main() -> None:
    print('=' * 72)
    print('DDQN-vs-vanilla DQN: faithful intervention at the graph level')
    print('=' * 72)

    inputs = _synthetic_inputs()

    # --- Vanilla DQN trace ---
    with trace_context() as records_vanilla:
        _ = bootstrap(**inputs)  # pyright: ignore[reportArgumentType]
    g_vanilla = build_computation_graph(records_vanilla)
    sig_vanilla = signature(g_vanilla)

    # --- DDQN trace (intervention: greedification=double_greedify) ---
    ddqn_bootstrap = partial(bootstrap, greedification=double_greedify)
    with trace_context() as records_ddqn:
        _ = ddqn_bootstrap(**inputs)  # pyright: ignore[reportArgumentType]
    g_ddqn = build_computation_graph(records_ddqn)
    sig_ddqn = signature(g_ddqn)

    # --- Render both ---
    _print_graph('VANILLA DQN', g_vanilla)
    _print_graph('DDQN       ', g_ddqn)

    # --- Diff ---
    print('\n  --- diff (DDQN vs VANILLA) ---')
    diff = g_ddqn.diff(g_vanilla)
    if diff.is_empty():
        print('  (graphs are structurally identical — no intervention)')
    else:
        if diff.nodes_only_in_self:
            print(f'  nodes only in DDQN:    '
                  f'{sorted(diff.nodes_only_in_self)}')
        if diff.nodes_only_in_other:
            print(f'  nodes only in VANILLA: '
                  f'{sorted(diff.nodes_only_in_other)}')
        if diff.edges_only_in_self:
            print(f'  edges only in DDQN:')
            for e in diff.edges_only_in_self:
                print(f'    {e.source} → {e.target} [{e.metadata}]')
        if diff.edges_only_in_other:
            print(f'  edges only in VANILLA:')
            for e in diff.edges_only_in_other:
                print(f'    {e.source} → {e.target} [{e.metadata}]')

    # --- Signatures ---
    print('\n  --- structural signatures ---')
    sv_nodes, sv_edges = sig_vanilla
    sd_nodes, sd_edges = sig_ddqn
    print(f'  signature(vanilla) = {len(sv_nodes)} nodes, '
          f'{len(sv_edges)} edges, hash={hash(sig_vanilla):>20}')
    print(f'  signature(ddqn)    = {len(sd_nodes)} nodes, '
          f'{len(sd_edges)} edges, hash={hash(sig_ddqn):>20}')
    print(f'  signatures equal?  = {sig_vanilla == sig_ddqn}')

    # Hard assertions for smoke value.
    assert sig_vanilla != sig_ddqn, (
        'expected vanilla and DDQN to produce different signatures'
    )
    print('\n  ✓ slot-swap intervention produces structurally distinct '
          'graphs.')

    # --- Bonus: an HP-only tweak should NOT change the graph ---
    print()
    print('=' * 72)
    print('control: pure HP tweak (gamma=0.95) — should be empty diff')
    print('=' * 72)

    tweaked = partial(bootstrap, gamma=0.95)
    inputs_no_gamma = {k: v for k, v in inputs.items() if k != 'gamma'}
    with trace_context() as records_tweak:
        _ = tweaked(**inputs_no_gamma)  # pyright: ignore[reportArgumentType]
    g_tweak = build_computation_graph(records_tweak)
    sig_tweak = signature(g_tweak)

    diff_tweak = g_vanilla.diff(g_tweak)
    print(f'\n  diff is_empty? = {diff_tweak.is_empty()}')
    print(f'  signatures equal? = {sig_vanilla == sig_tweak}')
    assert sig_vanilla == sig_tweak, (
        'expected pure HP tweak to leave the graph signature unchanged'
    )
    print('\n  ✓ HP tweak (gamma) leaves the structural signature '
          'identical — anti-laundering.')


if __name__ == '__main__':
    main()
