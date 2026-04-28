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
from corroborate.measurable_graph import (
    correlation_matrix_table,
    explained_by_claim_graph,
    pairwise_correlations,
)
from corroborate.rl.dqn.claims.bootstrap import (
    bootstrap,
    double_greedify,
)
from corroborate.rl.dqn.dqn import dqn


def _trace_full_dqn(
    intervention: dict[str, object] | None = None,
) -> tuple[ComputationGraph, dict[str, object]]:
    """Run the FULL `dqn` (the outermost claim — nested scan over
    training+eval bursts) under `trace_context()` and build the
    resulting computation graph.

    Returns `(claim_graph, record_dict)`. The record dict is what
    `dqn` returned — per-step measurable trajectories which we
    feed into `pairwise_correlations` to derive the statistical
    measurable-graph dual to the claim graph.

    Tiny config so the smoke runs fast on CPU (a few super-steps,
    one eval burst, one episode). Every `@claim` that fires inside
    the scan's tracing pass shows up in the trace because
    `record_call` records under jit/scan/vmap tracers (v10
    parity).

    `intervention` is merged into dqn kwargs — same shape as
    `cell_runner`'s composition via `partial(dqn, **intervention)`.
    """
    env, env_params = gymnax.make('CartPole-v1')
    # 2000 steps: enough for measurable-graph correlations to be
    # meaningful (Pearson r tightens with sample count). Still
    # ~10s on CPU. v10's smoke uses 5000; matched here at the
    # tier where DQN's structure is already legible.
    eff_kwargs: dict[str, object] = {
        'rng_key': jax.random.PRNGKey(0),
        'env': env, 'env_params': env_params,
        'obs_dim': 4, 'n_actions': 2,
        'eval_episode_cap': 8,
        'total_steps': 2000, 'eval_every': 500, 'n_episodes': 1,
        **(intervention or {}),
    }
    with trace_context() as records:
        record = dqn(**eff_kwargs)  # pyright: ignore[reportArgumentType]
    return build_computation_graph(records), dict(record)


def _derive_measurables(rec: dict[str, object]) -> dict[str, object]:
    """Project the raw record dict to a richer measurable series.

    Mirrors v10's `test_measurable_graph.scalar_series`: scalar
    fields pass through; multi-D claim outputs (per-step Q-vectors,
    per-sample TD-errors) get inline numpy reductions to produce
    derived scalar-per-step series. These derived measurables are
    where the interesting correlations live (q_std, q_gap surface
    DDQN-relevant overestimation signals).

    Long-term these reductions belong in `apply_trace_reductions`
    (declarative polars exprs over the trace store); for the smoke
    they're inline numpy."""
    import numpy as np

    out: dict[str, object] = {}
    # Pass-through scalars.
    for k in ('loss', 'td_error', 'reward', 'done',
              'ep_return', 'action', 'epsilon', 'buf_size',
              'eps_value', 'max_q'):
        if k in rec:
            arr = np.asarray(rec[k])
            if arr.ndim == 1:
                out[k] = arr.astype(np.float64)
            elif arr.ndim == 2 and arr.shape[-1] == 1:
                out[k] = arr.squeeze(-1).astype(np.float64)
    # Q-vector reductions (online_q_per_action: (steps, n_actions)
    # — already batch-averaged in train_phase).
    if 'online_q_per_action' in rec:
        q = np.asarray(rec['online_q_per_action']).astype(np.float64)
        if q.ndim == 2:
            out['q_mean'] = q.mean(-1)
            out['q_max'] = q.max(-1)
            out['q_std'] = q.std(-1)
            sorted_q = np.sort(q, axis=-1)
            out['q_gap'] = sorted_q[..., -1] - sorted_q[..., -2]
    if 'target_q_per_action' in rec:
        tq = np.asarray(rec['target_q_per_action']).astype(np.float64)
        if tq.ndim == 2:
            out['target_q_mean'] = tq.mean(-1)
            out['target_q_max'] = tq.max(-1)
    return out


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
    print('§1: claim graph — auto-induced computation graph')
    print('=' * 72)

    g_vanilla, rec_vanilla = _trace_full_dqn()
    ddqn_intervention: dict[str, object] = {
        'bootstrap': partial(bootstrap, greedification=double_greedify),
    }
    g_ddqn, rec_ddqn = _trace_full_dqn(ddqn_intervention)

    _print_graph('VANILLA dqn', g_vanilla)
    _print_graph('DDQN dqn   ', g_ddqn)
    _print_diff('vanilla', 'ddqn', g_vanilla, g_ddqn)

    sig_v = signature(g_vanilla)
    sig_d = signature(g_ddqn)
    print(f'\n  claim-graph signatures equal? = {sig_v == sig_d}')
    assert sig_v != sig_d, 'expected DDQN slot swap to differ'
    print('  ✓ slot-swap is structurally distinct at the claim graph.')

    # ============================================================
    print()
    print('=' * 72)
    print('§2: measurable graph — Pearson r across per-step series')
    print('=' * 72)

    series_vanilla = _derive_measurables(rec_vanilla)
    mg_vanilla = pairwise_correlations(series_vanilla)
    print(f'\n  vanilla: {len(mg_vanilla.nodes)} measurables, '
          f'{len(mg_vanilla.edges)} pairs')
    print('\n  top correlations (|r| ≥ 0.3):')
    for line in correlation_matrix_table(mg_vanilla, threshold=0.3):
        print(line)

    # Note: `explained_by_claim_graph` requires the measurable's
    # name to be a node IN the claim graph. Measurables are RECORD
    # FIELDS (loss, td_error, ep_return, max_q…) authored by
    # phases; the claim-graph nodes are CLAIM CLASSES (MLP,
    # bootstrap, train_phase…). The two layers name different
    # things. For the diagnostic to bridge them, a producer map
    # is needed — `record_field → owning_claim_name`. Deferred.
    print('\n  --- claim-graph reachability of measurables ---')
    print('  (record-field names ≠ claim names, so this is mostly')
    print('   "unexplained" — bridging requires a producer map.)')
    measurables_in_claim_graph = sum(
        1 for n in mg_vanilla.nodes if n in g_vanilla.nodes
    )
    print(f'  {measurables_in_claim_graph} of {len(mg_vanilla.nodes)} '
          f'measurables also appear as claim-graph nodes.')


if __name__ == '__main__':
    main()
