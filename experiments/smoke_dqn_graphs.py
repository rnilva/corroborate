"""Smoke: auto-induced computation graphs for vanilla DQN vs DDQN.

Calls the actual `dqn_step` (Mnih 2015 Algorithm 1, one step) on
a degenerate single-step trace using a tiny CartPole rollout. No
jit, no scan, no vmap so `record_call` fires for every eager
claim invocation. Inside `value_and_grad(compute_loss)`,
JAX-tracer args trigger `record_call`'s tracing-skip logic, so
bootstrap (and its greedification slot) doesn't appear inside
the gradient pass — but a separate eager bootstrap call after
the dqn_step exposes the DDQN axis cleanly.

Output structure:

1. Vanilla `dqn_step` trace → graph A.
2. DDQN `dqn_step` trace (intervention: bootstrap=partial(
   bootstrap, greedification=double_greedify)) → graph B.
3. Diff (A, B): top-level claims that change.
4. Plus a focused bootstrap pipeline showing the DDQN slot swap
   at the greedification node (since bootstrap-internal calls
   are gated under value_and_grad in train_phase).

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
from corroborate.rl.dqn.claims.optimizer import Adam
from corroborate.rl.dqn.dqn import dqn_step, init_state


def _build_state_and_kwargs() -> tuple[object, dict[str, object]]:
    """Init a tiny CartPole DQN state. Returns (state, dqn_step_kwargs)."""
    env, env_params = gymnax.make('CartPole-v1')
    rng = jax.random.PRNGKey(0)
    optimizer = Adam(lr=1e-3)
    state = init_state(
        env=env, env_params=env_params,
        obs_dim=4, n_actions=2,
        rng_key=rng,
        optimizer=optimizer(),
    )
    # Push a few real transitions into the replay so train_phase's
    # batch sample isn't all zeros — gives the trace meaningful
    # claims to fire. Otherwise replay.sample_batch returns
    # garbage but still fires the @claim, so the graph is the same
    # shape regardless.
    kwargs: dict[str, object] = {
        'env': env, 'env_params': env_params, 'n_actions': 2,
        'optimizer': optimizer(),
    }
    return state, kwargs


def _trace_one_step(
    intervention: dict[str, object] | None = None,
) -> ComputationGraph:
    """Run one eager `dqn_step` under `trace_context()` and build
    the resulting computation graph.

    `intervention` is merged into dqn_step kwargs (intervention
    overrides defaults — same shape as `cell_runner`'s composition
    via partial)."""
    state, base_kwargs = _build_state_and_kwargs()
    eff_kwargs = {**base_kwargs, **(intervention or {})}
    with trace_context() as records:
        _new_state, _record = dqn_step(  # pyright: ignore[reportArgumentType]
            state, jnp.int32(0), **eff_kwargs,
        )
    return build_computation_graph(records)


def _trace_bootstrap_pipeline(
    use_double: bool,
) -> ComputationGraph:
    """Standalone bootstrap probe — exposes the greedification
    slot directly. Inputs are concrete arrays so `record_call`
    fires inside bootstrap (and its sub-claims max_greedify /
    double_greedify / semi_gradient).

    This is what the dqn_step trace can NOT show because train
    phase's compute_loss is wrapped in `value_and_grad`, whose
    tracer args trigger the jit-skip in `record_call`."""
    from corroborate.rl.dqn.claims.q_network import MLP
    rng = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(rng)
    qn = MLP(hidden=(8,))
    online = qn.init(k1, obs_dim=2, n_actions=3)
    target = qn.init(k2, obs_dim=2, n_actions=3)
    next_obs = jnp.array([[0.5, -0.3]])
    reward = jnp.array([1.0])
    done = jnp.array([0.0])

    boot = (
        partial(bootstrap, greedification=double_greedify)
        if use_double else bootstrap
    )
    with trace_context() as records:
        _ = boot(  # pyright: ignore[reportArgumentType]
            online_params=online, target_params=target,
            q_network=qn, next_obs=next_obs,
            reward=reward, done=done, gamma=0.99,
        )
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
    print('§1: dqn_step (one full step on CartPole, eager)')
    print('=' * 72)

    g_vanilla_step = _trace_one_step()
    ddqn_intervention: dict[str, object] = {
        'bootstrap': partial(bootstrap, greedification=double_greedify),
    }
    g_ddqn_step = _trace_one_step(ddqn_intervention)

    _print_graph('VANILLA dqn_step', g_vanilla_step)
    _print_graph('DDQN dqn_step   ', g_ddqn_step)
    _print_diff('vanilla', 'ddqn', g_vanilla_step, g_ddqn_step)

    sig_v = signature(g_vanilla_step)
    sig_d = signature(g_ddqn_step)
    print(f'\n  signatures equal at dqn_step level? = {sig_v == sig_d}')
    print('  (note: bootstrap fires inside value_and_grad in train_phase,')
    print('   whose JAX tracers trigger record_call\'s jit-skip — so the')
    print('   greedification slot swap is invisible at this trace level.)')

    print()
    print('=' * 72)
    print('§2: bootstrap pipeline (eager, no value_and_grad)')
    print('=' * 72)

    g_vanilla_boot = _trace_bootstrap_pipeline(use_double=False)
    g_ddqn_boot = _trace_bootstrap_pipeline(use_double=True)

    _print_graph('VANILLA bootstrap', g_vanilla_boot)
    _print_graph('DDQN bootstrap   ', g_ddqn_boot)
    _print_diff('vanilla', 'ddqn', g_vanilla_boot, g_ddqn_boot)

    sig_vb = signature(g_vanilla_boot)
    sig_db = signature(g_ddqn_boot)
    print(f'\n  signatures equal at bootstrap level? = {sig_vb == sig_db}')
    assert sig_vb != sig_db, 'expected DDQN slot swap to differ'
    print('  ✓ bootstrap-level trace surfaces the slot swap.')


if __name__ == '__main__':
    main()
