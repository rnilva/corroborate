"""Visualize the @claim ComputationGraph of one dqn run.

Runs one dqn arm under `trace_context()` so JAX's first-call abstract
trace pass fires every nested @claim once; that single pass is the
structural graph. Output:

- ASCII tree (sources-first BFS) — always.
- DOT file (for graphviz: `dot -Tpng dqn_graph.dot -o dqn_graph.png`)
  via `--dot OUT.dot`.

The graph is the per-(theory, intervention) static call graph: nodes
are @claim names, edges are data flow (an edge `dqn_step → bootstrap`
means dqn_step's output is consumed by bootstrap as one of its kwargs;
edge metadata carries the reader_arg + source_path).

Usage:
    JAX_PLATFORMS=cpu uv run python scripts/visualize_dqn_graph.py
    JAX_PLATFORMS=cpu uv run python scripts/visualize_dqn_graph.py \\
        --total-steps 5000 --dot experiments/figures/dqn_graph.dot
"""
from __future__ import annotations

import argparse
from functools import partial
from pathlib import Path

from corroborate.graph.computation import ComputationGraph
from corroborate_rl.cell_runner import run_dqn_arm
from corroborate_rl.dqn.dqn import dqn
from corroborate_rl.env_catalogue import get as get_env_spec


def _to_dot(g: ComputationGraph) -> str:
    """Emit a graphviz DOT source string. Box nodes, labelled edges
    (`reader_arg` + `source_path` from `ComputationEdge`)."""
    lines: list[str] = ['digraph dqn {']
    lines.append('  rankdir=TB;')
    lines.append(
        '  node [shape=box, fontname="Helvetica", fontsize=10];',
    )
    lines.append(
        '  edge [fontname="Helvetica", fontsize=8];',
    )
    for n in sorted(g.nodes):
        lines.append(f'  "{n}";')
    for e in g.edges:
        meta = e.metadata
        label = f'{meta.reader_arg}'
        if meta.source_path:
            label += f' ← .{meta.source_path}'
        lines.append(
            f'  "{e.source}" -> "{e.target}" [label="{label}"];',
        )
    lines.append('}')
    return '\n'.join(lines)


def main() -> None:
    p = argparse.ArgumentParser(prog='visualize_dqn_graph')
    p.add_argument('--env', default='CartPole-v1')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--total-steps', type=int, default=5_000)
    p.add_argument('--eval-every', type=int, default=5_000)
    p.add_argument('--n-episodes', type=int, default=5)
    p.add_argument(
        '--dot',
        type=Path,
        default=None,
        help='write DOT source to this path (render with graphviz: '
             '`dot -Tpng OUT.dot -o OUT.png`)',
    )
    args = p.parse_args()

    env_spec = get_env_spec(args.env)
    claim = partial(
        dqn,
        total_steps=args.total_steps,
        eval_every=args.eval_every,
        n_episodes=args.n_episodes,
    )
    print(
        f'Tracing 1 cell of dqn on {args.env} '
        f'(total_steps={args.total_steps}, seed={args.seed})...'
    )
    arm = run_dqn_arm(
        env_spec,
        seeds=(args.seed,),
        claim=claim,
        arm_key='graph_viz',
        measurables=(),
    )
    g = arm.graph
    print(f'  done. {len(g.nodes)} @claim nodes, {len(g.edges)} edges.')

    print('\n=== ASCII tree (sources-first) ===')
    print(g.to_tree())

    print('\n=== Edges (source → target [reader_arg ← source_path]) ===')
    for e in sorted(
        g.edges,
        key=lambda x: (x.source, x.target, x.metadata.reader_arg),
    ):
        meta = e.metadata
        path = f'.{meta.source_path}' if meta.source_path else '<bare>'
        print(f'  {e.source} → {e.target}  [{meta.reader_arg} ← {path}]')

    if args.dot is not None:
        args.dot.parent.mkdir(parents=True, exist_ok=True)
        args.dot.write_text(_to_dot(g))
        print(f'\nDOT source: {args.dot}')
        print(
            f'Render with:  dot -Tpng {args.dot} '
            f'-o {args.dot.with_suffix(".png")}'
        )


if __name__ == '__main__':
    main()
