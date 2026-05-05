"""Visualize the trace + measurable output of one DQN cell.

Runs `dqn` once on a small env with short training (so it finishes
in seconds on CPU), prints the full trace schema, plots every 1-D
trajectory in a grid, and prints a few canonical measurable scalars
computed from the trace.

Usage:
    JAX_PLATFORMS=cpu uv run python scripts/visualize_dqn_trace.py
    JAX_PLATFORMS=cpu uv run python scripts/visualize_dqn_trace.py \\
        --env FourRooms-misc --total-steps 50000 --seed 0
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import corroborate_rl  # noqa: F401  # pyright: ignore[reportUnusedImport]
from corroborate.measurables import (
    evaluate_with_measurables,
    get_registered,
)
from corroborate_rl.cell_runner import run_dqn_cell
from corroborate_rl.dqn.dqn import dqn
from corroborate_rl.env_catalogue import get as get_env_spec


_DEFAULT_MEASURABLES: tuple[str, ...] = (
    'jensen_gap',
    'bootstrap_fraction',
    'effective_horizon',
    'eval_best_burst_mean',
    'eval_final_mean',
    'q_mean',
    'q_max',
)


def _print_schema(trace: Mapping[str, object]) -> list[tuple[str, np.ndarray]]:
    """Print every trace key + shape; return the plottable 1-D
    series so the caller can grid-plot them."""
    series: list[tuple[str, np.ndarray]] = []
    print(f'Trace ({len(trace)} keys):')
    for k in sorted(trace):
        v = trace[k]
        if isinstance(v, np.ndarray):
            print(f'  {k}: shape={v.shape} dtype={v.dtype}')
            if v.ndim >= 1:
                series.append((k, v))
        else:
            print(f'  {k}: {v!r}')
    return series


def _print_measurables(
    trace: Mapping[str, object], names: tuple[str, ...],
) -> None:
    """Compute each requested measurable against the trace and
    print its scalar value. Skips names not in the registry; logs
    errors per-name without aborting."""
    print(f'\nMeasurable scalars:')
    cache: dict[str, object] = {}
    for name in names:
        m = get_registered(name)
        if m is None:
            print(f'  {name}: <not registered>')
            continue
        try:
            value = evaluate_with_measurables(m.fn, trace, cache=cache)
        except Exception as e:  # noqa: BLE001
            print(f'  {name}: <error: {type(e).__name__}: {e}>')
            continue
        if isinstance(value, np.ndarray):
            print(f'  {name}: shape={value.shape} mean={float(value.mean()):.4g}')
        else:
            print(f'  {name}: {value!r}')


def _grid_plot(
    series: list[tuple[str, np.ndarray]], out: Path,
) -> None:
    """Plot every 1-D trajectory in a grid; mean over trailing
    axes for higher-dim arrays. Saves to `out` at 100 DPI."""
    if not series:
        print('No 1-D series to plot.')
        return
    n = len(series)
    cols = min(3, n)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 3 * rows))
    axes_flat = (
        axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    )
    for i, (k, v) in enumerate(series):
        ax = axes_flat[i]
        if v.ndim == 1:
            ax.plot(v)
            ax.set_xlabel('step' if v.shape[0] >= 100 else 'index')
        else:
            collapsed = v.reshape(v.shape[0], -1).mean(axis=-1)
            ax.plot(collapsed)
            ax.set_xlabel(f'axis-0 (mean over {v.shape[1:]})')
        ax.set_title(k)
        ax.grid(True, alpha=0.3)
    for j in range(n, len(axes_flat)):
        axes_flat[j].axis('off')
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=100)
    print(f'\nSaved figure: {out}')


def main() -> None:
    p = argparse.ArgumentParser(prog='visualize_dqn_trace')
    p.add_argument('--env', default='CartPole-v1')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--total-steps', type=int, default=20_000)
    p.add_argument('--eval-every', type=int, default=5_000)
    p.add_argument('--n-episodes', type=int, default=10)
    p.add_argument(
        '--out',
        type=Path,
        default=Path('experiments/figures/dqn_trace.png'),
    )
    p.add_argument(
        '--measurables',
        nargs='*',
        default=list(_DEFAULT_MEASURABLES),
        help='measurable names to compute; defaults to a curated set',
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
        f'Running 1 cell of dqn on {args.env} '
        f'(total_steps={args.total_steps}, seed={args.seed})...'
    )
    cell = run_dqn_cell(
        env_spec,
        seed=args.seed,
        claim=claim,
        arm_key='trace_viz',
        measurables=(),
    )
    print(f'  done. cell.id={cell.run.id[:8]}…')

    trace = cell.trace.leaves
    series = _print_schema(trace)
    _print_measurables(trace, tuple(args.measurables))
    _grid_plot(series, args.out)


if __name__ == '__main__':
    main()
