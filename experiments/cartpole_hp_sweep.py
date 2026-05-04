"""HP sweep on CartPole-v1 — vanilla DQN only.

Tests the user's hypothesis: the convergence audit found CartPole
unsolved at 200k under the existing corpus's HPs (lr=1e-3,
batch=32, capacity=10k, sync=100), and CartPole *should* be
solvable with reasonable HPs without any new mechanism. If a
different HP region solves it, the unsolved-CartPole finding is
HP-conditioned, not mechanism-conditioned.

Sweep axes (Cartesian product, vanilla DQN only, no DDQN):
- `lr`: {2.5e-4, 5e-4, 1e-3}
- `batch_size`: {32, 64, 128}
- `sync_period`: {100, 500}
- `capacity`: {10_000, 50_000}

= 3 × 3 × 2 × 2 = 36 grid points × 5 seeds = 180 cells. CartPole
is fast; whole sweep is ~30 min on CPU (vmap-batched over seeds).

Output: `experiments/data/cartpole_hp/runs.parquet` with one row
per cell. Apply `convergence_audit` to the result to see which HP
configs solve CartPole.

Usage:
    uv run python experiments/cartpole_hp_sweep.py
    uv run python experiments/cartpole_hp_sweep.py --quick  # tiny grid for testing
    uv run python experiments/cartpole_hp_sweep.py --output-dir /tmp/cartpole_hp
"""
from __future__ import annotations

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')

import argparse
import gc
import itertools
import time
from functools import partial
from pathlib import Path
from typing import Any

import jax
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from corroborate.core.hypothesis import Hypothesis
from corroborate.persistence import (
    apply_trace_reductions,
    write_runrows,
    write_tracerows,
)
from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord


# ============ HP grid ============

_FULL_GRID: dict[str, list[Any]] = {
    'lr': [2.5e-4, 5e-4, 1e-3],
    'batch_size': [32, 64, 128],
    'sync_period': [100, 500],
    'capacity': [10_000, 50_000],
}

# Quick grid for verifying the script runs end-to-end.
_QUICK_GRID: dict[str, list[Any]] = {
    'lr': [5e-4, 1e-3],
    'batch_size': [64],
    'sync_period': [100],
    'capacity': [10_000],
}

ENV_NAME = 'CartPole-v1'
TOTAL_STEPS = 200_000
EVAL_EVERY = 20_000
N_EPISODES = 5
GAMMA = 0.99


def _intervention_for(
    *, capacity: int, batch_size: int, lr: float,
    sync_period: int,
) -> dict[str, object]:
    return {
        'total_steps': TOTAL_STEPS,
        'eval_every': EVAL_EVERY,
        'n_episodes': N_EPISODES,
        'gamma': GAMMA,
        'replay': Replay(capacity=capacity, batch_size=batch_size),
        'optimizer': WarmedUpdate(inner=Adam(lr=lr), warmup_steps=100),
        'sync_period': sync_period,
    }


def _make_hypothesis(
    grid_point: dict[str, Any],
) -> Hypothesis[DQNTrajectoryRecord]:
    from corroborate.rl.dqn.measurables import dqn_default_measurables
    return Hypothesis(
        name='vanilla_dqn',
        intervention=_intervention_for(**grid_point),
        predicted_direction=None,
        intervention_arms=(),
        measurables=dqn_default_measurables(),
    )


def _grid_tag(grid_point: dict[str, Any]) -> str:
    return '__'.join(
        f'{k}={v!r}' for k, v in sorted(grid_point.items())
    )


# ============ Trace reductions (mirror collect_ddqn_runs) ============

def _per_step_max_q(nested_list: pl.Series) -> list[float]:
    return [max(per_action) for per_action in nested_list.to_list()]


def _per_step_min_q(nested_list: pl.Series) -> list[float]:
    return [min(per_action) for per_action in nested_list.to_list()]


def _per_step_mean_q(nested_list: pl.Series) -> list[float]:
    return [
        sum(per_action) / len(per_action) if per_action else float('nan')
        for per_action in nested_list.to_list()
    ]


def _per_step_argmax_q(nested_list: pl.Series) -> list[int]:
    return [
        int(max(range(len(per_action)), key=lambda i: per_action[i]))
        if per_action else -1
        for per_action in nested_list.to_list()
    ]


_TRACE_POST_REDUCTIONS: tuple[pl.Expr, ...] = (
    pl.col('online_q_per_action').map_elements(
        _per_step_max_q, return_dtype=pl.List(pl.Float64),
    ).alias('online_max_q_per_step'),
    pl.col('target_q_per_action').map_elements(
        _per_step_max_q, return_dtype=pl.List(pl.Float64),
    ).alias('target_max_q_per_step'),
    pl.col('online_q_per_action').map_elements(
        _per_step_min_q, return_dtype=pl.List(pl.Float64),
    ).alias('online_min_q_per_step'),
    pl.col('online_q_per_action').map_elements(
        _per_step_mean_q, return_dtype=pl.List(pl.Float64),
    ).alias('online_mean_q_per_step'),
    pl.col('online_q_per_action').map_elements(
        _per_step_argmax_q, return_dtype=pl.List(pl.Int64),
    ).alias('online_argmax_per_step'),
    pl.col('target_q_per_action').map_elements(
        _per_step_argmax_q, return_dtype=pl.List(pl.Int64),
    ).alias('target_argmax_per_step'),
)

_TRACE_POST_DROPS: tuple[str, ...] = (
    'online_q_per_action',
    'target_q_per_action',
)


# ============ Per-config runner ============

def _run_one_config(
    grid_point: dict[str, Any],
    seeds: tuple[int, ...],
    tmp_dir: Path,
    cfg_idx: int,
) -> tuple[Path, Path]:
    from corroborate.rl.cell_runner import run_dqn_arm
    from corroborate.rl.env_catalogue import get

    cfg_tag = f'cfg{cfg_idx:03d}__{_grid_tag(grid_point)}'
    runs_path = tmp_dir / f'{cfg_tag}__runs.parquet'
    traces_path = tmp_dir / f'{cfg_tag}__traces.parquet'

    h = _make_hypothesis(grid_point)
    arm = run_dqn_arm(
        get(ENV_NAME), seeds, hypothesis=h,
    )
    cells = arm.cells
    write_runrows(tuple(c.run for c in cells), runs_path)
    reduced = apply_trace_reductions(
        [c.trace for c in cells],
        add=_TRACE_POST_REDUCTIONS,
        drop=_TRACE_POST_DROPS,
    )
    write_tracerows(reduced, traces_path)

    del arm, cells, reduced
    jax.clear_caches()
    gc.collect()
    return runs_path, traces_path


# ============ Merge ============

def _grid_points(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    return [
        dict(zip(keys, point))
        for point in itertools.product(*grid.values())
    ]


def _union_schema(paths: list[Path]) -> pa.Schema:
    fields: dict[str, pa.Field] = {}
    for p in paths:
        for f in pq.ParquetFile(p).schema_arrow:
            fields.setdefault(f.name, f)
    return pa.schema(list(fields.values()))


def _merge_parquets(paths: list[Path], dest: Path) -> None:
    """Merge per-config parquets into one. Uses union schema to
    handle minor column-presence drift across configs."""
    schema = _union_schema(paths)
    writer = pq.ParquetWriter(dest, schema)
    for p in paths:
        table = pq.read_table(p)
        # Coerce to the unified schema (null-fill missing cols).
        for f in schema:
            if f.name not in table.column_names:
                table = table.append_column(
                    f, pa.nulls(len(table), type=f.type),
                )
        table = table.select(schema.names)
        writer.write_table(table)
    writer.close()


# ============ Driver ============

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        '--output-dir', type=Path,
        default=Path('experiments/data/cartpole_hp'),
    )
    _ = parser.add_argument(
        '--quick', action='store_true',
        help='Tiny grid for sanity-testing the pipeline.',
    )
    _ = parser.add_argument(
        '--n-seeds', type=int, default=5,
    )
    args = parser.parse_args()
    output_dir: Path = Path(args.output_dir)  # pyright: ignore[reportAny]
    quick: bool = bool(args.quick)  # pyright: ignore[reportAny]
    n_seeds: int = int(args.n_seeds)  # pyright: ignore[reportAny]

    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / 'tmp'
    tmp_dir.mkdir(exist_ok=True)

    grid = _QUICK_GRID if quick else _FULL_GRID
    points = _grid_points(grid)
    seeds = tuple(range(n_seeds))

    print(f'sweep: env={ENV_NAME}, total_steps={TOTAL_STEPS}, '
          f'n_configs={len(points)}, n_seeds={n_seeds}, '
          f'mode={"quick" if quick else "full"}')
    print(f'output: {output_dir}')
    print()

    runs_paths: list[Path] = []
    traces_paths: list[Path] = []
    t_start = time.time()
    for i, gp in enumerate(points):
        t_cfg = time.time()
        print(f'  [{i+1}/{len(points)}] {_grid_tag(gp)} ...', flush=True)
        rp, tp = _run_one_config(gp, seeds, tmp_dir, i)
        runs_paths.append(rp)
        traces_paths.append(tp)
        elapsed = time.time() - t_cfg
        total = time.time() - t_start
        print(
            f'    done in {elapsed:.1f}s '
            f'(cumulative {total/60:.1f}min)',
            flush=True,
        )

    print()
    print('merging per-config parquets ...')
    runs_dest = output_dir / 'runs.parquet'
    traces_dest = output_dir / 'traces.parquet'
    _merge_parquets(runs_paths, runs_dest)
    _merge_parquets(traces_paths, traces_dest)
    print(f'  → {runs_dest} ({pq.read_table(runs_dest).num_rows} rows)')
    print(f'  → {traces_dest} ({pq.read_table(traces_dest).num_rows} rows)')


if __name__ == '__main__':
    main()
