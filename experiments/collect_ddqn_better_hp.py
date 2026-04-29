"""DDQN-vs-vanilla sweep at the *converging* HP regime.

The original 200k corpus uses `capacity=10_000` — which the
CartPole HP sweep showed is too small for stable convergence
(CartPole peaks at 99.34 then forgets). A 5× larger replay buffer
(`capacity=50_000`) fixes the stability problem on CartPole
*without any new mechanism*.

This sweep tests whether the existing DDQN-vs-vanilla finding
(mechanism HELD, link to outcome BROKEN) is robust to the
HP regime, or whether it was conditioned on under-converged
training. Same shape as `collect_ddqn_runs.py` with two
restrictions:

- `capacity=50_000` (the load-bearing fix from the CartPole HP
  sweep, with `lr=1e-3, batch=32, sync=100` matching the
  original corpus on every other axis).
- 6 iteration envs, balanced by family for diversity.
- Single `total_steps=200_000` grid point.

Output: `experiments/data/ddqn_better_hp/runs.parquet` +
`traces.parquet`. Apply the convergence audit + §3 verdict
afterwards to see how the §3 pattern shifts under the better-HP
regime.

Usage:
    uv run python experiments/collect_ddqn_better_hp.py
"""
from __future__ import annotations

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')

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

from corroborate.hypothesis import Hypothesis
from corroborate.intervention import Intervention
from corroborate.persistence import (
    apply_trace_reductions,
    write_runrows,
    write_tracerows,
)
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord


# ============ Iteration env subset ============

# 6 envs balanced by family. Picked for diversity, not for
# pre-existing convergence — we want to see how the better-HP
# regime classifies them.
ENV_NAMES: tuple[str, ...] = (
    'CartPole-v1',          # classic, HP-sensitive
    'Catch-bsuite',         # bsuite, fast-converging
    'MNISTBandit-bsuite',   # bsuite, was unsolved at capacity=10k
    'Freeway-MinAtar',      # MinAtar, was unsolved
    'GaussianBandit-misc',  # misc, no threshold
    'MetaMaze-misc',        # misc, no threshold
)

HYPOTHESIS_NAMES = ('vanilla_dqn', 'ddqn')
# 30 seeds total, run as 3 chunks of 10 (vmap can't fit all 30 ×
# capacity=50k × 784 obs in 16GB GPU). Chunked sequentially per
# arm; cells concatenated before parquet write so the output is
# indistinguishable from an n=30 vmapped arm.
SEEDS: tuple[int, ...] = tuple(range(30))
SEED_CHUNK_SIZE: int = 10


# ============ HP grid (converging regime) ============

HP_GRID: dict[str, list[Any]] = {
    'capacity': [50_000],   # 5× larger than original corpus's 10_000
    'batch_size': [32],
    'lr': [1e-3],
    'total_steps': [200_000],
}


# ============ Trace post-reductions (mirror collect_ddqn_runs) ============

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


TRACE_POST_REDUCTIONS: tuple[pl.Expr, ...] = (
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


TRACE_POST_DROPS: tuple[str, ...] = (
    'online_q_per_action',
    'target_q_per_action',
)


def _intervention_for(
    hypothesis_name: str, *,
    capacity: int, batch_size: int, lr: float, total_steps: int,
) -> dict[str, object]:
    base: dict[str, object] = {
        'total_steps': total_steps,
        'eval_every': total_steps // 10,
        'n_episodes': 5,
        'gamma': 0.99,
        'replay': Replay(capacity=capacity, batch_size=batch_size),
        'optimizer': WarmedUpdate(inner=Adam(lr=lr), warmup_steps=100),
        'sync_period': 100,
    }
    if hypothesis_name == 'ddqn':
        base['bootstrap'] = partial(
            bootstrap, greedification=double_greedify,
        )
    return base


def _make_hypothesis(
    hypothesis_name: str, grid_point: dict[str, Any],
) -> Hypothesis[DQNTrajectoryRecord]:
    intervention = _intervention_for(hypothesis_name, **grid_point)
    if hypothesis_name == 'vanilla_dqn':
        return Hypothesis(
            name='vanilla_dqn',
            intervention=intervention,
            bridges=(),
            predicted_direction=None,
            intervention_arms=(),
        )
    if hypothesis_name == 'ddqn':
        return Hypothesis(
            name='ddqn',
            intervention=intervention,
            bridges=(),
            predicted_direction='a_gt_b',
            intervention_arms=(
                Intervention(
                    slot_path='bootstrap',
                    replacement=partial(
                        bootstrap, greedification=double_greedify,
                    ),
                ),
            ),
        )
    raise ValueError(f'unknown hypothesis name: {hypothesis_name!r}')


def _grid_tag(grid_point: dict[str, Any]) -> str:
    return '__'.join(
        f'{k}={v!r}' for k, v in sorted(grid_point.items())
    )


def _run_one_arm(
    env_name: str, hypothesis_name: str, grid_point: dict[str, Any],
    seeds: tuple[int, ...], tmp_dir: Path, arm_idx: int,
    *,
    chunk_size: int = SEED_CHUNK_SIZE,
) -> tuple[Path, Path]:
    """Run an arm's seeds in sequential chunks to fit GPU memory.

    With capacity=50k the f32[50000, n_seeds, obs] replay tensor
    OOMs at n_seeds=30 on a 16GB GPU. We split into 3 chunks of 10,
    run each as a separate `run_dqn_arm` (vmap-batched), then
    concatenate cells before parquet write. Output is identical to
    a single n=30 vmap call would have produced."""
    from corroborate.rl.cell_runner import CellResult, run_dqn_arm
    from corroborate.rl.env_catalogue import get

    arm_tag = (
        f'arm{arm_idx:03d}__{env_name}__{hypothesis_name}__'
        f'{_grid_tag(grid_point)}'
    )
    runs_path = tmp_dir / f'{arm_tag}__runs.parquet'
    traces_path = tmp_dir / f'{arm_tag}__traces.parquet'

    chunks = [
        seeds[i:i + chunk_size]
        for i in range(0, len(seeds), chunk_size)
    ]
    all_cells: list[CellResult] = []
    env_spec = get(env_name)
    for chunk_idx, chunk in enumerate(chunks):
        h = _make_hypothesis(hypothesis_name, grid_point)
        arm = run_dqn_arm(
            env_spec, chunk, hypothesis=h,
        )
        all_cells.extend(arm.cells)
        del arm
        # Clear between chunks so the next chunk's compile gets
        # fresh device memory; without this the prior chunk's
        # arrays stay rooted until function return.
        jax.clear_caches()
        gc.collect()

    write_runrows(tuple(c.run for c in all_cells), runs_path)
    reduced_traces = apply_trace_reductions(
        [c.trace for c in all_cells],
        add=TRACE_POST_REDUCTIONS,
        drop=TRACE_POST_DROPS,
    )
    write_tracerows(reduced_traces, traces_path)

    del all_cells, reduced_traces
    jax.clear_caches()
    gc.collect()
    return runs_path, traces_path


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


def _cast_to_target(tbl: pa.Table, target: pa.Schema) -> pa.Table:
    n = len(tbl)
    cols: list[pa.Array | pa.ChunkedArray] = []
    for f in target:
        if f.name in tbl.column_names:
            col = tbl.column(f.name)
            if col.type != f.type:
                col = col.cast(f.type)
            cols.append(col)
        else:
            cols.append(pa.nulls(n, type=f.type))
    return pa.Table.from_arrays(cols, schema=target)


def _stream_concat(inputs: list[Path], out: Path) -> None:
    if not inputs:
        raise ValueError('no input parquets to concat')
    target = _union_schema(inputs)
    if out.exists():
        out.unlink()
    writer = pq.ParquetWriter(
        out, target, compression='zstd', compression_level=3,
    )
    try:
        for p in inputs:
            pf = pq.ParquetFile(p)
            for i in range(pf.num_row_groups):
                tbl = pf.read_row_group(i)
                tbl = _cast_to_target(tbl, target)
                writer.write_table(tbl)
                del tbl
            del pf
    finally:
        writer.close()


def main() -> None:
    out_dir = Path(__file__).parent / 'data' / 'ddqn_better_hp'
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    final_runs_path = out_dir / 'runs.parquet'
    final_traces_path = out_dir / 'traces.parquet'

    grid_points = _grid_points(HP_GRID)
    n_arms = len(ENV_NAMES) * len(HYPOTHESIS_NAMES) * len(grid_points)

    arms_specs = [
        (idx, env_name, h_name, gp)
        for idx, (env_name, h_name, gp) in enumerate(
            itertools.product(ENV_NAMES, HYPOTHESIS_NAMES, grid_points),
        )
    ]

    print(
        f'sweep: {n_arms} arms ({len(ENV_NAMES)} envs × '
        f'{len(HYPOTHESIS_NAMES)} hypotheses × {len(grid_points)} '
        f'grid points), {len(SEEDS)} seeds vmapped per arm — '
        f'{n_arms * len(SEEDS)} cells total',
        flush=True,
    )
    print(f'HP: {dict(HP_GRID)}', flush=True)
    print(f'envs: {list(ENV_NAMES)}', flush=True)
    print(flush=True)

    runs_paths: list[Path] = []
    traces_paths: list[Path] = []
    t_start = time.time()
    for idx, env_name, h_name, gp in arms_specs:
        t_arm = time.time()
        print(
            f'  [{idx+1}/{n_arms}] {env_name} {h_name} '
            f'{_grid_tag(gp)} ...',
            flush=True,
        )
        rp, tp = _run_one_arm(env_name, h_name, gp, SEEDS, tmp_dir, idx)
        runs_paths.append(rp)
        traces_paths.append(tp)
        elapsed = time.time() - t_arm
        total = time.time() - t_start
        print(
            f'    done in {elapsed:.1f}s '
            f'(cumulative {total/60:.1f} min)',
            flush=True,
        )

    print()
    print('merging per-arm parquets ...', flush=True)
    _stream_concat(runs_paths, final_runs_path)
    _stream_concat(traces_paths, final_traces_path)
    runs_count = pq.ParquetFile(final_runs_path).metadata.num_rows
    traces_count = pq.ParquetFile(final_traces_path).metadata.num_rows
    print(f'  → {final_runs_path} ({runs_count} rows)')
    print(f'  → {final_traces_path} ({traces_count} rows)')


if __name__ == '__main__':
    main()
