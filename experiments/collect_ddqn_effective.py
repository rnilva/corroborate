"""DDQN-effective cohort sweep — envs and HPs where DDQN should
*actually improve outcome* (not just reduce the gap).

The mediator-search analysis needs at least some envs where the
link DDQN→outcome HELDs at non-trivial g; otherwise per-candidate
Pearson r between mediator g and outcome g is constructed from
~zero outcome g across the board and can't distinguish among
mediators.

Cohort selection (Hasselt-style + our prior signal):
- **Asterix-MinAtar** (|A|=5): MinAtar family canonical.
- **Breakout-MinAtar** (|A|=3): MinAtar family canonical.
- **SpaceInvaders-MinAtar** (|A|=4): showed g=−0.53 HELD on
  time-to-solve in the original 200k corpus.
- **MNISTBandit-bsuite** (|A|=10): high-|A| anchor; the Jensen
  floor scales with √(2 log |A|) ≈ 2.15.

HP regime: **sub-saturating**.
- capacity=20000 — between the under-converged cap=10k and the
  saturating cap=50k that ceilinged outcomes in
  collect_action_dim_sweep. Aim is for outcome to have headroom
  while the buffer is large enough to avoid Q-collapse.
- 200k training steps (matches the rest of the corpus).
- batch=32, lr=1e-3, sync=100 (canonical).

Seed budget: 30 seeds per arm, chunked at 15 to keep the f32[
20000, n, 784] replay tensor under ~1 GB per chunk on a 16 GB GPU.

Output:
  experiments/data/ddqn_effective_cohort/runs.parquet
  experiments/data/ddqn_effective_cohort/traces.parquet
  with online_std_q_per_step persisted.

Usage:
  uv run python experiments/collect_ddqn_effective.py
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
from corroborate.rl.env_catalogue import get as _get_env_spec
from corroborate.rl.sweep import DQNRunner
from corroborate.sweep import SweepCellResult


ENV_CONFIG: tuple[tuple[str, int, int], ...] = (
    # (env_name, n_seeds, chunk_size). All MinAtar/large-obs need chunking
    # at cap=20k to stay under GPU memory.
    ('Asterix-MinAtar', 30, 15),
    ('Breakout-MinAtar', 30, 15),
    ('SpaceInvaders-MinAtar', 30, 15),
    ('MNISTBandit-bsuite', 30, 10),  # 784-D obs + |A|=10
)

HYPOTHESIS_NAMES = ('vanilla_dqn', 'ddqn')

HP_GRID: dict[str, list[Any]] = {
    'capacity': [20_000],     # sub-saturating; below cap=50k ceiling
    'batch_size': [32],
    'lr': [1e-3],
    'total_steps': [200_000],
}


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


def _per_step_std_q(nested_list: pl.Series) -> list[float]:
    import statistics as _stat
    out: list[float] = []
    for per_action in nested_list.to_list():
        if per_action is None or len(per_action) < 2:
            out.append(float('nan'))
        else:
            out.append(float(_stat.pstdev(per_action)))
    return out


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
        _per_step_std_q, return_dtype=pl.List(pl.Float64),
    ).alias('online_std_q_per_step'),
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
            name='vanilla_dqn', intervention=intervention,
            bridges=(), predicted_direction=None,
            intervention_arms=(),
        )
    if hypothesis_name == 'ddqn':
        return Hypothesis(
            name='ddqn', intervention=intervention,
            bridges=(), predicted_direction='a_gt_b',
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
    seeds: tuple[int, ...], chunk_size: int,
    tmp_dir: Path, arm_idx: int,
    runner: DQNRunner,
) -> tuple[Path, Path]:
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
    h = _make_hypothesis(hypothesis_name, grid_point)
    all_runs: list[Any] = []
    all_traces: list[Any] = []
    for chunk in chunks:
        cell_result: SweepCellResult = runner(
            h, {'env_name': env_name, 'seeds': chunk},
        )
        all_runs.extend(cell_result.runs)
        all_traces.extend(cell_result.traces)
        del cell_result
        jax.clear_caches()
        gc.collect()

    write_runrows(tuple(all_runs), runs_path)
    reduced_traces = apply_trace_reductions(
        list(all_traces),
        add=TRACE_POST_REDUCTIONS,
        drop=TRACE_POST_DROPS,
    )
    write_tracerows(reduced_traces, traces_path)

    del all_runs, all_traces, reduced_traces
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
    out_dir = Path(__file__).parent / 'data' / 'ddqn_effective_cohort'
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    final_runs_path = out_dir / 'runs.parquet'
    final_traces_path = out_dir / 'traces.parquet'

    grid_points = _grid_points(HP_GRID)
    n_arms = len(ENV_CONFIG) * len(HYPOTHESIS_NAMES) * len(grid_points)
    total_cells = sum(n for _, n, _ in ENV_CONFIG) * len(HYPOTHESIS_NAMES)

    arms_specs: list[tuple[int, str, int, int, str, dict[str, Any]]] = []
    arm_idx = 0
    for env_name, n_seeds, chunk_size in ENV_CONFIG:
        for h_name in HYPOTHESIS_NAMES:
            for gp in grid_points:
                arms_specs.append((arm_idx, env_name, n_seeds, chunk_size, h_name, gp))
                arm_idx += 1

    print(
        f'sweep: {n_arms} arms ({len(ENV_CONFIG)} envs × '
        f'{len(HYPOTHESIS_NAMES)} hypotheses × {len(grid_points)} '
        f'grid points), {total_cells} cells total',
        flush=True,
    )
    print(f'HP: {dict(HP_GRID)}', flush=True)
    print(f'envs: {[(n, s, c) for n, s, c in ENV_CONFIG]}', flush=True)
    print(flush=True)

    env_specs = {name: _get_env_spec(name) for name, _, _ in ENV_CONFIG}
    runner = DQNRunner(env_specs)

    runs_paths: list[Path] = []
    traces_paths: list[Path] = []
    t_start = time.time()
    for idx, env_name, n_seeds, chunk_size, h_name, gp in arms_specs:
        t_arm = time.time()
        seeds = tuple(range(n_seeds))
        print(
            f'  [{idx+1}/{n_arms}] {env_name} {h_name} '
            f'(seeds={n_seeds}, chunk={chunk_size}) ...',
            flush=True,
        )
        rp, tp = _run_one_arm(
            env_name, h_name, gp, seeds, chunk_size, tmp_dir, idx, runner,
        )
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
