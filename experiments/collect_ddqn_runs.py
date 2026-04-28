"""Collect DDQN-vs-vanilla DQN runs for §3 acceptance test.

Three nested parallelism layers:

1. **Subprocess over arms** — `ProcessPoolExecutor` spawns one
   Python process per (env, hypothesis, hp-grid-point) cell.
   Each subprocess has its own JAX / CUDA context, isolating
   per-cell JIT-cache memory. Concurrency capped at 2 — concurrent
   JAX init on a single GPU races on CUDA context allocation
   and crashes the worker pool at higher concurrency.
2. **Vmap over seeds** — within each subprocess, `run_dqn_arm`
   batches all seeds in one jit-compiled call.
3. **Nested scan over training steps** — `train_with_eval` is a
   single nested-scan: outer over super-steps, inner over
   `eval_every` training steps + one eval burst.

**HP grid is multi-axis.** `HP_GRID` is a dict of axes; each
axis is a list of values. Cartesian product produces the grid
points. `_intervention_for(hypothesis_name, **grid_point)`
constructs the intervention dict for one cell. Adding an axis
is two lines: a key on `HP_GRID` + the corresponding kwarg in
`_intervention_for`.

**Output is a SINGLE pair of parquets.** Per-arm files live in
a `tmp/` subdir as intermediate cache; after all workers
complete, the orchestrator merges into `runs.parquet` +
`traces.parquet` and (optionally) cleans up the tempfiles.
Downstream analysis reads two files, not 2 × N.

Run: `uv run python experiments/collect_ddqn_runs.py`."""
from __future__ import annotations

# Set JAX memory env vars BEFORE any import that touches jax.
# `mp.get_context('spawn')` workers re-execute this module as
# `__mp_main__`, so module-level imports trigger jax initialisation
# at line ~60 (env_catalogue.py uses `jnp.array(...)` at module
# top). With JAX's default preallocate=true at 0.9 GPU, the second
# worker OOMs before its function body can set its own caps.
# Setting these at the very top makes both parent and every spawned
# worker share the device cleanly.
import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.45')

import itertools
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path
from typing import Any

import polars as pl

from corroborate.hypothesis import Hypothesis
from corroborate.persistence import (
    apply_trace_reductions,
    read_runrows,
    read_tracerows,
    write_runrows,
    write_tracerows,
)
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.schema import RunRow, TraceRow


# ============ Experiment grid ============

from corroborate.rl.env_catalogue import ENV_REGISTRY


# All envs registered in the catalogue. v9-corpus parity: 18-19
# envs spanning classic-control, bsuite, MinAtar, misc, bandit
# families. The §3 schema's cross-env aggregation (random-effects
# PI, link Pearson r) needs n_envs ≥ ~10 to have meaningful
# corpus-level power. Pong-misc is known to drop seeds (see
# PAPER_NOTES §3.9 caveat 4); included anyway, downstream filters
# out failed cells.
ENV_NAMES: tuple[str, ...] = tuple(ENV_REGISTRY.keys())

HYPOTHESIS_NAMES = ('vanilla_dqn', 'ddqn')

SEEDS: tuple[int, ...] = tuple(range(30))

# v9 parity: TrainingProtocol.total_steps = 50_000 (their default).
# v9 parallelises 64 envs per cell, so their 50k env-steps is
# ~781 gradient steps; our 1:1 ratio gives 50k gradient steps —
# strictly more learning than v9's nominal 50k. Sufficient for
# CartPole convergence (literature converges in 30-50k steps).
TOTAL_STEPS = 50_000


# Multi-axis HP grid. Each key is a flat axis name; each value is
# a list of grid points along that axis. Cartesian product over
# values yields the full grid. Add axes by adding keys here AND
# the corresponding kwarg in `_intervention_for` below.
#
# Single-element lists collapse the axis (no variation along it);
# this lets the script stay flexible without changing structure
# when only some axes vary.
# For the all-envs §3 sweep, single-point grid (capacity=10k is
# the v9 / literature default for CartPole-class). HP-sensitivity
# can be a separate sweep on a smaller env subset.
HP_GRID: dict[str, list[Any]] = {
    'capacity': [10_000],
    'batch_size': [32],
    'lr': [1e-3],
}


# ============ Trace post-reductions (polars exprs) ============

# 3-D Q-tensors (`online_q_values`, `target_q_values`) are
# `(steps, batch, n_actions)` — at 50k steps × 30 seeds they
# dominate disk (~800 MB per arm). For DDQN §3 acceptance we
# only need per-step max-Q (online & target) — `hasselt_
# covariance_gap` reads max-Q across time, not per-batch detail.
#
# `map_elements` is the right tool for nested-list aggregation:
# `pl.list.eval` has known limitations with deeply nested lists.
# The UDF runs once per cell at write time — overhead negligible.

def _per_step_max_q(nested_list: pl.Series) -> list[float]:
    """Per-step max-Q over (batch, n_actions). Input is a nested
    Python list shaped `(steps, batch, n_actions)`; output is
    `(steps,)`."""
    return [
        max(max(per_action_q) for per_action_q in per_batch)
        for per_batch in nested_list.to_list()
    ]


TRACE_POST_REDUCTIONS: tuple[pl.Expr, ...] = (
    pl.col('online_q_values').map_elements(
        _per_step_max_q, return_dtype=pl.List(pl.Float64),
    ).alias('online_max_q_per_step'),
    pl.col('target_q_values').map_elements(
        _per_step_max_q, return_dtype=pl.List(pl.Float64),
    ).alias('target_max_q_per_step'),
)


# Source columns to drop after reductions. Explicit acknowledgment
# that the raw 3-D arrays are gone from the trace store; bridges
# that need them must be precomputed at training time. The 2-D
# `sample_indices` (~50 MB) is kept for `lin_iid_gap` post-hoc.
TRACE_POST_DROPS: tuple[str, ...] = (
    'online_q_values',
    'target_q_values',
)


def _intervention_for(
    hypothesis_name: str, *,
    capacity: int, batch_size: int, lr: float,
) -> dict[str, object]:
    """Build the intervention dict for one (hypothesis, grid-point).
    Pure function so workers can reconstruct it from picklable
    args (kwargs from `HP_GRID`'s keys).

    Adding a new HP-grid axis: extend `HP_GRID` with the new key,
    add a matching kwarg here, and use it inside the dict."""
    base: dict[str, object] = {
        'total_steps': TOTAL_STEPS,
        'eval_every': TOTAL_STEPS // 10,
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
    """Reconstruct hypothesis from a string id + a grid point.
    Workers can't pickle closures cleanly across spawn-mode
    processes, so each worker rebuilds the hypothesis from
    stable args (string + dict of primitives)."""
    intervention = _intervention_for(hypothesis_name, **grid_point)
    if hypothesis_name == 'vanilla_dqn':
        return Hypothesis(
            name='vanilla_dqn',
            intervention=intervention,
            bridges=(),
            predicted_direction=None,
        )
    if hypothesis_name == 'ddqn':
        return Hypothesis(
            name='ddqn',
            intervention=intervention,
            bridges=(),
            predicted_direction='a_gt_b',
        )
    raise ValueError(f'unknown hypothesis name: {hypothesis_name!r}')


def _grid_tag(grid_point: dict[str, Any]) -> str:
    """Compact tag for arm-file naming. Format:
    `axis1=v1__axis2=v2__...`. Used only for tempfile names; the
    leaves are also written as RunRow measurements (the source of
    truth for analysis)."""
    return '__'.join(
        f'{k}={v!r}' for k, v in sorted(grid_point.items())
    )


# ============ Worker: one (env, hypothesis, grid-point) cell ============

def _run_arm_worker(
    args: tuple[str, str, dict[str, Any], tuple[int, ...], str, float, int],
) -> tuple[Path, Path]:
    """Subprocess worker. Runs `run_dqn_arm` for one (env,
    hypothesis, grid-point) cell + writes per-arm parquets. Returns
    (runs_path, traces_path).

    Disables XLA preallocation BEFORE importing jax — by default
    each JAX process grabs ~75% of GPU memory at init, which
    starves sibling workers. With `XLA_PYTHON_CLIENT_PREALLOCATE=
    false` plus a per-worker `XLA_PYTHON_CLIENT_MEM_FRACTION`
    cap, N workers can share one device cleanly."""
    (
        env_name, hypothesis_name, grid_point,
        seeds, tmp_dir_str, mem_fraction, arm_idx,
    ) = args

    # Must set before any jax import in this process.
    import os
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault(
        'XLA_PYTHON_CLIENT_MEM_FRACTION', f'{mem_fraction:.3f}',
    )

    tmp_dir = Path(tmp_dir_str)
    arm_tag = f'arm{arm_idx:03d}__{env_name}__{hypothesis_name}__{_grid_tag(grid_point)}'
    runs_path = tmp_dir / f'{arm_tag}__runs.parquet'
    traces_path = tmp_dir / f'{arm_tag}__traces.parquet'

    from corroborate.persistence import (
        apply_trace_reductions as _apply_reductions,
        write_runrows as _write_runrows,
        write_tracerows as _write_tracerows,
    )
    from corroborate.rl.cell_runner import run_dqn_arm
    from corroborate.rl.dqn.claims.optimizer import Adam as _Adam
    from corroborate.rl.env_catalogue import get

    h = _make_hypothesis(hypothesis_name, grid_point)
    cells = run_dqn_arm(
        get(env_name), seeds, hypothesis=h, optimizer=_Adam(),
    )
    _write_runrows(tuple(c.run for c in cells), runs_path)
    # Apply polars-expr reductions in-memory before writing the
    # trace parquet. Drops 3-D Q-tensors after computing per-step
    # max-Q reductions — the trace file shrinks ~10× with the
    # bridge-relevant signal preserved.
    reduced_traces = _apply_reductions(
        [c.trace for c in cells],
        add=TRACE_POST_REDUCTIONS,
        drop=TRACE_POST_DROPS,
    )
    _write_tracerows(reduced_traces, traces_path)

    # Free compiled XLA programs from the worker's JIT cache. Each
    # env shape pulls a fresh compilation; without clearing, the
    # cache grows unboundedly and the ~6-8th env OOMs the worker
    # (observed BrokenProcessPool from arm 11/32 onwards on 18-env
    # sweeps). `jax.clear_caches()` releases the python-side
    # compilation references; the XLA runtime then frees the device
    # buffers. Avoids `max_tasks_per_child` recycle, which deadlocked
    # under the spawn-context pool (parent + worker race during
    # restart-bookkeeping).
    #
    # Drop the per-arm cell payload BEFORE clearing caches so the
    # arrays themselves go away too — otherwise compiled programs
    # are freed but the JAX arrays from `cells` keep their device
    # buffers rooted until return.
    del cells, reduced_traces
    import jax as _jax
    _jax.clear_caches()
    import gc as _gc
    _gc.collect()

    return runs_path, traces_path


# ============ Orchestrator ============

def _grid_points(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of `grid`. Returns a list of dicts, one
    per grid point, keyed identically to `grid`."""
    keys = list(grid.keys())
    return [
        dict(zip(keys, point))
        for point in itertools.product(*grid.values())
    ]


def main() -> None:
    out_dir = Path(__file__).parent / 'data' / 'ddqn'
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    final_runs_path = out_dir / 'runs.parquet'
    final_traces_path = out_dir / 'traces.parquet'

    grid_points = _grid_points(HP_GRID)
    n_arms = len(ENV_NAMES) * len(HYPOTHESIS_NAMES) * len(grid_points)

    # Cap worker concurrency. 2 is the safe default — concurrent
    # JAX init on a single GPU races on CUDA context allocation
    # at higher concurrency (observed `BrokenProcessPool` killing
    # all workers at 4 with capacity=50_000 + 30 seeds + 50k steps).
    n_workers = min(2, n_arms, mp.cpu_count())
    mem_fraction = 0.9 / n_workers

    arms_args: list[
        tuple[str, str, dict[str, Any], tuple[int, ...], str, float, int]
    ] = [
        (
            env_name, h_name, gp,
            SEEDS, str(tmp_dir), mem_fraction, idx,
        )
        for idx, (env_name, h_name, gp) in enumerate(
            itertools.product(ENV_NAMES, HYPOTHESIS_NAMES, grid_points),
        )
    ]
    print(
        f'spawning up to {n_workers} subprocess workers for '
        f'{n_arms} arms ({len(ENV_NAMES)} envs × '
        f'{len(HYPOTHESIS_NAMES)} hypotheses × '
        f'{len(grid_points)} grid points), '
        f'{len(SEEDS)} seeds vmapped per arm — '
        f'{n_arms * len(SEEDS)} cells total',
        flush=True,
    )
    print(f'HP grid axes: {list(HP_GRID.keys())}')
    for axis, values in HP_GRID.items():
        print(f'  {axis}: {values}')
    print(flush=True)

    # Resume: skip arms whose tmp parquets already exist OR whose
    # (env_name, intervention_name, leaf-signature) tuple is already
    # present in the merged `runs.parquet` from a previous sweep.
    # Two layers because the previous sweep may have:
    #   (a) been killed mid-run → tmp parquets remain, merge final
    #       not yet executed; OR
    #   (b) completed merge for some arms → tmp parquets cleaned,
    #       data lives only in the merged final.
    def _arm_tag(
        env_name: str, h_name: str,
        grid_point: dict[str, Any], idx: int,
    ) -> str:
        return f'arm{idx:03d}__{env_name}__{h_name}__{_grid_tag(grid_point)}'

    # Project (env_name, intervention_name, replay.capacity, batch_size,
    # lr) tuples already in the merged runs.parquet so we can skip
    # those arms.
    completed_keys: set[tuple[str, str, int, int, float]] = set()
    if final_runs_path.exists():
        df_existing = pl.read_parquet(final_runs_path)
        if not df_existing.is_empty():
            for row in df_existing.select(
                'env_name', 'intervention_name', 'replay.capacity',
                'replay.batch_size', 'optimizer.inner.lr',
            ).unique().iter_rows(named=False):
                if row[0] is None or row[1] is None:
                    continue
                completed_keys.add((
                    str(row[0]), str(row[1]), int(row[2]),
                    int(row[3]), float(row[4]),
                ))
            print(
                f'resume: {len(completed_keys)} (env, intervention, hp) '
                f'tuples already in {final_runs_path.name}',
                flush=True,
            )

    pre_existing: list[tuple[Path, Path]] = []
    pending_args: list[
        tuple[str, str, dict[str, Any], tuple[int, ...], str, float, int]
    ] = []
    for args in arms_args:
        env_name, h_name, gp, *_, idx = args
        # Skip if already in the merged file.
        gp_key = (
            env_name, h_name,
            int(gp.get('capacity', 0)),
            int(gp.get('batch_size', 0)),
            float(gp.get('lr', 0.0)),
        )
        if gp_key in completed_keys:
            continue
        # Skip if tmp parquets exist (interrupted-mid-merge case).
        tag = _arm_tag(env_name, h_name, gp, idx)
        runs_p = tmp_dir / f'{tag}__runs.parquet'
        traces_p = tmp_dir / f'{tag}__traces.parquet'
        if runs_p.exists() and traces_p.exists():
            pre_existing.append((runs_p, traces_p))
        else:
            pending_args.append(args)
    if pre_existing:
        print(f'resume: {len(pre_existing)} arms in tmp/; '
              f'{len(pending_args)} pending', flush=True)

    # Spawn-start so each worker gets a clean JAX state.
    ctx = mp.get_context('spawn')

    t0 = time.time()
    parquet_pairs: list[tuple[Path, Path]] = list(pre_existing)
    # No `max_tasks_per_child` recycle — observed deadlock on the
    # transition: a recycled worker's spawn races with the parent's
    # future-completion bookkeeping under spawn-context. Accept JIT
    # cache growth instead; with capacity=10_000 the per-arm memory
    # cost is small enough that 18-env compilations fit per worker.
    with ProcessPoolExecutor(
        max_workers=n_workers, mp_context=ctx,
    ) as pool:
        future_to_arm = {
            pool.submit(_run_arm_worker, args): args[:3]
            for args in pending_args
        }
        for i, future in enumerate(as_completed(future_to_arm), start=1):
            env_name, h_name, grid_point = future_to_arm[future]
            try:
                runs_path, traces_path = future.result()
            except Exception as e:
                print(
                    f'[{i:>3}/{len(pending_args)}] FAILED  {env_name:<22} '
                    f'h={h_name:<12} {_grid_tag(grid_point):<40} '
                    f'{type(e).__name__}: {e}',
                    flush=True,
                )
                continue
            parquet_pairs.append((runs_path, traces_path))
            elapsed = time.time() - t0
            print(
                f'[{i:>3}/{len(pending_args)}] done    {env_name:<22} '
                f'h={h_name:<12} {_grid_tag(grid_point):<40} '
                f'elapsed={elapsed:>6.0f}s',
                flush=True,
            )

    print(f'\nworkers complete in {time.time() - t0:.0f}s; '
          f'merging {len(parquet_pairs)} per-arm parquet pairs')

    # Union-merge per-arm parquets into one pair of files. If a
    # final file already exists (resume case), seed `all_runs` /
    # `all_traces` with it so we APPEND rather than OVERWRITE.
    all_runs: list[RunRow] = []
    all_traces: list[TraceRow] = []
    if final_runs_path.exists():
        all_runs.extend(read_runrows(final_runs_path))
    if final_traces_path.exists():
        all_traces.extend(read_tracerows(final_traces_path))
    for runs_p, traces_p in parquet_pairs:
        all_runs.extend(read_runrows(runs_p))
        all_traces.extend(read_tracerows(traces_p))
    write_runrows(all_runs, final_runs_path)
    write_tracerows(all_traces, final_traces_path)
    print(
        f'written {len(all_runs)} runs → {final_runs_path.name}\n'
        f'written {len(all_traces)} traces → {final_traces_path.name}',
    )

    # Clean up per-arm intermediate files. Comment this out to
    # keep them for debugging.
    for runs_p, traces_p in parquet_pairs:
        runs_p.unlink(missing_ok=True)
        traces_p.unlink(missing_ok=True)
    try:
        tmp_dir.rmdir()
    except OSError:
        pass  # tmp dir not empty (e.g., FAILED arms left files); leave it
    print(f'cleaned up per-arm files in {tmp_dir.name}/')


if __name__ == '__main__':
    main()
