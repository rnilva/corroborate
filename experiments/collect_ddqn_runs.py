"""Collect DDQN-vs-vanilla DQN runs for §3 acceptance test.

Single-process sequential. Two parallelism layers (no
subprocesses):

1. **Vmap over seeds** — `run_dqn_arm` batches all seeds in one
   jit-compiled call.
2. **Nested scan over training steps** — `train_with_eval` is a
   single nested-scan: outer over super-steps, inner over
   `eval_every` training steps + one eval burst.

The earlier `ProcessPoolExecutor` design had two recurring failure
modes on multi-env sweeps:
- `BrokenProcessPool` from JIT-cache OOMs after ~6-8 distinct env
  shapes filled the per-worker memory budget.
- Spawn-context recycle deadlocks when `max_tasks_per_child` was
  set, leaving workers in a half-restarted state.

`jax.clear_caches()` after each arm is the same primitive workers
were trying to use, lifted into the main process where it's not
fighting spawn-context bookkeeping. Single-process is ~2× wall-
clock vs 2 workers but reliable: 36 arms × ~5 min = ~3 hours.

**HP grid is multi-axis.** `HP_GRID` is a dict of axes; each axis
is a list of values. Cartesian product produces the grid points.
`_intervention_for(hypothesis_name, **grid_point)` constructs the
intervention dict for one cell.

**Output is a SINGLE pair of parquets.** Per-arm files live in a
`tmp/` subdir as intermediate cache; after each arm completes the
file is on disk, so a killed sweep can resume cleanly. After all
arms complete the orchestrator merges into `runs.parquet` +
`traces.parquet` and cleans up the tempfiles.

Run: `uv run python experiments/collect_ddqn_runs.py`."""
from __future__ import annotations

# Cap JAX memory: with single process we own the whole GPU; tell
# JAX not to preallocate the full device so other things can run
# alongside without waiting on this script. Set BEFORE any jax
# import.
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
from corroborate.rl.sweep import DQNRunner


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


# Multi-axis HP grid. Each key is a flat axis name; each value is
# a list of grid points along that axis. Cartesian product over
# values yields the full grid. Add axes by adding keys here AND
# the corresponding kwarg in `_intervention_for` below.
#
# Single-element lists collapse the axis (no variation along it);
# this lets the script stay flexible without changing structure
# when only some axes vary.
#
# `total_steps` is a grid axis so we can stack training-duration
# regimes on top of the existing 50k-step corpus. v9 used 50k as
# the default; the 200k point gives 4× more late-window samples
# for §5's mediator reductions and lets envs that converge slowly
# (Acrobot, Pong) settle. `eval_every = total_steps // 10` keeps
# the eval cadence proportional. The resume key (in `main`) keys
# on total_steps too so prior 50k-step cells aren't re-run.
HP_GRID: dict[str, list[Any]] = {
    'capacity': [10_000],
    'batch_size': [32],
    'lr': [1e-3],
    'total_steps': [50_000, 200_000],
}


# ============ Trace post-reductions (polars exprs) ============

# Q-tensor reductions now happen in-loop (`train_phase` returns
# `online_q_per_action` / `target_q_per_action` shape (n_actions,)
# per step + 5-tuple `pearson_stats`). High-action-env OOM is
# avoided at the source. We reduce the per-action vectors here to
# per-step scalars (max, min, argmax, mean) so PAPER §5's mediator
# features (q_gap_late, q_gap_growth, greedy_match_late,
# v_vs_max_delta_late) are derivable downstream without keeping
# the (steps, n_actions) tensors in trace.

def _per_step_max_q(nested_list: pl.Series) -> list[float]:
    """Per-step max over the (n_actions,) per-step Q vector.
    Input is a nested list shaped `(steps, n_actions)`; output is
    `(steps,)`."""
    return [max(per_action) for per_action in nested_list.to_list()]


def _per_step_min_q(nested_list: pl.Series) -> list[float]:
    """Per-step min over the (n_actions,) per-step Q vector.
    `q_gap = max - min` per step is the §5 action-margin signal."""
    return [min(per_action) for per_action in nested_list.to_list()]


def _per_step_mean_q(nested_list: pl.Series) -> list[float]:
    """Per-step mean over actions of the per-step Q vector. Proxies
    V; `|V - max| late` is §5's `v_vs_max_delta_late`."""
    return [
        sum(per_action) / len(per_action) if per_action else float('nan')
        for per_action in nested_list.to_list()
    ]


def _per_step_argmax_q(nested_list: pl.Series) -> list[int]:
    """Per-step argmax over actions. Online-vs-target argmax
    disagreement is §5's `greedy_match_late = mean(online_argmax ==
    target_argmax)` over the late window."""
    return [
        int(max(range(len(per_action)), key=lambda i: per_action[i]))
        if per_action else -1
        for per_action in nested_list.to_list()
    ]


def _per_step_std_q(nested_list: pl.Series) -> list[float]:
    """Per-step std-across-actions of the (n_actions,) Q vector.
    σ_action input to `jensen_floor_late = σ × √(2 log |A|)`. The
    action-axis collapse is named explicitly here; offline analysis
    averages across the time axis to recover the scalar floor."""
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


# Drop the per-action vectors after reducing — bridges that consume
# the per-step reductions read the named columns directly. Keeps
# parquet compact: 6 per-step 1-D columns instead of 2 per-step
# 2-D tensors.
TRACE_POST_DROPS: tuple[str, ...] = (
    'online_q_per_action',
    'target_q_per_action',
)


def _intervention_for(
    hypothesis_name: str, *,
    capacity: int, batch_size: int, lr: float, total_steps: int,
) -> dict[str, object]:
    """Build the intervention dict for one (hypothesis, grid-point).
    Pure function so workers can reconstruct it from picklable
    args (kwargs from `HP_GRID`'s keys).

    Adding a new HP-grid axis: extend `HP_GRID` with the new key,
    add a matching kwarg here, and use it inside the dict."""
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
    """Reconstruct hypothesis from a string id + a grid point.
    Workers can't pickle closures cleanly across spawn-mode
    processes, so each worker rebuilds the hypothesis from
    stable args (string + dict of primitives).

    `intervention_arms` carries the typed identity of mechanism
    swaps (DDQN's `bootstrap` → `partial(bootstrap, greedification=
    double_greedify)`); HPs (capacity, batch_size, lr,
    total_steps) stay in `intervention` as covariates."""
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
    """Compact tag for arm-file naming. Format:
    `axis1=v1__axis2=v2__...`. Used only for tempfile names; the
    leaves are also written as RunRow measurements (the source of
    truth for analysis)."""
    return '__'.join(
        f'{k}={v!r}' for k, v in sorted(grid_point.items())
    )


# ============ Per-arm runner ============

def _run_one_arm(
    env_name: str, hypothesis_name: str, grid_point: dict[str, Any],
    seeds: tuple[int, ...], tmp_dir: Path, arm_idx: int,
    runner: 'DQNRunner',
) -> tuple[Path, Path]:
    """Run one (env, hypothesis, grid-point) cell via the
    framework's `DQNRunner` Protocol + write per-arm parquets.
    Returns (runs_path, traces_path).

    Plain function (no subprocess). After completion, drops the
    per-arm payload and clears the JIT cache so the next arm gets
    a fresh compilation budget — without this, accumulated XLA
    programs OOM the device after ~6-8 distinct env shapes.

    `runner` is a shared `DQNRunner` instance — the env catalogue
    is cached in it, so repeated calls don't re-resolve env_specs."""
    arm_tag = f'arm{arm_idx:03d}__{env_name}__{hypothesis_name}__{_grid_tag(grid_point)}'
    runs_path = tmp_dir / f'{arm_tag}__runs.parquet'
    traces_path = tmp_dir / f'{arm_tag}__traces.parquet'

    h = _make_hypothesis(hypothesis_name, grid_point)
    cell_result = runner(h, {'env_name': env_name, 'seeds': seeds})
    # cell_result.graph is the captured ComputationGraph — held in
    # memory only (per the parquets-are-for-measurables principle);
    # no current consumer in the §3-§7 path. Forward-investment for
    # the redundancy / register / mechanism-key bundle.
    write_runrows(cell_result.runs, runs_path)
    reduced_traces = apply_trace_reductions(
        list(cell_result.traces),
        add=TRACE_POST_REDUCTIONS,
        drop=TRACE_POST_DROPS,
    )
    write_tracerows(reduced_traces, traces_path)

    # Drop the per-arm cell payload BEFORE clearing caches so the
    # arrays go away too — otherwise compiled programs are freed
    # but the JAX arrays from `cells` keep their device buffers
    # rooted until function return.
    del cell_result, reduced_traces
    jax.clear_caches()
    gc.collect()

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


def _union_schema(paths: list[Path]) -> pa.Schema:
    """Compute the union of all input parquets' schemas. Each
    field appears at most once; first-seen type wins on collision
    (per-arm parquets share types per-column by construction —
    only the column SET differs across step-counts when new
    measurables get added mid-corpus)."""
    fields: dict[str, pa.Field] = {}
    for p in paths:
        for f in pq.ParquetFile(p).schema_arrow:
            fields.setdefault(f.name, f)
    return pa.schema(list(fields.values()))


def _cast_to_target(tbl: pa.Table, target: pa.Schema) -> pa.Table:
    """Project / null-pad `tbl` to match `target`. Existing columns
    cast to the target type; missing columns inserted as full-null
    arrays of the target type."""
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


def _stream_concat_traces(inputs: list[Path], out: Path) -> None:
    """pyarrow incremental union-merge of `inputs` → `out`.
    Per-row-group streaming: never materialises the full corpus,
    bounded at ~one row group + the writer's pending buffer.
    Heterogeneous schemas (new columns added mid-corpus) null-pad
    via `_cast_to_target`."""
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
    out_dir = Path(__file__).parent / 'data' / 'ddqn'
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    final_runs_path = out_dir / 'runs.parquet'
    final_traces_path = out_dir / 'traces.parquet'

    grid_points = _grid_points(HP_GRID)
    n_arms = len(ENV_NAMES) * len(HYPOTHESIS_NAMES) * len(grid_points)

    arms_specs: list[
        tuple[int, str, str, dict[str, Any]]
    ] = [
        (idx, env_name, h_name, gp)
        for idx, (env_name, h_name, gp) in enumerate(
            itertools.product(ENV_NAMES, HYPOTHESIS_NAMES, grid_points),
        )
    ]
    print(
        f'sequential single-process sweep: {n_arms} arms '
        f'({len(ENV_NAMES)} envs × {len(HYPOTHESIS_NAMES)} hypotheses '
        f'× {len(grid_points)} grid points), '
        f'{len(SEEDS)} seeds vmapped per arm — '
        f'{n_arms * len(SEEDS)} cells total',
        flush=True,
    )
    print(f'HP grid axes: {list(HP_GRID.keys())}')
    for axis, values in HP_GRID.items():
        print(f'  {axis}: {values}')
    print(flush=True)

    # Resume: skip arms whose tmp parquets already exist OR whose
    # (env_name, intervention_name, hp) tuple is already in the
    # merged `runs.parquet` from a previous sweep.
    def _arm_tag(
        env_name: str, h_name: str,
        grid_point: dict[str, Any], idx: int,
    ) -> str:
        return f'arm{idx:03d}__{env_name}__{h_name}__{_grid_tag(grid_point)}'

    completed_keys: set[tuple[str, str, int, int, float, int]] = set()
    if final_runs_path.exists():
        df_existing = pl.read_parquet(final_runs_path)
        if not df_existing.is_empty():
            for row in df_existing.select(
                'env_name', 'intervention_name', 'replay.capacity',
                'replay.batch_size', 'optimizer.inner.lr', 'total_steps',
            ).unique().iter_rows(named=False):
                if row[0] is None or row[1] is None:
                    continue
                completed_keys.add((
                    str(row[0]), str(row[1]), int(row[2]),
                    int(row[3]), float(row[4]), int(row[5]),
                ))
            print(
                f'resume: {len(completed_keys)} (env, intervention, hp) '
                f'tuples already in {final_runs_path.name}',
                flush=True,
            )

    pre_existing: list[tuple[Path, Path]] = []
    pending_specs: list[tuple[int, str, str, dict[str, Any]]] = []
    for idx, env_name, h_name, gp in arms_specs:
        gp_key = (
            env_name, h_name,
            int(gp.get('capacity', 0)),
            int(gp.get('batch_size', 0)),
            float(gp.get('lr', 0.0)),
            int(gp.get('total_steps', 0)),
        )
        if gp_key in completed_keys:
            continue
        tag = _arm_tag(env_name, h_name, gp, idx)
        runs_p = tmp_dir / f'{tag}__runs.parquet'
        traces_p = tmp_dir / f'{tag}__traces.parquet'
        if runs_p.exists() and traces_p.exists():
            pre_existing.append((runs_p, traces_p))
        else:
            pending_specs.append((idx, env_name, h_name, gp))
    if pre_existing:
        print(f'resume: {len(pre_existing)} arms in tmp/; '
              f'{len(pending_specs)} pending', flush=True)

    # Shared DQNRunner — caches the full env catalogue once,
    # avoids per-arm `gymnax.make` lookups + lets a future
    # scheduler introspect runner state.
    runner = DQNRunner(ENV_REGISTRY)

    t0 = time.time()
    parquet_pairs: list[tuple[Path, Path]] = list(pre_existing)
    failures: list[tuple[str, str, str]] = []  # (env, hyp, exc)
    for i, (idx, env_name, h_name, gp) in enumerate(pending_specs, start=1):
        try:
            runs_path, traces_path = _run_one_arm(
                env_name, h_name, gp, SEEDS, tmp_dir, idx, runner,
            )
        except Exception as e:
            print(
                f'[{i:>3}/{len(pending_specs)}] FAILED  {env_name:<22} '
                f'h={h_name:<12} {_grid_tag(gp):<40} '
                f'{type(e).__name__}: {e}',
                flush=True,
            )
            failures.append((env_name, h_name, repr(e)))
            # Best-effort cache clear so the next arm gets a clean
            # state even after a failure.
            jax.clear_caches()
            gc.collect()
            continue
        parquet_pairs.append((runs_path, traces_path))
        elapsed = time.time() - t0
        print(
            f'[{i:>3}/{len(pending_specs)}] done    {env_name:<22} '
            f'h={h_name:<12} {_grid_tag(gp):<40} '
            f'elapsed={elapsed:>6.0f}s',
            flush=True,
        )

    print(f'\nsweep complete in {time.time() - t0:.0f}s; '
          f'merging {len(parquet_pairs)} per-arm parquet pairs '
          f'({len(failures)} failures)')

    # Runs: small (<1 MB total even at full corpus); polars in-memory
    # `diagonal_relaxed` concat is fine. Heterogeneous columns
    # across arms (e.g. arms with different `eval_step_index`
    # lengths from different `total_steps`) null-pad cleanly.
    runs_inputs: list[Path] = []
    traces_inputs: list[Path] = []
    if final_runs_path.exists():
        runs_inputs.append(final_runs_path)
    if final_traces_path.exists():
        traces_inputs.append(final_traces_path)
    for runs_p, traces_p in parquet_pairs:
        runs_inputs.append(runs_p)
        traces_inputs.append(traces_p)

    runs_tmp = final_runs_path.with_suffix('.parquet.tmp')
    traces_tmp = final_traces_path.with_suffix('.parquet.tmp')
    pl.concat(
        [pl.scan_parquet(p) for p in runs_inputs],
        how='diagonal_relaxed',
    ).sink_parquet(runs_tmp)

    # Traces: pyarrow incremental writer. polars' `sink_parquet`
    # over a `diagonal_relaxed` lazy plan dies silently at multi-tens
    # of GB on cross-corpus merges (observed when bringing a 200k-
    # step corpus alongside an existing 50k-step corpus —
    # `concat([scan(50k), scan(200k_arms...)]).sink_parquet(...)`
    # exits at 0 bytes with no traceback). pyarrow's per-row-group
    # read-cast-write is bounded at ~one row group + the writer's
    # pending buffer and survives at the multi-tens-of-GB scale
    # (proven by `scripts/tighten_traces.py` and
    # `scripts/merge_50k_200k.py`).
    _stream_concat_traces(traces_inputs, traces_tmp)

    # Atomic-rename only after both writes succeed.
    runs_tmp.replace(final_runs_path)
    traces_tmp.replace(final_traces_path)

    n_runs = pl.scan_parquet(final_runs_path).select(
        pl.len(),
    ).collect().item()
    n_traces = pl.scan_parquet(final_traces_path).select(
        pl.len(),
    ).collect().item()
    print(
        f'written {n_runs} runs → {final_runs_path.name}\n'
        f'written {n_traces} traces → {final_traces_path.name}',
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
