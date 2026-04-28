"""Collect DDQN-vs-vanilla DQN runs for §3 acceptance test.

Three nested parallelism layers:

1. **Subprocess over arms** — `ProcessPoolExecutor` spawns one
   Python process per (env, hypothesis) pair. Each subprocess
   has its own JAX / CUDA context, isolating per-env JIT-cache
   memory (matches v9's pattern; prevents cross-env JIT-cache
   OOM on long sweeps). Concurrency capped at CPU count or 4
   for memory headroom.
2. **Vmap over seeds** — within each subprocess, `run_dqn_arm`
   batches all seeds in one jit-compiled call. Gymnax envs vmap
   natively; gradient steps batch.
3. **Nested scan over training steps** — `train_with_eval` is a
   single nested-scan: outer over super-steps, inner over
   `eval_every` training steps + one eval burst.

Outputs `experiments/data/ddqn/runs.parquet` (union of per-arm
parquets). Step 5's `paired_comparison_from_runs` reads this
and produces ComparisonRows.

Run: `uv run python experiments/collect_ddqn_runs.py`."""
from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from functools import partial

from corroborate.hypothesis import Hypothesis
from corroborate.persistence import read_runrows, write_runrows
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.schema import RunRow


# ============ Experiment grid ============

ENV_NAMES = (
    'CartPole-v1',
    'Acrobot-v1',
    'MountainCar-v0',
    'Catch-bsuite',
    'DeepSea-bsuite',
)

SEEDS: tuple[int, ...] = tuple(range(30))

# v9 parity: TrainingProtocol.total_steps = 50_000 (their default).
# v9 parallelises 64 envs per cell, so their 50k env-steps is
# ~781 gradient steps; our 1:1 ratio gives 50k gradient steps —
# strictly more learning than v9's nominal 50k. Sufficient for
# CartPole convergence (literature converges in 30-50k steps).
TOTAL_STEPS = 50_000

# HP-sensitivity grid for buffer capacity. The principle: don't
# guess at HP defaults; let the framework's columnar parquet show
# us which HPs the outcome is actually sensitive to. Three levels
# spread far enough apart to detect non-trivial sensitivity, and
# the path-keyed leaf `replay.capacity` gets a distinct value per
# cell — one-line `df.group_by('replay.capacity').agg(...)` reveals
# the sensitivity post-hoc.
CAPACITIES: tuple[int, ...] = (2_000, 10_000, 50_000)


def _hparams(capacity: int) -> dict[str, object]:
    """HP bundle parameterised on `replay.capacity`. The other
    leaves stay fixed across the grid; capacity is the swept
    dimension. Adding more HP-grid axes (lr, batch_size, ...) is
    one line each here + the corresponding tuple at module top."""
    return {
        'total_steps': TOTAL_STEPS,
        'eval_every': TOTAL_STEPS // 10,
        'n_episodes': 5,
        'gamma': 0.99,
        'replay': Replay(capacity=capacity, batch_size=32),
        'optimizer': WarmedUpdate(inner=Adam(), warmup_steps=100),
        'sync_period': 100,
    }


def _make_hypothesis(
    name: str, capacity: int,
) -> Hypothesis[DQNTrajectoryRecord]:
    """Reconstruct hypothesis from a string id + capacity. Workers
    can't pickle closures cleanly across spawn-mode processes, so
    each worker rebuilds the hypothesis from stable args.

    Configurational leaves live as flat kwargs in `intervention`
    (matching `dqn`'s signature). Cell-runner records each (kwarg,
    value) pair as a measurement at its dotted topology path;
    downstream `leaf_signature` projects the configurational
    subset so two arms differing only in `replay.capacity` produce
    distinct group-by keys."""
    hparams = _hparams(capacity)
    if name == 'vanilla_dqn':
        return Hypothesis(
            name='vanilla_dqn',
            intervention={**hparams},
            bridges=(),
            predicted_direction=None,
        )
    if name == 'ddqn':
        return Hypothesis(
            name='ddqn',
            intervention={
                **hparams,
                'bootstrap': partial(
                    bootstrap, greedification=double_greedify,
                ),
            },
            bridges=(),
            predicted_direction='a_gt_b',
        )
    raise ValueError(f'unknown hypothesis name: {name!r}')


HYPOTHESIS_NAMES = ('vanilla_dqn', 'ddqn')


# ============ Worker: one (env, hypothesis) arm ============

def _run_arm_worker(
    args: tuple[str, str, int, tuple[int, ...], str, float],
) -> Path:
    """Subprocess worker. Runs `run_dqn_arm` for one
    (env, hypothesis, capacity) cell + writes the resulting
    RunRows + TraceRows to per-arm parquets. Returns the runs
    parquet path.

    Disables XLA preallocation BEFORE importing jax — by default
    each JAX process grabs ~75% of GPU memory at init, which
    starves sibling workers. With `XLA_PYTHON_CLIENT_PREALLOCATE=
    false` plus a per-worker `XLA_PYTHON_CLIENT_MEM_FRACTION`
    cap, N workers can share one device cleanly."""
    env_name, hypothesis_name, capacity, seeds, out_dir_str, mem_fraction = args

    # Must set before any jax import in this process.
    import os
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault(
        'XLA_PYTHON_CLIENT_MEM_FRACTION', f'{mem_fraction:.3f}',
    )

    out_dir = Path(out_dir_str)
    arm_tag = f'{env_name}__{hypothesis_name}__cap{capacity}'
    runs_path = out_dir / f'{arm_tag}__runs.parquet'
    traces_path = out_dir / f'{arm_tag}__traces.parquet'

    from corroborate.persistence import write_tracerows
    from corroborate.rl.cell_runner import run_dqn_arm
    from corroborate.rl.dqn.claims.optimizer import Adam
    from corroborate.rl.env_catalogue import get

    h = _make_hypothesis(hypothesis_name, capacity)
    cells = run_dqn_arm(
        get(env_name), seeds, hypothesis=h,
        optimizer=Adam(),
    )
    write_runrows(tuple(c.run for c in cells), runs_path)
    write_tracerows(tuple(c.trace for c in cells), traces_path)
    # Caller assembles the corpus from the per-arm runs files;
    # traces stay co-located so a downstream consumer can rejoin
    # by id without a separate manifest.
    return runs_path


# ============ Orchestrator ============

def main() -> None:
    out_dir = Path(__file__).parent / 'data' / 'ddqn'
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / 'runs.parquet'

    n_arms = len(ENV_NAMES) * len(HYPOTHESIS_NAMES) * len(CAPACITIES)

    # Cap worker concurrency. With `XLA_PYTHON_CLIENT_PREALLOCATE
    # =false` + per-worker `MEM_FRACTION = 0.9 / n_workers`,
    # N workers share one GPU cleanly. 2 is the safe default —
    # 4 workers concurrently initialising JAX on a single GPU
    # races on CUDA context allocation and crashes (observed at
    # capacity=50_000 + 30 seeds + 50k steps; single-arm path
    # fine, multi-process worker pool kills all workers).
    n_workers = min(2, n_arms, mp.cpu_count())
    mem_fraction = 0.9 / n_workers

    arms_args: list[tuple[str, str, int, tuple[int, ...], str, float]] = [
        (env_name, h_name, capacity, SEEDS, str(out_dir), mem_fraction)
        for env_name in ENV_NAMES
        for h_name in HYPOTHESIS_NAMES
        for capacity in CAPACITIES
    ]
    print(
        f'spawning up to {n_workers} subprocess workers for '
        f'{n_arms} arms ({len(ENV_NAMES)} envs × '
        f'{len(HYPOTHESIS_NAMES)} hypotheses × '
        f'{len(CAPACITIES)} capacities), '
        f'{len(SEEDS)} seeds vmapped per arm — '
        f'{n_arms * len(SEEDS)} cells total',
        flush=True,
    )

    # Spawn-start so each worker gets a clean JAX state (fork
    # would inherit the parent's CUDA context, breaking GPU
    # isolation if the parent had touched JAX).
    ctx = mp.get_context('spawn')

    t0 = time.time()
    parquet_paths: list[Path] = []
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        future_to_arm = {
            pool.submit(_run_arm_worker, args): args[:3]
            for args in arms_args
        }
        for i, future in enumerate(as_completed(future_to_arm), start=1):
            env_name, h_name, capacity = future_to_arm[future]
            try:
                p = future.result()
            except Exception as e:
                print(
                    f'[{i:>2}/{n_arms}] FAILED  {env_name:<22} '
                    f'h={h_name:<12} cap={capacity:<7} '
                    f'{type(e).__name__}: {e}',
                    flush=True,
                )
                continue
            parquet_paths.append(p)
            elapsed = time.time() - t0
            print(
                f'[{i:>2}/{n_arms}] done    {env_name:<22} '
                f'h={h_name:<12} cap={capacity:<7} '
                f'elapsed={elapsed:>6.0f}s',
                flush=True,
            )

    print(f'\nworkers complete in {time.time() - t0:.0f}s; '
          f'merging {len(parquet_paths)} per-arm parquets')

    # Union-merge per-arm parquets into one combined file.
    all_rows: list[RunRow] = []
    for p in parquet_paths:
        all_rows.extend(read_runrows(p))
    write_runrows(all_rows, final_path)
    print(f'written {len(all_rows)} rows → {final_path}')


if __name__ == '__main__':
    main()
