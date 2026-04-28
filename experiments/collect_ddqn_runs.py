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

SEEDS: tuple[int, ...] = tuple(range(10))

TOTAL_STEPS = 5000  # short-but-meaningful for v0 stats validation


# Shared HP bundle — author commitments. HPs spread as flat kwargs
# into `hypothesis.intervention` so `mechanism_key` distinguishes
# (vanilla, total_steps=5000) from (ddqn, total_steps=5000) AND
# from any future re-run with a different HP setting.
_HPARAMS: dict[str, object] = {
    'total_steps': TOTAL_STEPS,
    'eval_every': TOTAL_STEPS // 10,
    'n_episodes': 5,
    'gamma': 0.99,
    'replay': Replay(capacity=2000, batch_size=32),
    'optimizer': WarmedUpdate(inner=Adam(), warmup_steps=100),
    'sync_period': 100,
}


def _make_hypothesis(name: str) -> Hypothesis[DQNTrajectoryRecord]:
    """Reconstruct hypothesis from a string id. Workers can't
    pickle closures cleanly across spawn-mode processes, so each
    worker rebuilds the hypothesis from a stable name.

    HPs live as flat kwargs in `intervention` (matching `dqn`'s
    signature). `mechanism_key` canonicalises each (kwarg, value)
    pair separately so two arms differing only in `gamma` get
    distinct mechanism_keys. The `bootstrap` slot swap and
    `predicted_direction` are the only fields that differ between
    vanilla and DDQN at the structural level."""
    if name == 'vanilla_dqn':
        return Hypothesis(
            name='vanilla_dqn',
            intervention={**_HPARAMS},
            bridges=(),
            predicted_direction=None,
        )
    if name == 'ddqn':
        return Hypothesis(
            name='ddqn',
            intervention={
                **_HPARAMS,
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
    args: tuple[str, str, tuple[int, ...], str, float],
) -> Path:
    """Subprocess worker. Runs `run_dqn_arm` for one
    (env, hypothesis) pair and writes the resulting RunRows to a
    per-arm parquet. Returns the parquet path.

    Disables XLA preallocation BEFORE importing jax — by default
    each JAX process grabs ~75% of GPU memory at init, which
    starves sibling workers. With `XLA_PYTHON_CLIENT_PREALLOCATE=
    false` plus a per-worker `XLA_PYTHON_CLIENT_MEM_FRACTION`
    cap, N workers can share one device cleanly."""
    env_name, hypothesis_name, seeds, out_dir_str, mem_fraction = args

    # Must set before any jax import in this process.
    import os
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
    os.environ.setdefault(
        'XLA_PYTHON_CLIENT_MEM_FRACTION', f'{mem_fraction:.3f}',
    )

    out_dir = Path(out_dir_str)
    out_path = out_dir / f'{env_name}__{hypothesis_name}.parquet'

    from corroborate.rl.cell_runner import run_dqn_arm
    from corroborate.rl.dqn.claims.optimizer import Adam
    from corroborate.rl.env_catalogue import get

    h = _make_hypothesis(hypothesis_name)
    rows = run_dqn_arm(
        get(env_name), seeds, hypothesis=h,
        optimizer=Adam(),
    )
    write_runrows(rows, out_path)
    return out_path


# ============ Orchestrator ============

def main() -> None:
    out_dir = Path(__file__).parent / 'data' / 'ddqn'
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / 'runs.parquet'

    n_arms = len(ENV_NAMES) * len(HYPOTHESIS_NAMES)

    # Cap worker concurrency. With `XLA_PYTHON_CLIENT_PREALLOCATE
    # =false` + per-worker `MEM_FRACTION = 0.9 / n_workers`,
    # N workers share one GPU cleanly. 4 is a conservative
    # default; raise if GPU memory is plentiful.
    n_workers = min(4, n_arms, mp.cpu_count())
    mem_fraction = 0.9 / n_workers

    arms_args: list[tuple[str, str, tuple[int, ...], str, float]] = [
        (env_name, h_name, SEEDS, str(out_dir), mem_fraction)
        for env_name in ENV_NAMES
        for h_name in HYPOTHESIS_NAMES
    ]
    print(
        f'spawning up to {n_workers} subprocess workers for '
        f'{n_arms} arms ({len(ENV_NAMES)} envs × '
        f'{len(HYPOTHESIS_NAMES)} hypotheses), '
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
            pool.submit(_run_arm_worker, args): args[:2]
            for args in arms_args
        }
        for i, future in enumerate(as_completed(future_to_arm), start=1):
            env_name, h_name = future_to_arm[future]
            try:
                p = future.result()
            except Exception as e:
                print(
                    f'[{i:>2}/{n_arms}] FAILED  {env_name:<22} '
                    f'h={h_name:<12} {type(e).__name__}: {e}',
                    flush=True,
                )
                continue
            parquet_paths.append(p)
            elapsed = time.time() - t0
            print(
                f'[{i:>2}/{n_arms}] done    {env_name:<22} '
                f'h={h_name:<12} elapsed={elapsed:>6.0f}s',
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
