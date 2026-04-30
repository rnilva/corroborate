"""Vanilla DQN × n-step sweep — completes the 2×2 factorial.

The existing `nstep_intervention` corpus has DDQN-1step vs
DDQN-3step on 4 sparse-reward envs (FourRooms missed because of
a now-fixed dtype bug). To test the over-correction reading of
the n-step result — DDQN+3step backfires because two bias-
corrections compound — we need the missing two cells:

  | greedification | n_step=1 | n_step=3 |
  |---|---|---|
  | max (vanilla) | (in 200k corpus, but recompute here for HP-match) | NEW |
  | double (DDQN) | nstep_intervention arm A | nstep_intervention arm B |

If vanilla+3step *helps* on Catch/Acrobot (where DDQN+3step
HURT), the over-correction reading wins: n-step alone gives
useful bias-reduction, and stacking it on DDQN's correction is
the problem. If vanilla+3step *also hurts* on the same envs, the
variance-amplification reading wins (longer rollouts dilute the
rare-positive-reward signal regardless of greedification).

CPU-only (GPU is busy with the MinAtar 1M sweep). MLP q-network,
classic envs — small enough to run on CPU in reasonable time.

Storage: archive_remote='s3://corroborate-archive/
nstep_vanilla_arms'. Each arm uploaded to R2 right after
completion; final merge from remote URIs.

Usage:
  set -a; source .env; set +a
  JAX_PLATFORMS=cpu uv run python experiments/collect_nstep_vanilla_arms.py
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

from functools import partial
from pathlib import Path

from corroborate.hypothesis import Hypothesis
from corroborate.intervention import Intervention
from corroborate.rl.dqn.claims.bootstrap import bootstrap
from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.collect import EnvConfig
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import get as _get_env_spec


# Same 5 sparse-reward envs as the original nstep_intervention
# (now FourRooms works after the int32-obs dtype fix).
ENV_CONFIGS: tuple[EnvConfig, ...] = (
    EnvConfig('Catch-bsuite', n_seeds=30, chunk_size=15),
    EnvConfig('DiscountingChain-bsuite', n_seeds=30, chunk_size=15),
    EnvConfig('MountainCar-v0', n_seeds=30, chunk_size=15),
    EnvConfig('Acrobot-v1', n_seeds=30, chunk_size=15),
    EnvConfig('FourRooms-misc', n_seeds=30, chunk_size=15),
)

GAMMA: float = 0.99
REMOTE: str = 's3://corroborate-archive/nstep_vanilla_arms'


def _hypothesis(
    name: str, n_step: int,
) -> Hypothesis[DQNTrajectoryRecord]:
    """Vanilla (max_greedify) DQN with n_step ∈ {1, 3}.

    Replay's `n_step` and `gamma` HPs travel as a pair with
    bootstrap's `n_step` kwarg. greedification is bootstrap's
    default (max_greedify), so we don't pass it."""
    base: dict[str, object] = {
        'total_steps': 200_000,
        'eval_every': 20_000,
        'n_episodes': 5,
        'gamma': GAMMA,
        'replay': Replay(
            capacity=50_000, batch_size=32,
            n_step=n_step, gamma=GAMMA,
        ),
        'optimizer': WarmedUpdate(
            inner=Adam(lr=1e-4), warmup_steps=100,
        ),
        'sync_period': 100,
        'bootstrap': partial(bootstrap, n_step=n_step),
    }
    return Hypothesis(
        name=name, intervention=base,
        bridges=(),
        predicted_direction=None,  # no predicted direction —
        # we want clean per-(env, burst) g without sign assumption
        intervention_arms=(
            Intervention(
                slot_path='bootstrap',
                replacement=partial(bootstrap, n_step=n_step),
            ),
        ),
    )


def main() -> None:
    hypotheses: list[Hypothesis[DQNTrajectoryRecord]] = []
    env_configs_aligned: list[EnvConfig] = []
    for ec in ENV_CONFIGS:
        for h_name, n in (('vanilla_1step', 1), ('vanilla_3step', 3)):
            hypotheses.append(_hypothesis(h_name, n))
            env_configs_aligned.append(ec)

    out_dir = Path(__file__).parent / 'data' / 'nstep_vanilla_arms'

    from corroborate.rl.dqn.collect import _run_one_arm  # type: ignore[reportPrivateUsage]
    from corroborate.persistence import stream_concat_parquets
    from corroborate.cloud import archive
    from corroborate.rl.sweep import DQNRunner
    from corroborate.rl.dqn.trace_reductions import (
        Q_TRACE_DROPS, Q_TRACE_REDUCTIONS,
    )
    import time

    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / 'tmp'
    tmp_dir.mkdir(parents=True, exist_ok=True)

    env_specs = {ec.env_name: _get_env_spec(ec.env_name) for ec in ENV_CONFIGS}
    runner = DQNRunner(env_specs)

    archived_runs_uris: list[str] = []
    archived_traces_uris: list[str] = []
    print(f'sweep: {len(hypotheses)} arms = '
          f'{len(ENV_CONFIGS)} envs × 2 hypotheses, '
          f'platform={os.environ.get("JAX_PLATFORMS", "<unset>")}, '
          f'remote={REMOTE}', flush=True)
    t_start = time.time()
    for idx, (h, ec) in enumerate(zip(hypotheses, env_configs_aligned)):
        t_arm = time.time()
        tag = f'{ec.env_name}__{h.name}'
        print(
            f'  [{idx+1}/{len(hypotheses)}] {tag} '
            f'(seeds={ec.n_seeds}, chunk={ec.chunk_size}) ...',
            flush=True,
        )
        rp, tp = _run_one_arm(
            h, ec, runner, tmp_dir, idx, tag,
            trace_reductions=Q_TRACE_REDUCTIONS,
            trace_drops=Q_TRACE_DROPS,
        )
        rp_rel = rp.relative_to(out_dir).as_posix()
        tp_rel = tp.relative_to(out_dir).as_posix()
        archive(
            out_dir, REMOTE,
            files=[rp_rel, tp_rel], purge_local=True,
        )
        archived_runs_uris.append(f'{REMOTE.rstrip("/")}/{rp_rel}')
        archived_traces_uris.append(f'{REMOTE.rstrip("/")}/{tp_rel}')
        elapsed = time.time() - t_arm
        total = time.time() - t_start
        print(f'    done in {elapsed:.1f}s '
              f'(cumulative {total/60:.1f} min) '
              f'archived → {REMOTE}/{rp_rel}',
              flush=True)

    print()
    print('merging per-arm parquets from remote ...', flush=True)
    final_runs = out_dir / 'runs.parquet'
    final_traces = out_dir / 'traces.parquet'
    stream_concat_parquets(archived_runs_uris, final_runs)
    stream_concat_parquets(archived_traces_uris, final_traces)
    archive(
        out_dir, REMOTE,
        files=['runs.parquet', 'traces.parquet'],
        purge_local=False,
    )
    print(f'  → {final_runs}')
    print(f'  → {final_traces}')


if __name__ == '__main__':
    main()
