"""DDQN-effective cohort sweep at HPO-validated stable HPs.

The HPO sweep on Freeway-MinAtar with CNN
(`experiments/hpo_freeway_cnn.py`) found capacity=50k +
lr=1e-4 + CNN gives the lowest jensen_gap (0.38), highest
sustained final return (0.79), and lowest seed variance — the
most stable training regime among 6 HP combinations tested.

This cohort sweep applies that HP to MinAtar / high-|A| envs
where DDQN's bias-correction has theoretical leverage:
- Asterix-MinAtar (|A|=5)
- Breakout-MinAtar (|A|=3)
- SpaceInvaders-MinAtar (|A|=4) — showed g=-0.53 HELD on
  time-to-solve in the original 200k MLP corpus
- Freeway-MinAtar (|A|=3) — HPO target; included for direct
  comparison
- MNISTBandit-bsuite (|A|=10) — high-|A| anchor

Thin call site of `collect_sweep_to_parquet` — author 2 ×
n_envs hypotheses (DDQN + vanilla per env, with CNN configured
on the env's obs_shape) and dispatch.

Usage:
  uv run python experiments/collect_ddqn_effective.py
"""
from __future__ import annotations

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')

from functools import partial
from pathlib import Path

from corroborate.hypothesis import Hypothesis
from corroborate.intervention import Intervention
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate.rl.dqn.claims.q_network import CNN
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.collect import EnvConfig, collect_sweep_to_parquet
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import get as _get_env_spec


# (env_name, n_seeds, chunk_size). MinAtar 784-D obs at cap=50k
# requires chunking at 10 to stay under 16 GB GPU.
ENV_CONFIGS: tuple[EnvConfig, ...] = (
    EnvConfig('Asterix-MinAtar', n_seeds=30, chunk_size=10),
    EnvConfig('Breakout-MinAtar', n_seeds=30, chunk_size=10),
    EnvConfig('SpaceInvaders-MinAtar', n_seeds=30, chunk_size=10),
    EnvConfig('Freeway-MinAtar', n_seeds=30, chunk_size=10),
    EnvConfig('MNISTBandit-bsuite', n_seeds=30, chunk_size=10),
)


def _hypothesis(
    name: str, env_name: str,
) -> Hypothesis[DQNTrajectoryRecord]:
    """One env requires its own Hypothesis because CNN.obs_shape
    is env-specific."""
    spec = _get_env_spec(env_name)
    base: dict[str, object] = {
        'total_steps': 200_000,
        'eval_every': 20_000,
        'n_episodes': 5,
        'gamma': 0.99,
        'replay': Replay(capacity=50_000, batch_size=32),
        'optimizer': WarmedUpdate(
            inner=Adam(lr=1e-4), warmup_steps=100,
        ),
        'sync_period': 100,
        'q_network': CNN(
            obs_shape=spec.observation_shape,
            channels=(16, 32), kernel_size=3, hidden=(128,),
        ),
    }
    if name == 'vanilla_dqn':
        return Hypothesis(
            name='vanilla_dqn', intervention=base,
            bridges=(), predicted_direction=None,
            intervention_arms=(),
        )
    if name == 'ddqn':
        base['bootstrap'] = partial(
            bootstrap, greedification=double_greedify,
        )
        return Hypothesis(
            name='ddqn', intervention=base,
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
    raise ValueError(name)


def main() -> None:
    # Each hypothesis is env-specific (CNN.obs_shape varies);
    # we author n_envs × 2 hypotheses and pair each with its
    # corresponding env_config.
    hypotheses: list[Hypothesis[DQNTrajectoryRecord]] = []
    env_configs_aligned: list[EnvConfig] = []
    for ec in ENV_CONFIGS:
        for h_name in ('vanilla_dqn', 'ddqn'):
            hypotheses.append(_hypothesis(h_name, ec.env_name))
            env_configs_aligned.append(ec)

    out_dir = Path(__file__).parent / 'data' / 'ddqn_effective_cohort'

    # Manual arm iteration since each hypothesis is env-paired.
    # The orchestrator's default Cartesian product would mis-pair
    # hypotheses with envs; supply arm_tag to encode the pair
    # uniquely. We iterate via a single-env list per hypothesis:
    from corroborate.rl.dqn.collect import _run_one_arm  # type: ignore[reportPrivateUsage]
    from corroborate.persistence import stream_concat_parquets
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

    runs_paths: list[Path] = []
    traces_paths: list[Path] = []
    print(f'sweep: {len(hypotheses)} arms = '
          f'{len(ENV_CONFIGS)} envs × 2 hypotheses', flush=True)
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
        runs_paths.append(rp)
        traces_paths.append(tp)
        elapsed = time.time() - t_arm
        total = time.time() - t_start
        print(f'    done in {elapsed:.1f}s '
              f'(cumulative {total/60:.1f} min)', flush=True)

    print()
    print('merging per-arm parquets ...', flush=True)
    final_runs = out_dir / 'runs.parquet'
    final_traces = out_dir / 'traces.parquet'
    stream_concat_parquets(runs_paths, final_runs)
    stream_concat_parquets(traces_paths, final_traces)
    print(f'  → {final_runs}')
    print(f'  → {final_traces}')


if __name__ == '__main__':
    main()
