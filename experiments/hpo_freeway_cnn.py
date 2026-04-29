"""HPO sweep on Freeway-MinAtar with the CNN q-network.

Finds stable HPs (capacity × lr) for the cohort sweep that
follows. Vanilla DQN only — DDQN/vanilla parity is established
once we know the env converges.

This script is a thin call-site of `collect_sweep_to_parquet`:
just authors hypotheses + env_configs and dispatches. All the
trace-reduction + parquet-merge boilerplate lives in the
framework now (`corroborate.rl.dqn.collect`).

Output:
  experiments/data/hpo_freeway_cnn/runs.parquet
  experiments/data/hpo_freeway_cnn/traces.parquet

Usage:
  uv run python experiments/hpo_freeway_cnn.py
"""
from __future__ import annotations

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')

from pathlib import Path

from corroborate.hypothesis import Hypothesis
from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate.rl.dqn.claims.q_network import CNN
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.collect import EnvConfig, collect_sweep_to_parquet
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import get


ENV_NAME = 'Freeway-MinAtar'


def _hypothesis(capacity: int, lr: float) -> Hypothesis[DQNTrajectoryRecord]:
    obs_shape = get(ENV_NAME).observation_shape
    return Hypothesis(
        name=f'vanilla_dqn__cap{capacity}__lr{lr:.0e}',
        intervention={
            'total_steps': 200_000,
            'eval_every': 20_000,
            'n_episodes': 5, 'gamma': 0.99,
            'replay': Replay(capacity=capacity, batch_size=32),
            'optimizer': WarmedUpdate(inner=Adam(lr=lr), warmup_steps=100),
            'sync_period': 100,
            'q_network': CNN(
                obs_shape=obs_shape,
                channels=(16, 32), kernel_size=3, hidden=(128,),
            ),
        },
        bridges=(), predicted_direction=None,
        intervention_arms=(),
    )


def main() -> None:
    hypotheses = [
        _hypothesis(cap, lr)
        for cap in (10_000, 20_000, 50_000)
        for lr in (1e-4, 1e-3)
    ]
    env_configs = [EnvConfig(env_name=ENV_NAME, n_seeds=30, chunk_size=10)]
    out_dir = Path(__file__).parent / 'data' / 'hpo_freeway_cnn'

    collect_sweep_to_parquet(
        hypotheses=hypotheses,
        env_configs=env_configs,
        out_dir=out_dir,
        arm_tag=lambda h, ec: f'{ec.env_name}__{h.name}',
    )


if __name__ == '__main__':
    main()
