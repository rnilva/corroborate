"""MinAtar 1M-step DDQN-vs-vanilla cohort, incremental archive.

The first slice of the long-horizon multi-env replication. Anchored
to the same HPs as the 200k corpus (HPO-validated cap=50k +
lr=1e-4 + CNN, sync=100, γ=0.99) but at 5× more training steps
because the 200k cohort showed MinAtar barely learning (mean
final return ≈ 0.5–1.5 across the 4 envs at 200k). At 1M the
mechanism has time to activate; the chain decomposition recovered
on the classic-env 200k corpus can be tested at MinAtar scale.

Probe (Freeway, the largest-channel MinAtar env): chunk_size=15
fits in 14.6 GB of the 16 GB GPU; chunk_size=20 OOMs. So 30 seeds
split into 2 chunks of 15.

Storage: archive_remote='s3://corroborate-archive/minatar_1M'.
collect_sweep_to_parquet uploads each arm's tmp parquet pair
right after the arm completes and purges local; the final merge
reads back from the remote URIs. Peak local-disk during the run
≈ one arm's worth (~700 MB at 1M steps after Q_TRACE_REDUCTIONS).

Estimated wall: 60-70 min/chunk × 2 chunks/arm × 8 arms ≈
16-18 hours. The MNISTBandit-bsuite env from the 200k cohort is
deliberately omitted — its 2-D obs (28, 28) is a CNN.init type
mismatch, and as a bandit it has no spatial dynamics worth a CNN.

Usage:
  uv run python experiments/collect_minatar_1M.py
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
from corroborate.rl.dqn.collect import EnvConfig
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import get as _get_env_spec


# 4 MinAtar envs at 30 seeds × chunk=15. Probe on Freeway
# (worst case, 7 channels) confirmed chunk=15 fits in 14.6 GB
# of the 16 GB GPU.
ENV_CONFIGS: tuple[EnvConfig, ...] = (
    EnvConfig('Asterix-MinAtar', n_seeds=30, chunk_size=15),
    EnvConfig('Breakout-MinAtar', n_seeds=30, chunk_size=15),
    EnvConfig('Freeway-MinAtar', n_seeds=30, chunk_size=15),
    EnvConfig('SpaceInvaders-MinAtar', n_seeds=30, chunk_size=15),
)

REMOTE: str = 's3://corroborate-archive/minatar_1M'


def _hypothesis(
    name: str, env_name: str,
) -> Hypothesis[DQNTrajectoryRecord]:
    """Per-env hypothesis (CNN.obs_shape varies by env)."""
    spec = _get_env_spec(env_name)
    base: dict[str, object] = {
        'total_steps': 1_000_000,
        'eval_every': 50_000,
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
    from corroborate.rl.dqn.collect import (
        env_arm_tag, paired_arms,
    )
    from corroborate.rl.dqn.trace_reductions import (
        Q_TRACE_DROPS, Q_TRACE_REDUCTIONS,
    )
    from corroborate.rl.sweep import DQNRunner
    from corroborate.sweep import run_hypotheses

    hypotheses: list[Hypothesis[DQNTrajectoryRecord]] = []
    env_configs_aligned: list[EnvConfig] = []
    for ec in ENV_CONFIGS:
        for h_name in ('vanilla_dqn', 'ddqn'):
            hypotheses.append(_hypothesis(h_name, ec.env_name))
            env_configs_aligned.append(ec)

    out_dir = Path(__file__).parent / 'data' / 'minatar_1M'
    env_specs = {ec.env_name: _get_env_spec(ec.env_name) for ec in ENV_CONFIGS}

    run_hypotheses(
        paired_arms(hypotheses, env_configs_aligned),
        runner=DQNRunner(env_specs),
        out_dir=out_dir,
        archive_remote=REMOTE,
        arm_tag=env_arm_tag,
        trace_reductions=Q_TRACE_REDUCTIONS,
        trace_drops=Q_TRACE_DROPS,
    )


if __name__ == '__main__':
    main()
