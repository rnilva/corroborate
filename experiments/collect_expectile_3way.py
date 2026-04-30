"""Expectile-greedify vs DDQN vs vanilla — Strategy 2 contrast.

The 200k DDQN corpus produced a residual `bootstrap_fraction →
g_link | g_mech` (DoWhy ATE = +0.88) that DDQN's argmax-decoupling
mechanism does not eliminate. Two hypotheses for the residual:

1. **DDQN-specific** — only DDQN's selection-evaluation
   decoupling produces the `g_link → outcome` lift; an alternative
   bias-correction operator yields a smaller (or zero) residual.
2. **Sparse-reward-intrinsic** — any consistent bias-correction
   bites the same way; the residual is a property of the
   sparse-reward link, not of DDQN's particular mechanism.

Strategy 2 contrasts a structurally distinct bias-correction
operator: expectile-pessimistic greedification at τ=0.7 (Garg et
al 2023, "Extreme Q-Learning"). Expectile-greedify does NOT
decouple selection from evaluation — both sides of v(s') come
from Q_target — but it pessimistically dampens the σ-proportional
max-bias residual. If the `g_link → outcome` ATE shifts under
expectile-greedify by the same magnitude as under DDQN, the
residual is intrinsic. If only DDQN's variant produces the lift,
the residual is selection-evaluation-decoupling-specific.

Envs: the historical sparse-reward 200k cohort (Catch,
DiscountingChain, MountainCar, Acrobot, FourRooms). HPs match
the HPO-validated stable regime (cap=50k, lr=1e-4, sync=100,
γ=0.99, 200k steps, eval every 20k). MLP architecture matches
the original 200k corpus exactly.

Usage:
  uv run python experiments/collect_expectile_3way.py
"""
from __future__ import annotations

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')

from functools import partial
from pathlib import Path

from corroborate.hypothesis import Hypothesis
from corroborate.intervention import Intervention
from corroborate.rl.dqn.claims.bootstrap import (
    bootstrap, double_greedify, expectile_greedify,
)
from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate.rl.dqn.claims.q_network import MLP
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.collect import EnvConfig
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import get as _get_env_spec


ENV_CONFIGS: tuple[EnvConfig, ...] = (
    EnvConfig('Catch-bsuite', n_seeds=30, chunk_size=15),
    EnvConfig('DiscountingChain-bsuite', n_seeds=30, chunk_size=15),
    EnvConfig('MountainCar-v0', n_seeds=30, chunk_size=15),
    EnvConfig('Acrobot-v1', n_seeds=30, chunk_size=15),
    EnvConfig('FourRooms-misc', n_seeds=30, chunk_size=15),
)


def _hypothesis(name: str) -> Hypothesis[DQNTrajectoryRecord]:
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
        'q_network': MLP(hidden=(64, 64)),
    }
    if name == 'vanilla_dqn':
        return Hypothesis(
            name='vanilla_dqn', intervention=base,
            bridges=(), predicted_direction=None,
            intervention_arms=(),
        )
    if name == 'ddqn':
        boot = partial(bootstrap, greedification=double_greedify)
        base['bootstrap'] = boot
        return Hypothesis(
            name='ddqn', intervention=base,
            bridges=(), predicted_direction='a_gt_b',
            intervention_arms=(
                Intervention(slot_path='bootstrap', replacement=boot),
            ),
        )
    if name == 'expectile_dqn':
        boot = partial(
            bootstrap,
            greedification=partial(expectile_greedify, tau=0.7),
        )
        base['bootstrap'] = boot
        return Hypothesis(
            name='expectile_dqn', intervention=base,
            bridges=(), predicted_direction='a_gt_b',
            intervention_arms=(
                Intervention(slot_path='bootstrap', replacement=boot),
            ),
        )
    raise ValueError(name)


def main() -> None:
    from corroborate.rl.dqn.collect import (
        chunked_arms, env_arm_tag,
    )
    from corroborate.rl.dqn.trace_reductions import (
        Q_TRACE_DROPS, Q_TRACE_REDUCTIONS,
    )
    from corroborate.rl.sweep import DQNRunner
    from corroborate.sweep import run_hypotheses

    hypotheses = [
        _hypothesis(n)
        for n in ('vanilla_dqn', 'ddqn', 'expectile_dqn')
    ]
    out_dir = Path(__file__).parent / 'data' / 'expectile_3way'
    env_specs = {
        ec.env_name: _get_env_spec(ec.env_name)
        for ec in ENV_CONFIGS
    }

    run_hypotheses(
        chunked_arms(hypotheses, ENV_CONFIGS),
        runner=DQNRunner(env_specs),
        out_dir=out_dir,
        arm_tag=env_arm_tag,
        trace_reductions=Q_TRACE_REDUCTIONS,
        trace_drops=Q_TRACE_DROPS,
    )


if __name__ == '__main__':
    main()
