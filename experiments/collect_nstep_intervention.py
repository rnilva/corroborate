"""N-step intervention sweep on DDQN baseline.

Strategy 1 from the user's intervention design: hold DDQN fixed
as the baseline (action-noise bias-correction is on in both
arms) and intervene on n-step return — the bootstrap-axis
variance-reduction knob.

Hypothesis. The 200k DDQN corpus has a residual sparse-reward →
outcome direct edge (`bootstrap_fraction → g_link | g_mech`,
ATE=+0.88) that is not mediated by the bias-reduction mechanism
g_mech and is not captured by Q-spread / action-margin /
argmax-disagreement / state-coverage proxies. n-step return
reduces the share of the TD target that comes from the bootstrap
term (Σ γ^k r vs γ·v), which is the exact knob most plausibly
behind a sparse-reward → outcome benefit not mediated by bias.

Design:
- arm A (baseline): DDQN + n_step=1 — recovers single-step DDQN
  exactly (smoke-tested in tests/rl/dqn/test_replay.py).
- arm B (treatment): DDQN + n_step=3 — folds 3 raw transitions
  into each stored transition; bootstrap discount becomes γ^3.

Env subset chosen to maximise the residual signal: high-
bootstrap_fraction sparse-reward envs from the 200k corpus
(Catch, DiscountingChain, MountainCar, Acrobot, FourRooms),
where the residual edge is theoretically strongest.

Read: if g_link(arm B) − g_link(arm A) > 0 and the chain edge
g_mech ⟷ g_link is similar across arms, the residual was a
TD-target-variance / multi-step credit-assignment effect, NOT a
bias effect; n-step closes the gap by reducing reliance on the
bootstrap chain.

Usage:
  uv run python experiments/collect_nstep_intervention.py
"""
from __future__ import annotations

import os
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION', '0.9')

from dataclasses import replace
from functools import partial
from pathlib import Path

from corroborate.hypothesis import Hypothesis
from corroborate.intervention import Intervention
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.collect import EnvConfig
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import get as _get_env_spec


# Sparse-reward envs from the 200k DDQN corpus where the
# residual sparse-reward → outcome direct edge is theoretically
# strongest. MLP Q-network suffices (no MinAtar / no CNN).
ENV_CONFIGS: tuple[EnvConfig, ...] = (
    EnvConfig('Catch-bsuite', n_seeds=30, chunk_size=15),
    EnvConfig('DiscountingChain-bsuite', n_seeds=30, chunk_size=15),
    EnvConfig('MountainCar-v0', n_seeds=30, chunk_size=15),
    EnvConfig('Acrobot-v1', n_seeds=30, chunk_size=15),
    EnvConfig('FourRooms-misc', n_seeds=30, chunk_size=15),
)

GAMMA: float = 0.99


def _hypothesis(
    name: str, n_step: int, env_name: str,
) -> Hypothesis[DQNTrajectoryRecord]:
    """Per-env hypothesis with DDQN held fixed (double_greedify on
    both arms) and n_step varied.

    n-step lives at the dqn-level as a top-level HP (single
    source of truth for `n_step_return` during rollout AND for
    bootstrap's γⁿ discount); `Replay` is a vanilla FIFO ring
    that's unaware of n-step semantics."""
    del env_name  # MLP q_network handles all classic-env obs shapes
    base: dict[str, object] = {
        'total_steps': 200_000,
        'eval_every': 20_000,
        'n_episodes': 5,
        'gamma': GAMMA,
        'n_step': n_step,
        'replay': Replay(capacity=50_000, batch_size=32),
        'optimizer': WarmedUpdate(
            inner=Adam(lr=1e-4), warmup_steps=100,
        ),
        'sync_period': 100,
        'bootstrap': partial(bootstrap, greedification=double_greedify),
    }
    return Hypothesis(
        name=name, intervention=base,
        bridges=(),
        predicted_direction='a_gt_b' if n_step > 1 else None,
        intervention_arms=(
            Intervention(
                slot_path='bootstrap',
                replacement=partial(bootstrap, greedification=double_greedify),
            ),
        ),
    )


def main() -> None:
    hypotheses: list[Hypothesis[DQNTrajectoryRecord]] = []
    env_configs_aligned: list[EnvConfig] = []
    for ec in ENV_CONFIGS:
        for h_name, n in (('ddqn_1step', 1), ('ddqn_3step', 3)):
            hypotheses.append(_hypothesis(h_name, n, ec.env_name))
            env_configs_aligned.append(ec)

    out_dir = Path(__file__).parent / 'data' / 'nstep_intervention'

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

    # Hint: run the same analysis pipeline as the 200k corpus on
    # the new corpus to compare g_link / g_mech across n_step:
    # uv run python experiments/analyze_per_burst_summary.py \
    #   --corpus nstep_intervention
    # uv run python experiments/analyze_per_burst_meta_regression.py \
    #   --corpus nstep_intervention
    # uv run python experiments/causal_discovery_link_moderators.py \
    #   --corpus nstep_intervention
    _ = replace  # ensure import is used (replace not yet leveraged)


if __name__ == '__main__':
    main()
