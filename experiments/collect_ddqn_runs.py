"""Collect DDQN-vs-vanilla DQN runs for §3 acceptance test.

Sweeps a grid of (env, seed, hypothesis) cells: each env runs
N seeds of vanilla DQN and N seeds of DDQN at matched seeds
(same seed → same env reset → paired comparison once Step 5
ships paired statistics).

Outputs `experiments/data/ddqn/runs.parquet` with one RunRow per
cell. Step 5's `paired_comparison_from_runs` will read this and
produce ComparisonRows.

Scope: classical-control envs (CartPole, Acrobot, MountainCar)
+ a few bsuite envs. Image envs (MinAtar) deferred — episode
length and total_steps would be 100× longer; collect them when
the v0 stats are validated on the cheap envs first.

Run: `uv run python experiments/collect_ddqn_runs.py`. ~10
minutes wall-clock for the default config (5 envs × 5 seeds × 2
hypotheses = 50 cells, ~10s/cell on CPU)."""
from __future__ import annotations

import time
from pathlib import Path

import optax

from corroborate.hypothesis import Hypothesis
from corroborate.persistence import write_runrows
from corroborate.rl.cell_runner import EvalConfig, run_dqn_arm
from corroborate.rl.dqn.claims.bootstrap import ddqn_bootstrap
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import get
from corroborate.schema import RunRow


# ============ Experiment grid ============

ENV_NAMES = (
    'CartPole-v1',
    'Acrobot-v1',
    'MountainCar-v0',
    'Catch-bsuite',
    'DeepSea-bsuite',
)

SEEDS = tuple(range(5))  # 5 seeds per (env, hypothesis)

# Total steps per cell. Short enough for v0 wall-clock budget,
# long enough that the masked_window_mean catches several
# episode terminations in the late window.
TOTAL_STEPS = 2000

# Matches `EvalConfig.n_evals` divisibility constraint.
EVAL_CONFIG = EvalConfig.n_evals(
    total_steps=TOTAL_STEPS, n_evals=10, n_episodes=5,
)


def _make_hypotheses() -> tuple[
    Hypothesis[DQNTrajectoryRecord],
    Hypothesis[DQNTrajectoryRecord],
]:
    """Vanilla and DDQN hypotheses over the DQN record schema.
    Same intervention SHAPE (single slot swap), so the
    matched-seed pair runs the same env / training schedule with
    only the bootstrap slot differing."""
    vanilla: Hypothesis[DQNTrajectoryRecord] = Hypothesis(
        name='vanilla_dqn',
        intervention={},
        bridges=(),
        predicted_direction=None,
    )
    ddqn: Hypothesis[DQNTrajectoryRecord] = Hypothesis(
        name='ddqn',
        intervention={'bootstrap': ddqn_bootstrap},
        bridges=(),
        predicted_direction='a_gt_b',
    )
    return vanilla, ddqn


def main() -> None:
    out_dir = Path(__file__).parent / 'data' / 'ddqn'
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'runs.parquet'

    vanilla, ddqn = _make_hypotheses()
    arms = [
        (env_name, h)
        for env_name in ENV_NAMES
        for h in (vanilla, ddqn)
    ]
    n_arms = len(arms)
    print(f'collecting {n_arms} arms ({len(ENV_NAMES)} envs × 2 '
          f'hypotheses), {len(SEEDS)} seeds vmapped per arm — '
          f'{n_arms * len(SEEDS)} cells total')

    rows: list[RunRow] = []
    t0 = time.time()
    for i, (env_name, h) in enumerate(arms):
        env_spec = get(env_name)
        arm_t0 = time.time()
        arm_rows = run_dqn_arm(
            env_spec, SEEDS, hypothesis=h,
            total_steps=TOTAL_STEPS,
            optimizer=optax.adam(1e-3),
            eval_config=EVAL_CONFIG,
            warmup_steps=100, sync_period=100,
            buffer_capacity=2000, batch_size=32,
        )
        arm_t = time.time() - arm_t0
        outcomes = [r.primary_outcome_summary for r in arm_rows]
        outcome_summary = sum(outcomes) / len(outcomes)
        elapsed = time.time() - t0
        eta = elapsed / (i + 1) * (n_arms - i - 1)
        print(
            f'[{i + 1:>2}/{n_arms}] {env_name:<22} h={h.name:<12} '
            f'mean_outcome={outcome_summary:>8.3f} '
            f'arm={arm_t:>5.1f}s eta={eta:>6.0f}s',
            flush=True,
        )
        rows.extend(arm_rows)

    print(f'\nwriting {len(rows)} rows → {out_path}')
    write_runrows(rows, out_path)
    print(f'done. total wall-clock: {time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
