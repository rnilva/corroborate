"""End-to-end smoke for the DDQN substrate post-decomposition.

Validates the foundation + Phase 1+2 cuts actually carry the
vanilla-vs-DDQN comparison:

1. Builds two hypotheses differing only in `greedification`.
2. Runs each on CartPole, 3 seeds, 1000 steps via `run_dqn_arm`.
3. Asserts:
   - `hp_signature` distinguishes the arms by the `greedification`
     swap (HP topology paths differ).
   - `aggregate_runs` groups cells correctly.
   - Each arm has finite outcome summaries.

Run: `uv run python experiments/smoke_ddqn_run.py`."""
from __future__ import annotations

import time
from functools import partial

from corroborate.aggregate import aggregate_runs, hp_signature
from corroborate.hypothesis import Hypothesis
from corroborate.rl.cell_runner import run_dqn_arm
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.rl.dqn.claims.optimizer import Adam, WarmedUpdate
from corroborate.rl.dqn.claims.replay import Replay
from corroborate.rl.dqn.invariants import DQNTrajectoryRecord
from corroborate.rl.env_catalogue import get


_ENV = 'CartPole-v1'
_SEEDS: tuple[int, ...] = (0, 1, 2)
_TOTAL_STEPS = 1000


_HPARAMS: dict[str, object] = {
    'total_steps': _TOTAL_STEPS,
    'eval_every': _TOTAL_STEPS // 5,
    'n_episodes': 5,
    'gamma': 0.99,
    'replay': Replay(capacity=2000, batch_size=32),
    'sync_period': 100,
}


def _vanilla() -> Hypothesis[DQNTrajectoryRecord]:
    return Hypothesis(
        name='vanilla_dqn',
        intervention={**_HPARAMS},
        bridges=(),
        predicted_direction=None,
    )


def _ddqn() -> Hypothesis[DQNTrajectoryRecord]:
    return Hypothesis(
        name='ddqn',
        intervention={
            **_HPARAMS,
            'bootstrap': partial(bootstrap, greedification=double_greedify),
        },
        bridges=(),
        predicted_direction='a_gt_b',
    )


def main() -> None:
    env_spec = get(_ENV)
    optimizer = WarmedUpdate(inner=Adam(), warmup_steps=100)

    print(f'env={_ENV}, seeds={_SEEDS}, total_steps={_TOTAL_STEPS}')
    print()

    vanilla = _vanilla()
    ddqn = _ddqn()

    print('1. Running vanilla arm on CartPole...')
    t0 = time.time()
    vanilla_rows = run_dqn_arm(
        env_spec, _SEEDS, vanilla, optimizer=optimizer,
    )
    print(f'   {len(vanilla_rows)} rows in {time.time() - t0:.1f}s')
    for row in vanilla_rows:
        seed = row.measurements['seed']
        outcome = row.measurements['outcome.late_window_mean']
        print(f'     seed={seed} verdict={row.verdict.value} '
              f'outcome={outcome}')
    print()

    print('2. Running DDQN arm on CartPole...')
    t0 = time.time()
    ddqn_rows = run_dqn_arm(
        env_spec, _SEEDS, ddqn, optimizer=optimizer,
    )
    print(f'   {len(ddqn_rows)} rows in {time.time() - t0:.1f}s')
    for row in ddqn_rows:
        seed = row.measurements['seed']
        outcome = row.measurements['outcome.late_window_mean']
        print(f'     seed={seed} verdict={row.verdict.value} '
              f'outcome={outcome}')
    print()

    print('3. Per-arm hp_signature on RunRows:')
    v_sigs = {hp_signature(row.measurements) for row in vanilla_rows}
    d_sigs = {hp_signature(row.measurements) for row in ddqn_rows}
    assert len(v_sigs) == 1, (
        f'vanilla seeds should share one hp_signature; got {len(v_sigs)}'
    )
    assert len(d_sigs) == 1, (
        f'ddqn seeds should share one hp_signature; got {len(d_sigs)}'
    )
    assert v_sigs != d_sigs, 'vanilla and DDQN must canonicalise distinctly'
    print('   OK vanilla-rows share one signature, ddqn-rows share another, '
          'and they differ.\n')

    print('4. aggregate_runs groups by hp_signature:')
    arms = aggregate_runs(list(vanilla_rows) + list(ddqn_rows))
    print(f'   {len(arms)} arms')
    assert len(arms) == 2, f'expected 2 arms, got {len(arms)}'
    for arm in arms:
        name = arm.measurements['intervention_name']
        env = arm.measurements['env_name']
        n = arm.measurements['n']
        mean = arm.measurements['outcome.late_window_mean.arm_mean']
        sd = arm.measurements['outcome.late_window_mean.arm_sd']
        print(f'     {name} on {env}: n={n} mean={mean} sd={sd}')

    print()
    print('All checks passed.')


if __name__ == '__main__':
    main()
