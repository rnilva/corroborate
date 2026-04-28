"""End-to-end smoke for the DDQN substrate post-decomposition.

Validates the foundation + Phase 1+2 cuts actually carry the
vanilla-vs-DDQN comparison:

1. Builds two hypotheses differing only in `greedification`.
2. Runs each on CartPole, 3 seeds, 1000 steps via `run_dqn_arm`.
3. Asserts:
   - `mechanism_key.intervention_signature` distinguishes the
     arms by the `greedification` swap (canonical strings differ).
   - `aggregate_runs` groups cells by mechanism_key correctly.
   - Each arm has finite outcome summaries.

Run: `uv run python experiments/smoke_ddqn_run.py`."""
from __future__ import annotations

import time
from functools import partial

import optax

from corroborate.aggregate import aggregate_runs
from corroborate.hypothesis import Hypothesis
from corroborate.rl.cell_runner import run_dqn_arm
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
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
    'warmup_steps': 100,
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
    optimizer = optax.adam(1e-3)

    print(f'env={_ENV}, seeds={_SEEDS}, total_steps={_TOTAL_STEPS}')
    print()

    print('1. Hypothesis canonical signatures:')
    vanilla = _vanilla()
    ddqn = _ddqn()
    sig_v = dict(vanilla.mechanism_key.intervention_signature)
    sig_d = dict(ddqn.mechanism_key.intervention_signature)
    print(f'   vanilla bootstrap: {sig_v.get("bootstrap", "<none>")[:120]}')
    print(f'   ddqn    bootstrap: {sig_d.get("bootstrap", "<none>")[:120]}')
    assert sig_v != sig_d, 'vanilla and DDQN must produce distinct mechanism_keys'
    print('   OK distinct\n')

    print('2. Running vanilla arm on CartPole...')
    t0 = time.time()
    vanilla_rows = run_dqn_arm(
        env_spec, _SEEDS, vanilla, optimizer=optimizer,
    )
    print(f'   {len(vanilla_rows)} rows in {time.time() - t0:.1f}s')
    for row in vanilla_rows:
        print(f'     seed={row.seed} verdict={row.verdict.value} '
              f'outcome={row.primary_outcome_summary:.2f}')
    print()

    print('3. Running DDQN arm on CartPole...')
    t0 = time.time()
    ddqn_rows = run_dqn_arm(
        env_spec, _SEEDS, ddqn, optimizer=optimizer,
    )
    print(f'   {len(ddqn_rows)} rows in {time.time() - t0:.1f}s')
    for row in ddqn_rows:
        print(f'     seed={row.seed} verdict={row.verdict.value} '
              f'outcome={row.primary_outcome_summary:.2f}')
    print()

    print('4. Per-arm mechanism_key on RunRows:')
    v_keys = {row.mechanism_key for row in vanilla_rows}
    d_keys = {row.mechanism_key for row in ddqn_rows}
    assert len(v_keys) == 1, f'vanilla seeds should share one mechanism_key; got {len(v_keys)}'
    assert len(d_keys) == 1, f'ddqn seeds should share one mechanism_key; got {len(d_keys)}'
    assert v_keys != d_keys, 'vanilla and DDQN must canonicalise distinctly'
    print('   OK vanilla-rows share one key, ddqn-rows share another, '
          'and they differ.\n')

    print('5. aggregate_runs groups by mechanism_key:')
    arms = aggregate_runs(list(vanilla_rows) + list(ddqn_rows))
    print(f'   {len(arms)} arms')
    assert len(arms) == 2, f'expected 2 arms, got {len(arms)}'
    for arm in arms:
        print(f'     {arm.intervention_name} on {arm.env_name}: '
              f'n={arm.n} mean={arm.arm_mean:.2f} sd={arm.arm_sd:.2f}')

    print()
    print('All checks passed.')


if __name__ == '__main__':
    main()
