"""Per-env DoWhy mediation with refutations.

For each env, identifies the ATE of do(DDQN) on outcome via:
  - direct backdoor regression (no mediator adjustment)
  - backdoor with mediator as adjustment (direct effect — what
    survives after accounting for the mediator's path)

Then refutes both with:
  - placebo treatment (should yield ATE ≈ 0)
  - random common cause (ATE should be invariant to synthetic
    confounder)

Verdict per (env, mediator):
  HELD-as-mediator: ATE_with_mediator_adjustment is materially
    smaller in magnitude than ATE_marginal (the mediator absorbs
    the effect)
  REFUTED: placebo gives non-zero ATE (model is wrong) or
    RCC gives high drift (estimate is unstable)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'src'))
sys.path.insert(0, str(REPO / 'src/corroborate_rl'))

from corroborate.analyses.dowhy import (
    backdoor_ate, placebo_refutation, random_common_cause_refutation,
)


def main() -> int:
    df = pl.read_parquet(REPO / 'experiments/data/cache/ddqn.parquet').filter(
        ~pl.col('env_name').str.contains('bsuite')
    )
    learning = [
        'Acrobot-v1', 'Asterix-MinAtar', 'Breakout-MinAtar',
        'Freeway-MinAtar', 'MetaMaze-misc', 'MountainCar-v0',
        'PacMan-jumanji', 'SpaceInvaders-MinAtar',
    ]
    mediators = ['jensen_gap', 'argmax_entropy_late']

    arms = df['arm_key'].unique().to_list()
    ddqn = [a for a in arms if 'double_greedify' in a][0]

    # Need a binary 'is_ddqn' column for DoWhy treatment
    df = df.with_columns(
        (pl.col('arm_key') == ddqn).cast(pl.Float64).alias('is_ddqn')
    )

    print(f'{"env":<20} {"mediator":<22} {"ATE_marg":>9} {"ATE|M":>8} {"absorbed":>10} {"placebo_ATE":>13} {"RCC_drift":>10}')
    print('-' * 102)

    for env in learning:
        sub = df.filter(pl.col('env_name') == env)
        cells = sub.to_dicts()
        n = len(cells)
        if n < 10:
            continue

        # Marginal ATE: just treatment → outcome
        dag_marg: list[tuple[str, str]] = [
            ('is_ddqn', 'eval_best_burst_raw_mean'),
        ]
        try:
            r_marg = backdoor_ate.fn(
                cells, treatment='is_ddqn',
                outcome='eval_best_burst_raw_mean', dag=dag_marg,
            )
        except Exception as exc:
            print(f'{env:<20}  marg FAIL: {str(exc)[:50]}')
            continue

        # Placebo refutation on the marginal model
        try:
            placebo = placebo_refutation.fn(
                cells, treatment='is_ddqn',
                outcome='eval_best_burst_raw_mean', dag=dag_marg,
            )
        except Exception:
            placebo = None
        try:
            rcc = random_common_cause_refutation.fn(
                cells, treatment='is_ddqn',
                outcome='eval_best_burst_raw_mean', dag=dag_marg,
            )
        except Exception:
            rcc = None

        for med in mediators:
            # ATE conditioning on mediator: mediator as backdoor adjustment
            # DAG: T → Y; M → Y; M → T (mediator on path AND confounder
            # surface). DoWhy's backdoor adjustment will condition on M.
            dag_med: list[tuple[str, str]] = [
                ('is_ddqn', 'eval_best_burst_raw_mean'),
                (med, 'eval_best_burst_raw_mean'),
                (med, 'is_ddqn'),  # forces M into backdoor set
            ]
            try:
                r_med = backdoor_ate.fn(
                    cells, treatment='is_ddqn',
                    outcome='eval_best_burst_raw_mean', dag=dag_med,
                )
                ate_med = r_med.ate
            except Exception:
                ate_med = float('nan')

            absorbed = (1 - ate_med / r_marg.ate) * 100 if r_marg.ate != 0 else float('nan')
            placebo_ate = placebo.refuted_ate if placebo else float('nan')
            rcc_drift = rcc.drift if rcc else float('nan')

            print(
                f'{env:<20} {med:<22} {r_marg.ate:>+9.3f} {ate_med:>+8.3f} '
                f'{absorbed:>9.0f}% {placebo_ate:>+13.4f} {rcc_drift:>+10.4f}'
            )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
