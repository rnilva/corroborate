"""n-step return sweep — falsification probe for the bootstrap-bias-
compounding mechanism.

Hypothesis (Hasselt-2010 chain): DDQN's outcome benefit comes from
correcting the over-estimation that compounds along the bootstrap
chain. As n-step grows (1 → 10), the target shifts from full-
bootstrap (Hasselt floor maximal) toward Monte Carlo (no
bootstrap, no bias to correct). Predict: paired Δ on outcome
shrinks MONOTONICALLY with n.

`nstep_lambda_fourrooms` sweep: 5 n × 2 arms × 30 seeds × 200k
on FourRooms at γ=0.99.

A negative-prediction test — most of our prior corroborations
asked "where does Δ become large?". This asks "where SHOULD Δ
become small?". Confounds tend to add positive correlations,
not subtract them; if Δ → 0 cleanly as n grows, the bias-
compounding mechanism is the right story. If Δ stays nonzero
at high n, DDQN is doing something other than bias correction.

Also note: existing `nstep_intervention` 2×2 factorial showed
Δ shrinks 4× from n=1 → n=3. The wider sweep here gives the
full curve.

Usage:
    uv run python -m experiments.findings.nstep_lambda_fourrooms
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

import corroborate.rl.dqn.measurables as _m  # registers outcome_native
from corroborate.analyses.paired_g import paired_g

assert _m  # keep registration import live for type checker


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = (
    REPO_ROOT / 'experiments' / 'data' / 'nstep_lambda_fourrooms'
    / 'runs.parquet'
)


def main() -> None:
    if not RUNS.exists():
        print(f'(skip — {RUNS} missing)')
        return
    df = pl.read_parquet(RUNS)
    cells = list(df.iter_rows(named=True))

    print('# n-step sweep on FourRooms — bootstrap-bias-compounding falsification')
    print('=' * 90)
    print('Hypothesis: Δ ~ outcome shrinks monotonically as n grows '
          '(less bootstrap → less bias to fix).')
    print()
    print(f'{"n":>3}  {"vanilla outcome":>16}  {"ddqn outcome":>13}  {"Δ":>10}  {"Δ_se":>8}  {"p":>8}')
    print('-' * 80)

    n_values = sorted(df['n_step'].unique().to_list())
    for n in n_values:
        sub = [c for c in cells if c.get('n_step') == n]
        v_out = [
            c['outcome.eval_best_burst_mean']
            for c in sub
            if c.get('intervention_name') == f'vanilla_n{n}'
        ]
        d_out = [
            c['outcome.eval_best_burst_mean']
            for c in sub
            if c.get('intervention_name') == f'ddqn_n{n}'
        ]
        v_mean = sum(v_out) / len(v_out) if v_out else float('nan')
        d_mean = sum(d_out) / len(d_out) if d_out else float('nan')
        r = paired_g.fn(
            sub, treatment_arm=f'ddqn_n{n}', baseline_arm=f'vanilla_n{n}',
            pair_by=('seed',), source='outcome.eval_best_burst_mean',
        )
        print(
            f'{n:>3}  {v_mean:>+16.4f}  {d_mean:>+13.4f}  '
            f'{r.mean_diff:>+10.4f}  {r.mean_diff_se:>8.4f}  '
            f'{r.mean_diff_p_value:>8.4f}',
        )

    print()
    print('Reading: monotonic decline of Δ → 0 corroborates bootstrap-bias-')
    print('compounding as the mechanism. Non-monotonic or persistent Δ at')
    print('high n refutes it (DDQN must be doing something else).')


if __name__ == '__main__':
    main()
