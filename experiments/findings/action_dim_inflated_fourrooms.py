"""Action-dim inflation on FourRooms — does declared |A| drive the link?

The ddqn 200k corpus showed `|A| ∈ {3, 4}` predicts LINK_ACTIVE
with 92% accuracy across 13 non-saturated envs. But all our
|A|=5+ envs were confounded (Asterix dense MinAtar, DiscountingChain
bsuite); we couldn't tell if the upper boundary is real or a
missing-data artifact.

`action_dim_inflated_fourrooms` sweep: ActionDuplicate(k) wraps
FourRooms, mapping action `i ∈ [0, k*4)` to inner action `i % 4`.
Same dynamics, same optimal Q* — only declared |A| changes:
  k=1 → |A|=4   (baseline)
  k=2 → |A|=8
  k=3 → |A|=12
  k=4 → |A|=16

Three discriminating predictions:
  1. If `|A|∈{3,4}` is strict: link dies at k≥2, DDQN benefit
     collapses despite chain-amp env structure.
  2. If Hasselt floor monotone: DDQN benefit grows or saturates.
  3. If link saturates somewhere between |A|=4 and |A|=16: find
     the boundary.

These are mutually exclusive — 240 cells (4 × 2 × 30) resolve
between them.

Usage:
    uv run python -m experiments.findings.action_dim_inflated_fourrooms
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import polars as pl

import corroborate.rl.dqn.measurables as _m  # registers measurables
from corroborate.analyses.paired_g import paired_g

assert _m


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = (
    REPO_ROOT / 'experiments' / 'data' / 'action_dim_inflated_fourrooms'
    / 'runs.parquet'
)


def main() -> None:
    if not RUNS.exists():
        print(f'(skip — {RUNS} missing)')
        return
    df = pl.read_parquet(RUNS)
    cells = list(df.iter_rows(named=True))

    print('# Action-dim inflation on FourRooms')
    print('=' * 100)
    print('Hypothesis: if |A|∈{3,4} sweet spot is real (Hasselt floor active '
          'but bounded), DDQN benefit')
    print('saturates or shrinks as k grows. If just monotone in √(2 log|A|), '
          'DDQN benefit grows.')
    print()
    print(f'{"k":>3} {"|A|":>3} {"Hasselt floor":>14}  '
          f'{"vanilla":>10} {"ddqn":>10} {"Δ":>10} {"Δ_se":>8} {"p":>8}')
    print('-' * 100)

    k_values = sorted(df['action_duplicate_k'].unique().to_list())
    rows = []
    for k in k_values:
        sub = [c for c in cells if c.get('action_duplicate_k') == k]
        n_actions = int(4 * k)
        hasselt_floor = math.sqrt(2 * math.log(n_actions))
        v_out = [
            c['outcome.eval_best_burst_mean']
            for c in sub
            if c.get('intervention_name') == 'vanilla_dqn'
        ]
        d_out = [
            c['outcome.eval_best_burst_mean']
            for c in sub
            if c.get('intervention_name') == 'ddqn'
        ]
        v_mean = np.mean(v_out) if v_out else float('nan')
        d_mean = np.mean(d_out) if d_out else float('nan')
        r = paired_g.fn(
            sub, treatment_arm='ddqn', baseline_arm='vanilla_dqn',
            pair_by=('seed',), source='outcome.eval_best_burst_mean',
        )
        rows.append({'k': k, 'A': n_actions, 'floor': hasselt_floor,
                     'v': v_mean, 'd': d_mean,
                     'delta': r.mean_diff, 'se': r.mean_diff_se,
                     'p': r.mean_diff_p_value})
        print(f'{int(k):>3} {n_actions:>3} {hasselt_floor:>14.3f}  '
              f'{v_mean:>+10.4f} {d_mean:>+10.4f} '
              f'{r.mean_diff:>+10.4f} {r.se:>8.4f} '
              f'{r.mean_diff_p_value:>8.4f}'
              if hasattr(r, 'se') else
              f'{int(k):>3} {n_actions:>3} {hasselt_floor:>14.3f}  '
              f'{v_mean:>+10.4f} {d_mean:>+10.4f} '
              f'{r.mean_diff:>+10.4f} {r.mean_diff_se:>8.4f} '
              f'{r.mean_diff_p_value:>8.4f}')

    # Mechanism check: does jensen_gap also vary with k?
    print()
    print('# Mechanism: vanilla jensen_gap by k')
    print('=' * 70)
    print(f'{"k":>3} {"|A|":>3}  {"v_jens":>9} {"d_jens":>9} {"Δ_jens":>10} '
          f'{"p":>8}')
    print('-' * 70)
    for k in k_values:
        sub = [c for c in cells if c.get('action_duplicate_k') == k]
        v_jens = [c['mechanism.jensen_gap'] for c in sub
                  if c.get('intervention_name') == 'vanilla_dqn']
        d_jens = [c['mechanism.jensen_gap'] for c in sub
                  if c.get('intervention_name') == 'ddqn']
        v_m = np.mean(v_jens) if v_jens else float('nan')
        d_m = np.mean(d_jens) if d_jens else float('nan')
        r = paired_g.fn(
            sub, treatment_arm='ddqn', baseline_arm='vanilla_dqn',
            pair_by=('seed',), source='mechanism.jensen_gap',
        )
        n_actions = int(4 * k)
        print(f'{int(k):>3} {n_actions:>3}  {v_m:>9.4f} {d_m:>9.4f} '
              f'{r.mean_diff:>+10.4f} {r.mean_diff_p_value:>8.4f}')

    # Scope verdict
    print()
    print('# Verdict')
    print('=' * 70)
    if not rows:
        print('(no data)')
        return
    deltas = [r['delta'] for r in rows]
    if all(p < 0.05 and d > 0 for p, d in [(r['p'], r['delta']) for r in rows]):
        if abs(deltas[-1] - deltas[0]) / max(abs(deltas[0]), 0.01) < 0.3:
            verdict = 'SATURATED — Δ flat across k (mechanism not bottlenecked by |A|)'
        elif deltas[-1] > deltas[0]:
            verdict = 'MONOTONE INCREASING — DDQN benefit grows with |A| (Hasselt floor scales)'
        else:
            verdict = 'MONOTONE DECREASING — DDQN benefit shrinks with |A| (saturation past |A|=4)'
    else:
        first_null = next((i for i, r in enumerate(rows)
                          if r['p'] >= 0.05 or r['delta'] <= 0), None)
        if first_null is not None and first_null > 0:
            verdict = (
                f'BOUNDARY — link dies at |A|={rows[first_null]["A"]} '
                f'(k={int(rows[first_null]["k"])}); the |A|∈3-4 sweet spot is real'
            )
        else:
            verdict = 'AMBIGUOUS — see per-row p-values'
    print(f'  {verdict}')


if __name__ == '__main__':
    main()
