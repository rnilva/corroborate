"""Why does r_min change the sign of staleness's effect on DDQN
benefit?

Hypothesis: r_min sets the sign of the Q-trajectory regime, which
sets the direction of vanilla's overestimation bias.

Mechanism (sparse-terminal-positive, r_min = 0, e.g. FourRooms):
  - Reward only at terminal (e.g. +1 at goal, 0 elsewhere)
  - True Q* ∈ [0, R_max / (1−γ)] — POSITIVE, bounded above.
  - Vanilla's max-bootstrap: Q grows from 0 toward bounded-positive
    target. max-overestimation pushes Q ABOVE true value → policy
    becomes confidently wrong on non-goal-reaching actions →
    failure mode is "Q too high, policy degenerate".
  - DDQN's argmax/max separation removes this upward bias →
    bigger benefit when bias is bigger.
  - Staleness amplifies bias accumulation in target → DDQN's
    correction has more bite → POSITIVE ATE(stale → Δ_o).

Mechanism (dense-penalty, r_min = −1, e.g. Acrobot):
  - Per-step penalty −1 until terminal (terminal r=0).
  - True Q* ∈ [−1/(1−γ), 0] — NEGATIVE, bounded below.
  - Vanilla's max-bootstrap: Q starts at 0, descends toward
    bounded-negative true value. max-overestimation here means
    "less negative than truth" — preserves rank ordering of
    actions but inflates expected return → mild OPTIMISM that
    aids exploration through the long-horizon penalty floor.
  - DDQN's correction REMOVES this exploration optimism →
    sometimes hurts.
  - Staleness amplification of the optimism → vanilla's relative
    advantage grows → DDQN's deficit grows → NEGATIVE ATE.

The signature: vanilla's mean late-window Q value is POSITIVE in
sparse-terminal-positive envs and NEGATIVE in dense-penalty envs,
because Q* itself has those signs. We can verify by computing
`mean(online_max_q_per_step in late window)` per env per arm,
across the polyak τ sweep.

Output: per-env Q-regime panel + mechanism-narrative writeup.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
from pathlib import Path

import numpy as np
import polars as pl

import corroborate_rl.dqn.measurables  # register
from corroborate.runner.runner import _join_required_traces, _measurable_signature
from corroborate.corpus.measurements import build_measurements, load_measurements

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'


def _q_late_mean(arr: list[float] | None) -> float:
    if arr is None or len(arr) == 0:
        return float('nan')
    a = np.asarray(arr, dtype=np.float64)
    n = a.shape[0]
    if n == 0:
        return float('nan')
    lo = int(0.5 * n)
    return float(a[lo:].mean())


def main() -> None:
    base_dirs = [
        Path('experiments/data/polyak_tau_intervention'),
        Path('experiments/data/polyak_tau_asterix'),
    ]
    panels = []
    for base in base_dirs:
        for sub in sorted(base.iterdir()):
            if not sub.is_dir() or not (sub / 'runs.parquet').exists():
                continue
            traces = sub / 'traces.parquet'
            if not traces.exists():
                continue
            runs = pl.read_parquet(sub / 'runs.parquet')
            df = _join_required_traces(
                runs, traces,
                frozenset(['online_max_q_per_step']),
            )
            df_compact = df.select([
                'env_name', 'arm_key', 'seed', 'target_sync.tau',
                'online_max_q_per_step',
            ])
            panels.append(df_compact)

    df = pl.concat(panels, how='diagonal_relaxed')
    print(f'cells with traces: {df.height}', flush=True)

    rows = []
    for r in df.iter_rows(named=True):
        rows.append({
            'env': r['env_name'],
            'arm': 'baseline' if r['arm_key'] == 'baseline' else 'ddqn',
            'tau': r['target_sync.tau'],
            'q_late_mean': _q_late_mean(r['online_max_q_per_step']),
        })

    panel = pl.DataFrame(rows).filter(pl.col('q_late_mean').is_finite())

    print()
    print('=== Mean Q (late window) by (env, arm, τ) — Q-regime sign ===\n')
    agg = panel.group_by(['env', 'arm', 'tau']).agg(
        pl.col('q_late_mean').mean().alias('q_mean'),
        pl.col('q_late_mean').std().alias('q_std'),
        pl.len().alias('n'),
    ).sort(['env', 'arm', 'tau'])

    print(f'{"env":<22} {"arm":<10} {"τ":>7} {"Q̄_late":>10} {"σ_Q":>9} {"n":>4}')
    print('-' * 75)
    for row in agg.iter_rows(named=True):
        env = row['env']
        arm = row['arm']
        tau = row['tau']
        qm = row['q_mean']
        qs = row['q_std']
        n = row['n']
        print(f'{env:<22} {arm:<10} {tau:>7.3f} {qm:>+10.3f} {qs:>9.3f} {n:>4d}',
              flush=True)

    # Cross-env: vanilla's Q-regime sign at the lowest-τ (highest staleness)
    print()
    print('=== Q-regime SIGN per env (vanilla arm, low τ) ===\n')
    print(f'{"env":<22} {"r_min":>6} {"polarity":>9} {"mean Q̄ vanilla":>18} {"sign":<10}')
    print('-' * 80)

    # Hard-coded r_min from env catalogue (these are env-structural,
    # NOT cell-derived). Captures the structural regime.
    r_min_table = {
        'FourRooms-misc': 0,
        'Acrobot-v1': -1,
        'MountainCar-v0': -1,
        'Asterix-MinAtar': 0,
        'Breakout-MinAtar': 0,
    }
    polarity_table = {
        'FourRooms-misc': -0.92,
        'Acrobot-v1': -0.94,
        'MountainCar-v0': -1.00,
        'Asterix-MinAtar': +0.50,
        'Breakout-MinAtar': +0.99,
    }

    summary = []
    for env in sorted(panel['env'].unique()):
        envrows = panel.filter(
            (pl.col('env') == env) & (pl.col('arm') == 'baseline'),
        )
        if envrows.height == 0:
            continue
        q_mean = float(envrows['q_late_mean'].mean())
        sign = 'POSITIVE' if q_mean > 0.5 else (
            'NEGATIVE' if q_mean < -0.5 else 'near-zero'
        )
        rmin = r_min_table.get(env, '?')
        pol = polarity_table.get(env, float('nan'))
        print(f'{env:<22} {str(rmin):>6} {pol:>+9.3f} {q_mean:>+18.3f} {sign:<10}',
              flush=True)
        summary.append({
            'env': env, 'r_min': rmin, 'polarity': pol,
            'q_late_mean_vanilla': q_mean, 'q_regime_sign': sign,
        })

    print()
    print('=== Mechanism reading ===')
    print()
    print('  r_min ≥ 0 (sparse-terminal-positive):')
    print('    True Q* ∈ [0, R_max/(1−γ)] — POSITIVE bounded above.')
    print('    Vanilla overestimates → Q grows ABOVE true → policy degenerates.')
    print('    DDQN corrects → POSITIVE ATE(staleness → DDQN benefit).')
    print()
    print('  r_min < 0 (dense-penalty):')
    print('    True Q* ∈ [−|r_min|/(1−γ), 0] — NEGATIVE bounded below.')
    print('    Vanilla overestimates "upward" → less-negative Q → optimism aids exploration.')
    print('    DDQN removes optimism → sometimes hurts → NEGATIVE ATE.')

    out = Path('experiments/findings/sync_curve_breakout/rmin_q_regime_panel.json')
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
