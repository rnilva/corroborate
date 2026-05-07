"""Why does NEGATIVE Q regime invert DDQN's effect?

The mechanism story we're testing:

    DDQN's update target = r + γ·Q_target(s', argmax_a Q_online(s', a))
    Vanilla's update target = r + γ·max_a Q_target(s', a)

    The difference: DDQN evaluates Q_target at the action online
    thinks is best (argmax_online), instead of the action target
    thinks is best (argmax_target). When online and target agree
    (low staleness), they're identical. When they disagree (high
    staleness), DDQN's bootstrap value ≤ vanilla's bootstrap value
    (because vanilla picks the argmax of target's Q, while DDQN
    can pick a non-max action under target's Q).

Empirical signature we can compute from existing traces:
  `argmax_disagreement_rate` = fraction of steps where
    online_argmax != target_argmax. Should grow with staleness.
  (Same across envs — argmax disagreement is policy-side, not
  outcome-side.)

The asymmetric outcome effect (DDQN helps in sparse-positive,
hurts in dense-penalty) must therefore come from how the
disagreement maps to outcomes, NOT from the disagreement rate
itself.

Tests:
1. Disagreement rate grows with staleness universally.
2. The CONSEQUENCE of disagreement (Q_target at argmax_online
   vs max(Q_target)) is sign-asymmetric across regimes.
3. Per-env: how does disagreement-rate × Q-regime-sign predict
   Δ_outcome?
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
from pathlib import Path

import numpy as np
import polars as pl

import corroborate_rl.dqn.measurables  # register

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'


def _disagreement_rate(
    online_argmax: list[int] | None,
    target_argmax: list[int] | None,
) -> float:
    if online_argmax is None or target_argmax is None:
        return float('nan')
    a = np.asarray(online_argmax)
    b = np.asarray(target_argmax)
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return float('nan')
    a, b = a[:n], b[:n]
    return float((a != b).mean())


def _disagreement_rate_late(
    online_argmax: list[int] | None,
    target_argmax: list[int] | None,
) -> float:
    """Late-50% disagreement rate — matches `target_staleness_late`
    window."""
    if online_argmax is None or target_argmax is None:
        return float('nan')
    a = np.asarray(online_argmax)
    b = np.asarray(target_argmax)
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return float('nan')
    lo = int(0.5 * n)
    a, b = a[lo:n], b[lo:n]
    if len(a) == 0:
        return float('nan')
    return float((a != b).mean())


def main() -> None:
    rows = []
    for sweep_dir in (
        'polyak_tau_intervention', 'polyak_tau_asterix',
    ):
        base = Path(f'experiments/data/{sweep_dir}')
        for sub in sorted(base.iterdir()):
            if not sub.is_dir() or not (sub / 'runs.parquet').exists():
                continue
            traces_path = sub / 'traces.parquet'
            if not traces_path.exists():
                continue
            runs = pl.read_parquet(sub / 'runs.parquet')
            traces = pl.read_parquet(traces_path, columns=[
                'id', 'online_argmax_per_step', 'target_argmax_per_step',
            ])
            df = runs.join(traces, on='id', how='inner')
            for cell in df.iter_rows(named=True):
                rows.append({
                    'env': cell['env_name'],
                    'arm': 'baseline' if cell['arm_key'] == 'baseline' else 'ddqn',
                    'tau': cell.get('target_sync.tau'),
                    'seed': cell.get('seed'),
                    'disagreement_late': _disagreement_rate_late(
                        cell.get('online_argmax_per_step'),
                        cell.get('target_argmax_per_step'),
                    ),
                })

    df = pl.DataFrame(rows).filter(pl.col('disagreement_late').is_finite())
    print(f'cells with disagreement: {df.height}', flush=True)

    print()
    print('=== argmax_disagreement_rate (late 50%) by (env, arm, τ) ===\n')
    print(f'{"env":<22} {"arm":<10} {"τ":>7} {"disagree_rate":>15}')
    print('-' * 60)
    agg = df.group_by(['env', 'arm', 'tau']).agg(
        pl.col('disagreement_late').mean().alias('mean'),
        pl.len().alias('n'),
    ).sort(['env', 'arm', 'tau'])
    for r in agg.iter_rows(named=True):
        print(f'{r["env"]:<22} {r["arm"]:<10} {r["tau"]:>7.3f} {r["mean"]:>15.4f}')

    print()
    print('=== Test 1: disagreement_rate ↑ with staleness (τ ↓)? ===')
    print()
    for env in sorted(df['env'].unique()):
        sub = df.filter(pl.col('env') == env)
        # Spearman ρ(tau, disagree_rate) — should be NEGATIVE
        # (more τ ⇒ less staleness ⇒ less disagreement).
        from scipy.stats import spearmanr
        if sub.height >= 10:
            rho, p = spearmanr(sub['tau'].to_numpy(),
                               sub['disagreement_late'].to_numpy())
            print(f'  {env:<22}: ρ(τ, disagree_late) = {rho:+.4f} (p={p:.3g}, n={sub.height})')

    print()
    print('Reading: ρ < 0 across all envs would confirm "staleness universally drives')
    print('argmax disagreement". The asymmetric outcome effect (DDQN helps in')
    print('sparse-positive, hurts in dense-penalty) is THEN a consequence of how')
    print('disagreement maps to outcome, not of disagreement rate itself.')

    out = Path(
        'experiments/findings/sync_curve_breakout/disagreement_panel.json'
    )
    out.write_text(json.dumps(rows, indent=2, default=str))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
