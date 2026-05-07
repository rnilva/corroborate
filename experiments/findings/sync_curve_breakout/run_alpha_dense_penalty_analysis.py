"""Within-env α-sweep analysis on dense-penalty envs (Acrobot,
MountainCar) at default config (γ=0.99, sync=100).

The α sweep directly varies bias-correction strength without
varying τ — the cleanest within-env test of mechanism direction.
This is the dense-penalty companion to `dampened_alpha_envs`
(sparse-positive: FourRooms + DeepSea + DiscountingChain +
MNISTBandit at rs=0.1).

Three-tier test:
1. **Mechanism**: α → jensen_gap (Hasselt's premise firing).
2. **Link**: α → eval_best_burst_mean (does mech translate?).
3. **Joint diagnosis**: per env classify as
   - mech HELD + link HELD: the canonical chain
   - mech HELD + link NULL: bias correction works but doesn't
     reach outcome (already-documented `mech HELD ↛ link HELD`
     pattern, see `findings_dampened_alpha_envs` on
     DiscountingChain)
   - mech NULL: Hasselt's premise dormant — DDQN's correction
     has nothing to bite (already-documented per
     `ddqn_refuted_when_dormancy_fires` bridge,
     `findings_l2_acrobot_goldilocks`)
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import t as t_dist


def _slope_test(
    a: np.ndarray, y: np.ndarray,
) -> dict[str, float] | None:
    mask = np.isfinite(a) & np.isfinite(y)
    if mask.sum() < 10:
        return None
    a, y = a[mask], y[mask]
    if a.std() == 0:
        return None
    slope = float(np.cov(a, y, ddof=1)[0, 1]) / float(np.var(a, ddof=1))
    pred = slope * a + (y.mean() - slope * a.mean())
    resid = y - pred
    n = len(a)
    sigma_r = math.sqrt(float((resid ** 2).sum() / max(n - 2, 1)))
    ss_x = float(((a - a.mean()) ** 2).sum())
    se = sigma_r / math.sqrt(ss_x)
    t_stat = slope / se
    p_val = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 2))
    t_crit = float(t_dist.ppf(0.975, df=n - 2))
    return {
        'n': n, 'slope': slope, 'se': se, 't': t_stat,
        'p': p_val, 'ci_lo': slope - t_crit * se,
        'ci_hi': slope + t_crit * se,
    }


def main() -> None:
    r = pl.read_parquet(
        'experiments/data/dampened_alpha_dense_penalty/runs.parquet',
    )
    r = r.filter(pl.col('arm_key') != 'baseline')
    alpha_col = 'bootstrap.greedification.alpha'
    print(f'cells: {r.height}')

    print()
    print('=== α → jensen_gap (MECHANISM) ===\n')
    print(f'{"env":<22} {"slope":>10} {"se":>9} {"t":>6} {"p":>9} {"95% CI":>22}')
    print('-' * 80)
    out_panel = []
    for env in sorted(r['env_name'].unique()):
        sub = r.filter(pl.col('env_name') == env)
        res = _slope_test(
            sub[alpha_col].to_numpy(),
            sub['jensen_gap'].to_numpy(),
        )
        if res is None:
            continue
        verdict = (
            'MECH HELD' if (res['t'] < -2.0 and res['p'] < 0.05)
            else 'MECH NULL'
        )
        print(
            f'{env:<22} {res["slope"]:>+10.4f} {res["se"]:>9.4f} '
            f'{res["t"]:>+6.2f} {res["p"]:>9.4g} '
            f'[{res["ci_lo"]:>+7.3f},{res["ci_hi"]:>+7.3f}]  {verdict}'
        )
        out_panel.append({'env': env, 'metric': 'jensen_gap', **res, 'verdict': verdict})

    print()
    print('=== α → outcome (LINK) ===\n')
    print(f'{"env":<22} {"slope":>10} {"se":>9} {"t":>6} {"p":>9} {"95% CI":>22}')
    print('-' * 80)
    for env in sorted(r['env_name'].unique()):
        sub = r.filter(pl.col('env_name') == env)
        res = _slope_test(
            sub[alpha_col].to_numpy(),
            sub['eval_best_burst_mean'].to_numpy(),
        )
        if res is None:
            continue
        verdict = (
            'LINK +' if (res['t'] > 2.0 and res['p'] < 0.05)
            else 'LINK NULL'
        )
        print(
            f'{env:<22} {res["slope"]:>+10.4f} {res["se"]:>9.4f} '
            f'{res["t"]:>+6.2f} {res["p"]:>9.4g} '
            f'[{res["ci_lo"]:>+7.3f},{res["ci_hi"]:>+7.3f}]  {verdict}'
        )
        out_panel.append({'env': env, 'metric': 'outcome', **res, 'verdict': verdict})

    print()
    print('=== Per-α jensen_gap means (mechanism monotonicity) ===\n')
    agg = r.group_by(['env_name', alpha_col]).agg(
        pl.col('jensen_gap').mean().alias('jens_mean'),
        pl.col('jensen_gap').std().alias('jens_std'),
        pl.col('eval_best_burst_mean').mean().alias('out_mean'),
        pl.col('eval_best_burst_mean').std().alias('out_std'),
    ).sort(['env_name', alpha_col])
    for row in agg.iter_rows(named=True):
        print(
            f'  {row["env_name"]:<22} α={row[alpha_col]:>5.2f}  '
            f'jens={row["jens_mean"]:>+8.3f} ± {row["jens_std"]:>6.3f}  '
            f'out={row["out_mean"]:>+10.3f} ± {row["out_std"]:>7.3f}'
        )

    out = Path(
        'experiments/findings/sync_curve_breakout/'
        'alpha_dense_penalty_panel.json'
    )
    out.write_text(json.dumps(out_panel, indent=2, default=str))
    print(f'\nwrote: {out}')


if __name__ == '__main__':
    main()
