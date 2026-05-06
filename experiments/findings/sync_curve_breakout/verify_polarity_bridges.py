"""Verify the polarity bridges in ddqn_universe.py after cache backfill.

Checks:
1. Cache has finite env_reward_polarity / effective_horizon for sufficient cells.
2. Per-env polarity classification matches the formal proof (8 envs).
3. Run the polarity bridges via run_hypothesis-equivalent and report verdicts.
4. Cross-tabulate slope_y_on_m for goal vs survival scopes.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import sys
from pathlib import Path

import polars as pl

from corroborate.runner.runner import run

CACHE_PATH = Path('experiments/data/cache/ddqn_universe.parquet')


def main() -> None:
    cache = pl.read_parquet(CACHE_PATH)
    print(f'cache: {len(cache)} rows × {len(cache.columns)} cols')

    n_pol = cache.filter(pl.col('env_reward_polarity').is_finite()).height
    n_eh = cache.filter(pl.col('effective_horizon').is_finite()).height
    print(f'finite polarity: {n_pol}/{len(cache)}')
    print(f'finite eff_h:    {n_eh}/{len(cache)}')

    # --- Per-env polarity (vanilla baseline, vs formal proof's 8 envs) ---
    print()
    print('=== Per-env polarity (vanilla baseline cells, mean ± std) ===')
    sub = cache.filter(
        (pl.col('arm_key') == 'baseline')
        & pl.col('env_reward_polarity').is_finite()
    )
    panel = sub.group_by('env_name').agg([
        pl.col('env_reward_polarity').mean().alias('pol_mean'),
        pl.col('env_reward_polarity').std().alias('pol_std'),
        pl.len().alias('n')
    ]).sort('pol_mean')
    print(panel.to_pandas().to_string())

    # --- Polarity classification expected vs measured ---
    print()
    print('=== Classification (expected ↔ measured) ===')
    expected = {
        'Acrobot-v1': 'GOAL',
        'FourRooms-misc': 'GOAL',
        'MountainCar-v0': 'GOAL',
        'DiscountingChain-bsuite': 'GOAL',
        'CartPole-v1': 'SURVIVAL',
        'Breakout-MinAtar': 'SURVIVAL',
        'SpaceInvaders-MinAtar': 'SURVIVAL',
        'Asterix-MinAtar': 'SURVIVAL',
    }
    panel_dict = dict(zip(panel['env_name'].to_list(), panel['pol_mean'].to_list()))
    n_match = 0
    n_total = 0
    for env, exp_class in expected.items():
        if env not in panel_dict:
            print(f'  {env:<28} NO DATA (expected {exp_class})')
            continue
        n_total += 1
        pol = panel_dict[env]
        if exp_class == 'GOAL':
            measured = 'GOAL' if pol < -0.3 else ('SURVIVAL' if pol > 0.3 else 'NEUTRAL')
        else:
            measured = 'SURVIVAL' if pol > 0.3 else ('GOAL' if pol < -0.3 else 'NEUTRAL')
        match = 'OK ' if measured == exp_class else 'X  '
        if measured == exp_class:
            n_match += 1
        print(f'  {match} {env:<28} pol={pol:+.3f} → {measured} (expected {exp_class})')
    print(f'  match rate: {n_match}/{n_total}')

    # --- Run the polarity bridges via the runner ---
    print()
    print('=== Polarity bridges via runner ===')
    results = run('experiments.findings.ddqn_universe', use_cache=True, write_cache=False, restore_from_cloud=False)
    polarity_bridge_names = [
        'eff_h_mediates_g_link__goal_envs',
        'eff_h_mediates_g_link__survival_envs',
    ]
    for name in polarity_bridge_names:
        ev = results.get(name)
        if ev is None:
            print(f'  {name}: NOT IN RESULTS')
            continue
        print(f'  {name}:')
        print(f'    verdict: {ev.verdict.value}')
        for k, ar in ev.analysis_results.items():
            print(f'    {type(ar).__name__}:')
            for attr in ('proportion', 'total', 'direct', 'indirect', 'slope_y_on_m', 'in_unit_interval', 'n_pairs'):
                v = getattr(ar, attr, None)
                if v is not None:
                    print(f'      {attr}: {v}')

    # --- Side analysis: slope_y_on_m within polarity-stratified subsets ---
    print()
    print('=== slope_y_on_m by env (vanilla vs DDQN, eff_h mediator) ===')
    DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'

    for env in sorted(panel_dict, key=panel_dict.get):
        env_pol = panel_dict[env]
        env_class = 'GOAL' if env_pol < -0.3 else ('SURVIVAL' if env_pol > 0.3 else 'NEUTRAL')
        sub = cache.filter(pl.col('env_name') == env)

        v = sub.filter((pl.col('arm_key') == 'baseline') & pl.col('effective_horizon').is_finite() & pl.col('eval_best_burst_mean').is_finite())
        d = sub.filter((pl.col('arm_key') == DDQN) & pl.col('effective_horizon').is_finite() & pl.col('eval_best_burst_mean').is_finite())

        pair_keys = ['corpus', 'gamma', 'sync_period', 'total_steps', 'seed']
        v_p = v.select(pair_keys + ['effective_horizon', 'eval_best_burst_mean']).rename({'effective_horizon': 'eh_v', 'eval_best_burst_mean': 'o_v'})
        d_p = d.select(pair_keys + ['effective_horizon', 'eval_best_burst_mean']).rename({'effective_horizon': 'eh_d', 'eval_best_burst_mean': 'o_d'})
        j = v_p.join(d_p, on=pair_keys, how='inner').filter(
            pl.col('eh_v').is_not_nan() & pl.col('eh_d').is_not_nan() & pl.col('o_v').is_not_nan() & pl.col('o_d').is_not_nan()
        )
        if len(j) < 3:
            continue
        delta_eh = (j['eh_d'] - j['eh_v']).to_numpy()
        delta_o = (j['o_d'] - j['o_v']).to_numpy()
        if delta_eh.std() == 0 or delta_o.std() == 0:
            continue
        import numpy as np
        slope, intercept = np.polyfit(delta_eh, delta_o, 1)
        r = np.corrcoef(delta_eh, delta_o)[0, 1]
        print(f'  {env_class:<8} {env:<28} pol={env_pol:+.3f}  n_pairs={len(j):>4}  slope={slope:+.3g}  r={r:+.3f}  '
              f'mean Δ_eh={delta_eh.mean():+.2f}  mean Δ_o={delta_o.mean():+.2f}')


if __name__ == '__main__':
    main()
