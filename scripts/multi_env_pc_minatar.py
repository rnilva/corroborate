"""Cross-env PC discovery + Δ analysis on 4 MinAtar envs × 2 γ.

Tests cross-env stability of:
1. DDQN's arm-effect on within-episode repeat-rate
2. The PC-selected mediator for arm→outcome edge
3. Within-arm ρ(repeat, outcome) (loops universally bad?)

Tests the cross-env claim: DDQN's Δ_repeat predicts DDQN's
Δ_outcome.
"""
from __future__ import annotations

import numpy as np
import polars as pl
import scipy.stats as st
import experiments.findings.ddqn_three_conditions  # populate registries

from corroborate.graph.discovery import discover_adjacency, _spearman_marginal


_VARS = (
    'arm',
    'jensen_gap',
    'policy_growth_fraction',
    'state_repeat_rate_within_episode_window64_late',
    'state_hash_entropy_late',
    'eval_best_burst_raw_mean',
)


def _prepare(df: pl.DataFrame, env: str, gamma: float) -> pl.DataFrame:
    scope = (pl.col('env_name') == env) & (pl.col('gamma') == gamma)
    cells = df.filter(scope).with_columns(
        (pl.col('arm_key') != 'baseline').cast(pl.Float64).alias('arm'),
    )
    for v in _VARS[1:]:
        cells = cells.filter(pl.col(v).is_finite())
    return cells


def main() -> None:
    df = pl.read_parquet('experiments/data/cache/ddqn_three_conditions.parquet')
    envs = ['Asterix-MinAtar', 'Breakout-MinAtar', 'Freeway-MinAtar', 'SpaceInvaders-MinAtar']
    gammas = [0.99, 0.999]
    print('=== Cross-env arm-effect on key measurables ===')
    print()
    print(f'{"env":25s} {"γ":6s} {"n":4s} {"Δ_repeat":>10s} {"d_rep":>7s} {"Δ_outcome":>10s} {"d_out":>7s} {"Δ_jens":>10s} {"d_jens":>7s}')
    print('-' * 90)
    rows = []
    for env in envs:
        for g in gammas:
            cells = _prepare(df, env, g)
            if len(cells) < 20: continue
            v = cells.filter(pl.col('arm')==0.0)
            d = cells.filter(pl.col('arm')==1.0)
            def diff(col):
                vn = v[col].to_numpy()
                dn = d[col].to_numpy()
                d_eff = (dn.mean()-vn.mean())/np.sqrt((dn.var(ddof=1)+vn.var(ddof=1))/2)
                return dn.mean()-vn.mean(), d_eff
            d_rep, d_rep_d = diff('state_repeat_rate_within_episode_window64_late')
            d_out, d_out_d = diff('eval_best_burst_raw_mean')
            d_jens, d_jens_d = diff('jensen_gap')
            row = {
                'env': env.replace('-MinAtar', ''),
                'gamma': g,
                'n': len(cells),
                'd_repeat': d_rep, 'd_repeat_cohen': d_rep_d,
                'd_outcome': d_out, 'd_outcome_cohen': d_out_d,
                'd_jens': d_jens, 'd_jens_cohen': d_jens_d,
            }
            rows.append(row)
            print(f'{row["env"]:25s} {g:>6} {len(cells):>4} {d_rep:>10.4f} {d_rep_d:>+7.2f} {d_out:>10.4f} {d_out_d:>+7.2f} {d_jens:>10.2f} {d_jens_d:>+7.2f}')
    print()
    print('=== Cross-env correlation: DDQN Δ_repeat vs DDQN Δ_outcome ===')
    drep = np.array([r['d_repeat'] for r in rows])
    dout = np.array([r['d_outcome'] for r in rows])
    if len(drep) >= 3:
        rho_s, p_s = st.spearmanr(drep, dout)
        rho_p, p_p = st.pearsonr(drep, dout)
        print(f'Spearman ρ(Δ_repeat, Δ_outcome) = {rho_s:+.3f}, p={p_s:.4g}, n={len(drep)}')
        print(f'Pearson  r(Δ_repeat, Δ_outcome) = {rho_p:+.3f}, p={p_p:.4g}')
    print()
    print('=== Within-vanilla ρ(repeat, outcome) per env (loops universally bad?) ===')
    print(f'{"env":25s} {"γ":6s} {"vanilla ρ":>14s} {"DDQN ρ":>14s}')
    print('-' * 60)
    for env in envs:
        for g in gammas:
            cells = _prepare(df, env, g)
            if len(cells) < 20: continue
            v = cells.filter(pl.col('arm')==0.0)
            d = cells.filter(pl.col('arm')==1.0)
            v_r = v['state_repeat_rate_within_episode_window64_late'].to_numpy()
            v_o = v['eval_best_burst_raw_mean'].to_numpy()
            d_r = d['state_repeat_rate_within_episode_window64_late'].to_numpy()
            d_o = d['eval_best_burst_raw_mean'].to_numpy()
            rho_v, p_v = _spearman_marginal(v_r, v_o)
            rho_d, p_d = _spearman_marginal(d_r, d_o)
            print(f'{env.replace("-MinAtar",""):25s} {g:>6} {rho_v:>+6.3f} (p={p_v:.2g})  {rho_d:>+6.3f} (p={p_d:.2g})')
    print()
    print('=== Per-env PC discovery (depth=2, α=0.05) — arm→outcome separator ===')
    print(f'{"env":25s} {"γ":6s} {"PC separator for arm→outcome edge":40s}')
    print('-' * 80)
    for env in envs:
        for g in gammas:
            cells = _prepare(df, env, g)
            if len(cells) < 20: continue
            adj = discover_adjacency(cells, variables=_VARS, max_conditioning=2, alpha=0.05)
            arm_outcome = frozenset({'arm', 'eval_best_burst_raw_mean'})
            if arm_outcome in adj.edges:
                msg = 'SURVIVED (PC could not separate)'
            else:
                sep_sets = adj.separating_sets.get(arm_outcome, frozenset())
                msg = ' | '.join(
                    '{' + ', '.join(s.replace('state_repeat_rate_within_episode_window64_late', 'repeat')
                                     .replace('state_hash_entropy_late', 'entropy')
                                     .replace('jensen_gap', 'jens')
                                     .replace('policy_growth_fraction', 'growth')
                                    for s in sorted(seps)) + '}'
                    for seps in sep_sets
                ) if sep_sets else '(removed marginally?)'
            print(f'{env.replace("-MinAtar",""):25s} {g:>6}  {msg}')


if __name__ == '__main__':
    main()
