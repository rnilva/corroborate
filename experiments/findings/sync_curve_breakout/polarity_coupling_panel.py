"""Save the per-env polarity coupling panel as JSON + check whether
|polarity| predicts |coupling r| across envs (deeper test of the
polarity-as-moderator story)."""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

CACHE_PATH = Path('experiments/data/cache/ddqn_universe.parquet')
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'


def main() -> None:
    cache = pl.read_parquet(CACHE_PATH)
    print(f'cache: {len(cache)} rows')

    panel = []
    for env in sorted(cache.filter(pl.col('env_name').is_not_null())['env_name'].unique()):
        sub = cache.filter(pl.col('env_name') == env)
        v = sub.filter((pl.col('arm_key') == 'baseline') & pl.col('effective_horizon').is_finite() & pl.col('eval_best_burst_mean').is_finite())
        d = sub.filter((pl.col('arm_key') == DDQN) & pl.col('effective_horizon').is_finite() & pl.col('eval_best_burst_mean').is_finite())

        v_with_pol = v.filter(pl.col('env_reward_polarity').is_finite())
        if len(v_with_pol) == 0:
            continue
        env_pol = float(v_with_pol['env_reward_polarity'].mean())

        pair_keys = ['corpus', 'gamma', 'sync_period', 'total_steps', 'seed']
        v_p = v.select(pair_keys + ['effective_horizon', 'eval_best_burst_mean']).rename({'effective_horizon': 'eh_v', 'eval_best_burst_mean': 'o_v'})
        d_p = d.select(pair_keys + ['effective_horizon', 'eval_best_burst_mean']).rename({'effective_horizon': 'eh_d', 'eval_best_burst_mean': 'o_d'})
        j = v_p.join(d_p, on=pair_keys, how='inner').filter(
            pl.col('eh_v').is_not_nan() & pl.col('eh_d').is_not_nan() & pl.col('o_v').is_not_nan() & pl.col('o_d').is_not_nan()
        )
        if len(j) < 5:
            continue
        delta_eh = (j['eh_d'] - j['eh_v']).to_numpy()
        delta_o = (j['o_d'] - j['o_v']).to_numpy()
        if delta_eh.std() == 0 or delta_o.std() == 0:
            continue
        r = float(np.corrcoef(delta_eh, delta_o)[0, 1])
        slope, _ = np.polyfit(delta_eh, delta_o, 1)
        panel.append({
            'env': env,
            'polarity': env_pol,
            'n_pairs': len(j),
            'r': r,
            'slope': float(slope),
            'mean_d_eh': float(delta_eh.mean()),
            'mean_d_o': float(delta_o.mean()),
            'sd_d_eh': float(delta_eh.std()),
            'sd_d_o': float(delta_o.std()),
        })

    print()
    print(f'{"env":<28} {"polarity":>10} {"|pol|":>6} {"n":>5} {"r":>7} {"|r|":>6} {"slope":>9} {"|Δo|":>7} {"|Δeh|":>9}')
    print('-' * 110)
    for p in sorted(panel, key=lambda x: -abs(x['polarity'])):
        print(f'{p["env"]:<28} {p["polarity"]:>+10.3f} {abs(p["polarity"]):>6.3f} {p["n_pairs"]:>5d} {p["r"]:>+7.3f} {abs(p["r"]):>6.3f} {p["slope"]:>+9.3g} {abs(p["mean_d_o"]):>7.3f} {abs(p["mean_d_eh"]):>9.3g}')

    print()
    print('=== Cross-env tests ===')
    pols = np.array([p['polarity'] for p in panel])
    rs = np.array([p['r'] for p in panel])
    n_envs = len(panel)
    print(f'n_envs: {n_envs}')

    # polarity sign vs r sign — match check
    sign_match = sum(1 for p in panel if np.sign(p['polarity']) == np.sign(p['r']) or abs(p['polarity']) < 0.1)
    print(f'sign(polarity) == sign(r): {sign_match}/{n_envs} envs (excludes |pol|<0.1 from check)')

    # Spearman: |polarity| vs |r| — is the moderation strength correlated?
    abs_pols = np.abs(pols)
    abs_rs = np.abs(rs)
    rho, p = spearmanr(abs_pols, abs_rs)
    print(f'Spearman ρ(|polarity|, |r|) = {rho:+.3f}, p = {p:.3g}')

    # Spearman: polarity vs r (signed) — should be highly positive
    rho_signed, p_signed = spearmanr(pols, rs)
    print(f'Spearman ρ(polarity, r) (signed) = {rho_signed:+.3f}, p = {p_signed:.3g}')

    # Save
    out = Path('experiments/findings/sync_curve_breakout/polarity_coupling_panel.json')
    out.write_text(json.dumps({'per_env': panel, 'cross_env': {
        'spearman_abs_pol_abs_r': {'rho': rho, 'p': p},
        'spearman_signed_pol_r': {'rho': rho_signed, 'p': p_signed},
        'sign_match': f'{sign_match}/{n_envs}',
    }}, indent=2))
    print(f'\nwrote: {out}')


if __name__ == '__main__':
    main()
