"""Does polarity moderate the DDQN outcome BENEFIT (not just the
eff_h mediator coupling)? If polarity is a true causal moderator,
we should see DDQN's outcome advantage vary with polarity.

Tests:
1. Per-env Δ_outcome (DDQN − vanilla) vs polarity — does outcome
   benefit polarity-stratify?
2. Per-env Δ_outcome vs |polarity| — does benefit scale with
   polarity strength?
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

CACHE_PATH = Path('experiments/data/cache/ddqn.parquet')
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'


def main() -> None:
    cache = pl.read_parquet(CACHE_PATH)

    panel = []
    for env in sorted(cache.filter(pl.col('env_name').is_not_null())['env_name'].unique()):
        sub = cache.filter(pl.col('env_name') == env)
        v = sub.filter((pl.col('arm_key') == 'baseline') & pl.col('eval_best_burst_mean').is_finite())
        d = sub.filter((pl.col('arm_key') == DDQN) & pl.col('eval_best_burst_mean').is_finite())

        v_with_pol = v.filter(pl.col('env_reward_polarity').is_finite())
        if len(v_with_pol) == 0:
            continue
        env_pol = float(v_with_pol['env_reward_polarity'].mean())

        pair_keys = ['corpus', 'gamma', 'sync_period', 'total_steps', 'seed']
        v_p = v.select(pair_keys + ['eval_best_burst_mean']).rename({'eval_best_burst_mean': 'o_v'})
        d_p = d.select(pair_keys + ['eval_best_burst_mean']).rename({'eval_best_burst_mean': 'o_d'})
        j = v_p.join(d_p, on=pair_keys, how='inner').filter(
            pl.col('o_v').is_not_nan() & pl.col('o_d').is_not_nan()
        )
        if len(j) < 5:
            continue
        delta_o = (j['o_d'] - j['o_v']).to_numpy()
        # Standardize Δ_outcome by within-env σ to make comparable across envs
        # Use Hedges' g-style: mean / pooled SD
        pooled_sd = float(np.sqrt(0.5 * (j['o_v'].to_numpy().var(ddof=1) + j['o_d'].to_numpy().var(ddof=1))))
        if pooled_sd <= 1e-9:
            continue
        g = float(delta_o.mean() / pooled_sd)
        panel.append({
            'env': env,
            'polarity': env_pol,
            'n_pairs': len(j),
            'mean_d_o': float(delta_o.mean()),
            'g_outcome': g,
            'sd_d_o': float(delta_o.std()),
        })

    print(f'{"env":<28} {"polarity":>10} {"n":>5} {"mean Δ_o":>10} {"g_outcome":>10}')
    print('-' * 80)
    for p in sorted(panel, key=lambda x: x['polarity']):
        print(f'{p["env"]:<28} {p["polarity"]:>+10.3f} {p["n_pairs"]:>5d} {p["mean_d_o"]:>+10.3f} {p["g_outcome"]:>+10.3f}')

    print()
    pols = np.array([p['polarity'] for p in panel])
    gs = np.array([p['g_outcome'] for p in panel])

    rho, pval = spearmanr(pols, gs)
    print(f'Spearman ρ(polarity, g_outcome) = {rho:+.3f}, p = {pval:.3g}')
    rho_abs, pval_abs = spearmanr(np.abs(pols), gs)
    print(f'Spearman ρ(|polarity|, g_outcome) = {rho_abs:+.3f}, p = {pval_abs:.3g}')
    rho_abs2, pval_abs2 = spearmanr(np.abs(pols), np.abs(gs))
    print(f'Spearman ρ(|polarity|, |g_outcome|) = {rho_abs2:+.3f}, p = {pval_abs2:.3g}')

    # Implication interpretation
    print()
    print('=== Interpretation ===')
    if abs(rho) < 0.3:
        print('  Polarity does NOT moderate g_outcome direction → DDQN benefit is NOT polarity-conditional.')
        print('  This is consistent with polarity being a *mediation-channel* moderator (it routes which')
        print('  per-cell mechanism (eff_h vs other) carries the outcome) but NOT a moderator of the')
        print('  AVERAGE outcome benefit. The polarity finding is about *how* DDQN helps, not *whether*.')
    else:
        print(f'  Polarity moderates g_outcome direction (ρ={rho:+.3f}). DDQN helps differently across polarities.')

    out = Path('experiments/findings/sync_curve_breakout/polarity_outcome_panel.json')
    out.write_text(json.dumps({'per_env': panel, 'cross_env': {
        'spearman_pol_g': {'rho': rho, 'p': pval},
        'spearman_abs_pol_g': {'rho': rho_abs, 'p': pval_abs},
        'spearman_abs_pol_abs_g': {'rho': rho_abs2, 'p': pval_abs2},
    }}, indent=2))
    print(f'\nwrote: {out}')


if __name__ == '__main__':
    main()
