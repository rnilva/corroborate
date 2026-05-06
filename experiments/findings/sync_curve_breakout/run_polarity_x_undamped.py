"""Test whether polarity × discount-undampedness predicts the per-env
coupling |r(Δeff_h, Δoutcome)|.

Hypothesis (refining the earlier H1+saturation finding into a single
two-variable model):

  r ≈ env_reward_polarity × γ^L_mean

Reasoning:
  - `env_reward_polarity` (within-cell r(length, return)) sets the
    SIGN and the local rigidity of how Δlength translates to Δreturn.
  - `γ^L_mean` (un-decayed fraction at typical episode length) gates
    HOW MUCH of a Δlength change actually moves Δreturn under a
    γ < 1 horizon. At small L: γ^L ≈ 1, full traction. At large L:
    γ^L → 0, Δreturn collapses regardless of polarity.

This combines the H1 (reward-identification) + discount-saturation
findings into a single predictor. If the two-factor model fits the
cross-env |r| panel tightly, the polarity-asymmetry story is a
fully closed-form env property — no DDQN-specific dynamics required.
"""
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
PAIR_KEYS = ['corpus', 'gamma', 'sync_period', 'total_steps', 'seed']


def main() -> None:
    cache = pl.read_parquet(CACHE_PATH)
    print(f'cache: {len(cache)} rows')

    panel = []
    finite = pl.all_horizontal([
        pl.col(c).is_finite() for c in
        ('bootstrap_fraction', 'effective_horizon', 'eval_best_burst_mean',
         'env_reward_polarity', 'gamma')
    ])

    for env in sorted(cache.filter(pl.col('env_name').is_not_null())['env_name'].unique()):
        sub = cache.filter(pl.col('env_name') == env)
        v = sub.filter((pl.col('arm_key') == 'baseline') & finite)
        d = sub.filter((pl.col('arm_key') == DDQN) & finite)
        if len(v) == 0 or len(d) == 0:
            continue

        v_with_pol = v.filter(pl.col('env_reward_polarity').is_finite())
        if len(v_with_pol) == 0:
            continue
        env_pol = float(v_with_pol['env_reward_polarity'].mean())

        v_p = v.select(PAIR_KEYS + ['bootstrap_fraction', 'effective_horizon', 'eval_best_burst_mean']).rename(
            {'bootstrap_fraction': 'bf_v', 'effective_horizon': 'eh_v',
             'eval_best_burst_mean': 'o_v'}
        )
        d_p = d.select(PAIR_KEYS + ['bootstrap_fraction', 'effective_horizon', 'eval_best_burst_mean']).rename(
            {'bootstrap_fraction': 'bf_d', 'effective_horizon': 'eh_d',
             'eval_best_burst_mean': 'o_d'}
        )
        j = v_p.join(d_p, on=PAIR_KEYS, how='inner').filter(
            pl.col('bf_v').is_not_nan() & pl.col('bf_d').is_not_nan()
            & pl.col('eh_v').is_not_nan() & pl.col('eh_d').is_not_nan()
            & pl.col('o_v').is_not_nan() & pl.col('o_d').is_not_nan()
        )
        if len(j) < 5:
            continue

        bf_v = j['bf_v'].to_numpy()
        gamma = j['gamma'].to_numpy()
        L_v = 1.0 / np.maximum(1.0 - bf_v, 1e-12)
        # γ^L: un-decayed fraction at typical L. For L large, → 0.
        gamma_L = np.power(gamma, L_v)
        # Saturation index: 1 − (1−γ^L) = γ^L. Same as gamma_L.
        # 'undampedness': larger = less dampened by discount.

        d_eh = (j['eh_d'] - j['eh_v']).to_numpy()
        d_o = (j['o_d'] - j['o_v']).to_numpy()

        if d_eh.std() == 0 or d_o.std() == 0:
            continue

        r_eh_o = float(np.corrcoef(d_eh, d_o)[0, 1])

        panel.append({
            'env': env,
            'polarity': env_pol,
            'mean_L_v': float(L_v.mean()),
            'mean_gamma': float(gamma.mean()),
            'mean_gamma_L': float(gamma_L.mean()),  # discount undampedness
            'pol_x_undamped': env_pol * float(gamma_L.mean()),
            'abs_pol_x_undamped': abs(env_pol) * float(gamma_L.mean()),
            'n_pairs': len(j),
            'r': r_eh_o,
            'abs_r': abs(r_eh_o),
        })

    print()
    print('=== Per-env |r| vs polarity × γ^L ===')
    print()
    print(f'{"env":<24} {"pol":>7} {"L_mean":>7} {"γ":>6} {"γ^L":>7} {"pol×γ^L":>8} {"r":>8} {"|r|":>6}')
    print('-' * 80)
    for p in sorted(panel, key=lambda x: x['polarity']):
        print(f'{p["env"]:<24} {p["polarity"]:>+7.3f} {p["mean_L_v"]:>7.1f} '
              f'{p["mean_gamma"]:>6.4f} {p["mean_gamma_L"]:>7.4f} '
              f'{p["pol_x_undamped"]:>+8.3f} {p["r"]:>+8.3f} {p["abs_r"]:>6.3f}')

    print()
    print('=== Cross-env predictive tests ===')
    pols = np.array([p['polarity'] for p in panel])
    rs = np.array([p['r'] for p in panel])
    gamma_Ls = np.array([p['mean_gamma_L'] for p in panel])
    pol_x_u = np.array([p['pol_x_undamped'] for p in panel])
    abs_pol_x_u = np.array([p['abs_pol_x_undamped'] for p in panel])

    for name, predictor in (('polarity', pols),
                            ('γ^L', gamma_Ls),
                            ('pol × γ^L', pol_x_u)):
        rho, p_val = spearmanr(predictor, rs)
        pearson_r, p_val_pearson = pearsonr(predictor, rs)
        print(f'  signed: {name:<14}  Spearman={rho:+.3f} (p={p_val:.3g})   '
              f'Pearson={pearson_r:+.3f} (p={p_val_pearson:.3g})')
    print()
    for name, predictor in (('|polarity|', np.abs(pols)),
                            ('γ^L', gamma_Ls),
                            ('|pol| × γ^L', abs_pol_x_u)):
        rho, p_val = spearmanr(predictor, np.abs(rs))
        pearson_r, p_val_pearson = pearsonr(predictor, np.abs(rs))
        print(f'  |·|:    {name:<14}  Spearman={rho:+.3f} (p={p_val:.3g})   '
              f'Pearson={pearson_r:+.3f} (p={p_val_pearson:.3g})')

    # Linear regression: r ~ pol × γ^L (signed)
    # OLS through origin: slope = Σ(x·y) / Σ(x²)
    slope_signed = float((pol_x_u * rs).sum() / (pol_x_u ** 2).sum())
    fitted = slope_signed * pol_x_u
    ss_res = float(((rs - fitted) ** 2).sum())
    ss_tot = float(((rs - rs.mean()) ** 2).sum())
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    print()
    print(f'OLS r ~ slope · (pol × γ^L), through origin:  '
          f'slope = {slope_signed:+.3f}, R² = {r_squared:+.3f}')

    # Save
    out = Path('experiments/findings/sync_curve_breakout/polarity_x_undamped_panel.json')
    out.write_text(json.dumps({
        'per_env': panel,
        'pair_keys': PAIR_KEYS,
        'cache_path': str(CACHE_PATH),
    }, indent=2))
    print(f'\nwrote: {out}')


if __name__ == '__main__':
    main()
