"""Per-env polarity coupling conditioned on mech HELD (Δ_jens < 0).

The polarity-coupling residual at scalar level (FourRooms over-strong,
CartPole/SI/Acrobot under-strong) may be explained by mech firing
heterogeneity. Per `CLAUDE.md`'s conditioning rule:

    Link analyses MUST condition on `mech HELD` (Δ_jens < 0 with
    the mechanism active, not just `jensen_gap > 0`). Otherwise
    "link null" claims silently mix mech-dormant (bias premise
    inactive) with mech-active-but-link-broken cells.

Audit shows:
  - Acrobot mean Δ_jens = +96.2 (REVERSED on average)
  - SpaceInvaders mean Δ_jens = +14309 (Q-amplification)
  - CartPole frac(Δ_jens<0) = 0.48 (only half fire)

Without conditioning, the polarity coupling pools mech-active and
mech-dormant pairs. The expected refinement: when restricted to
mech-active pairs, |r| should approach the polarity prediction
(r ≈ 0.535 × polarity).
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

CACHE_PATH = Path('experiments/data/cache/ddqn.parquet')
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
PAIR_KEYS = ['corpus', 'gamma', 'sync_period', 'total_steps', 'seed']
POLARITY_ENVS = [
    'Acrobot-v1', 'FourRooms-misc', 'MountainCar-v0', 'MetaMaze-misc',
    'CartPole-v1', 'Asterix-MinAtar', 'Breakout-MinAtar', 'SpaceInvaders-MinAtar',
]


def main() -> None:
    cache = pl.read_parquet(CACHE_PATH)
    print(f'cache: {len(cache)} rows')
    prior = json.loads(
        Path('experiments/findings/sync_curve_breakout/polarity_x_undamped_panel.json').read_text()
    )['per_env']
    prior_by_env = {p['env']: p for p in prior}

    panel = []
    finite = pl.all_horizontal([
        pl.col(c).is_finite() for c in
        ('bootstrap_fraction', 'effective_horizon', 'eval_best_burst_mean', 'jensen_gap')
    ])

    for env in POLARITY_ENVS:
        sub = cache.filter(pl.col('env_name') == env)
        v = sub.filter((pl.col('arm_key') == 'baseline') & finite)
        d = sub.filter((pl.col('arm_key') == DDQN) & finite)
        if len(v) == 0 or len(d) == 0:
            continue

        v_p = v.select(PAIR_KEYS + ['bootstrap_fraction', 'effective_horizon',
                                    'eval_best_burst_mean', 'jensen_gap']).rename({
            'bootstrap_fraction': 'bf_v', 'effective_horizon': 'eh_v',
            'eval_best_burst_mean': 'o_v', 'jensen_gap': 'jg_v',
        })
        d_p = d.select(PAIR_KEYS + ['bootstrap_fraction', 'effective_horizon',
                                    'eval_best_burst_mean', 'jensen_gap']).rename({
            'bootstrap_fraction': 'bf_d', 'effective_horizon': 'eh_d',
            'eval_best_burst_mean': 'o_d', 'jensen_gap': 'jg_d',
        })
        j = v_p.join(d_p, on=PAIR_KEYS, how='inner').filter(
            pl.col('bf_v').is_not_nan() & pl.col('bf_d').is_not_nan()
            & pl.col('eh_v').is_not_nan() & pl.col('eh_d').is_not_nan()
            & pl.col('o_v').is_not_nan() & pl.col('o_d').is_not_nan()
            & pl.col('jg_v').is_not_nan() & pl.col('jg_d').is_not_nan()
        )
        if len(j) < 5:
            continue

        d_eh_all = (j['eh_d'] - j['eh_v']).to_numpy()
        d_o_all = (j['o_d'] - j['o_v']).to_numpy()
        d_jens_all = (j['jg_d'] - j['jg_v']).to_numpy()

        # Unconditional r
        if d_eh_all.std() > 0 and d_o_all.std() > 0:
            r_all = float(np.corrcoef(d_eh_all, d_o_all)[0, 1])
        else:
            r_all = float('nan')

        # Mech HELD: Δ_jens < 0 (DDQN reduced Q estimation)
        mask_held = d_jens_all < 0
        n_held = int(mask_held.sum())
        if n_held >= 5 and d_eh_all[mask_held].std() > 0 and d_o_all[mask_held].std() > 0:
            r_held = float(np.corrcoef(d_eh_all[mask_held], d_o_all[mask_held])[0, 1])
        else:
            r_held = float('nan')

        # Mech REVERSED: Δ_jens > 0 (DDQN amplified Q)
        mask_reversed = d_jens_all > 0
        n_reversed = int(mask_reversed.sum())
        if n_reversed >= 5 and d_eh_all[mask_reversed].std() > 0 and d_o_all[mask_reversed].std() > 0:
            r_reversed = float(np.corrcoef(d_eh_all[mask_reversed], d_o_all[mask_reversed])[0, 1])
        else:
            r_reversed = float('nan')

        # Strong-mech: |Δ_jens| in top-25% by magnitude AND negative
        # (only the strongly-firing pairs)
        abs_d_jens = np.abs(d_jens_all)
        thresh = float(np.percentile(abs_d_jens, 75))
        mask_strong = (d_jens_all < 0) & (abs_d_jens >= thresh)
        n_strong = int(mask_strong.sum())
        if n_strong >= 5 and d_eh_all[mask_strong].std() > 0 and d_o_all[mask_strong].std() > 0:
            r_strong = float(np.corrcoef(d_eh_all[mask_strong], d_o_all[mask_strong])[0, 1])
        else:
            r_strong = float('nan')

        polarity = float(prior_by_env[env]['polarity'])
        predicted = 0.535 * polarity

        panel.append({
            'env': env,
            'polarity': polarity,
            'predicted': predicted,
            'n_all': len(j),
            'r_all': r_all,
            'n_held': n_held,
            'r_held': r_held,
            'n_reversed': n_reversed,
            'r_reversed': r_reversed,
            'n_strong': n_strong,
            'r_strong': r_strong,
            'frac_held': float(mask_held.mean()),
            'mean_d_jens': float(d_jens_all.mean()),
        })

    print()
    hdr = ('env', 'pol', 'pred', 'r_all', 'r_held', 'r_strong', 'r_rev', 'n_all', 'n_held', 'frac_h')
    print(f'{hdr[0]:<24} {hdr[1]:>6} {hdr[2]:>6} {hdr[3]:>7} {hdr[4]:>7} {hdr[5]:>9} {hdr[6]:>7} {hdr[7]:>6} {hdr[8]:>6} {hdr[9]:>7}')
    print('-' * 105)
    for p in sorted(panel, key=lambda x: x['polarity']):
        rh = f'{p["r_held"]:>+7.3f}' if not np.isnan(p['r_held']) else '     na'
        rs = f'{p["r_strong"]:>+9.3f}' if not np.isnan(p['r_strong']) else '       na'
        rr = f'{p["r_reversed"]:>+7.3f}' if not np.isnan(p['r_reversed']) else '     na'
        print(
            f'{p["env"]:<24} {p["polarity"]:>+6.2f} {p["predicted"]:>+6.3f} '
            f'{p["r_all"]:>+7.3f} {rh} {rs} {rr} '
            f'{p["n_all"]:>6d} {p["n_held"]:>6d} {p["frac_held"]:>7.3f}'
        )

    # Cross-env: how does conditioning on mech HELD change the prediction R²?
    print()
    print('=== Cross-env: r(polarity, r_*) and OLS slopes ===')
    pols = np.array([p['polarity'] for p in panel])
    for col in ('r_all', 'r_held', 'r_strong'):
        rs = np.array([p[col] for p in panel])
        if np.isnan(rs).any():
            mask = ~np.isnan(rs)
            pols_m = pols[mask]; rs_m = rs[mask]
        else:
            pols_m = pols; rs_m = rs
        if len(pols_m) < 3:
            continue
        rho, p_val = stats.spearmanr(pols_m, rs_m)
        slope = float((pols_m * rs_m).sum() / (pols_m ** 2).sum())
        fitted = slope * pols_m
        ss_res = float(((rs_m - fitted) ** 2).sum())
        ss_tot = float(((rs_m - rs_m.mean()) ** 2).sum())
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float('nan')
        print(f'  {col:<10} (n={len(pols_m)})  Spearman ρ = {rho:+.3f} (p={p_val:.3g})  '
              f'OLS slope = {slope:+.3f}  R² = {r2:+.3f}')

    out = Path('experiments/findings/sync_curve_breakout/polarity_mech_conditioned_panel.json')
    out.write_text(json.dumps({'per_env': panel}, indent=2))
    print(f'\nwrote: {out}')


if __name__ == '__main__':
    main()
