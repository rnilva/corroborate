"""Refined bimodality test: compute properties of the PAIRWISE Δ
distribution (Δ_o, Δ_L, |Δ_o| / σ_o) per env. The polarity-coupling
residual may track:

  1. Δ-distribution bimodality (FourRooms hypothesis): some pairs in
     'no-effect' mode, others in 'regime-jump' mode → bimodal |Δ_o|.
  2. Effect-size relative to baseline noise (`|Δ_o| / σ_o_baseline`):
     when interventions cross a regime boundary, the effect size
     dwarfs baseline cell-to-cell variability.
  3. Concordance of Δ direction (does DDQN consistently push pairs
     in one direction, or symmetric around zero?). Asymmetric Δ
     signals a regime-bias of the intervention.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

CACHE_PATH = Path('experiments/data/cache/ddqn_universe.parquet')
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
PAIR_KEYS = ['corpus', 'gamma', 'sync_period', 'total_steps', 'seed']
SLOPE_POLARITY = 0.535


def sarle_bc(x: np.ndarray) -> float:
    n = len(x)
    if n < 4 or x.std() == 0:
        return float('nan')
    skew = float(stats.skew(x, bias=False))
    kurt = float(stats.kurtosis(x, fisher=True, bias=False))
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denom = kurt + correction
    if denom <= 0:
        return float('nan')
    return (skew ** 2 + 1) / denom


def main() -> None:
    cache = pl.read_parquet(CACHE_PATH)
    print(f'cache: {len(cache)} rows')
    prior = json.loads(
        Path('experiments/findings/sync_curve_breakout/polarity_x_undamped_panel.json').read_text()
    )['per_env']
    by_env = {p['env']: p for p in prior}

    finite = pl.col('eval_best_burst_mean').is_finite() & pl.col('bootstrap_fraction').is_finite()
    panel = []
    for env, row in by_env.items():
        sub = cache.filter(pl.col('env_name') == env)
        v = sub.filter((pl.col('arm_key') == 'baseline') & finite)
        d = sub.filter((pl.col('arm_key') == DDQN) & finite)
        if len(v) < 4 or len(d) < 4:
            continue

        v_p = v.select(PAIR_KEYS + ['bootstrap_fraction', 'eval_best_burst_mean']).rename(
            {'bootstrap_fraction': 'bf_v', 'eval_best_burst_mean': 'o_v'}
        )
        d_p = d.select(PAIR_KEYS + ['bootstrap_fraction', 'eval_best_burst_mean']).rename(
            {'bootstrap_fraction': 'bf_d', 'eval_best_burst_mean': 'o_d'}
        )
        j = v_p.join(d_p, on=PAIR_KEYS, how='inner').filter(
            pl.col('bf_v').is_not_nan() & pl.col('bf_d').is_not_nan()
            & pl.col('o_v').is_not_nan() & pl.col('o_d').is_not_nan()
        )
        if len(j) < 5:
            continue

        bf_v = j['bf_v'].to_numpy()
        bf_d = j['bf_d'].to_numpy()
        o_v = j['o_v'].to_numpy()
        o_d = j['o_d'].to_numpy()
        d_o = o_d - o_v
        d_bf = bf_d - bf_v

        polarity = float(row['polarity'])
        actual_r = float(row['r'])
        fitted = SLOPE_POLARITY * polarity
        residual = actual_r - fitted

        # Effect-size relative to baseline variance:
        sigma_o_v = float(o_v.std()) if o_v.std() > 0 else 1e-12
        cohen_d = float(np.mean(d_o) / sigma_o_v)
        # Mean magnitude of Δ relative to baseline σ
        rel_effect = float(np.mean(np.abs(d_o)) / sigma_o_v)

        # Skewness / asymmetry of Δ_o (signed)
        skew_d_o = float(stats.skew(d_o, bias=False))
        # Kurtosis of Δ_o — heavy tails signal a few extreme regime jumps
        kurt_d_o = float(stats.kurtosis(d_o, fisher=True, bias=False))

        # BC of |Δ_o|: bimodal-shape indicates 'small effect / large effect' clusters
        bc_abs_d_o = sarle_bc(np.abs(d_o))
        # BC of Δ_o (signed): bimodal indicates two-regime (consistent +Δ vs no-effect)
        bc_d_o = sarle_bc(d_o)

        # Fraction of pairs with |Δ_o|/σ_o_v > 1 — 'large-effect' pair fraction
        large_effect_frac = float((np.abs(d_o) > sigma_o_v).mean())

        panel.append({
            'env': env,
            'polarity': polarity,
            'r': actual_r,
            'residual': residual,
            'abs_residual': abs(residual),
            'n_pairs': len(j),
            'sigma_o_v': sigma_o_v,
            'mean_d_o': float(d_o.mean()),
            'sigma_d_o': float(d_o.std()),
            'cohen_d': cohen_d,
            'rel_effect': rel_effect,
            'large_effect_frac': large_effect_frac,
            'skew_d_o': skew_d_o,
            'kurt_d_o': kurt_d_o,
            'bc_d_o': bc_d_o,
            'bc_abs_d_o': bc_abs_d_o,
        })

    print()
    print('=== Per-env Δ-distribution diagnostics ===')
    print()
    hdr = ['env', 'pol', 'r', 'resid', 'σ_o_v', 'cohen_d', 'rel|Δ|', 'large%', 'skew_Δ', 'kurt_Δ', 'BC_Δ']
    print(f'{hdr[0]:<24} {hdr[1]:>6} {hdr[2]:>6} {hdr[3]:>6} {hdr[4]:>7} {hdr[5]:>8} {hdr[6]:>7} {hdr[7]:>7} {hdr[8]:>7} {hdr[9]:>7} {hdr[10]:>6}')
    print('-' * 110)
    for p in sorted(panel, key=lambda x: -x['abs_residual']):
        print(
            f'{p["env"]:<24} {p["polarity"]:>+6.2f} {p["r"]:>+6.2f} {p["residual"]:>+6.2f} '
            f'{p["sigma_o_v"]:>7.3f} {p["cohen_d"]:>+8.3f} {p["rel_effect"]:>7.3f} '
            f'{p["large_effect_frac"]:>7.3f} {p["skew_d_o"]:>+7.3f} {p["kurt_d_o"]:>+7.2f} '
            f'{p["bc_d_o"]:>6.3f}'
        )

    print()
    print('=== Cross-env correlations with |residual| ===')
    abs_resid = np.array([p['abs_residual'] for p in panel])
    for name in ('rel_effect', 'large_effect_frac', 'kurt_d_o', 'bc_d_o', 'bc_abs_d_o', 'sigma_o_v', 'n_pairs', 'abs_skew_d_o'):
        if name == 'abs_skew_d_o':
            v = np.abs(np.array([p['skew_d_o'] for p in panel]))
        else:
            v = np.array([p[name] for p in panel])
        if np.isnan(v).any():
            mask = ~np.isnan(v)
            v_m = v[mask]; r_m = abs_resid[mask]
        else:
            v_m = v; r_m = abs_resid
        if len(v_m) < 3 or v_m.std() == 0:
            continue
        rho_s, p_s = stats.spearmanr(v_m, r_m)
        print(f'  ρ(|resid|, {name:<22}) = {rho_s:+.3f} (p={p_s:.3g})  (n={len(v_m)})')

    print()
    print('=== Cross-env correlations with SIGNED residual ===')
    signed_resid = np.array([p['residual'] for p in panel])
    for name in ('mean_d_o', 'cohen_d', 'skew_d_o', 'kurt_d_o', 'bc_d_o'):
        v = np.array([p[name] for p in panel])
        if np.isnan(v).any():
            mask = ~np.isnan(v)
            v_m = v[mask]; r_m = signed_resid[mask]
        else:
            v_m = v; r_m = signed_resid
        if len(v_m) < 3 or v_m.std() == 0:
            continue
        rho_s, p_s = stats.spearmanr(v_m, r_m)
        print(f'  ρ(signed_resid, {name:<22}) = {rho_s:+.3f} (p={p_s:.3g})')

    out = Path('experiments/findings/sync_curve_breakout/delta_bimodality_panel.json')
    out.write_text(json.dumps({'per_env': panel}, indent=2))
    print(f'\nwrote: {out}')


if __name__ == '__main__':
    main()
