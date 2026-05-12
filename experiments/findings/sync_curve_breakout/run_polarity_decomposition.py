"""Decomposition of `r(Δeff_h, Δoutcome)` per env to explain the GOAL/SURVIVAL
asymmetry.

For each pair of (vanilla, ddqn) cells joined on (corpus, gamma, sync_period,
total_steps, seed), we compute:

  bf  := bootstrap_fraction (cell-aggregate, ≈ 1 - 1/avg_episode_length)
  eh  := effective_horizon = 1/(1 - γ·bf)
  o   := eval_best_burst_mean (the outcome)

Pairwise Δ are taken (ddqn - vanilla). We then report per env:

  r(Δbf, Δo)     — bf → outcome ("length identification" — strong-negative in
                    GOAL where return ≈ −α·length, weak in SURVIVAL where
                    return is per-step skill-modulated)
  r(Δbf, Δeh)    — bf → eff_h ("gain factor"; near 1 by construction unless
                    γ varies within pair, which the pair_keys exclude)
  r(Δeh, Δo)     — original coupling (matches polarity-panel)
  ratio          — r(Δeh, Δo) / r(Δbf, Δo)  (close to 1 if eff_h is a clean
                    proxy for bf in this env; <1 means eff_h adds noise)
  saturation     — γ · eff_h^2 / L^2 evaluated at mean bf, mean γ — the
                    sensitivity of eff_h to length changes ('compressing' if
                    <<1, '1:1' if ~1)

The asymmetry hypothesis (H1+H2):
  - GOAL envs: r(Δbf, Δo) ≈ −1 (length IS outcome, up to step penalty);
                eff_h is a clean bf proxy → r(Δeh, Δo) inherits it.
  - SURVIVAL envs: r(Δbf, Δo) is moderate (length correlates with outcome but
                    skill modulates); plus eff_h saturation regime can compress
                    or amplify the signal.

If both predictions are confirmed, the asymmetry decomposes into:
  (length-as-outcome identification) × (eff_h-as-bf-proxy fidelity)
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import pearsonr, spearmanr

CACHE_PATH = Path('experiments/data/cache/ddqn.parquet')
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
PAIR_KEYS = ['corpus', 'gamma', 'sync_period', 'total_steps', 'seed']


def fisher_z(r: float) -> float:
    r = max(min(r, 1 - 1e-10), -1 + 1e-10)
    return 0.5 * math.log((1 + r) / (1 - r))


def main() -> None:
    cache = pl.read_parquet(CACHE_PATH)
    print(f'cache: {len(cache)} rows')

    panel = []
    needed = ['bootstrap_fraction', 'effective_horizon', 'eval_best_burst_mean', 'env_reward_polarity', 'gamma']
    finite_pred = pl.all_horizontal([pl.col(c).is_finite() for c in needed])

    for env in sorted(cache.filter(pl.col('env_name').is_not_null())['env_name'].unique()):
        sub = cache.filter(pl.col('env_name') == env)
        v = sub.filter((pl.col('arm_key') == 'baseline') & finite_pred)
        d = sub.filter((pl.col('arm_key') == DDQN) & finite_pred)

        if len(v) == 0 or len(d) == 0:
            continue

        v_with_pol = v.filter(pl.col('env_reward_polarity').is_finite())
        if len(v_with_pol) == 0:
            continue
        env_pol = float(v_with_pol['env_reward_polarity'].mean())

        v_p = v.select(PAIR_KEYS + ['bootstrap_fraction', 'effective_horizon', 'eval_best_burst_mean']).rename(
            {'bootstrap_fraction': 'bf_v', 'effective_horizon': 'eh_v', 'eval_best_burst_mean': 'o_v'}
        )
        d_p = d.select(PAIR_KEYS + ['bootstrap_fraction', 'effective_horizon', 'eval_best_burst_mean']).rename(
            {'bootstrap_fraction': 'bf_d', 'effective_horizon': 'eh_d', 'eval_best_burst_mean': 'o_d'}
        )
        j = v_p.join(d_p, on=PAIR_KEYS, how='inner').filter(
            pl.col('bf_v').is_not_nan() & pl.col('bf_d').is_not_nan()
            & pl.col('eh_v').is_not_nan() & pl.col('eh_d').is_not_nan()
            & pl.col('o_v').is_not_nan() & pl.col('o_d').is_not_nan()
        )
        if len(j) < 5:
            continue

        d_bf = (j['bf_d'] - j['bf_v']).to_numpy()
        d_eh = (j['eh_d'] - j['eh_v']).to_numpy()
        d_o = (j['o_d'] - j['o_v']).to_numpy()

        if d_bf.std() == 0 or d_eh.std() == 0 or d_o.std() == 0:
            continue

        r_bf_o = float(np.corrcoef(d_bf, d_o)[0, 1])
        r_bf_eh = float(np.corrcoef(d_bf, d_eh)[0, 1])
        r_eh_o = float(np.corrcoef(d_eh, d_o)[0, 1])

        # Saturation diagnostic at vanilla regime.
        bf_v = j['bf_v'].to_numpy()
        gamma = j['gamma'].to_numpy()
        eh_v = 1.0 / np.maximum(1.0 - gamma * bf_v, 1e-12)
        # ∂eh/∂L = γ² / ((1-γ)L + γ)² where L = 1/(1-bf)
        L = 1.0 / np.maximum(1.0 - bf_v, 1e-12)
        deh_dL = (gamma ** 2) / np.maximum(((1 - gamma) * L + gamma) ** 2, 1e-24)
        # vs. r=+1 mapping: |Δeh/ΔL| in [0,1]; closer to 0 = saturation.
        saturation = float(np.median(deh_dL))

        # OLS slope diagnostic
        b_bf_o, _ = np.polyfit(d_bf, d_o, 1)
        b_bf_eh, _ = np.polyfit(d_bf, d_eh, 1)

        # Partial r(Δeh, Δo | Δbf) — does eff_h add anything beyond bf?
        # residualize d_eh on d_bf, and d_o on d_bf, then correlate residuals
        beta_eh_on_bf = np.cov(d_eh, d_bf)[0, 1] / d_bf.var()
        beta_o_on_bf = np.cov(d_o, d_bf)[0, 1] / d_bf.var()
        eh_res = d_eh - beta_eh_on_bf * d_bf
        o_res = d_o - beta_o_on_bf * d_bf
        r_partial = float(np.corrcoef(eh_res, o_res)[0, 1]) if eh_res.std() > 0 and o_res.std() > 0 else float('nan')

        panel.append({
            'env': env,
            'polarity': env_pol,
            'n_pairs': len(j),
            'r_eh_o': r_eh_o,
            'r_bf_o': r_bf_o,
            'r_bf_eh': r_bf_eh,
            'r_eh_o_partial_bf': r_partial,
            'b_bf_o': float(b_bf_o),
            'b_bf_eh': float(b_bf_eh),
            'mean_bf_v': float(bf_v.mean()),
            'mean_L_v': float(L.mean()),
            'mean_eh_v': float(eh_v.mean()),
            'mean_gamma': float(gamma.mean()),
            'saturation_dL': saturation,
            'sd_d_bf': float(d_bf.std()),
            'sd_d_eh': float(d_eh.std()),
            'sd_d_o': float(d_o.std()),
            'mean_d_bf': float(d_bf.mean()),
            'mean_d_eh': float(d_eh.mean()),
            'mean_d_o': float(d_o.mean()),
        })

    # Print primary panel
    print()
    print('=== Per-env decomposition ===')
    print()
    print(f'{"env":<24} {"pol":>7} {"n":>5} {"r_eh_o":>8} {"r_bf_o":>8} {"r_bf_eh":>9} {"r_partial":>11} {"sat_dL":>9} {"L_mean":>8}')
    print('-' * 110)
    for p in sorted(panel, key=lambda x: x['polarity']):
        print(f'{p["env"]:<24} {p["polarity"]:>+7.3f} {p["n_pairs"]:>5d} '
              f'{p["r_eh_o"]:>+8.3f} {p["r_bf_o"]:>+8.3f} {p["r_bf_eh"]:>+9.3f} '
              f'{p["r_eh_o_partial_bf"]:>+11.3f} {p["saturation_dL"]:>9.4f} {p["mean_L_v"]:>8.1f}')

    # Pooled by polarity (Fisher-z weighted by n-3)
    print()
    print('=== Pooled by polarity (Fisher-z weighted) ===')
    for tag, predicate in [('GOAL', lambda p: p['polarity'] < -0.3), ('SURVIVAL', lambda p: p['polarity'] > +0.3)]:
        subset = [p for p in panel if predicate(p)]
        if not subset:
            continue
        for col in ('r_eh_o', 'r_bf_o', 'r_bf_eh'):
            zs = np.array([fisher_z(p[col]) for p in subset])
            ws = np.array([max(p['n_pairs'] - 3, 1) for p in subset])
            z_pool = float((zs * ws).sum() / ws.sum())
            r_pool = math.tanh(z_pool)
            n_envs = len(subset)
            print(f'  {tag:<10} {col:<9}  ρ_pool = {r_pool:+.3f}  (n_envs={n_envs}, sum_pairs={int(ws.sum())+3*n_envs})')
        print()

    # Cross-env tests on the decomposition
    print('=== Cross-env tests ===')
    pols = np.array([p['polarity'] for p in panel])
    abs_pols = np.abs(pols)

    for col in ('r_eh_o', 'r_bf_o', 'r_bf_eh'):
        rs = np.array([p[col] for p in panel])
        rho_signed, p_signed = spearmanr(pols, rs)
        rho_abs, p_abs = spearmanr(abs_pols, np.abs(rs))
        print(f'  {col:<10}  signed ρ(pol, r) = {rho_signed:+.3f}, p={p_signed:.3g}    |·| ρ = {rho_abs:+.3f}, p={p_abs:.3g}')

    # Save
    out = Path('experiments/findings/sync_curve_breakout/polarity_decomposition_panel.json')
    out.write_text(json.dumps({
        'per_env': panel,
        'pair_keys': PAIR_KEYS,
        'cache_path': str(CACHE_PATH),
    }, indent=2))
    print(f'\nwrote: {out}')


if __name__ == '__main__':
    main()
