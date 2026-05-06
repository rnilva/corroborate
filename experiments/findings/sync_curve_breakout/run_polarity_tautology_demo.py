"""Q2 — Explicit demonstration of the polarity-predicts-link-sign
soft tautology.

The argument: `env_reward_polarity` is by definition the within-cell
r(episode_length, mc_return). `effective_horizon` and `bootstrap_
fraction` are monotone in mean episode length per cell. Therefore:

  HARD  r(eff_h, outcome) within an env's cells       ≈ polarity
        (essentially polarity's definition with eff_h as L-proxy)

  SOFT  r(Δ_eff_h, Δ_outcome) across paired cells     ≈ polarity
        (DDQN traverses the env's L→outcome curve, so paired-Δ
        inherits the same structural correlation)

We compute both within-env r's per env and check cross-env Spearman
ρ against polarity. If the soft form tracks polarity at ρ ≈ +0.8,
the 8-of-8 sign-match observation in the original polarity finding
is structural, not an independent mechanistic discovery.

Three columns per env, all forming the same tautology family:
  r_within_eff_h:  within-env r(eff_h, outcome) over baseline cells
  r_pair_eff_h:    within-env r(Δ_eff_h, Δ_outcome) across pairs
  r_pair_bf:       within-env r(Δ_bf,    Δ_outcome) across pairs
                   (bf is L-proxy without the γ confound)
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy import stats

CORPUS_DIR = Path('experiments/data/ddqn')
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'

SKIP_ENVS = {
    'BernoulliBandit-misc', 'GaussianBandit-misc', 'MNISTBandit-bsuite',
    'Catch-bsuite', 'DeepSea-bsuite', 'DiscountingChain-bsuite',
    'Freeway-MinAtar', 'MemoryChain-bsuite', 'UmbrellaChain-bsuite',
}


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 5 or x.std() == 0 or y.std() == 0:
        return float('nan')
    r, _ = stats.pearsonr(x, y)
    return float(r)


def main() -> None:
    runs = pl.read_parquet(CORPUS_DIR / 'runs.parquet', columns=['id', 'env_name', 'arm_key', 'seed'])
    ms = pl.read_parquet(CORPUS_DIR / 'measurements.parquet')
    df = runs.join(ms, on='id', how='inner')

    print('Demonstrating the polarity-predicts-link-sign soft tautology.')
    print('Definition: env_reward_polarity = within-cell r(episode_length, mc_return).')
    print('eff_h and bf are both monotone in mean episode length per cell.')
    print()

    panel = []
    for env in sorted(df['env_name'].unique()):
        if env in SKIP_ENVS:
            continue
        sub = df.filter(pl.col('env_name') == env)
        v = sub.filter(pl.col('arm_key') == 'baseline').select(
            'seed', 'effective_horizon', 'bootstrap_fraction',
            'eval_best_burst_mean', 'env_reward_polarity',
        )
        d = sub.filter(pl.col('arm_key') == DDQN).select(
            'seed', 'effective_horizon', 'bootstrap_fraction',
            'eval_best_burst_mean',
        )
        if len(v) < 5 or len(d) < 5:
            continue

        # Polarity (env-level): mean of within-cell polarity across cells.
        polarity = float(v['env_reward_polarity'].drop_nans().mean())
        if math.isnan(polarity):
            continue

        # ----- HARD form -----
        # Within-env r(eff_h, outcome) over baseline cells.
        # This is essentially the cross-cell version of polarity (using
        # eff_h as L-proxy, outcome as return-proxy). Each cell's eff_h
        # IS a monotone transform of E[L_cell].
        v_eh = v['effective_horizon'].drop_nans().to_numpy()
        v_o  = v['eval_best_burst_mean'].drop_nans().to_numpy()
        if len(v_eh) >= 5 and len(v_o) >= 5:
            n = min(len(v_eh), len(v_o))
            r_within = safe_pearson(v_eh[:n], v_o[:n])
        else:
            r_within = float('nan')

        # ----- SOFT form -----
        # Pair v/d on seed; compute Δ_eff_h, Δ_bf, Δ_outcome.
        v_p = v.rename({c: f'{c}_v' for c in v.columns if c != 'seed'})
        d_p = d.rename({c: f'{c}_d' for c in d.columns if c != 'seed'})
        j = v_p.join(d_p, on='seed', how='inner').filter(
            pl.col('eval_best_burst_mean_v').is_finite()
            & pl.col('eval_best_burst_mean_d').is_finite()
            & pl.col('effective_horizon_v').is_finite()
            & pl.col('effective_horizon_d').is_finite()
            & pl.col('bootstrap_fraction_v').is_finite()
            & pl.col('bootstrap_fraction_d').is_finite()
        )
        if len(j) < 5:
            continue

        d_o  = (j['eval_best_burst_mean_d'] - j['eval_best_burst_mean_v']).to_numpy()
        d_eh = (j['effective_horizon_d'] - j['effective_horizon_v']).to_numpy()
        d_bf = (j['bootstrap_fraction_d'] - j['bootstrap_fraction_v']).to_numpy()
        r_pair_eh = safe_pearson(d_eh, d_o)
        r_pair_bf = safe_pearson(d_bf, d_o)

        panel.append({
            'env': env, 'polarity': polarity, 'n_pairs': len(j), 'n_cells_v': len(v_eh),
            'r_within_eff_h':  r_within,
            'r_pair_eff_h':    r_pair_eh,
            'r_pair_bf':       r_pair_bf,
        })

    print(f'{"env":<24} {"polarity":>9} {"r_within(eff_h,O)":>18} {"r_pair(Δeff_h,ΔO)":>18} {"r_pair(Δbf,ΔO)":>15}')
    print('-' * 90)
    for r in sorted(panel, key=lambda x: x['polarity']):
        def fmt(x: float) -> str:
            return f'{x:>+.3f}' if not math.isnan(x) else 'nan'
        print(
            f'  {r["env"]:<22} {r["polarity"]:>+9.3f}      {fmt(r["r_within_eff_h"]):>15} '
            f'    {fmt(r["r_pair_eff_h"]):>15} {fmt(r["r_pair_bf"]):>13}',
            flush=True,
        )

    print()
    print('=== Cross-env Spearman ρ vs polarity (the tautology test) ===\n')

    pols = np.array([r['polarity'] for r in panel])

    for predictor, label, kind in (
        ('r_within_eff_h',  'r(eff_h,    outcome) within env',  'HARD'),
        ('r_pair_eff_h',    'r(Δ_eff_h,  Δ_outcome) across pairs', 'SOFT'),
        ('r_pair_bf',       'r(Δ_bf,     Δ_outcome) across pairs', 'SOFT (no γ)'),
    ):
        rs = np.array([r[predictor] for r in panel])
        mask = ~np.isnan(rs) & ~np.isnan(pols)
        if mask.sum() < 3:
            continue
        rho_s, p_s = stats.spearmanr(pols[mask], rs[mask])
        # Also linear regression slope (the v2 form: r ≈ k × polarity)
        slope, intercept, r_val, p_val, _ = stats.linregress(pols[mask], rs[mask])
        r2 = float(r_val) ** 2
        print(
            f'  [{kind:<10}]  ρ(polarity, {label})\n'
            f'              = {rho_s:+.3f} (p={p_s:.3g}, n={mask.sum()})\n'
            f'              regression: r ≈ {slope:+.3f}·polarity + {intercept:+.3f},  R² = {r2:.3f}\n',
            flush=True,
        )

    print()
    print('=== Reading ===')
    print('If all three track polarity at ρ ≈ +0.8 (or higher), the soft')
    print('tautology is explicit: polarity-predicts-link-sign is the env\'s')
    print('structural L→outcome map measured in another guise. The 8-of-8')
    print('sign match in the original polarity bridge finding requires')
    print('only that DDQN traverses the env\'s L→outcome curve, not that')
    print('eff_h carries any independent mechanistic information about')
    print('how DDQN works.')

    out = Path('experiments/findings/sync_curve_breakout/polarity_tautology_demo.json')
    out.write_text(json.dumps(panel, indent=2))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
