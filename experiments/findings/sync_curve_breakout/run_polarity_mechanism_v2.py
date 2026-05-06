"""V2 — uses per-corpus measurements.parquet (universal-parquet
paradigm) for the full 18-env panel from the `ddqn` corpus.

Tests: is target_staleness mediation polarity-blind, while
eff_h coupling is polarity-coupled-by-definition?

Per env, compute pair-level (DDQN − vanilla) deltas and
within-env Pearson r between each candidate predictor and
Δ_outcome:
  - r(Δ_jens, Δ_o)    — mech step → outcome
  - r(Δ_stale, Δ_o)   — staleness mediator (predicted: polarity-blind)
  - r(Δ_eff_h, Δ_o)   — length channel (predicted: polarity-coupled)
  - r(Δ_bf, Δ_o)      — bf == eff_h structurally

Cross-env: test sign-consistency. The mechanism-blind hypothesis
predicts r(Δ_stale, Δ_o) is sign-consistent (negative everywhere),
r(Δ_eff_h, Δ_o) flips with polarity sign.
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

# Envs to skip (no polarity-coupling structure):
#   bandits = single-step;  fixed-L envs = no policy-driven length variation.
SKIP_ENVS = {
    'BernoulliBandit-misc', 'GaussianBandit-misc', 'MNISTBandit-bsuite',
    'Catch-bsuite', 'DeepSea-bsuite', 'DiscountingChain-bsuite',
    'Freeway-MinAtar', 'MemoryChain-bsuite', 'UmbrellaChain-bsuite',
}


def main() -> None:
    runs = pl.read_parquet(CORPUS_DIR / 'runs.parquet', columns=['id', 'env_name', 'arm_key', 'seed'])
    ms = pl.read_parquet(CORPUS_DIR / 'measurements.parquet')
    df = runs.join(ms, on='id', how='inner')
    print(f'ddqn corpus: {len(df)} cells', flush=True)

    panel = []
    for env in sorted(df['env_name'].unique()):
        if env in SKIP_ENVS:
            continue
        sub = df.filter(pl.col('env_name') == env)
        v = sub.filter(pl.col('arm_key') == 'baseline').select(
            'seed', 'jensen_gap', 'effective_horizon', 'bootstrap_fraction',
            'target_staleness_late', 'eval_best_burst_mean', 'env_reward_polarity',
        )
        d = sub.filter(pl.col('arm_key') == DDQN).select(
            'seed', 'jensen_gap', 'effective_horizon', 'bootstrap_fraction',
            'target_staleness_late', 'eval_best_burst_mean',
        )
        if len(v) == 0 or len(d) == 0:
            continue
        # Pair on seed
        v = v.rename({c: f'{c}_v' for c in v.columns if c != 'seed'})
        d = d.rename({c: f'{c}_d' for c in d.columns if c != 'seed'})
        j = v.join(d, on='seed', how='inner').filter(
            pl.col('eval_best_burst_mean_v').is_finite()
            & pl.col('eval_best_burst_mean_d').is_finite()
            & pl.col('jensen_gap_v').is_finite()
            & pl.col('jensen_gap_d').is_finite()
            & pl.col('target_staleness_late_v').is_finite()
            & pl.col('target_staleness_late_d').is_finite()
        )
        if len(j) < 5:
            continue

        polarity = float(j['env_reward_polarity_v'].mean())
        if math.isnan(polarity):
            polarity_alt = float(v['env_reward_polarity_v'].drop_nans().mean()) if 'env_reward_polarity_v' in v.columns else float('nan')
            polarity = polarity_alt

        d_o = (j['eval_best_burst_mean_d'] - j['eval_best_burst_mean_v']).to_numpy()
        d_jens = (j['jensen_gap_d'] - j['jensen_gap_v']).to_numpy()
        d_stale = (j['target_staleness_late_d'] - j['target_staleness_late_v']).to_numpy()
        d_eh = (j['effective_horizon_d'] - j['effective_horizon_v']).to_numpy()
        d_bf = (j['bootstrap_fraction_d'] - j['bootstrap_fraction_v']).to_numpy()

        if d_o.std() == 0:
            continue

        def safe_r(x, y):
            if x.std() == 0 or y.std() == 0:
                return float('nan'), float('nan')
            r, p = stats.pearsonr(x, y)
            return float(r), float(p)

        r_jens_o, p_jens_o = safe_r(d_jens, d_o)
        r_stale_o, p_stale_o = safe_r(d_stale, d_o)
        r_eh_o, p_eh_o = safe_r(d_eh, d_o)
        r_bf_o, p_bf_o = safe_r(d_bf, d_o)

        panel.append({
            'env': env, 'polarity': polarity, 'n_pairs': len(j),
            'mean_d_o': float(d_o.mean()),
            'mean_d_jens': float(d_jens.mean()),
            'mean_d_stale': float(d_stale.mean()),
            'mean_d_eh': float(d_eh.mean()),
            'r_jens_o': r_jens_o, 'p_jens_o': p_jens_o,
            'r_stale_o': r_stale_o, 'p_stale_o': p_stale_o,
            'r_eh_o': r_eh_o, 'p_eh_o': p_eh_o,
            'r_bf_o': r_bf_o, 'p_bf_o': p_bf_o,
        })

    print()
    print('=== Within-env r(Δ_predictor, Δ_outcome) panel ===\n')
    print(f'{"env":<24} {"polarity":>9} {"n":>4} {"r_jens":>8} {"r_stale":>9} {"r_eff_h":>9} {"r_bf":>9}', flush=True)
    print('-' * 80)
    for r in sorted(panel, key=lambda x: x['polarity']):
        print(
            f'  {r["env"]:<22} {r["polarity"]:>+9.3f} {r["n_pairs"]:>4d} '
            f'{r["r_jens_o"]:>+8.3f} {r["r_stale_o"]:>+9.3f} '
            f'{r["r_eh_o"]:>+9.3f} {r["r_bf_o"]:>+9.3f}',
            flush=True,
        )

    print()
    print('=== Sign-consistency (mechanism-blind hypothesis) ===')
    for predictor, name in (
        ('r_jens_o', 'Δ_jens'),
        ('r_stale_o', 'Δ_staleness'),
        ('r_eh_o', 'Δ_eff_h'),
        ('r_bf_o', 'Δ_bf'),
    ):
        rs = np.array([r[predictor] for r in panel if not math.isnan(r[predictor])])
        if len(rs) < 3:
            continue
        n_neg = int((rs < 0).sum())
        n_pos = int((rs > 0).sum())
        signs = '  '.join(f'{r:+.2f}' for r in rs)
        print(f'  {name:<14}  pos={n_pos}, neg={n_neg}  rs=[{signs}]', flush=True)

    print()
    print('=== Cross-env: ρ(within-env r, polarity) ===')
    pols = np.array([r['polarity'] for r in panel])
    for predictor, name in (
        ('r_jens_o', 'Δ_jens'),
        ('r_stale_o', 'Δ_staleness'),
        ('r_eh_o', 'Δ_eff_h'),
        ('r_bf_o', 'Δ_bf'),
    ):
        rs = np.array([r[predictor] for r in panel])
        mask = ~np.isnan(rs) & ~np.isnan(pols)
        if mask.sum() < 3:
            continue
        rho_s, p_s = stats.spearmanr(pols[mask], rs[mask])
        print(
            f'  ρ(pol, {name:<12}) = {rho_s:+.3f} (p={p_s:.3g})  '
            f'(n={mask.sum()})',
            flush=True,
        )

    out = Path('experiments/findings/sync_curve_breakout/polarity_mechanism_v2.json')
    out.write_text(json.dumps(panel, indent=2))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
