"""Continuous env-polarity measurable: per-cell Pearson(episode_length, mc_return)
across all (burst × episode) eval points, computed on vanilla baseline cells.

Validates the categorical polarity proof by replacing the env-codebook
(GOAL/SURVIVAL hand-assignment) with an endogenous data-driven scalar.
The continuous polarity should:
- correlate with the categorical sign assignment (goal=−1, survival=+1)
- predict per-env r(Δeff_h, Δoutcome) magnitude AND sign as well as
  the categorical assignment did

If yes, we can author `env_reward_polarity` as a proper @measurable
that the framework computes per cell — no hand-coded env catalogue.
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

# Local trace sources (paths + which env_filter to apply)
TRACE_SOURCES = [
    ('experiments/data/expectile_3way',           'all'),  # has 5 envs
    ('experiments/data/capacity_sweep_fourrooms', 'all'),  # FourRooms only
    ('experiments/data/minatar_sync_intervention', 'all'),  # 4 MinAtar envs
]

# Categorical polarity from previous analysis (for validation)
CAT_POLARITY = {
    'Acrobot-v1':              -1,  # goal
    'FourRooms-misc':          -1,
    'MountainCar-v0':          -1,
    'DiscountingChain-bsuite': -1,
    'DeepSea-bsuite':          -1,
    'MemoryChain-bsuite':      -1,
    'UmbrellaChain-bsuite':    -1,
    'MetaMaze-misc':           -1,
    'CartPole-v1':             +1,  # survival
    'Breakout-MinAtar':        +1,
    'SpaceInvaders-MinAtar':   +1,
    'Asterix-MinAtar':         +1,
    'Pong-misc':               +1,
    'Catch-bsuite':            0,   # excluded (degenerate)
    'Freeway-MinAtar':         0,
    'BernoulliBandit-misc':    0,
    'GaussianBandit-misc':     0,
    'MNISTBandit-bsuite':      0,
}


def _per_cell_polarity(row: dict) -> float:
    """Pearson r(episode_length, mc_return) across all (burst × episode)
    eval points for one cell. Positive = longer = better (survival);
    negative = shorter = better (goal); ~0 = decoupled."""
    el = row.get('episode_length')
    mc = row.get('mc_return')
    if el is None or mc is None:
        return float('nan')
    el_arr = np.asarray(el if isinstance(el, np.ndarray) else el).astype(np.float64)
    mc_arr = np.asarray(mc if isinstance(mc, np.ndarray) else mc).astype(np.float64)
    if el_arr.shape != mc_arr.shape:
        return float('nan')
    el_flat = el_arr.flatten()
    mc_flat = mc_arr.flatten()
    if len(el_flat) < 3 or el_flat.std() == 0 or mc_flat.std() == 0:
        return float('nan')
    r, _ = pearsonr(el_flat, mc_flat)
    return float(r) if math.isfinite(r) else float('nan')


def _process_corpus(path: Path) -> pl.DataFrame | None:
    runs_path = path / 'runs.parquet'
    traces_path = path / 'traces.parquet'
    if not runs_path.exists() or not traces_path.exists():
        return None
    runs = pl.read_parquet(runs_path, columns=['id', 'env_name', 'arm_key', 'seed'])
    traces = pl.read_parquet(traces_path, columns=['id', 'episode_length', 'mc_return'])
    df = runs.join(traces, on='id', how='inner')
    rows = []
    for r in df.iter_rows(named=True):
        rows.append({
            'env_name': r['env_name'],
            'arm_key': r['arm_key'],
            'seed': r['seed'],
            'polarity_continuous': _per_cell_polarity(r),
        })
    return pl.DataFrame(rows)


def main() -> None:
    pieces = []
    for src_dir, _ in TRACE_SOURCES:
        path = Path(src_dir)
        if path.is_dir():
            df = _process_corpus(path)
            if df is not None:
                pieces.append(df)
        # Also walk sub-directories (capacity_sweep_fourrooms has sub-arms)
        if path.is_dir():
            for sub in path.iterdir():
                if sub.is_dir() and (sub / 'runs.parquet').exists() and (sub / 'traces.parquet').exists():
                    sub_df = _process_corpus(sub)
                    if sub_df is not None:
                        pieces.append(sub_df)
    if not pieces:
        print('no local traces')
        return
    all_cells = pl.concat(pieces, how='vertical_relaxed')
    print(f'cells: {len(all_cells)}')

    # Per-env distribution of polarity_continuous (vanilla baseline only)
    baseline = 'baseline'
    baseline_only = all_cells.filter(pl.col('arm_key') == baseline)
    print()
    print(f'{"env":<25} {"n_baseline":>11} {"polarity_mean":>14} {"polarity_sd":>12} {"polarity_min":>13} {"polarity_max":>13} {"cat_polarity":>13}')
    print('-' * 110)
    panel = []
    for env in sorted(baseline_only['env_name'].unique()):
        sub = baseline_only.filter(pl.col('env_name') == env).filter(pl.col('polarity_continuous').is_not_nan())
        if len(sub) == 0:
            continue
        mean_p = float(sub['polarity_continuous'].mean() or float('nan'))
        sd_p = float(sub['polarity_continuous'].std() or float('nan'))
        min_p = float(sub['polarity_continuous'].min() or float('nan'))
        max_p = float(sub['polarity_continuous'].max() or float('nan'))
        cat = CAT_POLARITY.get(env, '?')
        panel.append({'env': env, 'n_baseline': len(sub), 'mean_p': mean_p, 'sd_p': sd_p, 'cat': cat})
        print(f'{env:<25} {len(sub):>11d} {mean_p:>+14.3f} {sd_p:>12.3f} {min_p:>+13.3f} {max_p:>+13.3f} {str(cat):>13}')

    # Validation: do continuous polarity means agree with categorical?
    print()
    print('=== Validation: continuous vs categorical polarity ===')
    cont = np.array([p['mean_p'] for p in panel if isinstance(p['cat'], int) and p['cat'] != 0])
    cat = np.array([p['cat'] for p in panel if isinstance(p['cat'], int) and p['cat'] != 0])
    if len(cont) > 1:
        rho, p = spearmanr(cont, cat)
        print(f'  Spearman ρ(continuous_polarity_mean, categorical) = {rho:+.3f}, p = {p:.3g}, n_envs = {len(cont)}')
        # sign agreement
        match = sum(1 for c, k in zip(cont, cat) if c * k > 0)
        from scipy.stats import binomtest
        bt = binomtest(match, len(cont), p=0.5, alternative='greater')
        print(f'  Sign agreement: {match}/{len(cont)} envs match sign (binomial p={bt.pvalue:.3g})')

    # Save
    out = Path('experiments/findings/sync_curve_breakout/polarity_continuous_panel.json')
    out.write_text(json.dumps({
        'per_env': panel,
    }, indent=2, default=str))
    print()
    print(f'wrote: {out}')


if __name__ == '__main__':
    main()
