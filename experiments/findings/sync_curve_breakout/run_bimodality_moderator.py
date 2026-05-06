"""Test whether per-env bimodality of cell-aggregate outcomes
moderates the polarity-coupling residual.

Setup:
  Baseline: r ≈ 0.535 · env_reward_polarity  (R² = 0.785, n_envs=8)
  Largest residual: FourRooms (resid = -0.40, |r|=0.87 vs polarity-
  predicted 0.46). Hypothesis: FourRooms's policy fluctuates between
  success-mode (L≈30, return≈0.74) and failure-mode (L≈500, return=0)
  across cells. The cross-cell Δ-coupling spans this regime boundary,
  amplifying r beyond what within-cell polarity would predict.

Operationalization:
  Per env, compute three bimodality measures over baseline-cell
  `eval_best_burst_mean` distribution:
    1. Sarle's bimodality coefficient (BC). BC > 0.555 → bimodal.
    2. Hartigan dip statistic (no scipy.diptest, so use a simple proxy).
    3. Mode-spread: range / (sd × √n) — a normalised range; large
       values suggest the distribution spans more than 1 SD.
    4. "Outcome modal share": min(frac_above_median, frac_below_median).
       Non-bimodal: 0.5; bimodal: peaks pulled to extremes.

  Also compute on the COMBINED (baseline + DDQN) distribution: if
  the intervention itself creates mode-switch (FourRooms case), the
  combined distribution's BC should be higher than baseline-only.

  Cross-env: Spearman ρ between each measure and |residual|.

Output: per_env panel + cross-env tests as JSON.
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

# From polarity_x_undamped_panel.json + the OLS fit:
SLOPE_POLARITY = 0.535


def sarle_bc(x: np.ndarray) -> float:
    """Sarle's bimodality coefficient. Returns NaN for n<4 or
    degenerate (zero variance) input."""
    n = len(x)
    if n < 4:
        return float('nan')
    if x.std() == 0:
        return float('nan')
    skew = float(stats.skew(x, bias=False))
    kurt = float(stats.kurtosis(x, fisher=True, bias=False))
    correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    denom = kurt + correction
    if denom <= 0:
        return float('nan')
    return (skew ** 2 + 1) / denom


def mode_modal_share(x: np.ndarray) -> float:
    """Fraction of mass NOT at the median. For unimodal distributions
    centred near the median, this is ~0.5. For bimodal distributions
    with peaks at extremes, mass concentrates away from the median.

    Specifically: (min(frac_far_from_median, n-frac_far_from_median) / n).
    Range ~0 (uniform) to 0.5 (perfectly bimodal). Higher = more
    bimodal-like."""
    if len(x) < 4:
        return float('nan')
    med = float(np.median(x))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    if iqr == 0:
        return 0.0
    far = np.abs(x - med) > 0.5 * iqr
    frac = float(far.mean())
    return min(frac, 1.0 - frac)


def normalised_range(x: np.ndarray) -> float:
    """Range divided by (sd × √n). Captures how much of the
    distribution's spread the support actually covers. Bimodal
    distributions tend to have larger normalised range."""
    if len(x) < 4 or x.std() == 0:
        return float('nan')
    return float((x.max() - x.min()) / (x.std() * np.sqrt(len(x))))


def main() -> None:
    cache = pl.read_parquet(CACHE_PATH)
    print(f'cache: {len(cache)} rows')

    # Per-env polarities + observed r (from prior decomposition)
    prior_panel = json.loads(
        Path('experiments/findings/sync_curve_breakout/polarity_x_undamped_panel.json').read_text()
    )['per_env']
    by_env = {p['env']: p for p in prior_panel}

    panel = []
    finite = pl.col('eval_best_burst_mean').is_finite()

    for env, row in by_env.items():
        sub = cache.filter(pl.col('env_name') == env)
        v = sub.filter((pl.col('arm_key') == 'baseline') & finite)
        d = sub.filter((pl.col('arm_key') == DDQN) & finite)
        if len(v) < 4 or len(d) < 4:
            continue

        v_o = v['eval_best_burst_mean'].to_numpy()
        d_o = d['eval_best_burst_mean'].to_numpy()
        all_o = np.concatenate([v_o, d_o])

        bc_v = sarle_bc(v_o)
        bc_d = sarle_bc(d_o)
        bc_all = sarle_bc(all_o)

        ms_v = mode_modal_share(v_o)
        ms_all = mode_modal_share(all_o)

        nr_v = normalised_range(v_o)
        nr_all = normalised_range(all_o)

        polarity = float(row['polarity'])
        actual_r = float(row['r'])
        fitted = SLOPE_POLARITY * polarity
        residual = actual_r - fitted

        panel.append({
            'env': env,
            'polarity': polarity,
            'r': actual_r,
            'fitted': fitted,
            'residual': residual,
            'abs_residual': abs(residual),
            'n_v': len(v_o),
            'n_d': len(d_o),
            'mean_o_v': float(v_o.mean()),
            'sd_o_v': float(v_o.std()),
            'min_o_v': float(v_o.min()),
            'max_o_v': float(v_o.max()),
            'bc_v': bc_v,
            'bc_d': bc_d,
            'bc_all': bc_all,
            'modal_share_v': ms_v,
            'modal_share_all': ms_all,
            'normalised_range_v': nr_v,
            'normalised_range_all': nr_all,
        })

    print()
    print('=== Per-env bimodality measures (baseline cells) ===')
    print()
    print(f'{"env":<24} {"pol":>7} {"r":>7} {"resid":>7} {"n_v":>4} {"BC_v":>6} {"BC_all":>7} {"MS_v":>6} {"NR_v":>6}')
    print('-' * 90)
    for p in sorted(panel, key=lambda x: -x['abs_residual']):
        bc_v = p['bc_v']
        bc_all = p['bc_all']
        ms_v = p['modal_share_v']
        nr_v = p['normalised_range_v']
        print(
            f'{p["env"]:<24} {p["polarity"]:>+7.3f} {p["r"]:>+7.3f} {p["residual"]:>+7.3f} '
            f'{p["n_v"]:>4d} '
            f'{bc_v:>6.3f} {bc_all:>7.3f} {ms_v:>6.3f} {nr_v:>6.3f}'
        )

    # Cross-env tests
    print()
    print('=== Cross-env tests: bimodality vs |residual| ===')
    abs_resid = np.array([p['abs_residual'] for p in panel])
    for name in ('bc_v', 'bc_d', 'bc_all', 'modal_share_v', 'modal_share_all',
                 'normalised_range_v', 'normalised_range_all'):
        v = np.array([p[name] for p in panel])
        if np.isnan(v).any():
            mask = ~np.isnan(v)
            v_m = v[mask]
            r_m = abs_resid[mask]
        else:
            v_m = v
            r_m = abs_resid
        if len(v_m) < 3 or v_m.std() == 0:
            print(f'  {name:<22}  (insufficient or degenerate)')
            continue
        rho_s, p_s = stats.spearmanr(v_m, r_m)
        rho_p, p_p = stats.pearsonr(v_m, r_m)
        print(f'  {name:<22}  Spearman={rho_s:+.3f} (p={p_s:.3g})   Pearson={rho_p:+.3f} (p={p_p:.3g})  (n={len(v_m)})')

    # Also: does bimodality predict signed residual (not just magnitude)?
    print()
    print('=== Bimodality vs SIGNED residual ===')
    signed_resid = np.array([p['residual'] for p in panel])
    for name in ('bc_all', 'modal_share_all', 'normalised_range_all'):
        v = np.array([p[name] for p in panel])
        if np.isnan(v).any():
            mask = ~np.isnan(v)
            v_m = v[mask]; r_m = signed_resid[mask]
        else:
            v_m = v; r_m = signed_resid
        if len(v_m) < 3 or v_m.std() == 0:
            continue
        rho_s, p_s = stats.spearmanr(v_m, r_m)
        print(f'  {name:<22}  ρ(., signed_resid) = {rho_s:+.3f} (p={p_s:.3g})')

    # Save
    out = Path('experiments/findings/sync_curve_breakout/bimodality_panel.json')
    out.write_text(json.dumps({'per_env': panel}, indent=2))
    print(f'\nwrote: {out}')


if __name__ == '__main__':
    main()
