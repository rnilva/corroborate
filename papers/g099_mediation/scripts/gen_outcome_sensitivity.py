"""Per-run outcome scalar sensitivity at γ=0.99 canonical.

Computes P(D>V) per env under FOUR per-run scalars:
  - peak    : max over bursts of (mean over eps)         [eval_best_burst_raw_mean]
  - final   : last burst's mean over eps                  [Agarwal-aligned converged]
  - late5   : last-5-burst window mean   (last 10% of 50) [Agarwal-aligned late-window]
  - late15  : last-15-burst window mean  (last 30% of 50) [larger late-window for stability]
  - auc     : mean over ALL bursts of mean over eps       [training-quality average]

Agarwal et al. 2021 ("Edge of the Statistical Precipice") explicitly
recommend AGAINST max-over-bursts ("best of N is upward-biased by chance")
and recommend averaging over a late window or whole training. We compare
the four to see how many env verdicts flip.

Writes:
  - figures/sensitivity_outcome_scalar_pxy.png  (heatmap-style table)
  - figures/sensitivity_outcome_scalar_pxy.csv  (raw numbers)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import load_g099_canonical_panel
from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM


OUT_FIG = SCRIPT_DIR.parent / 'figures' / 'sensitivity_outcome_scalar_pxy.png'
OUT_CSV = SCRIPT_DIR.parent / 'figures' / 'sensitivity_outcome_scalar_pxy.csv'
LATE_WINDOW = 5
LATE_WINDOW_15 = 15


def _per_burst_raw_mean(mc_raw_eps_list) -> np.ndarray:
    if mc_raw_eps_list is None:
        return np.zeros(0, dtype=np.float64)
    out: list[float] = []
    for burst_eps in mc_raw_eps_list:
        if burst_eps is None or len(burst_eps) == 0:
            out.append(np.nan)
        else:
            vals = np.asarray(burst_eps, dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            out.append(float(vals.mean()) if len(vals) else np.nan)
    return np.asarray(out, dtype=np.float64)


def _seed_scalars(arm_sub: pl.DataFrame) -> dict[str, np.ndarray]:
    """For each cell, compute the four per-run scalars."""
    peaks, finals, late5s, late15s, aucs = [], [], [], [], []
    for m in arm_sub['mc_return_raw_episodes'].to_list():
        traj = _per_burst_raw_mean(m)
        if traj.size == 0:
            continue
        finite = traj[np.isfinite(traj)]
        if finite.size == 0:
            continue
        peaks.append(np.nanmax(traj))
        finals.append(traj[-1] if np.isfinite(traj[-1]) else np.nanmean(traj[-3:]))
        k5 = min(LATE_WINDOW, traj.size)
        late5s.append(np.nanmean(traj[-k5:]))
        k15 = min(LATE_WINDOW_15, traj.size)
        late15s.append(np.nanmean(traj[-k15:]))
        aucs.append(np.nanmean(traj))
    return {
        'peak': np.asarray(peaks, dtype=np.float64),
        'final': np.asarray(finals, dtype=np.float64),
        'late5': np.asarray(late5s, dtype=np.float64),
        'late15': np.asarray(late15s, dtype=np.float64),
        'auc': np.asarray(aucs, dtype=np.float64),
    }


def _pxy(x: np.ndarray, y: np.ndarray) -> float:
    """P(X > Y) with 0.5 for ties — empirical pairwise."""
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float('nan')
    xx = x[:, None]; yy = y[None, :]
    return float(((xx > yy).sum() + 0.5 * (xx == yy).sum()) / (len(x) * len(y)))


def main() -> None:
    df = load_g099_canonical_panel()
    envs = sorted(df['env_name'].unique().to_list())

    metric_names = ('peak', 'final', 'late5', 'late15', 'auc')
    rows = []
    for env in envs:
        sub = df.filter(pl.col('env_name') == env)
        v_sub = sub.filter(pl.col('arm_key') == VANILLA_ARM)
        d_sub = sub.filter(pl.col('arm_key') == DDQN_ARM)
        if len(v_sub) == 0 or len(d_sub) == 0:
            continue
        v_scalars = _seed_scalars(v_sub)
        d_scalars = _seed_scalars(d_sub)
        row = {'env': env, 'n_V': len(v_scalars['peak']), 'n_D': len(d_scalars['peak'])}
        for m in metric_names:
            row[f'pxy_{m}'] = _pxy(d_scalars[m], v_scalars[m])
        rows.append(row)

    # Print table
    print(f'{"env":<22s} {"n_V":>4s} {"n_D":>4s} {"peak":>7s} {"final":>7s} {"late5":>7s} {"late15":>7s} {"auc":>7s}  flip?')
    print('-' * 88)
    flip_count = 0
    for r in rows:
        signs = [np.sign(r[f'pxy_{m}'] - 0.5) for m in metric_names]
        flip = (len(set(signs)) > 1) and not any(np.isnan(s) for s in signs)
        flag = ' ← FLIP' if flip else ''
        if flip:
            flip_count += 1
        print(f'{r["env"]:<22s} {r["n_V"]:>4d} {r["n_D"]:>4d} '
              f'{r["pxy_peak"]:>7.3f} {r["pxy_final"]:>7.3f} '
              f'{r["pxy_late5"]:>7.3f} {r["pxy_late15"]:>7.3f} {r["pxy_auc"]:>7.3f}{flag}')
    print('-' * 88)
    print(f'{flip_count}/{len(rows)} envs flip sign across the four metrics')

    # Cross-env aggregate: how many envs have P>0.5 in each metric?
    print()
    print('Cross-env aggregate (count of envs with P(D>V) > 0.5):')
    for m in metric_names:
        positive = sum(1 for r in rows if r[f'pxy_{m}'] > 0.5)
        n = sum(1 for r in rows if np.isfinite(r[f'pxy_{m}']))
        print(f'  {m:<7s}: {positive}/{n} envs favor DDQN')

    # Render figure: heatmap-style table
    arr = np.array([[r[f'pxy_{m}'] for m in metric_names] for r in rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 0.4 * len(rows) + 1.5))
    im = ax.imshow(arr, cmap='RdBu_r', vmin=0.2, vmax=0.8, aspect='auto')
    ax.set_xticks(range(len(metric_names)))
    ax.set_xticklabels(metric_names, fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r['env'] for r in rows], fontsize=8)
    for i in range(len(rows)):
        for j, m in enumerate(metric_names):
            val = arr[i, j]
            color = 'white' if abs(val - 0.5) > 0.18 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color)
    ax.set_title(f'P(D>V) by per-run scalar — {flip_count}/{len(rows)} envs flip sign\n'
                  'γ=0.99 canonical, Agarwal-style pairwise comparison',
                  fontsize=10)
    cbar = plt.colorbar(im, ax=ax, fraction=0.04)
    cbar.set_label('P(D>V); 0.5 = tied', fontsize=8)
    plt.tight_layout()
    plt.savefig(OUT_FIG, dpi=120, bbox_inches='tight')
    print(f'\nsaved → {OUT_FIG.name}')

    # Write CSV
    with OUT_CSV.open('w') as f:
        header = 'env,n_V,n_D,' + ','.join(f'pxy_{m}' for m in metric_names)
        f.write(header + '\n')
        for r in rows:
            cells = [r['env'], str(r['n_V']), str(r['n_D'])]
            cells.extend(f'{r[f"pxy_{m}"]:.4f}' for m in metric_names)
            f.write(','.join(cells) + '\n')
    print(f'saved → {OUT_CSV.name}')


if __name__ == '__main__':
    main()
