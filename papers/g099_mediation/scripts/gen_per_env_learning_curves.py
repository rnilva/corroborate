"""Generate figures/report_per_env_learning_curves.png — per-env seed-
aggregated learning curves (V vs D) at γ=0.99 canonical.

Uniform template across all envs, no per-env tailoring:
  - median (heavy line)
  - IQR band (25-75 percentile fill)
  - 5/95 percentile envelope (thin lines)

Units are RAW return per burst (recomputed from `mc_return_raw_episodes`),
matching the LINK outcome `eval_best_burst_raw_mean` rather than the
discounted variant. The envelope width is itself the finding per env —
narrow at saturated envs, wide at chaotic/high-variance envs.

5/95 (not min/max) because LL/Acrobot have 1-2 catastrophic-collapse
seeds that dominate true-min and compress the y-axis to illegibility.
5/95 keeps the bulk seed-spread (n=30 → excludes ~1.5 seeds per tail)
while letting the IQR band and median read.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import polars as pl

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import load_g099_canonical_panel

from corroborate.analyses.panel.cross_env_probability_of_improvement import (
    cross_env_probability_of_improvement,
)
from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM


OUT = SCRIPT_DIR.parent / 'figures' / 'report_per_env_learning_curves.png'


LATE_WINDOW_FRAC = 0.30  # last 30% of training (15 bursts of 50)


def _per_burst_raw_mean(mc_raw_eps_list) -> np.ndarray:
    """Convert `mc_return_raw_episodes` (per-burst K eps) into per-burst
    mean across eps, in RAW return units (not γ-discounted)."""
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


def _late_window_mean(traj: np.ndarray, frac: float = LATE_WINDOW_FRAC) -> float:
    """Mean over the last `frac` of bursts."""
    if traj.size == 0:
        return float('nan')
    k = max(1, int(round(traj.size * frac)))
    return float(np.nanmean(traj[-k:]))


def main() -> None:
    df = load_g099_canonical_panel()

    # Build a derived late30 per-cell column matching the panel's row order
    late_vals = {}
    for env in df['env_name'].unique():
        sub = df.filter(pl.col('env_name') == env)
        for row_id, mc in zip(sub['id'].to_list(),
                              sub['mc_return_raw_episodes'].to_list()):
            traj = _per_burst_raw_mean(mc)
            late_vals[row_id] = _late_window_mean(traj)
    df = df.with_columns(
        pl.col('id').map_elements(lambda i: late_vals.get(i, float('nan')),
                                    return_dtype=pl.Float64).alias('outcome_late30'))
    cells = df.to_dicts()

    link_peak = cross_env_probability_of_improvement.fn(
        cells, source='eval_best_burst_raw_mean',
        treatment_arm=DDQN_ARM, baseline_arm=VANILLA_ARM,
        stratify_by=('env_name',),
    )
    link_late = cross_env_probability_of_improvement.fn(
        cells, source='outcome_late30',
        treatment_arm=DDQN_ARM, baseline_arm=VANILLA_ARM,
        stratify_by=('env_name',),
    )
    link_pxy = {s.stratum_id[0]: s.p_xy for s in link_peak.per_stratum}
    link_late_pxy = {s.stratum_id[0]: s.p_xy for s in link_late.per_stratum}

    envs = sorted(df['env_name'].unique().to_list(),
                  key=lambda e: -link_pxy.get(e, 0.5))
    n = len(envs)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.2 * n_rows),
                              squeeze=False)
    axes_flat = axes.flatten()

    for i, env in enumerate(envs):
        ax = axes_flat[i]
        sub = df.filter(pl.col('env_name') == env)
        eval_every = int(sub['eval_every'].drop_nulls()[0]) if 'eval_every' in sub.columns else 20000

        results = {}
        for arm_label, arm_pred, color in [
            ('V', pl.col('arm_key') == VANILLA_ARM, 'steelblue'),
            ('D', pl.col('arm_key') == DDQN_ARM, 'crimson'),
        ]:
            arm_sub = sub.filter(arm_pred)
            if len(arm_sub) == 0:
                continue
            arrs = [_per_burst_raw_mean(m)
                    for m in arm_sub['mc_return_raw_episodes'].to_list()
                    if m is not None]
            arrs = [a for a in arrs if a.size > 0]
            if not arrs:
                continue
            max_len = max(len(a) for a in arrs)
            stack = np.full((len(arrs), max_len), np.nan, dtype=np.float64)
            for r, a in enumerate(arrs):
                stack[r, :len(a)] = a
            median = np.nanmedian(stack, axis=0)
            q25 = np.nanpercentile(stack, 25, axis=0)
            q75 = np.nanpercentile(stack, 75, axis=0)
            mn = np.nanpercentile(stack, 5, axis=0)
            mx = np.nanpercentile(stack, 95, axis=0)
            x = (np.arange(max_len) + 1) * eval_every
            ax.fill_between(x, q25, q75, color=color, alpha=0.20)
            ax.plot(x, median, color=color, linewidth=1.8,
                    label=f'{arm_label} (n={len(arrs)})')
            ax.plot(x, mn, color=color, linewidth=0.55, alpha=0.55)
            ax.plot(x, mx, color=color, linewidth=0.55, alpha=0.55)
            results[arm_label] = {'median_last': median[-1], 'n': len(arrs)}

        pxy_peak = link_pxy.get(env, 0.5)
        pxy_late = link_late_pxy.get(env, 0.5)
        # Draw the late-window region as a faint vertical band so the
        # reader sees what late30 is summarising.
        if 'V' in results and 'D' in results:
            x_max = max(results.get('x_max', 0),
                         results.get('x_max', 0))
        # Use eval_every × n_bursts × 0.7 as the band start
        band_left = (1 - LATE_WINDOW_FRAC) * 50 * eval_every
        band_right = 50 * eval_every
        ax.axvspan(band_left, band_right, color='goldenrod', alpha=0.07,
                   zorder=0)
        # Differential indicator on the title: ≈ if peak and late30 agree on sign
        sign_peak = '+' if pxy_peak > 0.5 else ('−' if pxy_peak < 0.5 else '·')
        sign_late = '+' if pxy_late > 0.5 else ('−' if pxy_late < 0.5 else '·')
        agree = '✓' if sign_peak == sign_late else '↕'
        ax.set_title(
            f'{env}  {agree}\n'
            f'P(D>V)  peak={pxy_peak:.2f}  late30={pxy_late:.2f}',
            fontsize=9,
        )
        ax.set_xlabel('training step', fontsize=8)
        ax.set_ylabel('raw return (median, IQR, 5-95%)', fontsize=8)
        ax.legend(fontsize=7, loc='best')
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_axis_off()

    n_agree = sum(1 for e in envs
                   if (link_pxy.get(e, 0.5) - 0.5) * (link_late_pxy.get(e, 0.5) - 0.5) >= 0)
    fig.suptitle(
        'γ=0.99 canonical — per-env seed-aggregated learning curves (raw return)\n'
        'per-burst median across seeds + IQR + 5/95 envelope; gold band = late30 window\n'
        f'Title shows P(D>V) under BOTH metrics — peak (best-burst) and late30 (last 30%); '
        f'{n_agree}/{len(envs)} envs agree on sign ✓; rest are ↕ metric-sensitive',
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT.name}')


if __name__ == '__main__':
    main()
