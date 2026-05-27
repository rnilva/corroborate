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


def main() -> None:
    df = load_g099_canonical_panel()
    cells = df.to_dicts()

    link = cross_env_probability_of_improvement.fn(
        cells, source='eval_best_burst_raw_mean',
        treatment_arm=DDQN_ARM, baseline_arm=VANILLA_ARM,
        stratify_by=('env_name',),
    )
    link_pxy = {s.stratum_id[0]: s.p_xy for s in link.per_stratum}

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

        pxy = link_pxy.get(env, 0.5)
        ax.set_title(
            f'{env}\n'
            f'P(D>V)={pxy:.2f}  '
            f'V={results.get("V", {}).get("median_last", float("nan")):.2f}, '
            f'D={results.get("D", {}).get("median_last", float("nan")):.2f}',
            fontsize=9,
        )
        ax.set_xlabel('training step', fontsize=8)
        ax.set_ylabel('raw return (median, IQR, 5-95%)', fontsize=8)
        ax.legend(fontsize=7, loc='best')
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.suptitle(
        'γ=0.99 canonical — per-env seed-aggregated learning curves (raw return)\n'
        'per-burst median across seeds + IQR fill + 5/95 envelope; envs sorted by P(D>V)',
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT.name}')


if __name__ == '__main__':
    main()
