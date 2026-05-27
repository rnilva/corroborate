"""Generate figures/report_per_env_learning_curves.png — per-env seed-
aggregated learning curves (V vs D) at γ=0.99 canonical.

For each env, plot the median across seeds (±IQR band) of per-burst
discounted MC return (`mc_return__mean_axis_-1`) over training. Both
arms overlaid per subplot. This is the LINK visualization that
complements the per-cell d / P(D>V) scalar tables in the report.

Median + IQR (not mean ± SE) per user's prior pedantic correction
about outlier-driven per-cell aggregation (§3.4).
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


def main() -> None:
    df = load_g099_canonical_panel()
    cells = df.to_dicts()

    # Per-env LINK strength (for the title annotation)
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
            arrs = [np.asarray(a, dtype=np.float64)
                    for a in arm_sub['mc_return__mean_axis_-1'].to_list()
                    if a is not None]
            if not arrs:
                continue
            max_len = max(len(a) for a in arrs)
            stack = np.full((len(arrs), max_len), np.nan, dtype=np.float64)
            for r, a in enumerate(arrs):
                stack[r, :len(a)] = a
            median = np.nanmedian(stack, axis=0)
            q25 = np.nanpercentile(stack, 25, axis=0)
            q75 = np.nanpercentile(stack, 75, axis=0)
            burst_idx = np.arange(max_len)
            x = (burst_idx + 1) * eval_every
            ax.plot(x, median, color=color, linewidth=1.8,
                    label=f'{arm_label} (n={len(arrs)})')
            ax.fill_between(x, q25, q75, color=color, alpha=0.18)
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
        ax.set_ylabel('discounted MC (median ± IQR)', fontsize=8)
        ax.legend(fontsize=7, loc='best')
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.suptitle(
        'γ=0.99 canonical — per-env seed-aggregated learning curves\n'
        'discounted MC return per-burst, median across seeds ± IQR band; envs sorted by P(D>V)',
        fontsize=12,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT.name}')


if __name__ == '__main__':
    main()
