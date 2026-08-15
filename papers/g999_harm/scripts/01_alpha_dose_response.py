"""Layer 1 — α (interpolation-strength) dose-response at Asterix γ=0.999.

The DDQN harm is not an artifact of the binary {vanilla, DDQN}
contrast: dialling the double-greedify interpolation continuously from α=0
(vanilla target) to α=1 (full DDQN) via `dampened_double_greedify(α)`
makes the outcome monotonically WORSE. A clean causal dose-response —
α is randomised across arms — establishes that the interpolation CAUSES the
harm (not a confound of the two algorithms differing elsewhere).

Panel: Asterix-MinAtar, γ=0.999, α ∈ {0, 0.25, 0.5, 0.75, 1.0},
n=15 seeds per α (75 cells). Outcome = best-burst discounted eval.

Input : papers/g999_harm/data/alpha_dose_cells.csv  (frozen)
Output: papers/g999_harm/figures/01_alpha_dose_response.{png,csv}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from _common import ALPHA_EVAL_COL, COLOR_HARMS, load_alpha_dose_cells

FIG_DIR = Path(__file__).resolve().parents[1] / 'figures'


def main() -> None:
    cells = load_alpha_dose_cells()
    alpha = cells.get_column('alpha').to_numpy()
    ev = cells.get_column(ALPHA_EVAL_COL).to_numpy()
    ok = np.isfinite(alpha) & np.isfinite(ev)
    rho, p = stats.spearmanr(alpha[ok], ev[ok])

    # Per-α summary: mean ± SEM.
    summary = (
        cells.group_by('alpha')
        .agg(
            pl.len().alias('n'),
            pl.col(ALPHA_EVAL_COL).mean().alias('eval_mean'),
            pl.col(ALPHA_EVAL_COL).std().alias('eval_sd'),
        )
        .with_columns((pl.col('eval_sd') / pl.col('n').sqrt()).alias('eval_sem'))
        .sort('alpha')
    )
    summary.write_csv(FIG_DIR / '01_alpha_dose_response.csv')

    xs = summary.get_column('alpha').to_numpy()
    ys = summary.get_column('eval_mean').to_numpy()
    es = summary.get_column('eval_sem').to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.errorbar(
        xs, ys, yerr=es, marker='o', ms=9, lw=2.2, capsize=4,
        color=COLOR_HARMS, ecolor=COLOR_HARMS, mfc=COLOR_HARMS, mec=COLOR_HARMS,
    )
    ax.set_xlabel('α  (interpolation strength: 0=vanilla → 1=full DDQN)', fontsize=11)
    ax.set_ylabel('eval_best_burst (discounted)', fontsize=11)
    ax.set_title(
        f'Asterix γ=0.999 dose-response (n={int(summary["n"][0])}/α)\n'
        f'stronger interpolation → monotonically worse: ρ={rho:+.2f}, p={p:.3f}',
        fontsize=13, fontweight='bold',
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / '01_alpha_dose_response.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(f'n={int(ok.sum())}  spearman ρ={rho:+.3f}  p={p:.4f}')
    print(summary)
    print('saved figures/01_alpha_dose_response.{png,csv}')


if __name__ == '__main__':
    main()
