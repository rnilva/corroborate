"""Layer 6 companion — 3-condition learning curve (single box) at Asterix γ=0.999.

Per-window mean RAW return, median across seeds (heavy line), IQR fill
(25-75%), ♦ = mean per-seed peak. vanilla vs DDQN vs DDQN-indp, all in ONE
box — the raw-eval story of Layer 6 as a trajectory: DDQN below vanilla
(harm), DDQN-indp recovering back toward (just below) vanilla.

Sized for a 0.49\\linewidth panel in the camera-ready: small canvas, larger
relative fonts, no in-figure title and no 5/95 envelope (the LaTeX caption
carries the claims; the envelope was unreadable at print size).

Input : experiments/data/cache/deep2010_g999_panel.parquet
Output: papers/g999_harm/figures/06b_learning_curve.{png,pdf}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

_PAPER_DIR = Path(__file__).resolve().parents[1]
CACHE = _PAPER_DIR.parents[1] / 'experiments/data/cache/deep2010_g999_panel.parquet'
OUT_PNG = _PAPER_DIR / 'figures' / '06b_learning_curve.png'
OUT_PDF = OUT_PNG.with_suffix('.pdf')
EVAL_EVERY = 20000

LAB = {
    'baseline': 'vanilla',
    'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)': 'DDQN',
    'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify_indep)': 'DDQN-indp',
}
ARMS = (('vanilla', '#2166ac'), ('DDQN', '#b2182b'), ('DDQN-indp', '#5e3c99'))

RC = {
    'font.size': 11,
    'axes.labelsize': 11.5,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
}


def per_window_raw_mean(mc_raw_eps_list: object) -> np.ndarray:
    """`mc_return_raw_episodes` (per-window K raw-return episodes) → per-window
    mean across episodes, in RAW return units."""
    if mc_raw_eps_list is None:
        return np.zeros(0, dtype=np.float64)
    out: list[float] = []
    for window_eps in mc_raw_eps_list:
        if window_eps is None or (hasattr(window_eps, '__len__') and len(window_eps) == 0):
            out.append(np.nan)
            continue
        vals = np.atleast_1d(np.asarray(window_eps, dtype=np.float64))
        vals = vals[np.isfinite(vals)]
        out.append(float(vals.mean()) if vals.size else np.nan)
    return np.asarray(out, dtype=np.float64)


def main() -> None:
    cache = pl.read_parquet(CACHE).with_columns(pl.col('arm_key').replace(LAB).alias('arm'))
    assert 'mc_return_raw_episodes' in cache.columns, 'cache lacks per-window raw trajectory'

    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(4.0, 3.15))
    for nm, color in ARMS:
        sub = cache.filter(pl.col('arm') == nm)
        arrs = [per_window_raw_mean(m) for m in sub['mc_return_raw_episodes'].to_list()]
        arrs = [a for a in arrs if a.size > 0 and np.any(np.isfinite(a))]
        if not arrs:
            continue
        max_len = max(len(a) for a in arrs)
        stack = np.full((len(arrs), max_len), np.nan, dtype=np.float64)
        for r, a in enumerate(arrs):
            stack[r, :len(a)] = a
        x = (np.arange(max_len) + 1) * EVAL_EVERY / 1e6
        median = np.nanmedian(stack, axis=0)
        q25, q75 = np.nanpercentile(stack, 25, axis=0), np.nanpercentile(stack, 75, axis=0)
        ax.fill_between(x, q25, q75, color=color, alpha=0.16, lw=0)
        ax.plot(x, median, color=color, lw=2.0, label=nm)
        # ♦ mean per-seed peak (step, value)
        psteps, pvals = [], []
        for a in arrs:
            if np.any(np.isfinite(a)):
                i = int(np.nanargmax(a))
                psteps.append((i + 1) * EVAL_EVERY / 1e6)
                pvals.append(float(a[i]))
        if psteps:
            ax.plot(float(np.mean(psteps)), float(np.mean(pvals)), marker='D',
                    color=color, markersize=7, markeredgecolor='black',
                    markeredgewidth=0.7, zorder=10)

    ax.set_xlabel('training steps (millions)')
    ax.set_ylabel('episode return')
    ax.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=3,
              frameon=False, handlelength=1.4, columnspacing=1.1,
              borderaxespad=0.0)
    ax.grid(alpha=0.22, lw=0.6)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches='tight')
    fig.savefig(OUT_PDF, bbox_inches='tight')  # vector PDF for the paper
    plt.close(fig)
    print(f'saved → {OUT_PNG.name}, {OUT_PDF.name}')


if __name__ == '__main__':
    main()
