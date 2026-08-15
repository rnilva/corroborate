"""Layer 6 companion — per-window UNCLIPPED Jensen-bias trajectory (single box).

The bias-REDUCTION story over training, for the 3 conditions, on the SIGNED gap
mean(predicted_q_at_start − mc_return) per evaluation window (median + IQR over
seeds). The framework's default `jensen_gap` clamps to max(0, ·) and so floors
DDQN-indp at 0; this uses the UNCLIPPED value, so DDQN-indp's slight
under-estimation (negative) is visible. Symlog y — the range spans −10 to ~+700.

Sized for a 0.49\\linewidth panel in the camera-ready: small canvas, larger
relative fonts, no in-figure title (the LaTeX caption carries the claims).

Input : experiments/data/cache/deep2010_g999_panel.parquet  (signed_bias_per_burst)
Output: papers/g999_harm/figures/06c_bias_trajectory.{png,pdf}
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
OUT_PNG = _PAPER_DIR / 'figures' / '06c_bias_trajectory.png'
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


def main() -> None:
    cache = pl.read_parquet(CACHE).with_columns(pl.col('arm_key').replace(LAB).alias('arm'))
    assert 'signed_bias_per_burst' in cache.columns, 'cache lacks signed_bias_per_burst'

    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(4.0, 3.15))
    for nm, color in ARMS:
        sub = cache.filter(pl.col('arm') == nm)
        arrs = [np.asarray(t, dtype=np.float64) for t in sub['signed_bias_per_burst'].to_list()
                if t is not None and len(t)]
        if not arrs:
            continue
        max_len = max(len(a) for a in arrs)
        stack = np.full((len(arrs), max_len), np.nan)
        for r, a in enumerate(arrs):
            stack[r, :len(a)] = a
        x = (np.arange(max_len) + 1) * EVAL_EVERY / 1e6
        median = np.nanmedian(stack, axis=0)
        q25, q75 = np.nanpercentile(stack, 25, axis=0), np.nanpercentile(stack, 75, axis=0)
        ax.fill_between(x, q25, q75, color=color, alpha=0.15, lw=0)
        ax.plot(x, median, color=color, lw=2.0, label=nm)

    ax.axhline(0, color='black', lw=0.8)
    ax.set_yscale('symlog', linthresh=1.0)
    ax.set_xlabel('training steps (millions)')
    ax.set_ylabel('signed Jensen bias (symlog)')
    ax.text(0.03, 0.96, 'over-estimation', transform=ax.transAxes,
            fontsize=9, va='top', color='#666666')
    ax.text(0.03, 0.05, 'under-estimation', transform=ax.transAxes,
            fontsize=9, va='bottom', color='#666666')
    ax.legend(loc='lower left', bbox_to_anchor=(-0.02, 1.0), ncol=3,
              frameon=False, handlelength=1.4, columnspacing=1.1,
              borderaxespad=0.0)
    ax.grid(alpha=0.22, lw=0.6)
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=160, bbox_inches='tight')
    fig.savefig(OUT_PDF, bbox_inches='tight')  # vector PDF for the paper
    plt.close(fig)
    print(f'saved → {OUT_PNG.name}, {OUT_PDF.name}')


if __name__ == '__main__':
    main()
