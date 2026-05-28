"""Snake 1M vs 3M learning curves — γ-discounted MC return per burst,
overlaid for both corpora and both arms.

Reads `mc_return__mean_axis_-1` (per-burst γ-discounted mean) from
the cache. Step-axis derived as `(burst_index + 1) × eval_every`
where eval_every is read from the cell record (50k for 1M, 20k
for 3M).

For each (corpus, arm), plot:
  - solid line: median across cells at each burst index
  - shaded ribbon: 25-75 percentile envelope
  - thin dashed: 5-95 percentile envelope

Two arms (V, D) × two corpora (1M, 3M) = 4 lines.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import COLOR_HARMS, COLOR_NULL, ARM_COLOR  # noqa: F401

OUT_PNG = SCRIPT_DIR.parent / 'figures' / '05c_snake_3M_vs_1M_curves.png'
OUTCOME_COL = 'mc_return__mean_axis_-1'


def _gather(sub: pl.DataFrame, col: str = OUTCOME_COL) -> np.ndarray:
    """Stack list-typed col into (n_cells, max_len) NaN-padded."""
    arrs = [np.asarray(a, dtype=np.float64) for a in sub[col].to_list() if a is not None]
    if not arrs:
        return np.zeros((0, 0))
    max_L = max(len(a) for a in arrs)
    padded = np.full((len(arrs), max_L), np.nan)
    for i, a in enumerate(arrs):
        padded[i, :len(a)] = a
    return padded


def main() -> None:
    df = pl.read_parquet('experiments/data/cache/hasselt_clean.parquet')
    snake = df.filter(pl.col('env_name') == 'Snake-jumanji')
    print(f'snake total cells: {snake.height}')

    fig, ax = plt.subplots(figsize=(10, 5.5))

    corpora = [
        ('snake_1M',                     '1M',  '-',   1.6),
        ('snake_g099_canonical_3M_ckpt', '3M',  '-',   2.0),
    ]
    arm_label = {
        'baseline': ('V', '#1f77b4'),
        'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)': ('D', '#d62728'),
    }

    for corp, corp_label, ls, lw in corpora:
        sub = snake.filter(pl.col('corpus') == corp)
        eval_every = sub['eval_every'][0]
        for arm_key in sub.select('arm_key').unique().to_series().to_list():
            label, color = arm_label[arm_key]
            arm_sub = sub.filter(pl.col('arm_key') == arm_key)
            stack = _gather(arm_sub)
            if stack.size == 0:
                continue
            n_bursts = stack.shape[1]
            # x-axis: burst INDEX × eval_every (= training step at end of burst)
            xs = (np.arange(n_bursts) + 1) * eval_every / 1e6  # in millions of steps
            med = np.nanmedian(stack, axis=0)
            q25 = np.nanpercentile(stack, 25, axis=0)
            q75 = np.nanpercentile(stack, 75, axis=0)
            q05 = np.nanpercentile(stack, 5, axis=0)
            q95 = np.nanpercentile(stack, 95, axis=0)
            ax.plot(xs, med, color=color, linestyle=ls, linewidth=lw,
                    label=f'{corp_label} · {label} (n={arm_sub.height})',
                    alpha=0.9 if corp_label == '3M' else 0.55)
            alpha_iqr = 0.16 if corp_label == '3M' else 0.08
            ax.fill_between(xs, q25, q75, color=color, alpha=alpha_iqr, linewidth=0)
            ax.plot(xs, q05, color=color, linestyle=':', linewidth=0.6,
                    alpha=0.4 if corp_label == '3M' else 0.2)
            ax.plot(xs, q95, color=color, linestyle=':', linewidth=0.6,
                    alpha=0.4 if corp_label == '3M' else 0.2)

    ax.set_xlabel('training steps (millions)', fontsize=10.5)
    ax.set_ylabel(r'eval MC return (γ=0.99-discounted, per burst)', fontsize=10.5)
    ax.set_title(
        'Snake-jumanji γ=0.99 — learning curves: 1M vs 3M corpus\n'
        '(median ± IQR ribbon; dotted = 5/95 percentile)',
        fontsize=11,
    )
    ax.axvline(1.0, color='#aaa', linestyle='--', linewidth=0.8)
    ax.text(1.02, ax.get_ylim()[1] * 0.02, '1M cutoff', fontsize=8, color='#888')
    ax.legend(loc='best', fontsize=9, frameon=False)
    ax.grid(alpha=0.25, linewidth=0.4)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT_PNG.name}')

    # Quick numerical comparison
    print('\nPer-corpus peak (median across cells of max-over-bursts) and final-30%-burst median:')
    for corp, corp_label, _, _ in corpora:
        sub = snake.filter(pl.col('corpus') == corp)
        for arm_key, (lbl, _) in arm_label.items():
            arm_sub = sub.filter(pl.col('arm_key') == arm_key)
            stack = _gather(arm_sub)
            if stack.size == 0:
                continue
            n_bursts = stack.shape[1]
            tail_start = int(n_bursts * 0.7)
            peak_per_cell = np.nanmax(stack, axis=1)
            tail_per_cell = np.nanmean(stack[:, tail_start:], axis=1)
            print(f'  {corp_label:>3s} {lbl}: peak={np.median(peak_per_cell):.4f}  '
                  f'tail70%={np.median(tail_per_cell):.4f}')


if __name__ == '__main__':
    main()
