"""Layer 2 — V-vs-D learning curves on the powered γ=0.999 envs.

The binary contrast that frames the case study: across the four
adequately-powered γ=0.999 envs, DDQN HARMS on Asterix (d_raw < 0) and
HELPS on Breakout / FourRooms / SpaceInvaders. Asterix is the sole
harm env at γ=0.999 — the spine of the whole study.

Per-env median raw-episode-return trajectory (IQR band) for each arm;
cell-aggregate Cohen's d on best-burst raw eval in the title.

Input : experiments/data/cache/hasselt_clean_gpanel.parquet  (via _common)
Output: papers/g999_harm/figures/02_powered_learning_curves.{png,csv}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from _common import (
    ARM_COLOR, ARM_LABEL, BASELINE_ARM, COLOR_HARMS, COLOR_HELPS,
    OUTCOME_PEAK_COL, OUTCOME_RAW_EPISODES_COL, TREATMENT_ARM, env_label,
    load_g999_panel,
)

FIG_DIR = Path(__file__).resolve().parents[1] / 'figures'

# Four adequately-powered γ=0.999 envs (n=30/arm).
POWERED = ('Asterix-MinAtar', 'Breakout-MinAtar', 'FourRooms-misc',
           'SpaceInvaders-MinAtar')


def per_burst_raw(lst: object) -> np.ndarray:
    """Per-burst mean over the raw-episode lists of one cell."""
    if lst is None:
        return np.zeros(0)
    out: list[float] = []
    for burst in lst:
        if burst is None or len(burst) == 0:
            out.append(np.nan)
            continue
        v = np.asarray(burst, dtype=np.float64)
        v = v[np.isfinite(v)]
        out.append(v.mean() if len(v) else np.nan)
    return np.asarray(out)


def cohen_d(v: np.ndarray, d: np.ndarray) -> float:
    return (d.mean() - v.mean()) / np.sqrt(
        (v.std(ddof=1) ** 2 + d.std(ddof=1) ** 2) / 2
    )


def main() -> None:
    g999 = load_g999_panel()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    rows: list[dict[str, object]] = []

    for ax, env in zip(axes.ravel(), POWERED):
        s = g999.filter(pl.col('env_name') == env)
        for arm in (BASELINE_ARM, TREATMENT_ARM):
            sub = s.filter(pl.col('arm_key') == arm)
            trajs = [t for t in (per_burst_raw(m)
                                 for m in sub[OUTCOME_RAW_EPISODES_COL].to_list())
                     if t.size > 0]
            if not trajs:
                continue
            length = min(t.size for t in trajs)
            stack = np.stack([t[:length] for t in trajs])
            x = np.arange(length)
            med = np.nanmedian(stack, axis=0)
            lo = np.nanpercentile(stack, 25, axis=0)
            hi = np.nanpercentile(stack, 75, axis=0)
            ax.plot(x, med, color=ARM_COLOR[arm], lw=2, label=ARM_LABEL[arm])
            ax.fill_between(x, lo, hi, color=ARM_COLOR[arm], alpha=0.18)

        v = s.filter(pl.col('arm_key') == BASELINE_ARM).get_column(OUTCOME_PEAK_COL).to_numpy()
        d = s.filter(pl.col('arm_key') == TREATMENT_ARM).get_column(OUTCOME_PEAK_COL).to_numpy()
        v = v[np.isfinite(v)]
        d = d[np.isfinite(d)]
        dr = cohen_d(v, d)
        direction = 'HARM' if dr < 0 else 'help'
        color = COLOR_HARMS if dr < 0 else COLOR_HELPS
        ax.set_title(f'{env_label(env)} γ=0.999   d_raw={dr:+.2f} ({direction})',
                     fontsize=12, color=color, fontweight='bold')
        ax.set_xlabel('eval burst')
        ax.set_ylabel('raw episode return (median, IQR)')
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.25)
        rows.append({'env': env_label(env), 'direction': direction,
                     'n_vanilla': int(v.size), 'n_ddqn': int(d.size),
                     'mean_vanilla': float(v.mean()), 'mean_ddqn': float(d.mean()),
                     'd_raw': float(dr)})

    plt.suptitle('DDQN vs vanilla learning curves — 4 powered γ=0.999 envs '
                 '(hasselt_clean_gpanel, n=30/arm)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG_DIR / '02_powered_learning_curves.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    pl.DataFrame(rows).write_csv(FIG_DIR / '02_powered_learning_curves.csv')
    print(pl.DataFrame(rows))
    print('saved figures/02_powered_learning_curves.{png,csv}')


if __name__ == '__main__':
    main()
