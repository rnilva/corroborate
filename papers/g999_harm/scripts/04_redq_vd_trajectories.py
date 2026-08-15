"""Layer 4 — REDQ overestimation (Chen 2021) V vs D, powered γ=0.999 envs.

REDQ relative bias = (Q − MC)/|MC|, per eval burst. Two views of the
same trajectory data across the four powered envs:

  04a (overlay)      — V and D curves with the shaded gap = how much the
                       DDQN clip removes. Harm (Asterix) leaves a large
                       residual; help envs are controlled to <10×.
  04b (line, log-y)  — V-arm and D-arm panels side-by-side, all envs
                       overlaid, to read absolute bias magnitude.

DDQN reduces overestimation in EVERY env (mech HELD universally); the
point of Layer 5 is that the reduction does NOT track the outcome.

Input : experiments/data/cache/hasselt_clean_gpanel.parquet  (via _common)
Output: papers/g999_harm/figures/04a_redq_overlay.png
        papers/g999_harm/figures/04b_redq_vd_lines.png
        papers/g999_harm/figures/04_redq_vd.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from _common import (
    BASELINE_ARM, COLOR_HARMS, COLOR_HELPS, REDQ_BIAS_COL, TREATMENT_ARM,
    env_label, load_g999_panel,
)

FIG_DIR = Path(__file__).resolve().parents[1] / 'figures'
REDQ_PER_BURST = 'normalized_bias_redq_per_burst'

# (env, direction, d_raw) — d_raw from Layer 2 cell-aggregate Cohen's d.
POWERED = (
    ('Asterix-MinAtar', 'HARM', -0.80, COLOR_HARMS),
    ('Breakout-MinAtar', 'help', +0.66, COLOR_HELPS),
    ('FourRooms-misc', 'help', +1.35, '#762a83'),
    ('SpaceInvaders-MinAtar', 'help', +2.16, '#2166ac'),
)


def traj(g999: pl.DataFrame, env: str, arm: str) -> np.ndarray | None:
    s = g999.filter((pl.col('env_name') == env) & (pl.col('arm_key') == arm))
    a = [np.asarray(x, dtype=float)
         for x in s.get_column(REDQ_PER_BURST).to_list()
         if x is not None and len(x) > 0]
    if not a:
        return None
    length = min(len(z) for z in a)
    return np.nanmean(np.stack([z[:length] for z in a]), axis=0)


def late(t: np.ndarray | None) -> float:
    return float(np.nanmean(t[-int(len(t) * 0.3):])) if t is not None else np.nan


def main() -> None:
    g999 = load_g999_panel()

    # ── 04a: V/D overlay per env, shaded gap ──
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    csv_rows: list[dict[str, object]] = []
    for ax, (env, direction, dr, _c) in zip(axes.ravel(), POWERED):
        tv = traj(g999, env, BASELINE_ARM)
        td = traj(g999, env, TREATMENT_ARM)
        if tv is None or td is None:
            continue
        x = np.arange(len(tv))
        ax.plot(x, tv, color='#2166ac', lw=2.2, label='V (vanilla)')
        ax.plot(x[:len(td)], td, color='#b2182b', lw=2.2, label='D (DDQN)')
        ax.fill_between(x[:len(td)], td, tv[:len(td)], color='gray', alpha=0.18)
        rv, rd = late(tv), late(td)
        color = COLOR_HARMS if dr < 0 else COLOR_HELPS
        ax.set_title(f'{env_label(env)} γ=0.999  ({direction}, d_raw={dr:+.2f})\n'
                     f'V={rv:.1f}× → D={rd:.1f}×  (clip leaves {rd:.0f}× residual)',
                     fontsize=11, color=color, fontweight='bold')
        ax.set_xlabel('eval burst')
        ax.set_ylabel('REDQ overestimation (Q−MC)/|MC|')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)
        csv_rows.append({'env': env_label(env), 'direction': direction, 'd_raw': dr,
                         'redq_V_late': rv, 'redq_D_late': rd,
                         'pct_reduced': (1 - rd / rv) * 100 if rv > 0 else np.nan})
    plt.suptitle('REDQ overestimation V vs D per env (γ=0.999): shaded gap = clip removal\n'
                 'Harm (Asterix) = clip leaves a large residual; help envs controlled to <10×',
                 fontsize=12, fontweight='bold')
    plt.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(FIG_DIR / '04a_redq_overlay.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── 04b: V-arm / D-arm line panels (log-y) ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ai, arm in enumerate((BASELINE_ARM, TREATMENT_ARM)):
        for env, direction, dr, c in POWERED:
            t = traj(g999, env, arm)
            if t is None:
                continue
            lab = (f'{env_label(env)} ({direction} d={dr:+.2f})'
                   if ai == 0 else env_label(env))
            axes[ai].plot(np.arange(len(t)), t, color=c, lw=2, label=lab)
        name = 'V (vanilla)' if arm == BASELINE_ARM else 'D (DDQN)'
        axes[ai].set_title(f'{name}-arm REDQ relative bias (Q−MC)/|MC|')
        axes[ai].set_yscale('log')
        axes[ai].set_xlabel('eval burst')
        axes[ai].grid(alpha=0.25)
        axes[ai].legend(fontsize=8)
    axes[0].set_ylabel('REDQ rel bias')
    plt.suptitle('REDQ overestimation (Chen 2021) V vs D, γ=0.999 — '
                 'does DDQN reduce it, and does reduction track outcome?',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG_DIR / '04b_redq_vd_lines.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    pl.DataFrame(csv_rows).write_csv(FIG_DIR / '04_redq_vd.csv')
    print(pl.DataFrame(csv_rows))
    print('saved figures/04a_redq_overlay.png, 04b_redq_vd_lines.png, 04_redq_vd.csv')


if __name__ == '__main__':
    main()
