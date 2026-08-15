"""Layer 3 — mediation absorption by channel, split by outcome direction.

Per-env partial Spearman ρ(arm, outcome | M) for a panel of candidate
mediators grouped into three channels (bias/Q-magnitude, state-coverage,
policy-shape). The bar = % of the arm→outcome association absorbed by
each mediator. The proximal (PC-selected) mediator differs by outcome
direction: HELP envs (FR / SI) route through STATE-COVERAGE; the HARM
env (Asterix) routes through BIAS — with the load-bearing caveat that
`jensen_gap` is soft-tautological (reads MC), marked with `*`.

Input : experiments/data/cache/hasselt_clean_gpanel.parquet  (via _common)
Output: papers/g999_harm/figures/03_powered_mediation.{png,csv}
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.patches import Patch
from scipy import stats

from _common import (
    COLOR_HARMS, COLOR_HELPS, OUTCOME_PEAK_COL, TREATMENT_ARM, env_label,
    load_g999_panel,
)

FIG_DIR = Path(__file__).resolve().parents[1] / 'figures'

POWERED = (
    ('Asterix-MinAtar', 'HARM'),
    ('Breakout-MinAtar', 'help'),
    ('FourRooms-misc', 'help'),
    ('SpaceInvaders-MinAtar', 'help'),
)

CANDIDATES = (
    'jensen_gap', 'q_late_mean', 'ddqn_bootstrap_gap_late',
    'state_hash_entropy_late', 'unique_states_visited_late',
    'mutual_info_state_argmax_late', 'policy_churn_late',
    'greedy_match_late', 'q_trajectory_autocorr_late',
)
# Soft-tautological mediators (read MC, the outcome's own input).
TAUTOLOGICAL = frozenset({'jensen_gap'})

CHANNEL_COLOR = {
    'bias/Q-mag': '#b2182b',
    'state-coverage': '#2166ac',
    'policy-shape': '#1a7d3a',
}
# PC-proximal node per env (from the depth-2 PC topology gate).
PROXIMAL = {
    'Asterix-MinAtar': 'jensen_gap',
    'Breakout-MinAtar': None,
    'FourRooms-misc': 'state_hash_entropy_late',
    'SpaceInvaders-MinAtar': 'state_hash_entropy_late',
}


def channel(m: str) -> str:
    if m in ('jensen_gap', 'q_late_mean', 'ddqn_bootstrap_gap_late'):
        return 'bias/Q-mag'
    if 'state_hash' in m or 'unique_states' in m:
        return 'state-coverage'
    return 'policy-shape'


def main() -> None:
    g999 = load_g999_panel()
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    csv_rows: list[dict[str, object]] = []

    for ax, (env, direction) in zip(axes.ravel(), POWERED):
        s = g999.filter(pl.col('env_name') == env).with_columns(
            (pl.col('arm_key') == TREATMENT_ARM).cast(pl.Float64).alias('arm')
        )
        arm = s.get_column('arm').to_numpy()
        ev = s.get_column(OUTCOME_PEAK_COL).to_numpy()
        ok0 = np.isfinite(arm) & np.isfinite(ev)
        marg = stats.spearmanr(arm[ok0], ev[ok0])[0]

        rows: list[tuple[str, float, str]] = []
        for m in CANDIDATES:
            if m not in s.columns:
                continue
            mv = s.get_column(m).to_numpy()
            ok = np.isfinite(mv) & ok0
            if ok.sum() < 8:
                continue
            rxy = stats.spearmanr(arm[ok], ev[ok])[0]
            rxz = stats.spearmanr(arm[ok], mv[ok])[0]
            ryz = stats.spearmanr(ev[ok], mv[ok])[0]
            den = np.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
            pr = (rxy - rxz * ryz) / den if den > 1e-9 else np.nan
            ab = (1 - pr / rxy) * 100 if abs(rxy) > 1e-9 else np.nan
            rows.append((m, float(np.clip(ab, -20, 140)), channel(m)))
            csv_rows.append({'env': env_label(env), 'direction': direction,
                             'mediator': m, 'channel': channel(m),
                             'marginal_rho': float(marg), 'absorbed_pct': float(ab),
                             'tautological': m in TAUTOLOGICAL,
                             'pc_proximal': PROXIMAL.get(env) == m})

        rows.sort(key=lambda r: r[1])
        labels = [r[0].replace('_late', '').replace('_', ' ')
                  + (' *' if r[0] in TAUTOLOGICAL else '') for r in rows]
        vals = [r[1] for r in rows]
        cols = [CHANNEL_COLOR[r[2]] for r in rows]
        ypos = np.arange(len(rows))
        ax.barh(ypos, vals, color=cols, alpha=0.85)
        for i, r in enumerate(rows):
            if r[0] == PROXIMAL[env]:
                ax.text(vals[i] + 2, i, '★ PC-proximal', va='center',
                        fontsize=9, fontweight='bold', color='black')
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(0, color='gray', lw=0.8)
        ax.axvline(100, color='gray', ls=':', lw=0.8)
        color = COLOR_HARMS if marg < 0 else COLOR_HELPS
        ax.set_title(f'{env_label(env)} γ=0.999  (marg ρ={marg:+.2f}, {direction})',
                     fontsize=12, color=color, fontweight='bold')
        ax.set_xlabel('% of arm→outcome absorbed by mediator')
        ax.grid(alpha=0.2, axis='x')

    legend = [Patch(facecolor=c, label=k) for k, c in CHANNEL_COLOR.items()]
    legend.append(Patch(facecolor='white', edgecolor='white',
                        label='* = soft-tautological (reads MC)'))
    fig.legend(handles=legend, loc='lower center', ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    plt.suptitle('Mediation absorption by channel — proximal mediator splits by '
                 'outcome direction\nHELP envs (FR/SI) → state-coverage proximal;  '
                 'HARM env (Asterix) → bias proximal', fontsize=13, fontweight='bold')
    plt.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG_DIR / '03_powered_mediation.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    pl.DataFrame(csv_rows).write_csv(FIG_DIR / '03_powered_mediation.csv')
    print(f'{len(csv_rows)} mediator rows across {len(POWERED)} envs')
    print('saved figures/03_powered_mediation.{png,csv}')


if __name__ == '__main__':
    main()
