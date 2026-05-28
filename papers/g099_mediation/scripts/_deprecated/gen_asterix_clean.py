"""Generate figures/report_asterix_clean_corrected.png — Asterix γ=0.99 per-burst
PC trajectory using the CLEAN best mediator (`q_argmax_margin`), with the
REDQ relative-bias diagnostic alongside.

Replaces v1's `report_asterix_clean.png` which used the TAUTOLOGICAL
pstate_bias. The corrected figure shows:
  - Left: rank-correlation trajectory at q_argmax_margin (clean)
  - Right: PC per-burst CI p-values
Plus an info panel summarizing the 3-number tautology audit:
  raw pstate_bias 97% / REDQ-normalized 83% / clean q_argmax_margin 59%
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import load_g099_canonical_panel

from corroborate.analyses.dynamic_mediation.pc_adjacency import dynamic_pc_adjacency
from corroborate.analyses.panel.cross_env_probability_of_improvement import (
    cross_env_probability_of_improvement,
)
from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM


OUT = SCRIPT_DIR.parent / 'figures' / 'report_asterix_clean_corrected.png'


def main() -> None:
    df = load_g099_canonical_panel()
    cells = df.to_dicts()

    # Asterix has CNN mediators (q_argmax_margin) available
    res = dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst='q_argmax_margin_per_burst',
        outcome_per_burst='mc_return__mean_axis_-1',
        stratify_by=('env_name',), min_n_per_burst=8,
    )
    r = res.get(('Asterix-MinAtar',))
    if r is None:
        print('No Asterix result')
        return

    link = cross_env_probability_of_improvement.fn(
        cells, source='eval_best_burst_raw_mean',
        treatment_arm=DDQN_ARM, baseline_arm=VANILLA_ARM,
        stratify_by=('env_name',),
    )
    link_pxy = next(s.p_xy for s in link.per_stratum if s.stratum_id[0] == 'Asterix-MinAtar')
    mech = cross_env_probability_of_improvement.fn(
        cells, source='jensen_gap',
        treatment_arm=DDQN_ARM, baseline_arm=VANILLA_ARM,
        stratify_by=('env_name',),
    )
    mech_pxy_lt = 1 - next(s.p_xy for s in mech.per_stratum if s.stratum_id[0] == 'Asterix-MinAtar')

    bursts = np.arange(len(r.rho_marginal))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(bursts, r.rho_marginal, color='steelblue', linewidth=2,
            marker='o', markersize=4, label='marginal ρ(arm, outcome)')
    ax.plot(bursts, r.rho_partial, color='crimson', linewidth=2,
            marker='s', markersize=4, label='partial ρ | q_argmax_margin')
    ax.fill_between(bursts, r.rho_marginal, r.rho_partial,
                    alpha=0.18, color='gold', label='absorbed by mediator')
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('burst')
    ax.set_ylabel('Spearman ρ')
    ax.set_title('Per-burst rank correlations\n(CLEAN mediator: q_argmax_margin)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(bursts, np.asarray(r.p_marginal), color='steelblue',
               s=42, alpha=0.85, label='marginal CI p-value')
    ax.scatter(bursts, np.asarray(r.p_conditional), color='crimson',
               s=42, alpha=0.85, marker='s', label='conditional CI p-value (|q_argmax_margin)')
    ax.axhline(0.05, color='black', linestyle='--', linewidth=0.7, label='α=0.05')
    ax.set_yscale('log')
    ax.set_xlabel('burst')
    ax.set_ylabel('CI test p-value (log scale)')
    pct = r.n_bursts_mediator_dseparates / max(r.n_bursts_marginal_edge, 1) * 100
    ax.set_title(
        f'Asterix γ=0.99 per-burst PC CI tests\n'
        f'q_argmax_margin d-separates at {r.n_bursts_mediator_dseparates}/'
        f'{r.n_bursts_marginal_edge} bursts = {pct:.0f}%'
    )
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f'Asterix γ=0.99 — corrected after tautology audit\n'
        f'P(D<V on jens)={mech_pxy_lt:.2f}, P(D>V on outcome)={link_pxy:.2f}; '
        f'best CLEAN mediator = q_argmax_margin. '
        f'Tautology audit: raw pstate_bias=97% (rejected), '
        f'REDQ-normalized=83% (partial), q_argmax_margin=59% (this).',
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(OUT, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT.name}')


if __name__ == '__main__':
    main()
