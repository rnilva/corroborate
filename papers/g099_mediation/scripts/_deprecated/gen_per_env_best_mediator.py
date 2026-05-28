"""Generate figures/report_per_env_best_mediator.png — per-env best CLEAN
per-burst mediator (tautology-corrected, replaces v1's report_mediator_attribution).

For each env, runs dynamic_pc_adjacency over the CLEAN_MEDIATORS set
(no MC-reading mediators), picks the mediator with highest d-separation%,
plots bar chart with d-sep% and per-env LINK strength.
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
from _common import CLEAN_MEDIATORS, load_g099_canonical_panel

from corroborate.analyses.dynamic_mediation.pc_adjacency import dynamic_pc_adjacency
from corroborate.analyses.panel.cross_env_probability_of_improvement import (
    cross_env_probability_of_improvement,
)
from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM


OUT = SCRIPT_DIR.parent / 'figures' / 'report_per_env_best_mediator.png'


def main() -> None:
    df = load_g099_canonical_panel()
    cells = df.to_dicts()
    link = cross_env_probability_of_improvement.fn(
        cells, source='eval_best_burst_raw_mean',
        treatment_arm=DDQN_ARM, baseline_arm=VANILLA_ARM,
        stratify_by=('env_name',),
    )
    link_pxy = {s.stratum_id[0]: s.p_xy for s in link.per_stratum}

    # Run dynamic_pc_adjacency over the clean candidate set
    results_by_med: dict[str, dict] = {}
    for label, col in CLEAN_MEDIATORS:
        try:
            res = dynamic_pc_adjacency.fn(
                df, arm_field='arm_key',
                mediator_per_burst=col, outcome_per_burst='mc_return__mean_axis_-1',
                stratify_by=('env_name',), min_n_per_burst=8,
            )
            results_by_med[label] = res
        except Exception:
            results_by_med[label] = {}

    # Per env: best mediator = highest dsep_rate, tie-break by highest n_marg
    rows = []
    for env in sorted(df['env_name'].unique().to_list()):
        candidates = []
        for label, _ in CLEAN_MEDIATORS:
            res = results_by_med.get(label, {}).get((env,))
            if res is None or res.n_bursts_marginal_edge == 0:
                continue
            rate = res.n_bursts_mediator_dseparates / res.n_bursts_marginal_edge
            candidates.append((rate, res.n_bursts_marginal_edge, label, res))
        if not candidates:
            rows.append({'env': env, 'best': None, 'rate': 0.0, 'marg': 0,
                         'link': link_pxy.get(env, 0.5)})
            continue
        candidates.sort(key=lambda c: (-c[0], -c[1]))
        rate, marg, label, res = candidates[0]
        rows.append({'env': env, 'best': label, 'rate': rate * 100, 'marg': marg,
                     'link': link_pxy.get(env, 0.5),
                     'dsep': res.n_bursts_mediator_dseparates,
                     'direct': res.n_bursts_direct_edge})

    # Sort envs by LINK
    rows.sort(key=lambda r: -r['link'])

    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    short = [r['env'].split('-')[0][:13] for r in rows]
    y = np.arange(len(rows))
    rates = [r['rate'] for r in rows]
    mediator_families = {
        'bg_magnitude': '#8c564b', 'bg_disagree': '#8c564b', 'bg_disagree_cond': '#8c564b',
        'greedy_match': '#8c564b',
        'argmax_ent': '#9467bd', 'state_cond_ent': '#9467bd',
        'q_argmax_margin': '#1f77b4', 'q_action_std': '#1f77b4',
        'q_autocorr': '#1f77b4', 'q_lambda_a': '#1f77b4',
        'state_n_unq': '#2ca02c', 'state_ent': '#2ca02c', 'state_repeat': '#2ca02c',
    }
    bar_colors = [mediator_families.get(r['best'], '#d3d3d3') if r['best'] else '#d3d3d3'
                  for r in rows]
    ax.barh(y, rates, color=bar_colors, edgecolor='black')
    for i, r in enumerate(rows):
        if r['best'] is None:
            ax.text(2, i, '(no per-burst LINK at α=0.05)',
                    fontsize=8, va='center', color='gray')
        else:
            ax.text(r['rate'] + 1, i,
                    f' {r["best"]} ({r["marg"]} bursts, P(D>V)={r["link"]:.2f})',
                    fontsize=8, va='center')
    ax.axvline(70, color='black', linestyle='--', linewidth=0.7,
               label='70% d-sep threshold')
    ax.set_yticks(y)
    ax.set_yticklabels(short, fontsize=9)
    ax.set_xlim(0, 120)
    ax.set_xlabel('PC d-separation rate (% of bursts with marg edge)')
    ax.set_title(
        'Per-env best CLEAN mediator from 13-candidate audit\n'
        '(γ=0.99 canonical; tautological MC-reading mediators excluded)\n'
        'color: brown=Bellman, blue=Q-shape, purple=action-policy, green=state-coverage'
    )
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(OUT, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT.name}')


if __name__ == '__main__':
    main()
