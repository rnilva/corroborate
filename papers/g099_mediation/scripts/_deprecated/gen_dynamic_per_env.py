"""Generate figures/report_dynamic_per_env_corrected.png — per-burst dynamic
mediation trajectories at the envs with detectable per-burst LINK, using
each env's best CLEAN mediator.

Replaces v1's `report_dynamic_5envs.png` which used the tautological
`pstate_bias` at several envs. The corrected version:
  - identifies per-env best clean mediator (from `gen_per_env_best_mediator`)
  - plots per-burst ρ_marg vs ρ_partial trajectories per env
  - shows absorption shading + per-burst CI p-values
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
from corroborate.analyses.dynamic_mediation.partial_spearman import (
    dynamic_partial_spearman,
)
from corroborate.analyses.panel.cross_env_probability_of_improvement import (
    cross_env_probability_of_improvement,
)
from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM


OUT = SCRIPT_DIR.parent / 'figures' / 'report_dynamic_per_env_corrected.png'


def main() -> None:
    df = load_g099_canonical_panel()
    cells = df.to_dicts()

    link_pxy = {s.stratum_id[0]: s.p_xy for s in
                cross_env_probability_of_improvement.fn(
                    cells, source='eval_best_burst_raw_mean',
                    treatment_arm=DDQN_ARM, baseline_arm=VANILLA_ARM,
                    stratify_by=('env_name',),
                ).per_stratum}

    # Run PC + partial-Spearman over CLEAN candidates per env, pick best
    pc_by_med: dict[str, dict] = {}
    ps_by_med: dict[str, dict] = {}
    for label, col in CLEAN_MEDIATORS:
        try:
            pc_by_med[label] = dynamic_pc_adjacency.fn(
                df, arm_field='arm_key',
                mediator_per_burst=col, outcome_per_burst='mc_return__mean_axis_-1',
                stratify_by=('env_name',), min_n_per_burst=8,
            )
            ps_by_med[label] = dynamic_partial_spearman.fn(
                df, arm_field='arm_key',
                mediator_per_burst=col, outcome_per_burst='mc_return__mean_axis_-1',
                stratify_by=('env_name',), min_n_per_burst=8,
            )
        except Exception:
            pc_by_med[label] = {}
            ps_by_med[label] = {}

    # Pick best per env: highest dsep%, tie-break highest n_marg
    per_env_best: dict[str, tuple] = {}
    for env in sorted(df['env_name'].unique().to_list()):
        candidates = []
        for label, _ in CLEAN_MEDIATORS:
            res_pc = pc_by_med.get(label, {}).get((env,))
            if res_pc is None or res_pc.n_bursts_marginal_edge == 0:
                continue
            rate = res_pc.n_bursts_mediator_dseparates / res_pc.n_bursts_marginal_edge
            res_ps = ps_by_med.get(label, {}).get((env,))
            candidates.append((rate, res_pc.n_bursts_marginal_edge, label, res_pc, res_ps))
        if candidates:
            candidates.sort(key=lambda c: (-c[0], -c[1]))
            per_env_best[env] = candidates[0]

    if not per_env_best:
        print('No env has detectable per-burst LINK')
        return

    # Sort by LINK strength
    detect_envs = sorted(per_env_best, key=lambda e: -link_pxy.get(e, 0.5))

    # Plot grid: 2 cols × ceil(N/2) rows
    n = len(detect_envs)
    n_cols = 2
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, 3.5 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for i, env in enumerate(detect_envs):
        ax = axes_flat[i]
        rate, marg, label, res_pc, res_ps = per_env_best[env]
        if res_ps is None:
            ax.set_axis_off()
            continue
        bursts = np.arange(len(res_ps.rho_marginal))
        marg_arr = np.asarray(res_ps.rho_marginal, dtype=np.float64)
        part_arr = np.asarray(res_ps.rho_partial, dtype=np.float64)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.plot(bursts, marg_arr, color='steelblue', linewidth=1.7,
                marker='o', markersize=3, label='ρ_marg(arm, outcome)')
        ax.plot(bursts, part_arr, color='crimson', linewidth=1.7,
                marker='s', markersize=3, label=f'ρ_part | {label}')
        ax.fill_between(bursts, marg_arr, part_arr, alpha=0.18, color='gold')
        pct = res_pc.n_bursts_mediator_dseparates / max(res_pc.n_bursts_marginal_edge, 1) * 100
        status = res_ps.aggregation_status.name[:7]
        ax.set_title(
            f'{env}\n'
            f'P(D>V)={link_pxy.get(env, 0.5):.2f} | '
            f'PC: {res_pc.n_bursts_marginal_edge} marg, {pct:.0f}% d-sep | '
            f'PS status: {status} | mediator: {label}',
            fontsize=8.5,
        )
        ax.set_xlabel('burst', fontsize=8)
        ax.set_ylabel('Spearman ρ', fontsize=8)
        ax.legend(fontsize=7, loc='best')
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.set_ylim(-0.9, 0.9)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_axis_off()

    fig.suptitle(
        f'Per-env dynamic mediation at the {n} envs with PC-detectable LINK\n'
        f'γ=0.99 canonical; per-env best mediator from CLEAN candidates '
        f'(tautological MC-reading mediators excluded)',
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT.name}')


if __name__ == '__main__':
    main()
