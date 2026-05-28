"""Generate figures/report_3layer_summary_corrected.png — three-panel
L1 directional × L1 magnitude × L3 per-env best-clean-mediator summary.

Replaces v1's `report_3layer_summary.png`. Uses the CLEAN_MEDIATORS set
(tautological MC-reading mediators excluded).
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

from corroborate.analyses.panel.cross_env_consistency_binomial import (
    cross_env_consistency_binomial,
)
from corroborate.analyses.panel.cross_env_probability_of_improvement import (
    cross_env_probability_of_improvement,
)
from corroborate.analyses.panel.stratified_arm_diff_pooled import (
    stratified_arm_diff_pooled,
)
from corroborate.analyses.dynamic_mediation.pc_adjacency import dynamic_pc_adjacency
from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM


OUT = SCRIPT_DIR.parent / 'figures' / 'report_3layer_summary_corrected.png'


def main() -> None:
    df = load_g099_canonical_panel()
    cells = df.to_dicts()

    # L1: directional + magnitude
    mech_binom = cross_env_consistency_binomial.fn(
        cells, source='jensen_gap',
        treatment_arm=DDQN_ARM, baseline_arm=VANILLA_ARM,
        stratify_by=('env_name',),
        predicted_direction='a_lt_b', null_floor=0.0,
    )
    mech_pool = stratified_arm_diff_pooled.fn(
        cells, source='jensen_gap',
        treatment_arm=DDQN_ARM, baseline_arm=VANILLA_ARM,
        stratify_by=('env_name',),
        min_baseline_predictor=0.0,
    )
    out_link = cross_env_probability_of_improvement.fn(
        cells, source='eval_best_burst_raw_mean',
        treatment_arm=DDQN_ARM, baseline_arm=VANILLA_ARM,
        stratify_by=('env_name',),
    )
    out_pool = stratified_arm_diff_pooled.fn(
        cells, source='eval_best_burst_raw_mean',
        treatment_arm=DDQN_ARM, baseline_arm=VANILLA_ARM,
        stratify_by=('env_name',),
        min_baseline_predictor=0.0,
    )
    mech_pxy_by_env = {
        s.stratum_id[0]: 1 - s.p_xy for s in
        cross_env_probability_of_improvement.fn(
            cells, source='jensen_gap',
            treatment_arm=DDQN_ARM, baseline_arm=VANILLA_ARM,
            stratify_by=('env_name',),
        ).per_stratum
    }
    out_pxy_by_env = {s.stratum_id[0]: s.p_xy for s in out_link.per_stratum}

    # L3: per-env best CLEAN mediator
    pc_best_per_env: dict[str, tuple[str, int, int, int]] = {}
    for env in sorted(df['env_name'].unique().to_list()):
        candidates = []
        for label, col in CLEAN_MEDIATORS:
            try:
                res = dynamic_pc_adjacency.fn(
                    df, arm_field='arm_key',
                    mediator_per_burst=col,
                    outcome_per_burst='mc_return__mean_axis_-1',
                    stratify_by=('env_name',), min_n_per_burst=8,
                ).get((env,))
            except Exception:
                continue
            if res is None or res.n_bursts_marginal_edge == 0:
                continue
            rate = res.n_bursts_mediator_dseparates / res.n_bursts_marginal_edge
            candidates.append((rate, res.n_bursts_marginal_edge, label, res))
        if not candidates:
            continue
        candidates.sort(key=lambda c: (-c[0], -c[1]))
        rate, marg, label, res = candidates[0]
        pc_best_per_env[env] = (label, marg, res.n_bursts_mediator_dseparates,
                                  res.n_bursts_direct_edge)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))

    # Panel A: rank-based (MECH, LINK) scatter
    ax = axes[0]
    for env in sorted(df['env_name'].unique().to_list()):
        m = mech_pxy_by_env.get(env, 0.5)
        l = out_pxy_by_env.get(env, 0.5)
        color = ('darkgreen' if (m > 0.6 and l > 0.55)
                 else 'darkorange' if (m > 0.6 and l <= 0.55)
                 else 'gray')
        ax.scatter(m, l, color=color, s=120, edgecolor='black', zorder=3, alpha=0.85)
        ax.annotate(env.split('-')[0][:11], (m, l),
                    fontsize=8, xytext=(4, 4), textcoords='offset points')
    ax.axhline(0.5, color='black', linewidth=0.5)
    ax.axvline(0.5, color='black', linewidth=0.5)
    ax.axhline(0.55, color='black', linestyle='--', linewidth=0.4)
    ax.axvline(0.6, color='black', linestyle='--', linewidth=0.4)
    ax.set_xlabel("MECH: rank P(D's jens < V's jens)")
    ax.set_ylabel("LINK: rank P(D > V on outcome)")
    ax.set_title(
        f'L1 cross-env directional\n'
        f'MECH binom p={mech_binom.p_value:.3f}; '
        f'LINK perm p={out_link.p_permutation:.3f}'
    )
    ax.set_xlim(0.4, 1.05)
    ax.set_ylim(0.4, 0.9)
    ax.grid(alpha=0.3)

    # Panel B: DL pool diagnostics
    ax = axes[1]
    metrics = ['MECH (d)', 'LINK (d)']
    ds = [mech_pool.pooled_d, out_pool.pooled_d]
    pis = [(mech_pool.pooled.pi_lo, mech_pool.pooled.pi_hi),
           (out_pool.pooled.pi_lo, out_pool.pooled.pi_hi)]
    i2s = [mech_pool.i_squared, out_pool.i_squared]
    y = np.arange(len(metrics))
    for i, (d, pi, i2) in enumerate(zip(ds, pis, i2s)):
        ax.barh(i, pi[1] - pi[0], left=pi[0], color='lightgray',
                edgecolor='black', alpha=0.6,
                label='95% PI' if i == 0 else None)
        ax.scatter(d, i, color='crimson', s=200, marker='|',
                   linewidth=3, label='pooled d' if i == 0 else None)
        ax.text(d + 0.3, i, f'I²={i2:.2f}', fontsize=11, va='center')
    ax.axvline(0, color='black', linewidth=0.7)
    ax.set_yticks(y); ax.set_yticklabels(metrics)
    ax.set_xlabel("Cohen's d")
    ax.set_title('L1 cross-env magnitude (DL pool)\n'
                 'Both PIs cross zero → verdict=no_effect (framework refuses)')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3, axis='x')

    # Panel C: per-env L3 dynamic PC (CLEAN candidates)
    ax = axes[2]
    envs_sorted = sorted(df['env_name'].unique().to_list(),
                          key=lambda e: -out_pxy_by_env.get(e, 0.5))
    short = [e.split('-')[0][:13] for e in envs_sorted]
    y = np.arange(len(envs_sorted))
    dsep = [pc_best_per_env[e][2] if e in pc_best_per_env else 0
            for e in envs_sorted]
    direct = [pc_best_per_env[e][3] if e in pc_best_per_env else 0
              for e in envs_sorted]
    marg = [pc_best_per_env[e][1] if e in pc_best_per_env else 0
            for e in envs_sorted]
    ax.barh(y, dsep, color='darkgreen', label='d-sep (mediated)', alpha=0.85)
    ax.barh(y, direct, left=dsep, color='crimson', label='direct edge', alpha=0.85)
    for i, e in enumerate(envs_sorted):
        if e in pc_best_per_env:
            label, m, d, dr = pc_best_per_env[e]
            pct = d / max(m, 1) * 100
            ax.text(m + 0.3, i, f' {label} ({pct:.0f}%)',
                    fontsize=8, va='center')
        else:
            ax.text(0.5, i, '(no LINK)', fontsize=8, va='center', color='gray')
    ax.set_yticks(y); ax.set_yticklabels(short, fontsize=9)
    ax.set_xlabel('n bursts with arm—outcome marg edge at α=0.05')
    ax.set_title(
        "L3 per-burst dynamic PC (CLEAN candidates only)\n"
        "tautological MC-reading mediators excluded"
    )
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3, axis='x')

    fig.suptitle(
        'γ=0.99 canonical DDQN mediation — three-layer framework analysis '
        '(tautology-corrected)', fontsize=13, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT.name}')


if __name__ == '__main__':
    main()
