"""Layer 5 — DYNAMIC MEDIATION + CLUSTER AGGREGATION across all 12 envs.

For each env, runs `dynamic_partial_spearman` with the canonical
clean mediator (`bootstrap_gap_magnitude_per_burst`, Bellman-residual,
no MC-leak). Surfaces per-env:

  - Per-burst trajectory: ρ_marginal(b), ρ_partial(b | bg)
  - `TimeAggregationStatus`: CONSISTENT_DIRECTION / WEAK_TIME_VARYING /
    SIGN_FLIP_DETECTED / UNDERPOWERED_BURSTS
  - DL random-effects pool: ρ_pooled, τ², I², plus cluster-bootstrap CI
    (n_bootstrap=1000) for publication-grade robustness.

Cluster aggregation across envs: DL pool over per-env pooled ρ values
(N=12 envs is too small for asymptotic-normal CIs; report as descriptive).

Output: 12-panel per-env trajectory grid + cluster aggregation table.

This is the framework's canonical RL-substrate mediation form per
CLAUDE.md vocabulary (per-burst is canonical for RL).
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
from _common import (
    COLOR_HELPS, COLOR_HARMS, COLOR_NULL, COLOR_UNDERPOWERED,
    ENV_ORDER, MEDIATOR_PER_BURST_COL, MEDIATOR_PER_BURST_LABEL,
    OUTCOME_PER_BURST_COL,
    env_label, load_g099_canonical_panel,
)

from corroborate.analyses.dynamic_mediation.partial_spearman import (
    dynamic_partial_spearman, TimeAggregationStatus,
)


OUT_PNG = SCRIPT_DIR.parent / 'figures' / '05_dynamic_mediation.png'
OUT_CSV = SCRIPT_DIR.parent / 'figures' / '05_dynamic_mediation.csv'

STATUS_COLOR = {
    TimeAggregationStatus.CONSISTENT_DIRECTION:  COLOR_HELPS,
    TimeAggregationStatus.WEAK_TIME_VARYING:     COLOR_UNDERPOWERED,
    TimeAggregationStatus.SIGN_FLIP_DETECTED:    COLOR_HARMS,
    TimeAggregationStatus.UNDERPOWERED_BURSTS:   COLOR_NULL,
}


def main() -> None:
    df = load_g099_canonical_panel()
    res = dynamic_partial_spearman.fn(
        df,
        arm_field='arm_key',
        mediator_per_burst=MEDIATOR_PER_BURST_COL,
        outcome_per_burst=OUTCOME_PER_BURST_COL,
        stratify_by=('env_name',),
        min_n_per_burst=8,
        n_bootstrap=1000,
        bootstrap_seed=42,
    )
    env_to_result = {sid[0]: r for sid, r in res.items()}

    # ─── figure: 4×3 trajectory grid in ENV_ORDER ───
    fig, axes = plt.subplots(4, 3, figsize=(13, 12), sharex=False)
    axes_flat = axes.flatten()

    for i, env in enumerate(ENV_ORDER):
        ax = axes_flat[i]
        if env not in env_to_result:
            ax.text(0.5, 0.5, '(no data)', ha='center', va='center',
                    transform=ax.transAxes, fontsize=10, color='gray')
            ax.set_title(env_label(env), fontsize=10)
            continue
        r = env_to_result[env]
        x = np.arange(len(r.burst_steps))
        rho_m = np.asarray(r.rho_marginal)
        rho_p = np.asarray(r.rho_partial)
        # NaN mask
        finite = np.isfinite(rho_m) & np.isfinite(rho_p)
        ax.plot(x[finite], rho_m[finite], color='steelblue', linewidth=1.2,
                marker='o', markersize=2, label='marginal', alpha=0.85)
        ax.plot(x[finite], rho_p[finite], color='goldenrod', linewidth=1.2,
                marker='D', markersize=2, label=f'partial | bg', alpha=0.85)
        ax.axhline(0, color='black', linewidth=0.4)
        ax.set_ylim(-1, 1)
        ax.grid(alpha=0.3)

        # Title with verdict color
        status = r.aggregation_status
        color = STATUS_COLOR.get(status, COLOR_NULL)
        ax.set_title(
            f'{env_label(env)}  [{status.name}]',
            fontsize=9.5, color=color, fontweight='bold', pad=4,
        )
        # DL annotation
        anno_lines = [
            f'DL marg ρ={r.dl_marginal.rho_pooled:+.2f}  I²={r.dl_marginal.i2:.2f}',
            f'DL part ρ={r.dl_partial.rho_pooled:+.2f}  I²={r.dl_partial.i2:.2f}',
        ]
        if r.bootstrap_marginal is not None:
            anno_lines.append(
                f'boot marg [{r.bootstrap_marginal.rho_lower:+.2f},'
                f'{r.bootstrap_marginal.rho_upper:+.2f}]'
            )
        ax.text(0.02, 0.04, '\n'.join(anno_lines), transform=ax.transAxes,
                va='bottom', fontsize=6.5, family='monospace',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='#fffaf0',
                          edgecolor='#cc9900', linewidth=0.4))

        if i == 0:
            ax.legend(fontsize=7, loc='upper right')

    for j in range(len(ENV_ORDER), len(axes_flat)):
        axes_flat[j].axis('off')

    # Bottom-row x-axis labels
    for col in range(3):
        axes[-1, col].set_xlabel('burst index', fontsize=8.5)
    for row in range(4):
        axes[row, 0].set_ylabel('ρ', fontsize=8.5)

    # Cluster aggregation summary (descriptive across-env DL pool of
    # per-env DL-pooled values is over-aggregating; we just report
    # status counts + per-env table.
    status_counts: dict[str, int] = {}
    for env in ENV_ORDER:
        if env not in env_to_result:
            continue
        nm = env_to_result[env].aggregation_status.name
        status_counts[nm] = status_counts.get(nm, 0) + 1
    cluster_summary = '   '.join(
        f'{k}: {v}' for k, v in sorted(status_counts.items())
    )
    fig.suptitle(
        f'Layer 5 (DYNAMIC MEDIATION): per-env trajectory + DL pool, '
        f'mediator = {MEDIATOR_PER_BURST_LABEL}\n'
        f'Cluster: {cluster_summary}',
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')

    # ─── CSV ───
    with OUT_CSV.open('w') as f:
        f.write('env,n_bursts,status,dl_marg_rho,dl_marg_i2,'
                'dl_part_rho,dl_part_i2,boot_marg_lo,boot_marg_hi,'
                'boot_part_lo,boot_part_hi\n')
        for env in ENV_ORDER:
            if env not in env_to_result:
                continue
            r = env_to_result[env]
            bm = r.bootstrap_marginal; bp = r.bootstrap_partial
            f.write(
                f'{env_label(env)},{len(r.burst_steps)},{r.aggregation_status.name},'
                f'{r.dl_marginal.rho_pooled:+.3f},{r.dl_marginal.i2:.3f},'
                f'{r.dl_partial.rho_pooled:+.3f},{r.dl_partial.i2:.3f},'
                f'{(bm.rho_lower if bm else float("nan")):+.3f},'
                f'{(bm.rho_upper if bm else float("nan")):+.3f},'
                f'{(bp.rho_lower if bp else float("nan")):+.3f},'
                f'{(bp.rho_upper if bp else float("nan")):+.3f}\n'
            )
        f.write(f'\n# cluster status counts: {status_counts}\n')

    print(f'saved → {OUT_PNG.name}, {OUT_CSV.name}')
    print(f'  cluster: {cluster_summary}')


if __name__ == '__main__':
    main()
