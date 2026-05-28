"""Generate figures/report_redq_mediator_comparison.png — per-env
dynamic mediation comparing three mediator families:

  1. `normalized_bias_redq_per_burst` (REDQ-style, scale-invariant,
     tautological by construction — reads mc_return_from_step)
  2. `mean_per_state_cumulative_bias_per_burst` (raw bias in env-units,
     tautological by construction — reads mc_return_from_step)
  3. Best CLEAN mediator per env (picked from CLEAN_MEDIATORS, no
     mc_return_from_step in reads — non-tautological)

The headline question: does REDQ-normalization tell a different story
from raw bias? And does either tautological diagnostic mediator track
the same per-burst dynamics as the best clean mediator?

Layout: one row per env, three columns (REDQ / raw bias / clean).
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

from corroborate.analyses.dynamic_mediation.partial_spearman import (
    dynamic_partial_spearman,
)
from corroborate.analyses.dynamic_mediation.pc_adjacency import (
    dynamic_pc_adjacency,
)
from corroborate.analyses.panel.cross_env_probability_of_improvement import (
    cross_env_probability_of_improvement,
)
from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM


OUT = SCRIPT_DIR.parent / 'figures' / 'report_redq_mediator_comparison.png'

REDQ = ('REDQ_norm_bias', 'normalized_bias_redq_per_burst')
RAW = ('raw_pstate_bias', 'mean_per_state_cumulative_bias_per_burst')


def _run_dynamic(df, mediator_col: str):
    return dynamic_partial_spearman.fn(
        df, arm_field='arm_key',
        mediator_per_burst=mediator_col,
        outcome_per_burst='mc_return__mean_axis_-1',
        stratify_by=('env_name',), min_n_per_burst=8,
    )


def _run_pc(df, mediator_col: str):
    return dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst=mediator_col,
        outcome_per_burst='mc_return__mean_axis_-1',
        stratify_by=('env_name',), min_n_per_burst=8,
    )


def main() -> None:
    df = load_g099_canonical_panel()
    cells = df.to_dicts()

    link_pxy = {s.stratum_id[0]: s.p_xy for s in
                cross_env_probability_of_improvement.fn(
                    cells, source='eval_best_burst_raw_mean',
                    treatment_arm=DDQN_ARM, baseline_arm=VANILLA_ARM,
                    stratify_by=('env_name',),
                ).per_stratum}

    # Run REDQ + raw bias once globally
    redq_ps = _run_dynamic(df, REDQ[1])
    redq_pc = _run_pc(df, REDQ[1])
    raw_ps = _run_dynamic(df, RAW[1])
    raw_pc = _run_pc(df, RAW[1])

    # Per env: pick best CLEAN mediator (highest d-sep%, tie-break highest marg)
    clean_ps_by_med: dict[str, dict] = {}
    clean_pc_by_med: dict[str, dict] = {}
    for label, col in CLEAN_MEDIATORS:
        try:
            clean_ps_by_med[label] = _run_dynamic(df, col)
            clean_pc_by_med[label] = _run_pc(df, col)
        except Exception:
            pass

    all_envs = sorted(df['env_name'].unique().to_list(),
                  key=lambda e: -link_pxy.get(e, 0.5))

    per_env_clean_best: dict[str, tuple] = {}
    for env in all_envs:
        candidates = []
        for label, _ in CLEAN_MEDIATORS:
            res_pc = clean_pc_by_med.get(label, {}).get((env,))
            if res_pc is None or res_pc.n_bursts_marginal_edge == 0:
                continue
            rate = res_pc.n_bursts_mediator_dseparates / res_pc.n_bursts_marginal_edge
            res_ps = clean_ps_by_med.get(label, {}).get((env,))
            candidates.append((rate, res_pc.n_bursts_marginal_edge,
                                label, res_pc, res_ps))
        if candidates:
            candidates.sort(key=lambda c: (-c[0], -c[1]))
            per_env_clean_best[env] = candidates[0]

    # Filter to envs with at least one mediator having marg≥3 (otherwise
    # the row is uninformative — "no data" across all three columns).
    def _max_marg(env: str) -> int:
        m_redq = redq_pc.get((env,))
        m_raw = raw_pc.get((env,))
        m_clean = per_env_clean_best.get(env, (0, 0, '', None))[3]
        return max(
            m_redq.n_bursts_marginal_edge if m_redq else 0,
            m_raw.n_bursts_marginal_edge if m_raw else 0,
            m_clean.n_bursts_marginal_edge if m_clean else 0,
        )
    envs = [e for e in all_envs if _max_marg(e) >= 3]

    # Render
    n = len(envs)
    fig, axes = plt.subplots(n, 3, figsize=(15, 2.7 * n), squeeze=False)
    col_titles = ['REDQ normalized bias (tautological)',
                   'Raw per-state bias (tautological)',
                   'Best CLEAN mediator (non-tautological)']
    for c, title in enumerate(col_titles):
        axes[0, c].set_title(title, fontsize=10)

    for r, env in enumerate(envs):
        for c, (label, res_ps, res_pc) in enumerate([
            (REDQ[0], redq_ps.get((env,)), redq_pc.get((env,))),
            (RAW[0], raw_ps.get((env,)), raw_pc.get((env,))),
            (per_env_clean_best.get(env, (None, None, None, None, None))[2],
             per_env_clean_best.get(env, (None, None, None, None, None))[4],
             per_env_clean_best.get(env, (None, None, None, None, None))[3]),
        ]):
            ax = axes[r, c]
            if c == 0:
                ax.set_ylabel(f'{env}\nP(D>V)={link_pxy.get(env, 0.5):.2f}',
                              fontsize=8.5)
            if res_ps is None or label is None:
                ax.text(0.5, 0.5, '—\nNo data', ha='center', va='center',
                        transform=ax.transAxes, fontsize=10)
                ax.set_axis_off()
                continue
            bursts = np.arange(len(res_ps.rho_marginal))
            marg_arr = np.asarray(res_ps.rho_marginal, dtype=np.float64)
            part_arr = np.asarray(res_ps.rho_partial, dtype=np.float64)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.plot(bursts, marg_arr, color='steelblue', linewidth=1.4,
                    marker='o', markersize=2.5, alpha=0.8,
                    label='ρ_marg')
            ax.plot(bursts, part_arr, color='crimson', linewidth=1.4,
                    marker='s', markersize=2.5, alpha=0.8,
                    label='ρ_part')
            ax.fill_between(bursts, marg_arr, part_arr,
                            alpha=0.15, color='gold')
            marg_n = res_pc.n_bursts_marginal_edge if res_pc else 0
            dsep = (res_pc.n_bursts_mediator_dseparates
                    if res_pc else 0)
            pct = (dsep / marg_n * 100) if marg_n > 0 else 0
            status = res_ps.aggregation_status.name[:7]
            ax.set_title(
                f'{label}: marg={marg_n}, dsep={pct:.0f}%, {status}',
                fontsize=8,
            )
            ax.set_xlabel('burst', fontsize=7)
            ax.legend(fontsize=6, loc='best')
            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.3)
            ax.set_ylim(-0.9, 0.9)

    fig.suptitle(
        'Per-env dynamic mediation comparison: REDQ vs raw bias vs best CLEAN\n'
        'γ=0.99 canonical; envs sorted by P(D>V). Tautological mediators (cols 1-2) '
        'read mc_return_from_step directly → diagnostic, not causal claim.',
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT, dpi=110, bbox_inches='tight')
    print(f'saved → {OUT.name}')

    # Also print a per-env summary table
    print()
    print(f'{"env":22s} {"REDQ_dsep":>10s} {"RAW_dsep":>10s} {"clean_best":>14s} {"clean_dsep":>10s}')
    for env in all_envs:
        rd = redq_pc.get((env,))
        rd_pct = (rd.n_bursts_mediator_dseparates / rd.n_bursts_marginal_edge * 100
                  if rd and rd.n_bursts_marginal_edge > 0 else 0)
        rd_n = rd.n_bursts_marginal_edge if rd else 0
        ra = raw_pc.get((env,))
        ra_pct = (ra.n_bursts_mediator_dseparates / ra.n_bursts_marginal_edge * 100
                  if ra and ra.n_bursts_marginal_edge > 0 else 0)
        ra_n = ra.n_bursts_marginal_edge if ra else 0
        cb = per_env_clean_best.get(env)
        cb_label = cb[2] if cb else '—'
        cb_pct = (cb[3].n_bursts_mediator_dseparates / cb[3].n_bursts_marginal_edge * 100
                  if cb and cb[3].n_bursts_marginal_edge > 0 else 0)
        cb_n = cb[3].n_bursts_marginal_edge if cb else 0
        print(f'{env:22s} {rd_pct:>6.0f}% (n={rd_n:>2d}) '
              f'{ra_pct:>6.0f}% (n={ra_n:>2d}) {cb_label:>14s} '
              f'{cb_pct:>6.0f}% (n={cb_n:>2d})')


if __name__ == '__main__':
    main()
