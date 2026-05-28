"""Dual test: is the bias mediator "mere tautology", or does it carry
Q-information beyond the MC-leak?

Two tests via multi-mediator `dynamic_pc_adjacency`:

  Test (1) — MC-residualization: compare three conditioning sets per env
    A. {bias}                 — what bias alone d-separates
    B. {MC_state}             — what MC alone d-separates (the leak baseline)
    C. {bias, MC_state}       — does bias add d-separation beyond MC?
    If C > B by a meaningful margin → bias carries Q-information.
    If C ≈ B → bias is just MC-leak.

  Test (2) — Incremental over clean Q-mediator: same logic, but using
    q_argmax_margin (a non-tautological Q-only mediator) as the baseline.
    A. {q_argmax_margin}            — clean Q-mediator alone
    B. {bias}                       — bias alone
    C. {q_argmax_margin, bias}      — does bias add d-separation beyond Q?
    If C > A → bias's signal extends beyond what q_argmax_margin captures
    (i.e., bias has unique Q-component or unique MC-component that
    q_argmax_margin misses).

Reports per-env: d-sep % for each conditioning set under the SAME marginal
edge count (marg edges are counted under the joint mediator set per the
PC primitive). Outputs a numerical table; the report can interpret.
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

from corroborate.analyses.dynamic_mediation.pc_adjacency import (
    dynamic_pc_adjacency,
)
from corroborate.analyses.panel.cross_env_probability_of_improvement import (
    cross_env_probability_of_improvement,
)
from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM


OUT = SCRIPT_DIR.parent / 'figures' / 'report_mc_leak_adjudication.png'


def _run(df, mediator):
    return dynamic_pc_adjacency.fn(
        df, arm_field='arm_key',
        mediator_per_burst=mediator,
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

    BIAS = 'mean_per_state_cumulative_bias_per_burst'
    MC = 'mean_mc_per_state_per_burst'
    Q = 'q_argmax_margin_per_burst'

    runs = {
        '{bias}': _run(df, BIAS),
        '{MC}': _run(df, MC),
        '{bias, MC}': _run(df, (BIAS, MC)),
        '{q_argmax}': _run(df, Q),
        '{bias, q_argmax}': _run(df, (BIAS, Q)),
    }

    envs = sorted(df['env_name'].unique().to_list(),
                  key=lambda e: -link_pxy.get(e, 0.5))

    # Per-env table
    print(f'{"env":22s} {"|":1s} {"marg_b":>7s} {"{bias}":>8s} {"{MC}":>7s} {"{B,MC}":>8s} {"|":1s} {"marg_q":>7s} {"{Q}":>7s} {"{B,Q}":>8s}')
    rows = []
    for env in envs:
        b = runs['{bias}'].get((env,))
        m = runs['{MC}'].get((env,))
        bm = runs['{bias, MC}'].get((env,))
        q = runs['{q_argmax}'].get((env,))
        bq = runs['{bias, q_argmax}'].get((env,))
        if b is None: continue
        def pct(r):
            if r is None or r.n_bursts_marginal_edge == 0:
                return float('nan'), 0
            return r.n_bursts_mediator_dseparates / r.n_bursts_marginal_edge * 100, r.n_bursts_marginal_edge
        p_b, n_b = pct(b); p_m, _ = pct(m); p_bm, _ = pct(bm)
        p_q, n_q = pct(q); p_bq, _ = pct(bq)
        rows.append((env, n_b, p_b, p_m, p_bm, n_q, p_q, p_bq))
        def fmt(x): return f'{x:>7.0f}%' if not np.isnan(x) else '    nan'
        print(f'{env:22s} | {n_b:>7d} {fmt(p_b):>8s} {fmt(p_m):>7s} {fmt(p_bm):>8s} | {n_q:>7d} {fmt(p_q):>7s} {fmt(p_bq):>8s}')

    # Filter to envs with marg ≥ 3 for visualization
    visible = [r for r in rows if r[1] >= 3]
    if not visible:
        print('\nNo env with n_marg >= 3 for visualization.')
        return

    # Render: two grouped-bar panels
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    env_labels = [r[0].replace('-MinAtar','').replace('-jumanji','').replace('-misc','').replace('-v1','').replace('-v0','').replace('-v2-jax','') for r in visible]
    x = np.arange(len(visible))
    w = 0.27

    # Test 1: {bias} vs {MC} vs {bias, MC}
    ax = axes[0]
    p_b = [r[2] for r in visible]; p_m = [r[3] for r in visible]; p_bm = [r[4] for r in visible]
    ax.bar(x - w, p_b, w, label='{bias}', color='steelblue', edgecolor='black')
    ax.bar(x, p_m, w, label='{MC}', color='goldenrod', edgecolor='black')
    ax.bar(x + w, p_bm, w, label='{bias, MC}', color='forestgreen', edgecolor='black')
    for i, r in enumerate(visible):
        ax.annotate(f'n={r[1]}', (x[i], 102), ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(env_labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('d-separation % among marg-edge bursts', fontsize=9)
    ax.set_title('Test 1: does bias add info beyond MC-leak?\n{bias,MC} > {MC} → bias carries Q-information',
                  fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y'); ax.set_ylim(0, 110)

    # Test 2: {q_argmax} vs {bias, q_argmax}
    ax = axes[1]
    p_q = [r[6] for r in visible]; p_bq = [r[7] for r in visible]
    ax.bar(x - w/2, p_q, w, label='{q_argmax}', color='crimson', edgecolor='black')
    ax.bar(x + w/2, p_bq, w, label='{bias, q_argmax}', color='forestgreen', edgecolor='black')
    for i, r in enumerate(visible):
        ax.annotate(f'n={r[5]}', (x[i], 102), ha='center', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(env_labels, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('d-separation % among marg-edge bursts', fontsize=9)
    ax.set_title('Test 2: does bias add info beyond clean Q-mediator?\n{bias,q_argmax} > {q_argmax} → bias adds beyond Q',
                  fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.3, axis='y'); ax.set_ylim(0, 110)

    fig.suptitle(
        'MC-leak adjudication: is the bias mediator "merely tautology"?\n'
        'γ=0.99 canonical; envs sorted by P(D>V). Larger green bar than '
        'corresponding baseline = bias carries info beyond the alternative.',
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT, dpi=120, bbox_inches='tight')
    print(f'\nsaved → {OUT.name}')


if __name__ == '__main__':
    main()
