"""Multi-input sibling adjudication: bias vs other Q-summaries at Asterix γ=0.99.

Closes the bias-vs-other-Q-property question left open after the
state-visitation sibling test refuted caveat (b). The state-vis
test showed the channel is Q-side; this test asks "which Q-property
inside the Q-side carries the load?"

Two passes:

**Pass 1: bias against the JOINT Q-summary set**
  mediator = bias
  sibling  = (q_argmax_margin, q_action_std, q_autocorr, q_lambda_a)

  GENUINE → bias adds info beyond all four other Q-summaries
            jointly → bias is uniquely informative among Q-summaries.
  LEAK    → bias is one Q-summary among many equivalents; the joint
            of the others already captures everything bias provides.

**Pass 2: each individual Q-summary against (bias + the other three)**
  For q_i ∈ {q_argmax_margin, q_action_std, q_autocorr, q_lambda_a}:
    mediator = q_i
    sibling  = (bias, *(others without q_i))

  GENUINE → q_i adds info beyond bias and the other Q-summaries →
            q_i is uniquely informative.
  LEAK    → q_i is subsumed by (bias + others).

If Pass 1 = LEAK and Pass 2 = LEAK for all q_i, no single
Q-summary among {bias, q_argmax, q_std, q_autocorr, q_lambda_a}
is uniquely load-bearing; they're a redundant cluster.

If Pass 1 = GENUINE and Pass 2 = LEAK for all q_i, bias is the
unique pivotal Q-summary; other Q-summaries are redundant given
bias.

Multiplicity: 5 tests (1 + 4), Bonferroni-corrected.

Depth budget: at Asterix γ=0.99, 60 cells/burst → depth-5 PC has
df ≈ 52 (n − 3 − 4). `min_n_per_burst=15` keeps Fisher-z stable.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import load_g099_canonical_panel

from corroborate.analyses.diagnostic.mediator_leak_adjudication import (
    LeakAdjudication, mediator_leak_adjudication,
)


Q_SUMMARIES = (
    ('q_argmax_margin', 'q_argmax_margin_per_burst'),
    ('q_action_std',    'q_action_std_per_burst'),
    ('q_autocorr',      'q_autocorr_per_burst'),
    ('q_lambda_a',      'q_lambda_a_per_burst'),
)
BIAS_TAG = 'bias'
BIAS_COL = 'mean_per_state_cumulative_bias_per_burst'
OUTCOME = 'mc_return__mean_axis_-1'
N_MULTIPLICITY = 5  # 1 main + 4 sibling-direction tests

OUT_TXT = SCRIPT_DIR.parent / 'figures' / 'report_q_summary_multi_input_test.txt'
OUT_PNG = SCRIPT_DIR.parent / 'figures' / 'report_q_summary_multi_input_test.png'


def _row(s, label: str) -> str:
    return (
        f'{label:48s} n_marg={s.n_marginal_edges:3d}  '
        f'dsep_sib={s.dsep_sibling_only:5.1f}%  '
        f'dsep_joint={s.dsep_joint:5.1f}%  '
        f'n01={s.n_discordant_joint_only:2d}  '
        f'n10={s.n_discordant_sibling_only:2d}  '
        f'z={s.z_mcnemar:+5.2f}  '
        f'→ {s.disposition.name}'
    )


def main() -> None:
    df = load_g099_canonical_panel()
    df_ast = df.filter(pl.col('env_name') == 'Asterix-MinAtar')
    print(f'panel: {df.height} cells; Asterix-MinAtar: {df_ast.height} cells')

    lines: list[str] = []
    lines.append('# Multi-input Q-summary adjudication at Asterix γ=0.99')
    lines.append(f'# Multiplicity: {N_MULTIPLICITY} (Bonferroni)')
    lines.append('# min_n_per_burst = 15 (depth budget for 4-5 conditioners)')
    lines.append('')
    panel_data: list[tuple[str, float, float, int, int, float, LeakAdjudication]] = []

    # ─── Pass 1: bias vs joint {q_argmax, q_std, q_autocorr, q_lambda_a} ───
    lines.append('## Pass 1: bias added to joint Q-summary set')
    lines.append('# GENUINE → bias is uniquely informative among Q-summaries')
    lines.append('# LEAK    → bias is one Q-summary among many equivalents')
    lines.append('')
    q_tuple = tuple(c for _, c in Q_SUMMARIES)
    res = mediator_leak_adjudication.fn(
        df, arm_field='arm_key',
        mediator_per_burst=BIAS_COL,
        sibling_per_burst=q_tuple,
        outcome_per_burst=OUTCOME,
        stratify_by=('env_name',),
        min_n_per_burst=15,
        n_strata_for_multiplicity=N_MULTIPLICITY,
    )
    s = next(x for x in res.per_stratum if x.stratum_id[0] == 'Asterix-MinAtar')
    lines.append(_row(s, f'bias | (q_argmax, q_std, q_autocorr, q_lambda_a)'))
    panel_data.append((
        'bias | {4 Q-summaries}',
        s.dsep_sibling_only, s.dsep_joint,
        s.n_discordant_joint_only, s.n_discordant_sibling_only,
        s.z_mcnemar, s.disposition,
    ))

    # ─── Pass 2: each Q-summary vs (bias + the other three) ───
    lines.append('')
    lines.append('## Pass 2: each Q-summary added to (bias + the other three)')
    lines.append('# GENUINE → that Q-summary is uniquely informative')
    lines.append('# LEAK    → that Q-summary is subsumed by (bias + others)')
    lines.append('')
    for tag, col in Q_SUMMARIES:
        others = tuple(c for t, c in Q_SUMMARIES if t != tag)
        sib = (BIAS_COL,) + others
        res2 = mediator_leak_adjudication.fn(
            df, arm_field='arm_key',
            mediator_per_burst=col,
            sibling_per_burst=sib,
            outcome_per_burst=OUTCOME,
            stratify_by=('env_name',),
            min_n_per_burst=15,
            n_strata_for_multiplicity=N_MULTIPLICITY,
        )
        s2 = next(x for x in res2.per_stratum if x.stratum_id[0] == 'Asterix-MinAtar')
        label = f'{tag} | (bias + {len(others)} other Q-summaries)'
        lines.append(_row(s2, label))
        panel_data.append((
            f'{tag} | (bias + 3 others)',
            s2.dsep_sibling_only, s2.dsep_joint,
            s2.n_discordant_joint_only, s2.n_discordant_sibling_only,
            s2.z_mcnemar, s2.disposition,
        ))

    txt = '\n'.join(lines)
    print(txt)
    OUT_TXT.write_text(txt + '\n')
    print(f'\nsaved text → {OUT_TXT.name}')

    # ─── figure ───
    fig, ax = plt.subplots(figsize=(13, 6.5))
    labels = [d[0] for d in panel_data]
    sib_alone = [d[1] for d in panel_data]
    joint = [d[2] for d in panel_data]
    x = np.arange(len(labels))
    w = 0.36
    ax.bar(x - w/2, sib_alone, w, label='sibling set alone',
           color='lightgray', edgecolor='black', linewidth=0.7)
    ax.bar(x + w/2, joint, w, label='joint (mediator + sibling set)',
           color='steelblue', edgecolor='black', linewidth=0.7)
    for i, d in enumerate(panel_data):
        ax.text(x[i] - w/2, d[1] + 1.5, f'{d[1]:.0f}',
                ha='center', fontsize=8)
        ax.text(x[i] + w/2, d[2] + 1.5, f'{d[2]:.0f}',
                ha='center', fontsize=8)
        verdict = d[6].name
        if verdict == 'GENUINE':
            color = '#1a8536'; fw = 'bold'
        elif verdict == 'LEAK':
            color = '#a23'; fw = 'bold'
        else:
            color = '#888'; fw = 'normal'
        ax.text(x[i], -8,
                f'n01={d[3]}, n10={d[4]}\nz={d[5]:+.2f}\n{verdict}',
                ha='center', fontsize=8, color=color, fontweight=fw)
    ax.set_ylim(-25, 110)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5, rotation=12, ha='right')
    ax.set_ylabel('d-sep % of Asterix marg-edge bursts', fontsize=10)
    ax.set_title(
        'Multi-input Q-summary adjudication at Asterix γ=0.99\n'
        f'Mediator added to sibling set; Bonferroni n={N_MULTIPLICITY}; min_n_per_burst=15',
        fontsize=11,
    )
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(alpha=0.3, axis='y')
    ax.axhline(y=0, color='black', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    print(f'saved fig  → {OUT_PNG.name}')


if __name__ == '__main__':
    main()
