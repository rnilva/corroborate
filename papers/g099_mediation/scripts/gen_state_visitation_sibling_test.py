"""State-visitation sibling test at Asterix γ=0.99.

Address the report's flagged ambiguity: GENUINE at Asterix says
the Q-channel is active, but doesn't pin down whether bias-as-
Q-summary is the load-bearing mediator vs Q-via-state-visitation
(longer pathway) vs some other Q-property.

For each state-visitation candidate ∈ {n_unique, entropy,
repeat_rate_window64}, run mediator_leak_adjudication in BOTH
directions:

  Direction A: mediator = bias, sibling = state_visitation
    GENUINE → bias adds info BEYOND state-visitation
            → Q-channel does work non-attributable to state-coverage
    LEAK    → state-visitation already subsumes what bias provides
            → bias-as-Q-summary may be Q-via-state-visitation
              (the (b) pathway in the report)

  Direction B: mediator = state_visitation, sibling = bias
    GENUINE → state-visitation adds info BEYOND bias
            → non-Q channel is parallel and independent
    LEAK    → bias already subsumes state-visitation's signal
            → no separate non-Q channel detectable here

Also outputs the state-visitation-alone d-sep rate (= dsep_sibling_only
when state-visitation is the sibling), giving the "does
state-coverage d-separate the arm→outcome edge at all?" baseline.
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


SIBLINGS = (
    ('state_n_unique', 'state_hash_n_unique_per_burst'),
    ('state_entropy', 'state_hash_entropy_per_burst'),
    ('state_repeat64', 'state_repeat_rate_window64_per_burst'),
)
BIAS = 'mean_per_state_cumulative_bias_per_burst'
OUTCOME = 'mc_return__mean_axis_-1'

OUT_TXT = SCRIPT_DIR.parent / 'figures' / 'report_state_visitation_sibling_test.txt'
OUT_PNG = SCRIPT_DIR.parent / 'figures' / 'report_state_visitation_sibling_test.png'


def _row(s, label: str) -> str:
    return (
        f'{label:38s} n_marg={s.n_marginal_edges:3d}  '
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

    # We only need Asterix verdicts. Multiplicity = 3 candidates per direction.
    lines: list[str] = []
    lines.append('# State-visitation sibling test at Asterix γ=0.99')
    lines.append('# Multiplicity: 3 candidate siblings per direction')
    lines.append('')
    panel_data: list[tuple[str, float, float, int, int, float, LeakAdjudication]] = []

    lines.append('## Direction A: mediator = bias, sibling = state_visitation')
    lines.append('# GENUINE → bias adds info beyond state-coverage (Q ≠ Q-via-coverage)')
    lines.append('')
    for tag, col in SIBLINGS:
        res = mediator_leak_adjudication.fn(
            df, arm_field='arm_key',
            mediator_per_burst=BIAS,
            sibling_per_burst=col,
            outcome_per_burst=OUTCOME,
            stratify_by=('env_name',),
            n_strata_for_multiplicity=3,
        )
        s = next(x for x in res.per_stratum
                  if x.stratum_id[0] == 'Asterix-MinAtar')
        lines.append(_row(s, f'A_bias_vs_{tag}'))
        panel_data.append((
            f'bias | {tag}',
            s.dsep_sibling_only, s.dsep_joint,
            s.n_discordant_joint_only, s.n_discordant_sibling_only,
            s.z_mcnemar, s.disposition,
        ))

    lines.append('')
    lines.append('## Direction B: mediator = state_visitation, sibling = bias')
    lines.append('# GENUINE → state-coverage adds info beyond bias (parallel non-Q channel)')
    lines.append('# dsep_sib here = bias-alone d-sep at Asterix (32 marg-edge bursts)')
    lines.append('')
    for tag, col in SIBLINGS:
        res = mediator_leak_adjudication.fn(
            df, arm_field='arm_key',
            mediator_per_burst=col,
            sibling_per_burst=BIAS,
            outcome_per_burst=OUTCOME,
            stratify_by=('env_name',),
            n_strata_for_multiplicity=3,
        )
        s = next(x for x in res.per_stratum
                  if x.stratum_id[0] == 'Asterix-MinAtar')
        lines.append(_row(s, f'B_{tag}_vs_bias'))
        panel_data.append((
            f'{tag} | bias',
            s.dsep_sibling_only, s.dsep_joint,
            s.n_discordant_joint_only, s.n_discordant_sibling_only,
            s.z_mcnemar, s.disposition,
        ))

    # Also compute the "state-visitation alone" d-sep baseline by
    # pulling dsep_sibling_only from Direction A (sibling = state_visitation).
    lines.append('')
    lines.append('## State-visitation-alone d-sep rate (= sibling-only from Direction A)')
    lines.append('')
    for tag, _col in SIBLINGS:
        rec = next(p for p in panel_data if p[0].startswith(f'bias | {tag}'))
        lines.append(f'  {tag:18s} d-sep = {rec[1]:5.1f}%  '
                     f'(of 32 marg-edge bursts at Asterix)')

    txt = '\n'.join(lines)
    print(txt)
    OUT_TXT.write_text(txt + '\n')
    print(f'\nsaved text → {OUT_TXT.name}')

    # ─── figure: two-row bar comparison ───
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    direction_a = panel_data[:3]
    direction_b = panel_data[3:]

    for ax, data, title, subtitle in [
        (axes[0], direction_a,
         'Direction A: mediator = bias, sibling = state-visitation',
         "GENUINE → bias adds info beyond state-coverage (Q ≠ via-state)"),
        (axes[1], direction_b,
         'Direction B: mediator = state-visitation, sibling = bias',
         "GENUINE → state-coverage adds info beyond bias (parallel non-Q channel)"),
    ]:
        labels = [d[0] for d in data]
        sib_alone = [d[1] for d in data]
        joint = [d[2] for d in data]
        x = np.arange(len(labels))
        w = 0.36
        ax.bar(x - w/2, sib_alone, w, label='sibling alone',
               color='lightgray', edgecolor='black', linewidth=0.7)
        ax.bar(x + w/2, joint, w, label='joint (med + sib)',
               color='steelblue', edgecolor='black', linewidth=0.7)
        for i, d in enumerate(data):
            ax.text(x[i] - w/2, d[1] + 1.5, f'{d[1]:.0f}',
                    ha='center', fontsize=8)
            ax.text(x[i] + w/2, d[2] + 1.5, f'{d[2]:.0f}',
                    ha='center', fontsize=8)
            verdict = d[6].name
            color = ('#1a8536' if verdict == 'GENUINE'
                     else '#a23' if verdict == 'LEAK'
                     else '#888')
            ax.text(x[i], -7, f'n01={d[3]}, n10={d[4]}\nz={d[5]:+.2f}\n{verdict}',
                    ha='center', fontsize=7.5, color=color,
                    fontweight=('bold' if verdict == 'GENUINE' else 'normal'))
        ax.set_ylim(-22, 105)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('d-sep % of 32 Asterix marg-edge bursts', fontsize=9)
        ax.set_title(f'{title}\n{subtitle}', fontsize=9.5)
        ax.legend(fontsize=8, loc='upper left')
        ax.grid(alpha=0.3, axis='y')
        ax.axhline(y=0, color='black', linewidth=0.5)

    fig.suptitle(
        'State-visitation sibling test at Asterix γ=0.99 — does the non-Q channel d-separate?',
        fontsize=11, y=1.00,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    print(f'saved fig  → {OUT_PNG.name}')


if __name__ == '__main__':
    main()
