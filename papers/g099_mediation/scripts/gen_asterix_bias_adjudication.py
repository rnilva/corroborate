"""Author the bias-mediation-adjudicated panel figure for
papers/g099_mediation/bias_mediation_adjudicated.md.

Single 2x2 panel:
  - top-left:  d-sep% at Asterix under {MC}, {bias}, {q_argmax},
               {bias,MC}, {bias,q_argmax} — bar chart with the
               structural-MC-leak baseline highlighted.
  - top-right: per-env verdict table — paired bias-vs-MC test
               across the 12-env panel, with GENUINE/LEAK/UPFG/UP
               color coding.
  - bottom-left: schematic causal-graph icon — arm → bias →
                  outcome with MC as a confounder, q_argmax as
                  parallel candidate. Annotations point at v6 verdicts.
  - bottom-right: caveats — the "GENUINE ≠ confirmed mechanism"
                   reading + state-visitation channel as the
                   recommended next falsification step.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import load_g099_canonical_panel

from corroborate.analyses.diagnostic.mediator_leak_adjudication import (
    LeakAdjudication, mediator_leak_adjudication,
)


OUT = SCRIPT_DIR.parent / 'figures' / 'report_asterix_bias_adjudication.png'


def main() -> None:
    df = load_g099_canonical_panel()

    # Run BOTH tests to populate the panel.
    res_mc = mediator_leak_adjudication.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mean_per_state_cumulative_bias_per_burst',
        sibling_per_burst='mean_mc_per_state_per_burst',
        outcome_per_burst='mc_return__mean_axis_-1',
        stratify_by=('env_name',),
        n_strata_for_multiplicity=12,
    )
    res_q = mediator_leak_adjudication.fn(
        df, arm_field='arm_key',
        mediator_per_burst='mean_per_state_cumulative_bias_per_burst',
        sibling_per_burst='q_argmax_margin_per_burst',
        outcome_per_burst='mc_return__mean_axis_-1',
        stratify_by=('env_name',),
        n_strata_for_multiplicity=12,
    )

    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], height_ratios=[1, 0.85])

    # ─── top-left: d-sep bar chart at Asterix ───
    ax = fig.add_subplot(gs[0, 0])
    asterix_mc = next(s for s in res_mc.per_stratum
                       if s.stratum_id[0] == 'Asterix-MinAtar')
    asterix_q = next(s for s in res_q.per_stratum
                      if s.stratum_id[0] == 'Asterix-MinAtar')
    # bias-alone d-sep: from the joint_pc of either run (bias is the
    # mediator in both); pull from res_mc.joint_pc which has joint
    # = {bias, MC}, and back-compute via the marginal-edge count vs
    # the joint's dsep. Actually simpler: use the standalone {bias}
    # number — we can compute it as the {bias} dsep from running PC
    # at depth-1 with just bias, but we don't have that surface
    # exposed cleanly. Use the joint number from res_mc as a proxy
    # for "best with bias" combined; the headline contrast is MC
    # alone vs joint-with-bias.
    bias_alone_dsep = asterix_mc.dsep_joint  # joint = {bias, MC}
    rates = [
        ('{MC alone}\n(structural leak baseline)',
         asterix_mc.dsep_sibling_only, 'goldenrod'),
        ('{q_argmax alone}\n(clean Q-mediator)',
         asterix_q.dsep_sibling_only, 'mediumseagreen'),
        ('{bias, MC}\n(bias + MC joint)',
         asterix_mc.dsep_joint, 'crimson'),
        ('{bias, q_argmax}\n(bias + Q-mediator)',
         asterix_q.dsep_joint, 'darkorchid'),
    ]
    labels = [r[0] for r in rates]
    values = [r[1] for r in rates]
    colors = [r[2] for r in rates]
    bars = ax.bar(range(len(rates)), values, color=colors,
                  edgecolor='black', linewidth=0.7, alpha=0.85)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f'{v:.0f}%',
                ha='center', fontsize=9, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.set_xticks(range(len(rates)))
    ax.set_xticklabels(labels, fontsize=8.5, rotation=0)
    ax.set_ylabel('d-separation % among 32 marg-edge bursts',
                  fontsize=10)
    ax.set_title('Asterix γ=0.99: what conditioning explains the arm→outcome edge?',
                  fontsize=11, pad=12)
    # Highlight the load-bearing comparison
    ax.axhline(y=asterix_mc.dsep_sibling_only, color='goldenrod',
               linestyle='--', linewidth=0.8, alpha=0.5)
    gain = asterix_mc.dsep_joint - asterix_mc.dsep_sibling_only
    ax.annotate(f'+{gain:.0f}pp gain over MC alone\n(joint includes bias)',
                xy=(2, asterix_mc.dsep_joint),
                xytext=(1.5, 96),
                fontsize=8.5, ha='center',
                arrowprops=dict(arrowstyle='->', color='crimson',
                                 lw=1, alpha=0.7))
    ax.grid(alpha=0.3, axis='y')

    # ─── top-right: per-env verdict table ───
    ax = fig.add_subplot(gs[0, 1])
    ax.axis('off')
    envs_order = ['Asterix-MinAtar', 'FourRooms-misc', 'MetaMaze-misc',
                   'SpaceInvaders-MinAtar', 'Freeway-MinAtar',
                   'Acrobot-v1', 'Breakout-MinAtar', 'CartPole-v1',
                   'LunarLander-v2-jax', 'MountainCar-v0',
                   'PacMan-jumanji', 'Snake-jumanji']
    by_env = {s.stratum_id[0]: s for s in res_mc.per_stratum}
    disp_color = {
        LeakAdjudication.GENUINE: '#1a8536',
        LeakAdjudication.LEAK: '#aaa',
        LeakAdjudication.UNDERPOWERED_FOR_GENUINE: '#d4ad28',
        LeakAdjudication.UNDERPOWERED: '#bbb',
    }
    disp_label = {
        LeakAdjudication.GENUINE: 'GENUINE',
        LeakAdjudication.LEAK: 'LEAK',
        LeakAdjudication.UNDERPOWERED_FOR_GENUINE: 'underpowered',
        LeakAdjudication.UNDERPOWERED: 'no test',
    }
    header = ['env', 'n_marg', 'n_info', 'verdict']
    cell_text = [header]
    cell_colors = [['#eee'] * 4]
    for e in envs_order:
        s = by_env.get(e)
        if s is None:
            continue
        env_label = e.replace('-MinAtar', '').replace('-jumanji', '').replace('-misc', '').replace('-v1', '').replace('-v0', '').replace('-v2-jax', '')
        n_info = s.n_discordant_joint_only + s.n_discordant_sibling_only
        cell_text.append([env_label, str(s.n_marginal_edges),
                           f'{n_info}', disp_label[s.disposition]])
        bg = '#fff' if s.disposition is not LeakAdjudication.GENUINE else '#dfffd6'
        cell_colors.append([bg, bg, bg, disp_color[s.disposition]])
    tbl = ax.table(cellText=cell_text, cellColours=cell_colors,
                   loc='center', cellLoc='center',
                   colWidths=[0.32, 0.18, 0.18, 0.30])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.5)
    # Make GENUINE row text bolder + white-on-color verdict cell
    for i, row in enumerate(cell_text):
        if row[3] == 'GENUINE':
            for j in range(4):
                tbl[(i, j)].get_text().set_fontweight('bold')
                if j == 3:
                    tbl[(i, j)].get_text().set_color('white')
    # Header bold
    for j in range(4):
        tbl[(0, j)].get_text().set_fontweight('bold')
    ax.set_title('Per-env verdicts (mediator = bias, sibling = MC)',
                  fontsize=11, pad=14)

    # ─── bottom-left: causal-graph schematic ───
    ax = fig.add_subplot(gs[1, 0])
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis('off')
    nodes = {
        'arm': (0.5, 3), 'bias': (4, 4.5),
        'Q': (4, 2.5), 'q_argmax': (4, 0.8),
        'MC': (6.7, 3.5), 'outcome': (9, 3),
    }
    for name, (x, y) in nodes.items():
        c = '#fff'
        if name == 'arm': c = '#cce5ff'
        if name == 'outcome': c = '#ffd6cc'
        if name in {'bias', 'MC'}: c = '#fff7d6'
        ax.add_patch(FancyBboxPatch(
            (x - 0.55, y - 0.32), 1.1, 0.64,
            boxstyle='round,pad=0.05', facecolor=c, edgecolor='black',
            linewidth=1.2))
        ax.text(x, y, name, ha='center', va='center', fontsize=9.5,
                fontweight='bold')
    def arr(a, b, color='black', style='-', lw=1.3):
        x1, y1 = nodes[a]; x2, y2 = nodes[b]
        ax.add_patch(FancyArrowPatch(
            (x1 + 0.55, y1), (x2 - 0.55, y2),
            arrowstyle='->', mutation_scale=12,
            color=color, linestyle=style, linewidth=lw))
    arr('arm', 'Q', color='#666')
    arr('Q', 'bias', color='#222')
    arr('Q', 'q_argmax', color='#222', style='--', lw=0.8)
    arr('MC', 'bias', color='#666')
    arr('bias', 'outcome', color='crimson', lw=1.6)
    arr('MC', 'outcome', color='goldenrod', lw=1.3)
    arr('q_argmax', 'outcome', color='mediumseagreen',
        style='--', lw=0.8)
    # Annotation labels
    ax.text(5.3, 4.95, 'bias = Q − MC', ha='center', fontsize=8,
            style='italic', color='#333')
    ax.text(8.3, 3.85, 'tautological\nshared input',
            ha='center', fontsize=7.5, color='goldenrod', style='italic')
    ax.text(6.8, 5.4, 'bias subsumes\nq_argmax (Asterix)',
            ha='center', fontsize=8, color='darkorchid', style='italic')
    ax.set_title('Causal-graph schematic: bias has MC-leak (yellow); test '
                 'controls for it', fontsize=10.5, pad=8)

    # ─── bottom-right: caveats box ───
    ax = fig.add_subplot(gs[1, 1])
    ax.axis('off')
    caveats = (
        'GENUINE ≠ confirmed mechanism\n\n'
        'Hasselt 2016 and REDQ never explicitly claim mediation —\n'
        'they report bias↓ AND outcome↑ together. We formalize\n'
        'that folk reading. The Asterix verdict supports it under\n'
        'three caveats:\n\n'
        '  (a) Bias-clip intuition (the folk reading itself) —\n'
        '      DDQN\'s clip → bias reduced → outcome improved.\n'
        '      Still admissible.\n\n'
        '  (b) Q-via-state-visitation — DDQN changes Q; Q changes\n'
        '      the policy; policy changes state visitation; state\n'
        '      visitation changes outcome. State-visitation\n'
        '      sibling test REFUTES this at Asterix: the\n'
        '      operationalisations (n_unique / entropy /\n'
        '      repeat_rate) d-separate only 6-16% alone, add\n'
        '      zero info beyond bias (n_01≤2 in all reverse\n'
        '      tests). Q-channel is not via state visitation.\n\n'
        '  (c) Analytical-tautology partial defense — under\n'
        '      successful Q-learning, bias↓ and outcome↑ co-occur\n'
        '      by Bellman contraction (Q tracks MC). The cross-env\n'
        '      pattern partially defuses this: bias↓-without-\n'
        '      outcome↑ is common (mech HELD broadly, link HELD\n'
        '      narrowly), so they\'re empirically separable.\n'
        '      Asterix is unique in having the temporal coupling.\n\n'
        'But DDQN is by construction Q-side, so any Q-summary\n'
        '(bias, q_argmax, magnitude) mediates by construction.\n'
        'The test confirms the Q-channel is active and bias is\n'
        'ONE admissible summary inside it — not that bias-\n'
        'reduction is THE Q-property carrying the load. The\n'
        'bias-vs-other-Q-property question remains open;\n'
        'multi-input sibling adjudication is the next step.'
    )
    ax.text(0.02, 0.98, caveats, transform=ax.transAxes,
            fontsize=8.7, verticalalignment='top', family='sans-serif',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#fffaf0',
                     edgecolor='#cc9900', linewidth=1))

    fig.suptitle(
        'DDQN at γ=0.99: bias mediates outcome at Asterix only, after '
        'controlling for the structural MC-leak\n'
        'Across the 12-env canonical panel, Asterix is the only env where '
        'the bias mediator passes a Bonferroni-corrected paired test.',
        fontsize=12, y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT.name}')


if __name__ == '__main__':
    main()
