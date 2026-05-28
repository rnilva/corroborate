"""Layer 3 companion — per-env best-mediator discovery.

The canonical per-env mediation discipline (CLAUDE.md "Mediation
recipe", refined v10 §2.11):

  1. PC adjacency over a plausible candidate set + arm + outcome
     — identifies which measurables are adjacent to BOTH arm and
     outcome (candidate mediators) and which one(s) d-separate the
     arm→outcome edge.
  2. partial_spearman with the discovered mediator(s) — rank-based,
     bounded-output, multicollinearity-robust mediation magnitude.
  3. DoWhy backdoor_ate with the inferred mediator DAG —
     parametric (linear regression) effect-size estimate on the
     same panel.
  4. Refutations (placebo + RCC) on the total ATE — sanity-checks
     the foundation.

This script runs the full pipeline per env on the canonical γ=0.99
panel, with a candidate mediator set drawn from the populated
scalar measurables in the hasselt_clean cache:

  Bellman-side (non-tautological):
    - bootstrap_action_mismatch_late
  Q-shape (non-tautological):
    - q_trajectory_autocorr_late
    - q_action_grad_overlap_late
  Bias-side (with tautology caveat):
    - jensen_gap (clamped Q − MC, MC-leak)
    - normalized_bias_redq_late (REDQ-style, partial de-tautology)

Numerics reported per env:
  - PC skeleton: list of mediators adjacent to arm AND outcome
  - For each candidate: marginal ρ, partial ρ | candidate, absorption %
  - DoWhy backdoor ATE on the best (highest-absorption) candidate
  - placebo refutation drift / RCC drift for sanity

Honest per-env table + figure showing which mediator survives PC
+ Spearman + DoWhy at each env. Most envs will have no PC-detected
mediation (small marginal ρ or no adjacency); a few (Asterix,
FourRooms, MetaMaze) will surface their canonical channel.
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
    BASELINE_ARM, COLOR_HARMS, COLOR_HELPS, COLOR_NULL,
    ENV_ORDER, OUTCOME_LATE_COL, TREATMENT_ARM,
    env_label, load_g099_canonical_panel,
)

from corroborate.analyses.pc_discovery import pc_discovery
from corroborate.analyses.spearman.partial_spearman import partial_spearman
from corroborate.analyses.dowhy.dowhy import (
    backdoor_ate, placebo_refutation, random_common_cause_refutation,
)


OUT_PNG = SCRIPT_DIR.parent / 'figures' / '03b_per_env_best_mediator.png'
OUT_CSV = SCRIPT_DIR.parent / 'figures' / '03b_per_env_best_mediator.csv'


# Candidate scalar mediator set (cell-level, populated in cache).
# We exclude jensen_gap-only candidates from the "PC-clean" set
# because they share MC inputs with the outcome (soft tautology),
# but report both for comparison.
CLEAN_MEDIATORS: tuple[str, ...] = (
    'bootstrap_action_mismatch_late',
    'q_trajectory_autocorr_late',
    'q_action_grad_overlap_late',
)
TAUT_MEDIATORS: tuple[str, ...] = (
    'jensen_gap',
    'normalized_bias_redq_late',
)


def _absorption(rho_marg: float, rho_part: float) -> float:
    if abs(rho_marg) < 1e-9 or np.isnan(rho_marg) or np.isnan(rho_part):
        return float('nan')
    return (1 - abs(rho_part) / abs(rho_marg)) * 100


def main() -> None:
    df_raw = load_g099_canonical_panel()
    df = df_raw.with_columns(
        pl.when(pl.col('arm_key') == TREATMENT_ARM).then(1)
          .when(pl.col('arm_key') == BASELINE_ARM).then(0)
          .otherwise(None).alias('arm_code')
    ).filter(pl.col('arm_code').is_not_null())
    print(f'panel: {df.height} cells')

    rows: list[dict[str, object]] = []
    for env in ENV_ORDER:
        sub = df.filter(pl.col('env_name') == env)
        if sub.height < 10:
            continue
        # Marginal arm→outcome ρ — base record for ALL exit paths.
        marg = partial_spearman.fn(
            sub, x='arm_code', y=OUTCOME_LATE_COL,
            conditioning=(), stratify_by='env_name', min_stratum_size=5,
        )
        record: dict[str, object] = {
            'env': env,
            'n': sub.height,
            'marg_rho': marg.rho_pooled,
        }

        # PC adjacency over (arm + candidates + outcome)
        all_cands = CLEAN_MEDIATORS + TAUT_MEDIATORS
        nodes = ('arm_code', *all_cands, OUTCOME_LATE_COL)
        # Filter to cells where ALL candidates are finite (PC needs them all)
        sub_pc = sub.filter(
            pl.all_horizontal([pl.col(c).is_finite() for c in all_cands])
        )
        if sub_pc.height < 15:
            record['note'] = 'PC underpowered'
            rows.append(record)
            continue
        try:
            pc_res = pc_discovery.fn(
                sub_pc.iter_rows(named=True), nodes=nodes,
                alpha=0.05, max_conditioning=2, conservative=True,
            )
        except Exception as e:
            record['note'] = f'PC failed: {type(e).__name__}'
            rows.append(record)
            continue

        # Identify candidate mediators: adjacent to BOTH arm and outcome.
        candidate_mediators = [
            c for c in all_cands
            if pc_res.is_in_skeleton('arm_code', c)
            and pc_res.is_in_skeleton(c, OUTCOME_LATE_COL)
        ]
        # If PC found no mediator-pattern adjacency, still report the
        # candidate with highest absorption for diagnostic purposes.
        per_cand: list[tuple[str, float, float, float]] = []
        for c in all_cands:
            p = partial_spearman.fn(
                sub, x='arm_code', y=OUTCOME_LATE_COL,
                conditioning=(c,), stratify_by='env_name', min_stratum_size=5,
            )
            per_cand.append((
                c, marg.rho_pooled, p.rho_pooled,
                _absorption(marg.rho_pooled, p.rho_pooled),
            ))
        # Best by absorption (positive direction only — sign-flip
        # absorptions don't count as mediation).
        best: tuple[str, float, float, float] | None = max(
            (t for t in per_cand
             if not np.isnan(t[3]) and t[3] > 0
             and (t[1] > 0) == (t[2] > 0)),  # no sign-flip
            key=lambda t: t[3], default=None,
        )

        # DoWhy backdoor + refutations on the best candidate's DAG:
        # arm → mediator → outcome with arm→outcome direct edge.
        # DAGLike accepts list/tuple of (src, tgt) edge pairs.
        dowhy_ate = float('nan')
        dowhy_placebo = float('nan')
        dowhy_rcc = float('nan')
        if best is not None and best[3] > 20:
            best_med = best[0]
            dag_edges: tuple[tuple[str, str], ...] = (
                ('arm_code', best_med),
                (best_med, OUTCOME_LATE_COL),
                ('arm_code', OUTCOME_LATE_COL),
            )
            try:
                bd = backdoor_ate.fn(
                    sub.iter_rows(named=True),
                    treatment='arm_code', outcome=OUTCOME_LATE_COL,
                    dag=dag_edges,
                )
                dowhy_ate = bd.ate if bd.identified else float('nan')
                if bd.identified:
                    pl_ref = placebo_refutation.fn(
                        sub.iter_rows(named=True),
                        treatment='arm_code', outcome=OUTCOME_LATE_COL,
                        dag=dag_edges, random_state=42,
                    )
                    dowhy_placebo = abs(pl_ref.refuted_ate)
                    rcc_ref = random_common_cause_refutation.fn(
                        sub.iter_rows(named=True),
                        treatment='arm_code', outcome=OUTCOME_LATE_COL,
                        dag=dag_edges, random_state=42,
                    )
                    dowhy_rcc = abs(rcc_ref.drift)
            except Exception as e:
                record['note'] = f'DoWhy failed: {type(e).__name__}'

        record.update({
            'pc_candidates_arm_adj': [
                c for c in all_cands if pc_res.is_in_skeleton('arm_code', c)
            ],
            'pc_candidates_outcome_adj': [
                c for c in all_cands
                if pc_res.is_in_skeleton(c, OUTCOME_LATE_COL)
            ],
            'pc_mediators': candidate_mediators,
            'best_med': best[0] if best else None,
            'best_part_rho': best[2] if best else float('nan'),
            'best_absorb': best[3] if best else float('nan'),
            'dowhy_ate': dowhy_ate,
            'dowhy_placebo_drift': dowhy_placebo,
            'dowhy_rcc_drift': dowhy_rcc,
        })
        rows.append(record)

    # ─── CSV ───
    with OUT_CSV.open('w') as f:
        f.write('env,n,marg_rho,pc_arm_adj,pc_outcome_adj,pc_mediators,'
                'best_med,best_part_rho,best_absorb_pct,'
                'dowhy_ate,dowhy_placebo_drift,dowhy_rcc_drift,note\n')
        for r in rows:
            pc_a = '|'.join(r.get('pc_candidates_arm_adj', []) or [])
            pc_o = '|'.join(r.get('pc_candidates_outcome_adj', []) or [])
            pc_m = '|'.join(r.get('pc_mediators', []) or [])
            f.write(
                f'{env_label(r["env"])},{r["n"]},'
                f'{r.get("marg_rho", float("nan")):+.3f},'
                f'"{pc_a}","{pc_o}","{pc_m}",'
                f'{r.get("best_med", "") or ""},'
                f'{r.get("best_part_rho", float("nan")):+.3f},'
                f'{r.get("best_absorb", float("nan")):+.1f},'
                f'{r.get("dowhy_ate", float("nan")):+.3f},'
                f'{r.get("dowhy_placebo_drift", float("nan")):.3f},'
                f'{r.get("dowhy_rcc_drift", float("nan")):.3f},'
                f'{r.get("note", "")}\n'
            )

    # ─── figure ───
    fig, axes = plt.subplots(1, 2, figsize=(15, 7),
                             gridspec_kw={'width_ratios': [1.6, 1]})
    rows_plot = [r for r in rows if not np.isnan(r.get('marg_rho', float('nan')))]
    rows_plot.sort(key=lambda r: -abs(r['marg_rho']))
    labels = [env_label(r['env']) for r in rows_plot]
    y_pos = np.arange(len(rows_plot))

    # LEFT: per-env marg / partial / absorption
    ax = axes[0]
    margs = [r['marg_rho'] for r in rows_plot]
    parts = [r.get('best_part_rho', float('nan')) for r in rows_plot]
    for i, r in enumerate(rows_plot):
        m = r['marg_rho']
        p = r.get('best_part_rho', float('nan'))
        if not np.isnan(p):
            absorb_pct = r.get('best_absorb', float('nan'))
            color = COLOR_HELPS if absorb_pct > 30 else COLOR_NULL
            ax.plot([m, p], [i, i], color=color, linewidth=2.2, alpha=0.7, zorder=2)
    ax.scatter(margs, y_pos, c='steelblue', s=80, edgecolor='black',
               linewidth=0.6, marker='o', label='marginal ρ', zorder=3)
    ax.scatter(parts, y_pos, c='goldenrod', s=80, edgecolor='black',
               linewidth=0.6, marker='D',
               label='partial ρ (best mediator)', zorder=3)
    # Best-mediator + absorption on right edge
    for i, r in enumerate(rows_plot):
        best = r.get('best_med')
        absorb_pct = r.get('best_absorb', float('nan'))
        if best is not None and not np.isnan(absorb_pct):
            ax.text(0.96, i, f'{best.replace("_late", "")}  {absorb_pct:+.0f}%',
                    transform=ax.get_yaxis_transform(),
                    ha='right', va='center', fontsize=7.5, style='italic',
                    color='#555')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.9, 0.9)
    ax.set_xlabel("Spearman ρ — partial conditions on PC-best mediator",
                  fontsize=10)
    ax.set_title('Per-env best mediator (highest non-sign-flip absorption)',
                 fontsize=11)
    ax.legend(loc='lower left', fontsize=8.5)
    ax.grid(alpha=0.3, axis='x')

    # RIGHT: DoWhy ATE + refutation panel
    ax = axes[1]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    for i, r in enumerate(rows_plot):
        ate = r.get('dowhy_ate', float('nan'))
        pl_drift = r.get('dowhy_placebo_drift', float('nan'))
        rcc_drift = r.get('dowhy_rcc_drift', float('nan'))
        best_med = r.get('best_med') or '(none)'
        absorb_pct = r.get('best_absorb', float('nan'))
        if np.isnan(ate):
            text = f'  no PC-clean mediator survives'
            color = '#aaa'
        else:
            text = (f'  {best_med.replace("_late","")[:24]:24s}  '
                    f'absorb={absorb_pct:+3.0f}%  '
                    f'ATE={ate:+5.2f}  '
                    f'pl_drift={pl_drift:.2f}  '
                    f'rcc_drift={rcc_drift:.2f}')
            color = COLOR_HELPS if pl_drift < 0.3 and rcc_drift < 0.3 else '#a23'
        ax.text(0.01, i, text, fontsize=7.5, family='monospace',
                color=color, va='center')
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_title('DoWhy backdoor + refutation gates', fontsize=11)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    fig.suptitle(
        'Layer 3 companion: per-env best-mediator pipeline\n'
        'PC adjacency → partial Spearman absorption → DoWhy backdoor ATE + refutations',
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT_PNG.name}, {OUT_CSV.name}')


if __name__ == '__main__':
    main()
