"""Layer 3 — STATIC MEDIATION: per-env partial Spearman.

For each env, compute marginal and partial Spearman ρ:
  - marginal: ρ(arm_code, outcome_late)
  - partial:  ρ(arm_code, outcome_late | jensen_gap)

`jensen_gap` is the framework's canonical bias measure — the same
scalar Layer 1's MECH verdict was about. Conditioning on it asks
the literature's mediation question: "does arm→outcome survive
when we hold bias constant?"

A note on the soft tautology: `jensen_gap = max(0, mean(Q − MC))`
and the outcome is a function of MC-related returns; the bias
mediator shares MC inputs with the outcome. The diagnostic primitive
`mediator_leak_adjudication` is available for substrate authors who
want to certify per-env GENUINE-vs-LEAK; we do not run it here as
this layer reports the literature-natural reading of the mediation
question. The clean (Bellman-residual) mediator is used at Layer 5.

Output: per-env forest plot showing marginal | partial ρ + absorption %.
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
    BASELINE_ARM, COLOR_HELPS, COLOR_HARMS, COLOR_NULL,
    ENV_ORDER, MECH_BIAS_COL, OUTCOME_LATE_COL, TREATMENT_ARM,
    env_label, load_g099_canonical_panel,
)

from corroborate.analyses.spearman.partial_spearman import partial_spearman


OUT_PNG = SCRIPT_DIR.parent / 'figures' / '03_static_mediation.png'
OUT_CSV = SCRIPT_DIR.parent / 'figures' / '03_static_mediation.csv'


def _absorption(rho_marg: float, rho_part: float) -> float:
    if abs(rho_marg) < 1e-9 or np.isnan(rho_marg) or np.isnan(rho_part):
        return float('nan')
    return (1 - abs(rho_part) / abs(rho_marg)) * 100


def main() -> None:
    df_raw = load_g099_canonical_panel()
    # Add 0/1 arm code so partial_spearman has a numeric arm var.
    df = df_raw.with_columns(
        pl.when(pl.col('arm_key') == TREATMENT_ARM).then(1)
          .when(pl.col('arm_key') == BASELINE_ARM).then(0)
          .otherwise(None).alias('arm_code')
    ).filter(pl.col('arm_code').is_not_null())
    print(f'panel: {df.height} cells (with arm_code)')

    rows = []
    for env in ENV_ORDER:
        sub = df.filter(pl.col('env_name') == env)
        if sub.height < 10:
            print(f'  {env}: skip (n={sub.height})')
            continue
        marg = partial_spearman.fn(
            sub, x='arm_code', y=OUTCOME_LATE_COL,
            conditioning=(), stratify_by='env_name',
            min_stratum_size=5,
        )
        part = partial_spearman.fn(
            sub, x='arm_code', y=OUTCOME_LATE_COL,
            conditioning=(MECH_BIAS_COL,), stratify_by='env_name',
            min_stratum_size=5,
        )
        rows.append((env, marg.rho_pooled, marg.p_value,
                     part.rho_pooled, part.p_value,
                     _absorption(marg.rho_pooled, part.rho_pooled),
                     marg.n_obs_total))

    # Sort by |marg ρ| descending — strongest arm→outcome signal first.
    rows.sort(key=lambda r: -abs(r[1]))

    with OUT_CSV.open('w') as f:
        f.write('env,rho_marginal,p_marg,rho_partial_given_jens,p_part,absorption_pct,n_cells\n')
        for r in rows:
            f.write(f'{env_label(r[0])},{r[1]:+.3f},{r[2]:.3f},'
                    f'{r[3]:+.3f},{r[4]:.3f},'
                    f'{r[5]:+.1f},{r[6]}\n')

    # ─── figure: per-env paired markers ───
    fig, ax = plt.subplots(figsize=(11, 6.5))
    y_pos = np.arange(len(rows))
    labels = [env_label(r[0]) for r in rows]
    marg_rho = [r[1] for r in rows]
    part_rho = [r[3] for r in rows]
    absorb = [r[5] for r in rows]

    # Line connecting marginal → partial per env
    for i, (m, p) in enumerate(zip(marg_rho, part_rho, strict=False)):
        color = (COLOR_HELPS if abs(p) < abs(m) * 0.6
                 else COLOR_HARMS if abs(p) > abs(m) * 1.4
                 else COLOR_NULL)
        ax.plot([m, p], [i, i], color=color, linewidth=2, alpha=0.7, zorder=2)
    ax.scatter(marg_rho, y_pos, c='steelblue', s=80, edgecolor='black',
               linewidth=0.6, label='marginal ρ(arm, outcome)',
               marker='o', zorder=3)
    ax.scatter(part_rho, y_pos, c='goldenrod', s=80, edgecolor='black',
               linewidth=0.6, label='partial ρ(arm, outcome | jensen_gap)',
               marker='D', zorder=3)

    # Absorption annotation on right
    xlim = (-0.9, 0.9)
    for i, a in enumerate(absorb):
        if not np.isnan(a):
            ax.text(xlim[1] - 0.05, i, f'{a:+.0f}%', ha='right',
                    va='center', fontsize=8.5, style='italic',
                    color='#555')

    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(xlim)
    ax.set_xlabel("Spearman ρ — rank correlation arm ↔ outcome (late-30%)",
                  fontsize=10)
    ax.set_title("Layer 3 (LINK): per-env static mediation — does conditioning on bias remove the arm→outcome signal?\n"
                 "Right column: absorption % = 1 − |partial ρ| / |marginal ρ|; "
                 "green = high absorption (bias mediates), gray = low (mediation null)",
                  fontsize=10.5, pad=8)
    ax.legend(loc='lower left', fontsize=9)
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT_PNG.name}, {OUT_CSV.name}')
    for r in rows[:5]:
        print(f'  {env_label(r[0]):15s}  marg={r[1]:+.2f}  partial={r[3]:+.2f}  absorb={r[5]:+.0f}%')


if __name__ == '__main__':
    main()
