"""Layer 2 — OUTCOME: does DDQN's bias reduction translate to outcome improvement?

Per-env independent-samples Cohen's d on TWO outcome metrics:
  - `eval_late_burst_raw_mean` — late-30% of bursts, raw return (steady-state)
  - `eval_best_burst_raw_mean` — best-of-bursts raw return (peak)

Plus cross-env probability-of-improvement (Agarwal et al. 2021 P(D > V))
via `cross_env_probability_of_improvement` — the framework's per-stratum
honest aggregator that refuses to assume Gaussian effect-size pooling.

Output: per-env forest plot, sorted by late-outcome d; CSV table.

The framework's MECH→LINK consequence story is per-env: bias goes down,
does outcome go up? At γ=0.99 the answer varies dramatically by env,
and the cross-env aggregate is honest about that.
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
    ENV_ORDER, OUTCOME_LATE_COL, OUTCOME_PEAK_COL, TREATMENT_ARM,
    env_label, load_g099_canonical_panel,
)

from corroborate.analyses.panel.stratified_arm_diff_pooled import (
    stratified_arm_diff_pooled,
)
from corroborate.analyses.panel.cross_env_probability_of_improvement import (
    cross_env_probability_of_improvement,
)


OUT_PNG = SCRIPT_DIR.parent / 'figures' / '02_outcome_per_env.png'
OUT_CSV = SCRIPT_DIR.parent / 'figures' / '02_outcome_per_env.csv'


def _arm_diff_per_env(df: pl.DataFrame, source: str):
    return stratified_arm_diff_pooled.fn(
        df,
        source=source,
        treatment_arm=TREATMENT_ARM, baseline_arm=BASELINE_ARM,
        stratify_by=('env_name',),
        min_baseline_predictor=float('-inf'),
        min_seeds_per_arm=5,
    )


def main() -> None:
    df = load_g099_canonical_panel()
    print(f'panel: {df.height} cells')

    late = _arm_diff_per_env(df, OUTCOME_LATE_COL)
    peak = _arm_diff_per_env(df, OUTCOME_PEAK_COL)

    # Saturation detection: an env's peak metric is uninformative
    # when most seeds in either arm reach the env's reward cap.
    # Flag envs where >70% of seeds in EITHER arm peak within 1% of
    # the joint env-cap. CartPole (29/30 V, 28/30 D at 500) and
    # FourRooms (1.00 cap, both arms saturate) qualify; the d_peak
    # at these envs is dominated by 1-2 below-cap outlier seeds.
    saturated_envs: set[str] = set()
    for env in ENV_ORDER:
        sub_e = df.filter(pl.col('env_name') == env)
        peaks_all = sub_e[OUTCOME_PEAK_COL].drop_nulls().to_list()
        if not peaks_all:
            continue
        env_cap = max(peaks_all)
        sat_per_arm: list[float] = []
        for arm in (BASELINE_ARM, TREATMENT_ARM):
            arm_peaks = (
                sub_e.filter(pl.col('arm_key') == arm)
                [OUTCOME_PEAK_COL].drop_nulls().to_list()
            )
            if arm_peaks and env_cap > 0:  # only positive-cap envs saturate
                sat_per_arm.append(
                    sum(1 for v in arm_peaks if v >= 0.99 * env_cap)
                    / len(arm_peaks)
                )
        if sat_per_arm and max(sat_per_arm) > 0.7:
            saturated_envs.add(env)

    # Per-env rows, ordered by late-outcome d.
    by_id_late = {s.stratum_id[0]: s for s in late.per_stratum}
    by_id_peak = {s.stratum_id[0]: s for s in peak.per_stratum}
    rows = []
    for env in ENV_ORDER:
        l = by_id_late.get(env); p = by_id_peak.get(env)
        if l is None or p is None:
            continue
        rows.append((env, l, p))
    rows.sort(key=lambda r: r[1].cohen_d)  # ascending: harms first, helps last

    # Cross-env P(D > V) for the late-30 outcome. The primitive
    # consumes record-dict iterables; convert from polars.
    p_xy = cross_env_probability_of_improvement.fn(
        df.iter_rows(named=True), source=OUTCOME_LATE_COL,
        treatment_arm=TREATMENT_ARM, baseline_arm=BASELINE_ARM,
        stratify_by=('env_name',),
    )

    # ─── CSV ───
    with OUT_CSV.open('w') as f:
        f.write('env,V_mean_late,D_mean_late,d_late,se_late,'
                'V_mean_peak,D_mean_peak,d_peak,se_peak,n_V,n_D\n')
        for env, l, p in rows:
            f.write(
                f'{env_label(env)},{l.arm_mean_baseline:.3f},{l.arm_mean_treatment:.3f},'
                f'{l.cohen_d:+.3f},{l.cohen_se:.3f},'
                f'{p.arm_mean_baseline:.3f},{p.arm_mean_treatment:.3f},'
                f'{p.cohen_d:+.3f},{p.cohen_se:.3f},'
                f'{l.n_seeds_baseline},{l.n_seeds_treatment}\n'
            )
        f.write(f'\n# late: DL d={late.pooled_d:+.3f}, I²={late.pooled.I2:.2f}, '
                f'PI=[{late.pooled.pi_lo:+.3f}, {late.pooled.pi_hi:+.3f}], '
                f'verdict={late.verdict.name}\n')
        f.write(f'# peak: DL d={peak.pooled_d:+.3f}, I²={peak.pooled.I2:.2f}, '
                f'PI=[{peak.pooled.pi_lo:+.3f}, {peak.pooled.pi_hi:+.3f}], '
                f'verdict={peak.verdict.name}\n')
        f.write(f'# P(D > V) cross-env: mean={p_xy.p_xy_mean:.3f}, '
                f'n_strata={p_xy.n_strata}, p_perm={p_xy.p_permutation:.3f}\n')

    # ─── figure: two-column forest plot (late | peak), shared y-order ───
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.5), sharey=True)
    y_pos = np.arange(len(rows))
    labels = [env_label(r[0]) for r in rows]
    for ax, source_d, source_se, pool, title in [
        (axes[0],
         [r[1].cohen_d for r in rows], [r[1].cohen_se for r in rows],
         late, 'late-30% (steady-state)'),
        (axes[1],
         [r[2].cohen_d for r in rows], [r[2].cohen_se for r in rows],
         peak, 'best-burst (peak)'),
    ]:
        colors = [
            COLOR_HELPS if d > 0.2 else COLOR_HARMS if d < -0.2 else COLOR_NULL
            for d in source_d
        ]
        ax.errorbar(source_d, y_pos, xerr=[1.96 * s for s in source_se],
                    fmt='o', color='black', ecolor='gray', capsize=3,
                    markersize=0, linewidth=0.8)
        ax.scatter(source_d, y_pos, c=colors, s=70, edgecolor='black',
                   linewidth=0.6, zorder=3)
        ax.axvline(0, color='black', linewidth=0.5)
        ax.axvspan(0, max(source_d) * 1.1 + 0.5, alpha=0.05,
                    color=COLOR_HELPS)
        # DL pool diamond at bottom
        diamond_y = len(rows) + 0.6
        ax.plot([pool.pooled.pi_lo, pool.pooled_d, pool.pooled.pi_hi,
                 pool.pooled_d, pool.pooled.pi_lo],
                [diamond_y, diamond_y - 0.3, diamond_y, diamond_y + 0.3, diamond_y],
                color='black', linewidth=1.2)
        ax.fill([pool.pooled.pi_lo, pool.pooled_d, pool.pooled.pi_hi, pool.pooled_d],
                [diamond_y, diamond_y - 0.3, diamond_y, diamond_y + 0.3],
                color='steelblue', alpha=0.6)
        ax.text(0.02, 0.98,
                f'DL d={pool.pooled_d:+.2f}\n'
                f'I²={pool.pooled.I2:.2f}\n'
                f'verdict: {pool.verdict.name}',
                transform=ax.transAxes,
                va='top', fontsize=8.5, style='italic',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffaf0',
                          edgecolor='#cc9900', linewidth=0.6))
        ax.set_xlabel("Cohen's d (D − V)", fontsize=10)
        ax.set_title(title, fontsize=10.5)
        ax.grid(alpha=0.3, axis='x')
    # Mark saturated envs on the y-tick labels with ⊥.
    labels_marked = [
        (lab + ' ⊥' if r[0] in saturated_envs else lab)
        for lab, r in zip(labels, rows, strict=False)
    ]
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(labels_marked)
    axes[0].invert_yaxis()
    axes[0].set_ylim(diamond_y + 0.8, -0.8)

    # Footnote about saturation flag — anchor to FIGURE coords below
    # the x-axis labels so it doesn't overlap.
    if saturated_envs:
        sat_list = ', '.join(env_label(e) for e in saturated_envs)
        fig.text(
            0.5, 0.01,
            f'⊥ saturated peak — d_peak dominated by below-cap outlier '
            f'seeds, not real treatment effect ({sat_list})',
            ha='center', va='bottom', fontsize=8, style='italic',
            color='#666',
        )

    # Title with cross-env P(D > V)
    fig.suptitle(
        f'Layer 2 (OUTCOME): per-env outcome consequences at γ=0.99 canonical\n'
        f'P(D > V) cross-env mean = {p_xy.p_xy_mean:.2f} over {p_xy.n_strata} strata; '
        f'permutation p = {p_xy.p_permutation:.3f}',
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT_PNG.name}, {OUT_CSV.name}')
    print(f'\nlate: DL d={late.pooled_d:+.3f}, I²={late.pooled.I2:.2f}, '
          f'verdict={late.verdict.name}')
    print(f'peak: DL d={peak.pooled_d:+.3f}, I²={peak.pooled.I2:.2f}, '
          f'verdict={peak.verdict.name}')
    print(f'P(D > V) = {p_xy.p_xy_mean:.3f}  (n_strata={p_xy.n_strata})')


if __name__ == '__main__':
    main()
