"""Layer 1 — MECH: does DDQN reduce overestimation bias?

Per-env independent-samples Cohen's d on `jensen_gap` (the
framework's canonical clamped bias measure, `max(0, mean(Q − MC))`).
Treatment arm = D (Double DQN), baseline = V (Vanilla DQN).

Output: forest plot of per-env d_jens (negative = DDQN reduces bias),
DL random-effects pool annotation, table CSV.

The MECH verdict is per-env HELD (DDQN reduces bias at that env) or
NULL (no detectable reduction). The cross-env aggregate is reported
honestly with τ² / I² heterogeneity.
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
    ENV_ORDER, MECH_BIAS_COL, TREATMENT_ARM,
    env_label, load_g099_canonical_panel,
)

from corroborate.analyses.panel.stratified_arm_diff_pooled import (
    stratified_arm_diff_pooled,
)


OUT_PNG = SCRIPT_DIR.parent / 'figures' / '01_mech_per_env.png'
OUT_CSV = SCRIPT_DIR.parent / 'figures' / '01_mech_per_env.csv'


def main() -> None:
    df = load_g099_canonical_panel()
    print(f'panel: {df.height} cells')

    # Predicted direction: DDQN should REDUCE bias → cohen_d < 0.
    # We pass the raw call without the predicted-direction filter so
    # we surface every stratum's d, then read sign at the panel layer.
    res = stratified_arm_diff_pooled.fn(
        df,
        source=MECH_BIAS_COL,
        treatment_arm=TREATMENT_ARM, baseline_arm=BASELINE_ARM,
        stratify_by=('env_name',),
        # MECH layer: we want EVERY env in the panel (no scope filter
        # on bias > X — that's a LINK-layer filter, not MECH).
        min_baseline_predictor=float('-inf'),
        min_seeds_per_arm=5,
    )

    # Build sorted rows in canonical env order.
    rows: list[tuple[str, float, float, float, float, int, int]] = []
    by_id = {s.stratum_id[0]: s for s in res.per_stratum}
    for env in ENV_ORDER:
        s = by_id.get(env)
        if s is None:
            print(f'  {env}: not present')
            continue
        rows.append((
            env,
            s.arm_mean_baseline, s.arm_mean_treatment,
            s.cohen_d, s.cohen_se,
            s.n_seeds_baseline, s.n_seeds_treatment,
        ))

    # ─── CSV ───
    with OUT_CSV.open('w') as f:
        f.write('env,V_mean_jens,D_mean_jens,cohen_d,cohen_se,n_V,n_D\n')
        for r in rows:
            f.write(f'{env_label(r[0])},{r[1]:.4f},{r[2]:.4f},'
                    f'{r[3]:+.3f},{r[4]:.3f},{r[5]},{r[6]}\n')
        f.write(f'\n# DL pool: d={res.pooled_d:+.3f} '
                f'SE={res.pooled_se:.3f} τ²={res.pooled.tau2:.3f} '
                f'I²={res.pooled.I2:.2f}\n')
        f.write(f'# PI: [{res.pooled.pi_lo:+.3f}, {res.pooled.pi_hi:+.3f}]\n')
        f.write(f'# Verdict: {res.verdict.name}\n')

    # ─── figure: forest plot ───
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    y_pos = np.arange(len(rows))
    ds = [r[3] for r in rows]
    ses = [r[4] for r in rows]
    labels = [env_label(r[0]) for r in rows]
    colors = [
        COLOR_HELPS if d < -0.2 else COLOR_HARMS if d > 0.2 else COLOR_NULL
        for d in ds
    ]

    # 95% CIs
    ax.errorbar(ds, y_pos, xerr=[1.96 * se for se in ses],
                fmt='o', color='black', ecolor='gray', capsize=3,
                markersize=0, linewidth=0.8)
    ax.scatter(ds, y_pos, c=colors, s=70, edgecolor='black',
               linewidth=0.6, zorder=3)
    ax.axvline(0, color='black', linewidth=0.5)
    # Predicted direction shading
    ax.axvspan(-3.5, 0, alpha=0.05, color=COLOR_HELPS)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Cohen's d (D − V) on jensen_gap "
                  "— negative = DDQN reduces bias", fontsize=10)
    ax.set_title("Layer 1 (MECH): per-env bias reduction at γ=0.99 canonical",
                  fontsize=11, pad=8)

    # DL pool diamond
    pooled_d = res.pooled_d
    pooled_pi = (res.pooled.pi_lo, res.pooled.pi_hi)
    diamond_y = len(rows) + 0.5
    ax.plot([pooled_pi[0], pooled_d, pooled_pi[1], pooled_d, pooled_pi[0]],
            [diamond_y, diamond_y - 0.3, diamond_y, diamond_y + 0.3, diamond_y],
            color='black', linewidth=1.2)
    ax.fill([pooled_pi[0], pooled_d, pooled_pi[1], pooled_d],
            [diamond_y, diamond_y - 0.3, diamond_y, diamond_y + 0.3],
            color='steelblue', alpha=0.6)
    ax.text(pooled_pi[1] + 0.05, diamond_y,
            f'DL pool d={pooled_d:+.2f} I²={res.pooled.I2:.2f}',
            va='center', fontsize=9, style='italic')

    ax.set_ylim(diamond_y + 0.8, -0.8)
    ax.grid(alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT_PNG.name}, {OUT_CSV.name}')
    print(f'\nDL pool: d={pooled_d:+.3f}  I²={res.pooled.I2:.2f}  '
          f'verdict={res.verdict.name}')


if __name__ == '__main__':
    main()
