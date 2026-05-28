"""Layer 1 — MECH: does DDQN reduce overestimation bias?

Per-env independent-samples Cohen's d on `jensen_gap` (the
framework's canonical clamped bias measure, `max(0, mean(Q − MC))`).
Treatment arm = D (Double DQN), baseline = V (Vanilla DQN).

This script mirrors `experiments/findings/hasselt_clean/chain.py`'s
canonical scope (`CANONICAL_DORMANCY_SCOPE & PREMISE_ACTIVE_PER_STRATUM`,
applied by `_common.load_g099_canonical_panel`) and surfaces TWO
verdict streams:

  - **Bridge verdict**: `cross_env_consistency_binomial` —
    the binomial sign-test that the chain.py bridge
    `ddqn_reduces_bias__consistently_cross_env` actually fires.
    HELD when ≥ X of N envs have d in the predicted direction.

  - **DL pool**: `stratified_arm_diff_pooled` — the per-env
    Cohen's d panel + random-effects pool with τ² / I². The DL
    pool is the heterogeneity diagnostic; it's deliberately
    PI-honest (verdict NO_EFFECT when PI includes zero), so the
    cross-env consistency bridge is the right summary for "did
    DDQN reduce bias broadly".

At γ=0.99 canonical the dormancy filter is empirically a no-op
(zero cells dormant across all 12 envs); LL et al would be filtered
at γ=0.999.
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
from corroborate.analyses.panel.cross_env_consistency_binomial import (
    cross_env_consistency_binomial,
)


OUT_PNG = SCRIPT_DIR.parent / 'figures' / '01_mech_per_env.png'
OUT_CSV = SCRIPT_DIR.parent / 'figures' / '01_mech_per_env.csv'


def main() -> None:
    df = load_g099_canonical_panel()
    print(f'panel: {df.height} cells (dormancy scope applied — no-op at γ=0.99)')

    # Predicted direction: DDQN should REDUCE bias → cohen_d < 0.
    res = stratified_arm_diff_pooled.fn(
        df,
        source=MECH_BIAS_COL,
        treatment_arm=TREATMENT_ARM, baseline_arm=BASELINE_ARM,
        stratify_by=('env_name',),
        min_baseline_predictor=float('-inf'),  # MECH-layer; no link-side filter
        min_seeds_per_arm=5,
    )
    # The chain.py bridge's actual verdict primitive: binomial sign-test.
    bridge = cross_env_consistency_binomial.fn(
        df.iter_rows(named=True),
        source=MECH_BIAS_COL,
        treatment_arm=TREATMENT_ARM, baseline_arm=BASELINE_ARM,
        stratify_by=('env_name',),
        predicted_direction='a_lt_b',  # DDQN reduces bias
        null_floor=0.0,
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
        f.write(f'# DL Verdict: {res.verdict.name}\n')
        f.write(f'# Bridge cross_env_consistency_binomial: '
                f'{bridge.n_signed_predicted}/{bridge.n_strata_above_floor} '
                f'envs in direction (of {bridge.n_strata_total} total), '
                f'p={bridge.p_value:.4f}\n')

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

    # ─── bridge-verdict + dormancy box ───
    bridge_verdict = (
        'HELD' if bridge.p_value <= 0.05
        else 'POWER_INSUFFICIENT' if bridge.p_value <= 0.15
        else 'NO_EFFECT'
    )
    bridge_color = (
        '#1a8536' if bridge_verdict == 'HELD'
        else '#d4ad28' if bridge_verdict == 'POWER_INSUFFICIENT'
        else '#a23'
    )
    bridge_text = (
        f'Bridge verdict (cross_env_consistency_binomial):\n'
        f'  {bridge.n_signed_predicted}/{bridge.n_strata_above_floor} envs in '
        f'predicted direction (of {bridge.n_strata_total} total)\n'
        f'  binomial sign-test p = {bridge.p_value:.4f} → {bridge_verdict}\n\n'
        f'DL pool diagnostic:  d={pooled_d:+.2f}, I²={res.pooled.I2:.2f}, '
        f'PI=[{pooled_pi[0]:+.2f},{pooled_pi[1]:+.2f}] → {res.verdict.name}\n'
        f'  (PI-honest NO_EFFECT: env heterogeneity is too high for a\n'
        f'   generalisable cross-env point estimate. The binomial sign-\n'
        f'   test is the right test for "DDQN reduces bias broadly".)\n\n'
        f'Dormancy filter (PREMISE_ACTIVE_PER_STRATUM): no-op at γ=0.99\n'
        f'  V\'s observed bias is ~10-100× the Jensen structural floor\n'
        f'  σ_Q × √(2 log K) in every env (e.g. LL: bias=45.6, floor=0.68).\n'
        f'  Premise is overwhelmingly active everywhere; no env dormant.\n'
        f'  (At γ=0.999, longer effective horizon flips this for some envs.)'
    )
    ax.text(0.02, 0.02, bridge_text, transform=ax.transAxes,
            va='bottom', fontsize=7.5, family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffaf0',
                      edgecolor=bridge_color, linewidth=1.2))

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT_PNG.name}, {OUT_CSV.name}')
    print(f'\nBridge: {bridge.n_signed_predicted}/{bridge.n_strata_above_floor} '
          f'envs in direction, p={bridge.p_value:.4f} → {bridge_verdict}')
    print(f'DL pool: d={pooled_d:+.3f}  I²={res.pooled.I2:.2f}  '
          f'verdict={res.verdict.name}')


if __name__ == '__main__':
    main()
