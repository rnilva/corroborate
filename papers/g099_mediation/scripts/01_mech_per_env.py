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

    # ─── robust (median + IQR) companion to the mean-based Cohen's d ───
    # Δ_med = (median_D − median_V) / pooled_IQR — median numerator, IQR
    # scale → scale-free + cross-env comparable (parallels Cohen's d =
    # Δmean/pooled_SD) but ROBUST to the heavy tails that make the MEAN
    # misleading. Snake is the canonical divergence: mean Cohen's d=+0.18
    # (4/30 D seeds Q-explode → heavy tail) yet median Δ<0 (typical seed:
    # DDQN reduces bias). Bootstrap 95% CI: independent per-arm resample,
    # B=2000, seed 0 (reproducible).
    rng = np.random.default_rng(0)

    def _robust_d(vv: np.ndarray, dd: np.ndarray) -> float:
        iqr_v = float(np.subtract(*np.percentile(vv, [75, 25])))
        iqr_d = float(np.subtract(*np.percentile(dd, [75, 25])))
        pooled = 0.5 * (iqr_v + iqr_d)
        if pooled <= 0:  # degenerate (identical values) — undefined scale
            return float('nan')
        return float((np.median(dd) - np.median(vv)) / pooled)

    # med_rows aligns 1:1 with `rows`: (median_V, median_D, robust_d, lo, hi)
    med_rows: list[tuple[float, float, float, float, float]] = []
    for r in rows:
        sub = df.filter(pl.col('env_name') == r[0])
        v = sub.filter(pl.col('arm_key') == BASELINE_ARM)[MECH_BIAS_COL].to_numpy()
        d = sub.filter(pl.col('arm_key') == TREATMENT_ARM)[MECH_BIAS_COL].to_numpy()
        v = v[np.isfinite(v)]
        d = d[np.isfinite(d)]
        rd = _robust_d(v, d)
        boot = np.array([
            _robust_d(rng.choice(v, v.size, replace=True),
                      rng.choice(d, d.size, replace=True))
            for _ in range(2000)
        ])
        boot = boot[np.isfinite(boot)]
        lo = float(np.percentile(boot, 2.5)) if boot.size else float('nan')
        hi = float(np.percentile(boot, 97.5)) if boot.size else float('nan')
        med_rows.append((float(np.median(v)), float(np.median(d)), rd, lo, hi))

    # ─── CSV ───
    with OUT_CSV.open('w') as f:
        f.write('env,V_mean_jens,D_mean_jens,cohen_d,cohen_se,n_V,n_D,'
                'V_median_jens,D_median_jens,robust_d,robust_ci_lo,robust_ci_hi\n')
        for r, m in zip(rows, med_rows):
            f.write(f'{env_label(r[0])},{r[1]:.4f},{r[2]:.4f},'
                    f'{r[3]:+.3f},{r[4]:.3f},{r[5]},{r[6]},'
                    f'{m[0]:.4f},{m[1]:.4f},{m[2]:+.3f},{m[3]:+.3f},{m[4]:+.3f}\n')
        f.write(f'\n# DL pool (mean): d={res.pooled_d:+.3f} '
                f'SE={res.pooled_se:.3f} τ²={res.pooled.tau2:.3f} '
                f'I²={res.pooled.I2:.2f}\n')
        f.write(f'# PI: [{res.pooled.pi_lo:+.3f}, {res.pooled.pi_hi:+.3f}]\n')
        f.write(f'# DL Verdict: {res.verdict.name}\n')
        f.write(f'# Bridge cross_env_consistency_binomial: '
                f'{bridge.n_signed_predicted}/{bridge.n_strata_above_floor} '
                f'envs in direction (of {bridge.n_strata_total} total), '
                f'p={bridge.p_value:.4f}\n')
        f.write('# robust_d = (median_D - median_V)/pooled_IQR; '
                'robust CI = bootstrap 2.5/97.5 (B=2000, per-arm resample)\n')

    pooled_d = res.pooled_d
    bridge_verdict = (
        'HELD' if bridge.p_value <= 0.05
        else 'POWER_INSUFFICIENT' if bridge.p_value <= 0.15
        else 'NO_EFFECT'
    )

    # ─── figure: two-panel forest — MEAN+CI (left) vs MEDIAN+IQR (right) ───
    # Colour by SIGNIFICANCE (95% CI excludes 0) so each dot's colour
    # agrees with its error bar; grey = CI straddles 0 (inconclusive).
    def _ci_color(lo: float, hi: float) -> str:
        if hi < 0:
            return COLOR_HELPS
        if lo > 0:
            return COLOR_HARMS
        return COLOR_NULL

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 6.6), sharey=True)
    y_pos = np.arange(len(rows))
    labels = [env_label(r[0]) for r in rows]

    # LEFT — mean-based Cohen's d ± 95% CI
    ds = [r[3] for r in rows]
    ses = [r[4] for r in rows]
    colL = [_ci_color(d - 1.96 * se, d + 1.96 * se) for d, se in zip(ds, ses)]
    axL.errorbar(ds, y_pos, xerr=[1.96 * se for se in ses], fmt='o',
                 color='black', ecolor='gray', capsize=3, markersize=0,
                 linewidth=0.8)
    axL.scatter(ds, y_pos, c=colL, s=70, edgecolor='black', linewidth=0.6,
                zorder=3)
    axL.axvline(0, color='black', linewidth=0.5)
    axL.axvspan(min(ds) - 1, 0, alpha=0.05, color=COLOR_HELPS)
    axL.set_yticks(y_pos)
    axL.set_yticklabels(labels)
    axL.invert_yaxis()
    axL.set_xlabel("Cohen's d (D − V) ± 95% CI", fontsize=9)
    axL.set_title("MEAN + CI  (outlier-sensitive)", fontsize=10)
    axL.grid(alpha=0.3, axis='x')

    # RIGHT — robust median effect Δmed/pooled-IQR ± bootstrap 95% CI
    rds = [m[2] for m in med_rows]
    rlo = [m[3] for m in med_rows]
    rhi = [m[4] for m in med_rows]
    colR = [_ci_color(lo, hi) for lo, hi in zip(rlo, rhi)]
    xerrR = [
        [rd - lo for rd, lo in zip(rds, rlo)],
        [hi - rd for rd, hi in zip(rds, rhi)],
    ]
    axR.errorbar(rds, y_pos, xerr=xerrR, fmt='o', color='black',
                 ecolor='gray', capsize=3, markersize=0, linewidth=0.8)
    axR.scatter(rds, y_pos, c=colR, s=70, edgecolor='black', linewidth=0.6,
                zorder=3)
    axR.axvline(0, color='black', linewidth=0.5)
    axR.axvspan(min(rlo) - 0.2, 0, alpha=0.05, color=COLOR_HELPS)
    axR.set_xlabel("Δmedian / pooled-IQR ± bootstrap 95% CI", fontsize=9)
    axR.set_title("MEDIAN + IQR  (robust to tails)", fontsize=10)
    axR.grid(alpha=0.3, axis='x')

    fig.suptitle(
        'Layer 1 (MECH): per-env DDQN bias effect at γ=0.99 canonical — '
        'mean vs robust median\n'
        '(negative = DDQN reduces bias; green = 95% CI excludes 0, '
        'grey = inconclusive)',
        fontsize=11.5,
    )

    note = (
        f'Bridge (cross_env_consistency_binomial): '
        f'{bridge.n_signed_predicted}/{bridge.n_strata_total} envs reduce bias, '
        f'sign-test p={bridge.p_value:.4f} → {bridge_verdict}.    '
        f'DL mean-pool d={pooled_d:+.2f}, I²={res.pooled.I2:.2f}, '
        f'PI=[{res.pooled.pi_lo:+.2f},{res.pooled.pi_hi:+.2f}] → {res.verdict.name} '
        f'(env heterogeneity too high for one point estimate).\n'
        f'Snake = the mean↔median divergence: mean d=+0.18 (4/30 D seeds '
        f'Q-explode → heavy tail) but median Δ<0 (typical seed: DDQN reduces '
        f'bias). Dormancy filter no-op at γ=0.99 (premise active everywhere).'
    )
    fig.text(0.5, -0.03, note, ha='center', va='top', fontsize=8,
             family='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffaf0',
                       edgecolor='#1a8536', linewidth=1.0))

    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT_PNG.name}, {OUT_CSV.name}')
    print(f'\nBridge: {bridge.n_signed_predicted}/{bridge.n_strata_above_floor} '
          f'envs in direction, p={bridge.p_value:.4f} → {bridge_verdict}')
    print(f'DL pool: d={pooled_d:+.3f}  I²={res.pooled.I2:.2f}  '
          f'verdict={res.verdict.name}')


if __name__ == '__main__':
    main()
