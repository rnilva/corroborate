"""Layer 2 companion — per-env learning curves (envelope + mean peak).

The Layer-2 scalar table reports `eval_late_burst_raw_mean` and
`eval_best_burst_raw_mean` per env. This script shows the underlying
per-burst trajectories so the reader can see WHY the cross-env P(D>V)
disagrees on peak vs late30 at some envs.

Per env, plot V and D arms:
  - heavy line  = median across seeds (per burst)
  - filled band = IQR (25-75 percentile)
  - thin lines  = 5/95 percentile envelope
  - ♦ marker    = (mean per-seed peak step, mean per-seed peak value)
  - gold band   = late-30% window (the scope of `eval_late_burst_raw_mean`)

Sorted by per-env `P(D>V)` on the PEAK outcome. Title shows P(D>V)
under BOTH metrics; ✓ if peak and late30 agree on direction, ↕ if
metric-sensitive (the methodological finding of Layer 2).

5/95 (not min/max) because LL/Acrobot have 1-2 catastrophic-collapse
seeds that dominate true-min and compress the y-axis. The 5/95
envelope keeps the bulk seed-spread (n=30 → excludes ~1.5 seeds per
tail) while letting the IQR band and median read.
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
    ARM_COLOR, BASELINE_ARM, ENV_ORDER, TREATMENT_ARM,
    env_label, load_g099_canonical_panel,
)

from corroborate.analyses.panel.cross_env_probability_of_improvement import (
    cross_env_probability_of_improvement,
)


OUT_PNG = SCRIPT_DIR.parent / 'figures' / '02b_learning_curves.png'

LATE_WINDOW_FRAC = 0.30

# Per-env y-axis floor (keyed by env_name). LunarLander and Acrobot
# have large-negative early returns (LL crash penalties; Acrobot's
# −500-ish until it first reaches the goal) that auto-scaling lets
# dominate the axis, compressing the informative late-training range.
# Clamp the bottom; top stays auto.
Y_MIN_FLOOR: dict[str, float] = {
    'LunarLander-v2-jax': -200.0,
    'Acrobot-v1': -200.0,
}


def _per_burst_raw_mean(mc_raw_eps_list) -> np.ndarray:
    """Convert `mc_return_raw_episodes` (per-burst K eps) into a
    per-burst mean across episodes, in RAW return units."""
    if mc_raw_eps_list is None:
        return np.zeros(0, dtype=np.float64)
    out: list[float] = []
    for burst_eps in mc_raw_eps_list:
        if burst_eps is None or len(burst_eps) == 0:
            out.append(np.nan)
        else:
            vals = np.asarray(burst_eps, dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            out.append(float(vals.mean()) if len(vals) else np.nan)
    return np.asarray(out, dtype=np.float64)


def _late_window_mean(traj: np.ndarray, frac: float = LATE_WINDOW_FRAC) -> float:
    if traj.size == 0:
        return float('nan')
    k = max(1, int(round(traj.size * frac)))
    return float(np.nanmean(traj[-k:]))


def main() -> None:
    df = load_g099_canonical_panel()
    print(f'panel: {df.height} cells')

    # Derived late30 per-cell column (recomputed from raw eps for
    # alignment with the trajectory we plot).
    late_vals: dict[str, float] = {}
    for env in df['env_name'].unique():
        sub = df.filter(pl.col('env_name') == env)
        for row_id, mc in zip(sub['id'].to_list(),
                              sub['mc_return_raw_episodes'].to_list()):
            traj = _per_burst_raw_mean(mc)
            late_vals[row_id] = _late_window_mean(traj)
    df = df.with_columns(
        pl.col('id').map_elements(
            lambda i: late_vals.get(i, float('nan')),
            return_dtype=pl.Float64,
        ).alias('outcome_late30')
    )
    cells = df.iter_rows(named=True)
    cells_list = list(cells)

    link_peak = cross_env_probability_of_improvement.fn(
        cells_list, source='eval_best_burst_raw_mean',
        treatment_arm=TREATMENT_ARM, baseline_arm=BASELINE_ARM,
        stratify_by=('env_name',),
    )
    link_late = cross_env_probability_of_improvement.fn(
        cells_list, source='outcome_late30',
        treatment_arm=TREATMENT_ARM, baseline_arm=BASELINE_ARM,
        stratify_by=('env_name',),
    )
    pxy_peak = {s.stratum_id[0]: s.p_xy for s in link_peak.per_stratum}
    pxy_late = {s.stratum_id[0]: s.p_xy for s in link_late.per_stratum}

    # Sort by P(D>V)_peak descending — strongest D-helps envs first.
    envs = [e for e in ENV_ORDER if e in pxy_peak]
    envs.sort(key=lambda e: -pxy_peak.get(e, 0.5))
    n = len(envs)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.3 * n_rows),
                             squeeze=False)
    axes_flat = axes.flatten()

    for i, env in enumerate(envs):
        ax = axes_flat[i]
        sub = df.filter(pl.col('env_name') == env)
        eval_every = (
            int(sub['eval_every'].drop_nulls()[0])
            if 'eval_every' in sub.columns else 20000
        )
        env_max_len = 0
        for arm_label, arm_pred, color in [
            ('V', pl.col('arm_key') == BASELINE_ARM, ARM_COLOR['V']),
            ('D', pl.col('arm_key') == TREATMENT_ARM, ARM_COLOR['D']),
        ]:
            arm_sub = sub.filter(arm_pred)
            if len(arm_sub) == 0:
                continue
            arrs = [_per_burst_raw_mean(m)
                    for m in arm_sub['mc_return_raw_episodes'].to_list()
                    if m is not None]
            arrs = [a for a in arrs if a.size > 0]
            if not arrs:
                continue
            max_len = max(len(a) for a in arrs)
            env_max_len = max(env_max_len, max_len)
            stack = np.full((len(arrs), max_len), np.nan, dtype=np.float64)
            for r, a in enumerate(arrs):
                stack[r, :len(a)] = a
            median = np.nanmedian(stack, axis=0)
            q25 = np.nanpercentile(stack, 25, axis=0)
            q75 = np.nanpercentile(stack, 75, axis=0)
            mn = np.nanpercentile(stack, 5, axis=0)
            mx = np.nanpercentile(stack, 95, axis=0)
            x = (np.arange(max_len) + 1) * eval_every
            ax.fill_between(x, q25, q75, color=color, alpha=0.20)
            ax.plot(x, median, color=color, linewidth=1.8,
                    label=f'{arm_label} (n={len(arrs)})')
            ax.plot(x, mn, color=color, linewidth=0.55, alpha=0.55)
            ax.plot(x, mx, color=color, linewidth=0.55, alpha=0.55)
            # Mean per-seed peak — both step (x) and value (y).
            peak_steps: list[float] = []
            peak_vals: list[float] = []
            for traj in arrs:
                if traj.size == 0 or not np.any(np.isfinite(traj)):
                    continue
                idx = int(np.nanargmax(traj))
                peak_steps.append((idx + 1) * eval_every)
                peak_vals.append(float(traj[idx]))
            if peak_steps:
                ax.plot(float(np.mean(peak_steps)),
                        float(np.mean(peak_vals)),
                        marker='D', color=color, markersize=8,
                        markeredgecolor='black', markeredgewidth=0.8,
                        zorder=10, label=f'{arm_label} mean peak')

        # Late-window band — width = LATE_WINDOW_FRAC of trajectory.
        if env_max_len > 0:
            band_left = (1 - LATE_WINDOW_FRAC) * env_max_len * eval_every
            band_right = env_max_len * eval_every
            ax.axvspan(band_left, band_right, color='goldenrod',
                       alpha=0.10, zorder=0)

        pk = pxy_peak.get(env, 0.5)
        lt = pxy_late.get(env, 0.5)
        # Saturation detection: per-env env-cap inferred from the
        # joint V+D max; if >70% of seeds in EITHER arm peak within
        # 1% of the joint-cap, the peak metric saturates and its
        # P(D>V) is a sampling-noise artifact of below-cap outliers,
        # not a real treatment effect (e.g. CartPole 29/30 V + 28/30
        # D seeds reach the env cap of 500).
        peak_vals_all = sub['eval_best_burst_raw_mean'].drop_nulls().to_list()
        env_cap = max(peak_vals_all) if peak_vals_all else 0.0
        sat_frac_per_arm: list[float] = []
        for arm in (BASELINE_ARM, TREATMENT_ARM):
            arm_peaks = (
                sub.filter(pl.col('arm_key') == arm)
                ['eval_best_burst_raw_mean'].drop_nulls().to_list()
            )
            if arm_peaks and env_cap > 0:
                sat_frac_per_arm.append(
                    sum(1 for v in arm_peaks if v >= 0.99 * env_cap)
                    / len(arm_peaks)
                )
        saturated = bool(sat_frac_per_arm) and max(sat_frac_per_arm) > 0.7

        sign_peak = '+' if pk > 0.5 else ('−' if pk < 0.5 else '·')
        sign_late = '+' if lt > 0.5 else ('−' if lt < 0.5 else '·')
        # Don't call ↕ vs ✓ on the peak axis when peak is saturated.
        if saturated:
            agree = '⊥'  # saturated: peak metric not informative
            peak_label = f'peak={pk:.2f} (SAT)'
        else:
            agree = '✓' if sign_peak == sign_late else '↕'
            peak_label = f'peak={pk:.2f}'
        ax.set_title(
            f'{env_label(env)}  {agree}\n'
            f'P(D>V)  {peak_label}  late30={lt:.2f}',
            fontsize=9,
        )
        ax.set_xlabel('training step', fontsize=8)
        ax.set_ylabel('raw return', fontsize=8)
        ax.legend(fontsize=7, loc='best')
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        floor = Y_MIN_FLOOR.get(env)
        if floor is not None:
            ax.set_ylim(bottom=floor)

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_axis_off()

    # Recompute agreement counts with saturation flag.
    n_agree = 0
    n_sat = 0
    n_disagree = 0
    for e in envs:
        sub_e = df.filter(pl.col('env_name') == e)
        peak_vals_all = sub_e['eval_best_burst_raw_mean'].drop_nulls().to_list()
        env_cap = max(peak_vals_all) if peak_vals_all else 0.0
        sat_frac_per_arm: list[float] = []
        for arm in (BASELINE_ARM, TREATMENT_ARM):
            arm_peaks = (
                sub_e.filter(pl.col('arm_key') == arm)
                ['eval_best_burst_raw_mean'].drop_nulls().to_list()
            )
            if arm_peaks and env_cap > 0:
                sat_frac_per_arm.append(
                    sum(1 for v in arm_peaks if v >= 0.99 * env_cap)
                    / len(arm_peaks)
                )
        if sat_frac_per_arm and max(sat_frac_per_arm) > 0.7:
            n_sat += 1
            continue
        pk = pxy_peak.get(e, 0.5)
        lt = pxy_late.get(e, 0.5)
        if (pk - 0.5) * (lt - 0.5) >= 0:
            n_agree += 1
        else:
            n_disagree += 1
    fig.suptitle(
        'Layer 2 companion: per-env learning curves at γ=0.99 canonical (raw return)\n'
        'median across seeds, IQR fill (25-75%), 5/95 envelope; ♦ = mean per-seed peak; '
        'gold band = late-30% window\n'
        f'Title shows P(D>V) under BOTH metrics — peak / late30. '
        f'Non-saturating envs: {n_agree} agree ✓, {n_disagree} metric-sensitive ↕; '
        f'{n_sat} saturated ⊥ (peak P(D>V) is a sampling artifact of below-cap outliers)',
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT_PNG.name}')
    print(f'  {n_agree}/{len(envs)} envs agree on peak/late30 sign')


if __name__ == '__main__':
    main()
