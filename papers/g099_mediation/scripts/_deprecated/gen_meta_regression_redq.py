"""Cross-env meta-regression: per-env Δ_mediator vs per-env Δ_outcome,
contrasting REDQ-normalized bias against raw env-units bias.

For each of 12 envs we compute:
  - Δ_outcome  = P(D > V) on `eval_late_burst_raw_mean` (late-window outcome)
  - Δ_REDQ     = P(D > V) on `normalized_bias_redq_late`
  - Δ_RAW      = P(D > V) on `mean_per_state_cumulative_bias_late`

Then meta-regress Δ_outcome ~ Δ_mediator across the 12 envs with OLS
+ Spearman ρ. The REDQ scale-invariance hypothesis predicts a tighter
Δ_REDQ → Δ_outcome relation than Δ_RAW → Δ_outcome, because RAW is
scaled by env-specific magnitude that biases the slope.

Why P(D > V) rather than d: P(D > V) is bounded in [0, 1] and
saturation-neutral, so envs with bounded outcomes (FR success rate,
CartPole 0–500 saturation) don't distort the slope. The per-env P(D > V)
is the right cross-env-comparable effect-size — it's what the
`cross_env_probability_of_improvement` primitive already computes,
and what the L1 LINK bridge uses.

Bias direction: at envs where D reduces bias relative to V, P(D > V)
on bias is < 0.5 (D's per-cell bias is LOWER than V's). At envs where
D improves outcome, P(D > V) on outcome is > 0.5. So if the
"bias-reduction → outcome-improvement" causal story holds, we'd expect
a NEGATIVE slope: lower Δ_bias predicts higher Δ_outcome.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import polars as pl
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
from _common import load_g099_canonical_panel
from corroborate.analyses.panel.cross_env_probability_of_improvement import (
    cross_env_probability_of_improvement,
)
from experiments.findings.ddqn._arms import DDQN_ARM, VANILLA_ARM


OUT = SCRIPT_DIR.parent / 'figures' / 'report_meta_regression_redq.png'


def _bootstrap_slope_ci(x: np.ndarray, y: np.ndarray, *, n_resamples: int = 2000,
                         rng_seed: int = 0) -> tuple[float, float, float, float]:
    """OLS slope + percentile bootstrap 95% CI + Spearman ρ + p-value.
    Strata are bootstrap units (envs)."""
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 4:
        return float('nan'), float('nan'), float('nan'), float('nan')
    slope = float(np.polyfit(x, y, 1)[0])
    rho, p = stats.spearmanr(x, y)
    rng = np.random.default_rng(rng_seed)
    slopes = np.empty(n_resamples, dtype=np.float64)
    n = len(x)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        slopes[i] = np.polyfit(x[idx], y[idx], 1)[0]
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    return slope, float(lo), float(hi), float(rho)


def _per_env_pxy(cells, *, source: str) -> dict[str, float]:
    res = cross_env_probability_of_improvement.fn(
        cells, source=source, treatment_arm=DDQN_ARM,
        baseline_arm=VANILLA_ARM, stratify_by=('env_name',),
    )
    return {s.stratum_id[0]: s.p_xy for s in res.per_stratum}


def main() -> None:
    df = load_g099_canonical_panel()
    cells = df.to_dicts()

    outcome_pxy = _per_env_pxy(cells, source='eval_late_burst_raw_mean')
    redq_pxy = _per_env_pxy(cells, source='normalized_bias_redq_late')
    raw_pxy = _per_env_pxy(cells, source='mean_per_state_cumulative_bias_late')

    envs = sorted(outcome_pxy.keys() & redq_pxy.keys() & raw_pxy.keys())
    out_arr = np.array([outcome_pxy[e] for e in envs])
    redq_arr = np.array([redq_pxy[e] for e in envs])
    raw_arr = np.array([raw_pxy[e] for e in envs])

    redq_slope, redq_lo, redq_hi, redq_rho = _bootstrap_slope_ci(redq_arr, out_arr)
    raw_slope, raw_lo, raw_hi, raw_rho = _bootstrap_slope_ci(raw_arr, out_arr)

    # ── plot ─────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    for ax, x_arr, x_lab, slope, lo, hi, rho, color in [
        (axes[0], redq_arr, 'P(D > V) on normalized_bias_redq_late',
         redq_slope, redq_lo, redq_hi, redq_rho, 'crimson'),
        (axes[1], raw_arr, 'P(D > V) on mean_per_state_cumulative_bias_late',
         raw_slope, raw_lo, raw_hi, raw_rho, 'steelblue'),
    ]:
        ax.scatter(x_arr, out_arr, s=60, color=color, edgecolor='black',
                   linewidth=0.6, alpha=0.8, zorder=3)
        for i, env in enumerate(envs):
            short = env.replace('-MinAtar', '').replace('-jumanji', '').replace('-misc', '').replace('-v1', '').replace('-v0', '').replace('-v2-jax', '')
            ax.annotate(short, (x_arr[i], out_arr[i]),
                        fontsize=7, alpha=0.85, xytext=(4, 3),
                        textcoords='offset points')
        # OLS line + bootstrap CI band
        xx = np.linspace(min(x_arr.min(), 0.0), max(x_arr.max(), 1.0), 50)
        intercept = float(np.polyfit(x_arr, out_arr, 1)[1])
        ax.plot(xx, slope * xx + intercept, color=color, linewidth=1.5,
                alpha=0.9, label=f'OLS slope={slope:+.3f} [{lo:+.2f}, {hi:+.2f}]')
        ax.axhline(0.5, color='gray', linewidth=0.5, alpha=0.6)
        ax.axvline(0.5, color='gray', linewidth=0.5, alpha=0.6)
        ax.set_xlabel(x_lab, fontsize=10)
        ax.set_ylabel('P(D > V) on eval_late_burst_raw_mean (outcome)', fontsize=10)
        ax.set_title(f'ρ_Spearman = {rho:+.3f}  (n={len(envs)} envs)', fontsize=10)
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.3)

    fig.suptitle(
        'Cross-env meta-regression: Δ_mediator vs Δ_outcome at γ=0.99 canonical\n'
        f'Each point = one env\'s P(D>V) under that mediator/outcome. '
        'Negative slope = bias-reduction → outcome-improvement story.\n'
        'REDQ left (scale-invariant); raw right (env-units).',
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(OUT, dpi=120, bbox_inches='tight')
    print(f'saved → {OUT.name}')

    # Per-env table
    print()
    print(f'{"env":25s} {"out P":>7s} {"REDQ P":>8s} {"RAW P":>7s}')
    for i, e in enumerate(envs):
        print(f'{e:25s} {out_arr[i]:>7.3f} {redq_arr[i]:>8.3f} {raw_arr[i]:>7.3f}')
    print()
    print('Meta-regression Δ_outcome ~ Δ_mediator:')
    print(f'  REDQ:  slope={redq_slope:+.3f} [{redq_lo:+.3f}, {redq_hi:+.3f}]  '
          f'Spearman ρ={redq_rho:+.3f}')
    print(f'  RAW:   slope={raw_slope:+.3f} [{raw_lo:+.3f}, {raw_hi:+.3f}]  '
          f'Spearman ρ={raw_rho:+.3f}')


if __name__ == '__main__':
    main()
