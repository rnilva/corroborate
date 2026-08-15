"""Layer 5 — overestimation is NECESSARY-not-SUFFICIENT for harm.

The clean V-vs-D contrast on four powered envs (Layer 4) suggested a
tidy "residual overestimation predicts harm" story. The full 8-env
γ=0.999 panel BREAKS it: cross-env ρ(d_raw, residual REDQ) is weak and
non-significant, and ρ(d_raw, absolute bias) fails outright —
LunarLander and PacMan match Asterix's absolute bias yet HELP. Asterix's
harm needs the deadly-triad CONJUNCTION, not bias magnitude alone.

  05a (full panel)  — outcome d_raw vs {residual REDQ, absolute bias},
                      8 envs, log-y, red = harm (d<0). Cross-env Spearman
                      in the suptitle.
  05b (decomp bars) — why Asterix's REDQ multiple is REAL: its absolute
                      bias (Q−MC) is 4-5× any other env, with a mid-range
                      MC denominator — not a small-denominator artifact.

Input : experiments/data/cache/hasselt_clean_gpanel.parquet  (via _common)
Output: papers/g999_harm/figures/05a_redq_full_panel.png
        papers/g999_harm/figures/05b_redq_decomp_bars.png
        papers/g999_harm/figures/05_redq_full_panel.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from scipy import stats

from _common import (
    BASELINE_ARM, COLOR_HARMS, COLOR_HELPS, OUTCOME_PEAK_COL, TREATMENT_ARM,
    env_label, load_g999_panel,
)

FIG_DIR = Path(__file__).resolve().parents[1] / 'figures'
REDQ_PER_BURST = 'normalized_bias_redq_per_burst'
ABS_BIAS_PER_BURST = 'mean_per_state_cumulative_bias_per_burst'
MC_PER_BURST = 'mean_mc_per_state_per_burst'

# Four powered envs carry the decomp panel (05b).
DECOMP = (
    ('Asterix-MinAtar', 'HARM', COLOR_HARMS),
    ('Breakout-MinAtar', 'help', COLOR_HELPS),
    ('FourRooms-misc', 'help', '#762a83'),
    ('SpaceInvaders-MinAtar', 'help', '#2166ac'),
)


def late_arm(g999: pl.DataFrame, env: str, arm: str, col: str) -> float:
    s = g999.filter((pl.col('env_name') == env) & (pl.col('arm_key') == arm))
    if col not in s.columns:
        return np.nan
    a = [np.asarray(x, dtype=float) for x in s.get_column(col).to_list()
         if x is not None and len(x) > 0]
    if not a:
        return np.nan
    length = min(len(z) for z in a)
    m = np.nanmean(np.stack([z[:length] for z in a]), axis=0)
    return float(np.nanmean(m[-int(max(1, length * 0.3)):]))


def d_raw(g999: pl.DataFrame, env: str, col: str) -> float:
    s = g999.filter(pl.col('env_name') == env)
    v = s.filter(pl.col('arm_key') == BASELINE_ARM).get_column(col).to_numpy()
    d = s.filter(pl.col('arm_key') == TREATMENT_ARM).get_column(col).to_numpy()
    v = v[np.isfinite(v)]
    d = d[np.isfinite(d)]
    if len(v) < 3 or len(d) < 3:
        return np.nan
    return (d.mean() - v.mean()) / np.sqrt(
        (v.std(ddof=1) ** 2 + d.std(ddof=1) ** 2) / 2
    )


def main() -> None:
    g999 = load_g999_panel()
    envs = sorted(g999.get_column('env_name').unique().to_list())

    # ── 05a: full 8-env panel ──
    rows: list[dict[str, object]] = []
    for e in envs:
        dr = d_raw(g999, e, OUTCOME_PEAK_COL)
        if not np.isfinite(dr):
            dr = d_raw(g999, e, 'eval_best_burst_mean')
        rows.append({
            'env': env_label(e), 'd_raw': dr,
            'redq_residual_D': late_arm(g999, e, TREATMENT_ARM, REDQ_PER_BURST),
            'abs_bias_D': late_arm(g999, e, TREATMENT_ARM, ABS_BIAS_PER_BURST),
            'mc_D': late_arm(g999, e, TREATMENT_ARM, MC_PER_BURST),
        })
    tbl = pl.DataFrame(rows)
    tbl.write_csv(FIG_DIR / '05_redq_full_panel.csv')

    fin = tbl.filter(
        pl.col('d_raw').is_finite() & pl.col('redq_residual_D').is_finite()
        & pl.col('abs_bias_D').is_finite()
    )
    dr = fin.get_column('d_raw').to_numpy()
    redq = fin.get_column('redq_residual_D').to_numpy()
    bias = fin.get_column('abs_bias_D').to_numpy()
    names = fin.get_column('env').to_list()
    rho_redq = stats.spearmanr(dr, redq)
    rho_bias = stats.spearmanr(dr, bias)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (y, ylab) in zip(
        axes,
        ((redq, 'D-arm REDQ residual (Q−MC)/MC'), (bias, 'D-arm absolute bias Q−MC')),
    ):
        for i, nm in enumerate(names):
            c = COLOR_HARMS if dr[i] < 0 else COLOR_HELPS
            ax.scatter(dr[i], y[i], c=c, s=70, zorder=3)
            ax.annotate(nm, (dr[i], y[i]), fontsize=8, xytext=(4, 3),
                        textcoords='offset points')
        ax.set_yscale('log')
        ax.axvline(0, color='gray', ls='--', lw=0.8)
        ax.set_xlabel('d_raw (DDQN − vanilla)')
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.25)
    axes[0].set_title(f'outcome vs RESIDUAL REDQ overestimation  '
                      f'(ρ={rho_redq[0]:+.2f}, p={rho_redq[1]:.2f})')
    axes[1].set_title(f'outcome vs ABSOLUTE bias  '
                      f'(ρ={rho_bias[0]:+.2f}, p={rho_bias[1]:.2f})')
    plt.suptitle(f'FULL γ=0.999 panel (gpanel, {len(names)} envs): does residual/absolute '
                 'overestimation predict harm?  (red=harm d<0)', fontsize=11, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG_DIR / '05a_redq_full_panel.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    # ── 05b: decomposition bars (4 powered envs) ──
    names_d = [env_label(e) for e, _, _ in DECOMP]
    cols = [c for _, _, c in DECOMP]
    bias_d = [late_arm(g999, e, TREATMENT_ARM, ABS_BIAS_PER_BURST) for e, _, _ in DECOMP]
    mc_d = [late_arm(g999, e, TREATMENT_ARM, MC_PER_BURST) for e, _, _ in DECOMP]
    redq_d = [b / m for b, m in zip(bias_d, mc_d)]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(names_d))
    w = 0.38
    ax.bar(x - w / 2, bias_d, w, color=cols, alpha=0.95,
           label='absolute bias  Q−MC  (numerator)')
    ax.bar(x + w / 2, mc_d, w, color=cols, alpha=0.45, hatch='//',
           label='true return  MC  (denominator)')
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{n}\n({dirn})' for n, (_, dirn, _) in zip(names_d, DECOMP)])
    ax.set_ylabel('value (log)')
    ax.legend(fontsize=9)
    for i, (b, m, r) in enumerate(zip(bias_d, mc_d, redq_d)):
        ax.text(i, max(b, m) * 1.4, f'REDQ={r:.0f}×', ha='center',
                fontsize=10, fontweight='bold', color=cols[i])
    ax.set_title('Why Asterix is special (D-arm, γ=0.999): absolute overestimation '
                 '(Q−MC), not small return\nAsterix bias is 4-5× any other env; its MC '
                 'is mid-range → the multiple is REAL, not a denominator artifact',
                 fontsize=10.5, fontweight='bold')
    plt.tight_layout()
    fig.savefig(FIG_DIR / '05b_redq_decomp_bars.png', dpi=120, bbox_inches='tight')
    plt.close(fig)

    print(tbl)
    print(f'\nρ(d_raw, residual REDQ) = {rho_redq[0]:+.3f}  p={rho_redq[1]:.3f}')
    print(f'ρ(d_raw, absolute bias) = {rho_bias[0]:+.3f}  p={rho_bias[1]:.3f}')
    print('saved figures/05a_redq_full_panel.png, 05b_redq_decomp_bars.png, 05_redq_full_panel.csv')


if __name__ == '__main__':
    main()
