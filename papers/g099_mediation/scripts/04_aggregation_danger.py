"""Layer 4 — AGGREGATION DANGER (cross-env static): pooling per-env
mediation results into one cross-env number doesn't represent any
individual env.

The natural cross-env aggregate of Layer-3-style per-env partial
Spearman is the Fisher-z-pooled `ρ(arm, outcome | jensen_gap)`
across all 12 envs. Polars-based primitives such as
`partial_spearman` compute it in one line. The aggregate is
publishable-looking:

  pooled marginal ρ  = +0.16    (across 680 cells, 12 strata)
  pooled partial  ρ  = −0.09    (after conditioning on jensen_gap)
  pooled absorption  = ~60%, sign-flipping

A naive reading: "DDQN's bias-clip absorbs the arm→outcome signal
AND flips its sign cross-env." Looks like a strong mediation
finding.

But: the per-env panel (Layer 3) reveals that the pool is averaging
THREE qualitatively different env regimes:

  - 2 envs with strong positive absorption (Asterix, MetaMaze):
    bias mediates without sign-flip, absorbs ~56% of marginal ρ
  - 2 envs with sign-flip under conditioning (FourRooms, Freeway):
    conditioning on bias REVERSES the arm→outcome direction —
    pooling these with the non-flippers averages opposite signs
  - 7 envs at near-zero marginal (CartPole, Acrobot, MountainCar,
    LunarLander, Snake, Breakout, SpaceInvaders, PacMan):
    they contribute mostly noise; their per-env partial reads are
    statistical artifacts rather than meaningful mediation signal

The pooled "−0.09 partial" doesn't represent any of these regimes.
It's the Fisher-z mean of a multimodal distribution.

This script generates a figure showing both: the pooled headline
on top, the per-env disaggregation below, and an annotation
spelling out why the pool misrepresents the data.

Companion: Layer 5 surfaces an analogous danger at a different
granularity — per-BURST trajectory aggregation hides sign-flip
WITHIN a single env (e.g. PacMan SIGN_FLIP_DETECTED). Layers 4
and 5 are two levels of the same Simpson's-paradox concern.
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
    ENV_ORDER, MECH_BIAS_COL, OUTCOME_LATE_COL, TREATMENT_ARM,
    env_label, load_g099_canonical_panel,
)

from corroborate.analyses.spearman.partial_spearman import partial_spearman


OUT_PNG = SCRIPT_DIR.parent / 'figures' / '04_aggregation_danger.png'
OUT_CSV = SCRIPT_DIR.parent / 'figures' / '04_aggregation_danger.csv'


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

    # ─── pooled cross-env ───
    pooled_marg = partial_spearman.fn(
        df, x='arm_code', y=OUTCOME_LATE_COL,
        conditioning=(), stratify_by='env_name', min_stratum_size=5,
    )
    pooled_part = partial_spearman.fn(
        df, x='arm_code', y=OUTCOME_LATE_COL,
        conditioning=(MECH_BIAS_COL,), stratify_by='env_name',
        min_stratum_size=5,
    )

    # ─── per-env (matches Layer 3) ───
    per_env: list[tuple[str, float, float]] = []
    for env in ENV_ORDER:
        sub = df.filter(pl.col('env_name') == env)
        if sub.height < 10:
            continue
        m = partial_spearman.fn(
            sub, x='arm_code', y=OUTCOME_LATE_COL,
            conditioning=(), stratify_by='env_name', min_stratum_size=5,
        )
        p = partial_spearman.fn(
            sub, x='arm_code', y=OUTCOME_LATE_COL,
            conditioning=(MECH_BIAS_COL,), stratify_by='env_name',
            min_stratum_size=5,
        )
        per_env.append((env, m.rho_pooled, p.rho_pooled))

    # Sort by marginal ρ descending so envs with real signal come first.
    per_env.sort(key=lambda r: -r[1])

    # ─── classify each env ───
    def classify(rho_m: float, rho_p: float) -> tuple[str, str]:
        if abs(rho_m) < 0.20:
            return 'near-zero marg', COLOR_NULL
        if (rho_m > 0) != (rho_p > 0) and abs(rho_p) > 0.1:
            return 'sign-flip', COLOR_HARMS
        if _absorption(rho_m, rho_p) > 30:
            return 'high absorption', COLOR_HELPS
        return 'low absorption', '#88a'

    regimes = [classify(m, p) for _, m, p in per_env]

    # ─── figure ───
    fig = plt.figure(figsize=(13, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.7],
                          height_ratios=[1, 1.4])

    # ── TOP-LEFT: the pooled headline ──
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    pooled_absorb = _absorption(pooled_marg.rho_pooled, pooled_part.rho_pooled)
    pool_text = (
        'Naive cross-env aggregate\n'
        '(`partial_spearman` over\n'
        '680 cells, 12 strata):\n\n'
        f'   marginal ρ  =  {pooled_marg.rho_pooled:+.3f}\n'
        f'   partial  ρ  =  {pooled_part.rho_pooled:+.3f}\n'
        f'   p (partial) =  {pooled_part.p_value:.4f}\n\n'
        f'absorption ≈ {pooled_absorb:.0f}%\n'
        f'partial SIGN-FLIPS under conditioning.\n\n'
        '   "DDQN\'s bias mediates the\n'
        '    arm→outcome signal and\n'
        '    reverses its sign\n'
        '    cross-env."\n\n'
        '— this reading doesn\'t represent\n'
        'any individual env (right panel).'
    )
    ax.text(0.05, 0.95, pool_text, transform=ax.transAxes,
            fontsize=10, va='top', family='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#fff4e0',
                      edgecolor='#cc6600', linewidth=1.2))

    # ── TOP-RIGHT + BOTTOM-RIGHT: per-env panel ──
    ax = fig.add_subplot(gs[:, 1])
    y_pos = np.arange(len(per_env))
    labels = [env_label(e) for e, _, _ in per_env]
    marg_vals = [m for _, m, _ in per_env]
    part_vals = [p for _, _, p in per_env]

    # Lines connecting marg → partial per env
    for i, (m, p) in enumerate(zip(marg_vals, part_vals, strict=False)):
        _, color = regimes[i]
        ax.plot([m, p], [i, i], color=color, linewidth=2.2, alpha=0.75, zorder=2)
    ax.scatter(marg_vals, y_pos, c='steelblue', s=85, edgecolor='black',
               linewidth=0.6, marker='o',
               label='marginal ρ(arm, outcome)', zorder=3)
    ax.scatter(part_vals, y_pos, c='goldenrod', s=85, edgecolor='black',
               linewidth=0.6, marker='D',
               label='partial ρ(arm, outcome | jensen_gap)', zorder=3)

    # Regime labels on right edge
    for i, (env, rho_m, rho_p) in enumerate(per_env):
        regime_label, color = regimes[i]
        ax.text(0.96, i, regime_label, transform=ax.get_yaxis_transform(),
                ha='right', va='center', fontsize=8.5, color=color,
                style='italic')

    ax.axvline(0, color='black', linewidth=0.5)
    # Pooled marker
    diamond_y = len(per_env) + 0.7
    ax.scatter([pooled_marg.rho_pooled], [diamond_y], marker='o',
               c='steelblue', s=160, edgecolor='black', linewidth=1.0,
               zorder=4)
    ax.scatter([pooled_part.rho_pooled], [diamond_y], marker='D',
               c='goldenrod', s=160, edgecolor='black', linewidth=1.0,
               zorder=4)
    ax.text(-0.95, diamond_y, '← POOLED cross-env (12 strata)',
            ha='left', va='center', fontsize=9, fontweight='bold',
            color='#cc6600')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(diamond_y + 0.8, -0.8)
    ax.set_xlabel("Spearman ρ — partial conditions on jensen_gap",
                  fontsize=10)
    ax.set_title('Per-env reveal: pool averages 3 distinct regimes',
                 fontsize=11, pad=8)
    ax.legend(loc='lower left', fontsize=8.5)
    ax.grid(alpha=0.3, axis='x')

    # ── BOTTOM-LEFT: regime counts + takeaway ──
    ax = fig.add_subplot(gs[1, 0])
    ax.axis('off')
    # Regime counts
    from collections import Counter
    counter = Counter(r[0] for r in regimes)
    counts_text_lines = [
        'Per-env regime breakdown:',
        '',
    ]
    for regime, c in counter.most_common():
        counts_text_lines.append(f'  {c:>2d}  {regime}')
    counts_text_lines += [
        '',
        'The pool averages over all 12,',
        'losing the regime structure.',
        '',
        'Framework discipline:',
        'report PER-STRATUM verdicts +',
        'heterogeneity diagnostics —',
        'never just the pool.',
    ]
    ax.text(0.05, 0.95, '\n'.join(counts_text_lines),
            transform=ax.transAxes, fontsize=9.5, va='top',
            family='monospace',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fffaf0',
                      edgecolor='#888', linewidth=0.8))

    fig.suptitle(
        'Layer 4 (AGGREGATION DANGER, cross-env): per-env static '
        'mediation pooled into one cross-env number\n'
        'hides regime structure (sign-flippers + high-absorption '
        'envs + near-zero envs averaged into a single ρ).',
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')

    # ─── CSV ───
    with OUT_CSV.open('w') as f:
        f.write('env,rho_marginal,rho_partial,absorption_pct,regime\n')
        for (env, m, p), (regime, _) in zip(per_env, regimes, strict=False):
            f.write(f'{env_label(env)},{m:+.3f},{p:+.3f},'
                    f'{_absorption(m, p):+.1f},{regime}\n')
        f.write(f'\n# POOLED cross-env (12 strata, 680 cells):\n')
        f.write(f'# marginal ρ = {pooled_marg.rho_pooled:+.3f}, '
                f'p = {pooled_marg.p_value:.4f}\n')
        f.write(f'# partial  ρ = {pooled_part.rho_pooled:+.3f}, '
                f'p = {pooled_part.p_value:.4f}\n')
        f.write(f'# absorption = {pooled_absorb:.0f}%\n')
    print(f'saved → {OUT_PNG.name}, {OUT_CSV.name}')
    print(f'\npooled marginal ρ = {pooled_marg.rho_pooled:+.3f}')
    print(f'pooled partial  ρ = {pooled_part.rho_pooled:+.3f}')
    print(f'pooled absorption = {pooled_absorb:.0f}%')
    print(f'\nregime counts: {dict(counter)}')


if __name__ == '__main__':
    main()
