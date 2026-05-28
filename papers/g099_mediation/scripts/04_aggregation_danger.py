"""Layer 4 — AGGREGATION DANGER: pooled mediation hides per-burst heterogeneity.

Two dramatic examples drawn from the canonical panel:

  - **PacMan γ=0.99**: pooled bg-mediation looks meaningful in
    naïve cross-env tables; the DL random-effects pool says ρ≈0
    with I²≈0 (no signal at all); the per-burst trajectory shows
    the marginal ρ FLIPS SIGN at mid-training. The pooled number
    was Simpson's-paradox averaging across opposite-sign bursts.

  - **Asterix γ=0.99**: pooled bg-mediation looks weak (small
    percent) but the per-burst trajectory shows STRONG
    heterogeneity (I²≈0.7) and SIGN_FLIP_DETECTED — bg mediates
    for some bursts but the arm→outcome edge persists for others.
    The pooled summary collapses two qualitatively different
    regimes into one number.

The framework's contribution is the per-stratum + per-burst typed
verdict surface that catches both pathologies without forcing the
analyst to know which one to look for.
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
    MEDIATOR_PER_BURST_COL, MEDIATOR_PER_BURST_LABEL,
    OUTCOME_PER_BURST_COL, TREATMENT_ARM,
    env_label, load_g099_canonical_panel,
)

from corroborate.analyses.dynamic_mediation.partial_spearman import (
    dynamic_partial_spearman,
)


OUT_PNG = SCRIPT_DIR.parent / 'figures' / '04_aggregation_danger.png'
OUT_CSV = SCRIPT_DIR.parent / 'figures' / '04_aggregation_danger.csv'

DRAMATIC_PAIR = ('PacMan-jumanji', 'Asterix-MinAtar')


def main() -> None:
    df = load_g099_canonical_panel()
    # Primitive expects raw `arm_key` and encodes internally.
    res = dynamic_partial_spearman.fn(
        df.filter(pl.col('env_name').is_in(DRAMATIC_PAIR)),
        arm_field='arm_key',
        mediator_per_burst=MEDIATOR_PER_BURST_COL,
        outcome_per_burst=OUTCOME_PER_BURST_COL,
        stratify_by=('env_name',),
        min_n_per_burst=8,
        n_bootstrap=1000,
        bootstrap_seed=42,
    )

    # ─── figure: 2-column ───
    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5),
                             sharex='col', height_ratios=[2, 1])

    print(f'  result keys: {list(res.keys())[:5]}...')
    # Build env→result lookup robust to whatever key shape the primitive uses.
    env_to_result = {}
    for sid, r in res.items():
        for v in (sid if isinstance(sid, tuple) else (sid,)):
            if v in DRAMATIC_PAIR:
                env_to_result[v] = r
                break

    for col, env in enumerate(DRAMATIC_PAIR):
        if env not in env_to_result:
            print(f'  {env}: not in result')
            continue
        r = env_to_result[env]
        n_b = len(r.rho_marginal)
        x = np.arange(n_b)

        ax_top = axes[0, col]
        # Plot per-burst marginal + partial trajectories
        ax_top.plot(x, r.rho_marginal, color='steelblue', linewidth=1.4,
                    label='marginal ρ(arm, outcome)', marker='o',
                    markersize=3, alpha=0.85)
        ax_top.plot(x, r.rho_partial, color='goldenrod', linewidth=1.4,
                    label=f'partial ρ(arm, outcome | {MEDIATOR_PER_BURST_LABEL})',
                    marker='D', markersize=3, alpha=0.85)
        ax_top.axhline(0, color='black', linewidth=0.4)
        ax_top.set_ylabel('Spearman ρ', fontsize=9.5)
        ax_top.set_title(f'{env_label(env)} γ=0.99', fontsize=11, pad=4)
        ax_top.legend(loc='upper right', fontsize=8)
        ax_top.grid(alpha=0.3)
        ax_top.set_ylim(-1, 1)

        # Annotate aggregation_status + DL pool
        dl_m = r.dl_marginal
        dl_p = r.dl_partial
        anno = (
            f'aggregation_status (marginal): {r.aggregation_status.name}\n'
            f'DL pool ρ (marginal): {dl_m.rho_pooled:+.3f}, '
            f'I²={dl_m.i2:.2f}\n'
            f'DL pool ρ (partial):  {dl_p.rho_pooled:+.3f}, '
            f'I²={dl_p.i2:.2f}\n'
            f'n_bursts={len(r.burst_steps)}, max n_per_burst={max(r.n_per_burst)}'
        )
        # Bootstrap CIs if present
        if r.bootstrap_marginal is not None and r.bootstrap_partial is not None:
            anno += (
                f'\nbootstrap CI ρ(marg): '
                f'[{r.bootstrap_marginal.rho_lower:+.2f}, '
                f'{r.bootstrap_marginal.rho_upper:+.2f}]'
                f'\nbootstrap CI ρ(part): '
                f'[{r.bootstrap_partial.rho_lower:+.2f}, '
                f'{r.bootstrap_partial.rho_upper:+.2f}]'
            )
        ax_top.text(0.02, 0.04, anno, transform=ax_top.transAxes,
                    va='bottom', fontsize=7.5, family='monospace',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#fffaf0',
                              edgecolor='#cc9900', linewidth=0.6))

        # Bottom panel: per-burst n samples
        ax_bot = axes[1, col]
        ax_bot.bar(x, r.n_per_burst, color='gray', alpha=0.6, width=0.8)
        ax_bot.set_ylabel('n per burst', fontsize=9)
        ax_bot.set_xlabel('burst index', fontsize=9)
        ax_bot.grid(alpha=0.3, axis='y')

    fig.suptitle(
        'Layer 4 (AGGREGATION DANGER): per-burst dynamic mediation reveals what pooling hides\n'
        f'mediator: {MEDIATOR_PER_BURST_LABEL} (clean Bellman-residual, no MC-leak)',
        fontsize=11.5,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(OUT_PNG, dpi=120, bbox_inches='tight')

    # ─── CSV ───
    with OUT_CSV.open('w') as f:
        f.write('env,burst,n,rho_marginal,rho_partial\n')
        for env in DRAMATIC_PAIR:
            if env not in env_to_result:
                continue
            r = env_to_result[env]
            for b in range(len(r.rho_marginal)):
                f.write(f'{env_label(env)},{b},{r.n_per_burst[b]},'
                        f'{r.rho_marginal[b]:+.4f},{r.rho_partial[b]:+.4f}\n')
            dl_m = r.dl_marginal; dl_p = r.dl_partial
            f.write(f'# {env_label(env)} agg_status={r.aggregation_status.name}; '
                    f'DL marg ρ={dl_m.rho_pooled:+.3f} I²={dl_m.i2:.2f}; '
                    f'DL part ρ={dl_p.rho_pooled:+.3f} I²={dl_p.i2:.2f}\n')
    print(f'saved → {OUT_PNG.name}, {OUT_CSV.name}')
    for env in DRAMATIC_PAIR:
        if env in env_to_result:
            r = env_to_result[env]
            print(f'  {env_label(env):12s}  status={r.aggregation_status.name:25s} '
                  f'DL marg ρ={r.dl_marginal.rho_pooled:+.3f} I²={r.dl_marginal.i2:.2f}')


if __name__ == '__main__':
    main()
