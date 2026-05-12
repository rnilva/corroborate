"""FourRooms vs Asterix-MinAtar comparison.

The do(τ) result from `polyak_tau_findings` shows DDQN's outcome
benefit is causally driven by target staleness on FourRooms (ATE=
−0.018, p=0.003). We don't have polyak data for Asterix yet (GPU
sweep queued in `polyak_tau_asterix.yaml`), but the existing
canonical corpora let us catalogue the structural differences
that might explain why staleness mediates DDQN on FourRooms but
the link is null/untestable on Asterix-sync=100.

Compares per env across all available cells (sync=100, gamma=0.99
where present):

1. **Reward polarity** (within-cell r(L, return)) — GOAL vs SURVIVAL
2. **Q-trajectory dynamics** (q_divergence_score) — bounded vs
   exploding
3. **Mechanism activation** (Δ_jens<0 fraction, |Δ_jens|) — whether
   DDQN's bias correction has bite
4. **Target staleness** (target_staleness_late) — the channel
   we're testing
5. **DDQN paired g(outcome)** — the outcome benefit
6. **Outcome variance** (σ across seeds) — power for the test

The hypothesis: FourRooms is a regime where staleness modulates
outcome cleanly because (a) Q stays bounded, (b) outcome variance
is non-trivial, (c) reward formula gives outcome dependence on
length-via-staleness routing. Asterix at sync=100 is in the
Q-explosion regime where mechanism HELD but link was previously
NULL — likely because Q dynamics dominate, dwarfing the staleness
channel.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import math
from pathlib import Path

import numpy as np
import polars as pl

import corroborate_rl.dqn.measurables  # register

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'


def _env_summary(df: pl.DataFrame, env: str) -> dict:
    """Per-env summary stats at sync=100, gamma=0.99 (matched
    regime)."""
    sub = df.filter(
        (pl.col('env_name') == env)
        & (pl.col('sync_period') == 100)
        & (pl.col('gamma') == 0.99)
    )
    if sub.height == 0:
        return {'env': env, 'n_cells': 0}

    out: dict[str, object] = {
        'env': env, 'n_cells': sub.height,
        'corpora': sorted(sub['corpus'].unique().to_list()),
    }

    # Per-arm cell count
    arms = sub['arm_key'].unique().to_list()
    out['arm_counts'] = {
        a: sub.filter(pl.col('arm_key') == a).height
        for a in arms
    }

    # Polarity
    pol = sub['env_reward_polarity'].drop_nans()
    out['env_reward_polarity'] = float(pol.mean()) if len(pol) > 0 else None

    # Q dynamics
    qd = sub['q_divergence_score'].drop_nans()
    if len(qd) > 0:
        out['q_divergence_score_mean'] = float(qd.mean())
        out['q_divergence_score_max'] = float(qd.max())

    # Mechanism (jensen_gap)
    jg = sub['jensen_gap'].drop_nans()
    if len(jg) > 0:
        out['jensen_gap_mean'] = float(jg.mean())
        out['jensen_gap_max'] = float(jg.max())

    # Outcome
    o = sub['eval_best_burst_mean'].drop_nans()
    if len(o) > 0:
        out['eval_best_burst_mean_mean'] = float(o.mean())
        out['eval_best_burst_mean_std_across_cells'] = float(o.std())
        out['eval_best_burst_mean_range'] = (float(o.min()), float(o.max()))

    # Target staleness
    if 'target_staleness_late' in sub.columns:
        ts = sub['target_staleness_late'].drop_nans()
        if len(ts) > 0:
            out['target_staleness_late_mean'] = float(ts.mean())
            out['target_staleness_late_range'] = (float(ts.min()), float(ts.max()))

    # Effective horizon
    if 'effective_horizon' in sub.columns:
        eh = sub['effective_horizon'].drop_nans()
        if len(eh) > 0:
            out['effective_horizon_mean'] = float(eh.mean())

    # Bootstrap fraction (≈ 1 - 1/E[L])
    if 'bootstrap_fraction' in sub.columns:
        bf = sub['bootstrap_fraction'].drop_nans()
        if len(bf) > 0:
            out['bootstrap_fraction_mean'] = float(bf.mean())

    # DDQN paired g(outcome) — within env, sync=100, paired by seed/gamma/sync/total_steps
    pair_keys = ['env_name', 'corpus', 'gamma', 'total_steps', 'sync_period', 'seed']
    v = sub.filter(pl.col('arm_key') == 'baseline').select(
        pair_keys + ['eval_best_burst_mean', 'jensen_gap']
    ).rename({'eval_best_burst_mean': 'out_v', 'jensen_gap': 'jens_v'})
    d = sub.filter(pl.col('arm_key') == DDQN).select(
        pair_keys + ['eval_best_burst_mean', 'jensen_gap']
    ).rename({'eval_best_burst_mean': 'out_d', 'jensen_gap': 'jens_d'})
    j = v.join(d, on=pair_keys, how='inner').filter(
        pl.col('out_v').is_finite() & pl.col('out_d').is_finite()
    )
    if j.height > 0:
        do = (j['out_d'] - j['out_v']).to_numpy()
        out['n_pairs'] = j.height
        out['mean_d_outcome'] = float(do.mean())
        out['sd_d_outcome'] = float(do.std(ddof=1)) if len(do) > 1 else float('nan')
        n = len(do)
        if out['sd_d_outcome'] > 0:
            g_raw = float(do.mean()) / out['sd_d_outcome']
            j_corr = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
            out['ddqn_paired_g'] = j_corr * g_raw
            out['ddqn_paired_g_se'] = math.sqrt(1.0 / n + g_raw ** 2 / (2.0 * n))
            out['ddqn_paired_g_t'] = out['ddqn_paired_g'] / out['ddqn_paired_g_se']
        # Δ_jens distribution
        if j['jens_v'].drop_nans().len() > 0 and j['jens_d'].drop_nans().len() > 0:
            jens_v_arr = j['jens_v'].to_numpy()
            jens_d_arr = j['jens_d'].to_numpy()
            mask = np.isfinite(jens_v_arr) & np.isfinite(jens_d_arr)
            if mask.sum() > 0:
                d_jens = jens_d_arr[mask] - jens_v_arr[mask]
                out['frac_mech_held'] = float((d_jens < 0).mean())
                out['mean_d_jens'] = float(d_jens.mean())

    return out


def main() -> None:
    df = pl.read_parquet('experiments/data/cache/ddqn.parquet')

    fr = _env_summary(df, 'FourRooms-misc')
    ax = _env_summary(df, 'Asterix-MinAtar')

    rows = [
        ('n_cells', 'n_cells'),
        ('arms', 'arm_counts'),
        ('env_reward_polarity (within-cell r(L,return))', 'env_reward_polarity'),
        ('Q-divergence score mean', 'q_divergence_score_mean'),
        ('Q-divergence score max', 'q_divergence_score_max'),
        ('jensen_gap mean', 'jensen_gap_mean'),
        ('jensen_gap max', 'jensen_gap_max'),
        ('outcome (eval_best_burst_mean) mean', 'eval_best_burst_mean_mean'),
        ('outcome std across cells', 'eval_best_burst_mean_std_across_cells'),
        ('outcome range', 'eval_best_burst_mean_range'),
        ('target_staleness_late mean', 'target_staleness_late_mean'),
        ('effective_horizon mean', 'effective_horizon_mean'),
        ('bootstrap_fraction mean', 'bootstrap_fraction_mean'),
        ('paired n', 'n_pairs'),
        ('DDQN mean Δ_outcome', 'mean_d_outcome'),
        ('DDQN paired-Δ outcome SD', 'sd_d_outcome'),
        ('DDQN paired g(outcome)', 'ddqn_paired_g'),
        ('DDQN paired g t-stat', 'ddqn_paired_g_t'),
        ('frac mech-HELD (Δ_jens<0)', 'frac_mech_held'),
        ('mean Δ_jens', 'mean_d_jens'),
    ]

    print()
    print('=== FourRooms-misc vs Asterix-MinAtar @ sync=100, γ=0.99 ===\n')
    print(f'{"property":<55} {"FourRooms":>20} {"Asterix":>20}')
    print('-' * 100)
    for label, key in rows:
        fv = fr.get(key, '-')
        av = ax.get(key, '-')
        if isinstance(fv, float) and not math.isnan(fv):
            fv_s = f'{fv:>+20.4g}' if 'paired' in label.lower() or 'std' in label.lower() else f'{fv:>20.4g}'
        elif isinstance(fv, dict):
            fv_s = f'{str(fv):>20}'
        elif isinstance(fv, tuple):
            fv_s = f'{fv[0]:.2g}…{fv[1]:.2g}'.rjust(20)
        else:
            fv_s = f'{str(fv):>20}'
        if isinstance(av, float) and not math.isnan(av):
            av_s = f'{av:>+20.4g}' if 'paired' in label.lower() or 'std' in label.lower() else f'{av:>20.4g}'
        elif isinstance(av, dict):
            av_s = f'{str(av):>20}'
        elif isinstance(av, tuple):
            av_s = f'{av[0]:.2g}…{av[1]:.2g}'.rjust(20)
        else:
            av_s = f'{str(av):>20}'
        print(f'{label:<55} {fv_s} {av_s}')

    print()
    print('=== Reading ===')
    print('FourRooms (where do(τ) HELD): GOAL polarity (negative), bounded')
    print('  Q (no explosion), substantial outcome variance, mech HELD with')
    print('  modest Δ_jens, outcome bounded in [0, 1] (sparse reward).')
    print()
    print('Asterix-MinAtar at sync=100: SURVIVAL polarity (positive), Q-')
    print('  EXPLOSION (q_divergence high), enormous Δ_jens (mech overfires),')
    print('  outcome on different scale entirely. The link τ→Δ_outcome may')
    print('  be drowned out by Q-dynamics-driven outcome variability.')

    import json
    out = Path('experiments/findings/sync_curve_breakout/fourrooms_vs_asterix.json')
    out.write_text(json.dumps({
        'FourRooms-misc': fr, 'Asterix-MinAtar': ax,
    }, indent=2, default=str))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
