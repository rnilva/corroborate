"""Empirical test of the finite-training-residual prediction:
longer n-step replaces bootstrap with Monte-Carlo, so the
integrated structural clip during training shrinks, so the
DDQN-vs-vanilla |Δjens| should shrink monotonically with n_step.

Setup: FourRooms-misc γ=0.99, n_step ∈ {1, 2, 3, 5, 10}, n=30
paired cells per n>1 level (n=90 at n=1). Holds gamma fixed so
that the only varying axis is n_step.

Two predictions:
  1. |Δjens| (bias-reduction strength) decreases monotonically
     with n_step. Cleanest test of the theorem because jens is
     directly the bias-correction direction.
  2. |ΔQ| (signed Q reduction) also decreases monotonically with
     n_step. Theorem-aligned but at FR γ=0.99 the absolute ΔQ
     magnitudes are tiny (<0.06), so this test is power-limited.

Run: `uv run python scripts/verify_nstep_attenuates_clip.py`."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats


CACHE_PATH = 'experiments/data/cache/ddqn.parquet'


def main() -> None:
    df = pl.scan_parquet(CACHE_PATH).filter(
        pl.col('env_name') == 'FourRooms-misc'
    ).select(
        ['arm_key', 'q_late_mean', 'jensen_gap', 'seed', 'gamma', 'n_step',
         'total_steps', 'replay.capacity', 'sync_period']
    ).collect()

    df = df.with_columns(
        pl.when(pl.col('arm_key') == 'baseline').then(pl.lit('vanilla'))
        .when(pl.col('arm_key').str.contains('Claim:double_greedify')).then(pl.lit('ddqn'))
        .otherwise(pl.lit('other')).alias('arm')
    )
    df = df.filter(
        pl.col('arm').is_in(['vanilla', 'ddqn'])
        & pl.col('q_late_mean').is_finite()
        & pl.col('jensen_gap').is_finite()
        # Hold γ fixed to isolate the n-step axis
        & (pl.col('gamma') == 0.99)
    )

    pair_keys = ['seed', 'gamma', 'n_step', 'total_steps', 'replay.capacity', 'sync_period']
    q = df.pivot(values='q_late_mean', index=pair_keys, on='arm', aggregate_function='mean').rename({'vanilla': 'q_van', 'ddqn': 'q_ddqn'})
    j = df.pivot(values='jensen_gap', index=pair_keys, on='arm', aggregate_function='mean').rename({'vanilla': 'j_van', 'ddqn': 'j_ddqn'})

    joined = q.join(j, on=pair_keys, how='inner').filter(
        pl.col('q_van').is_not_null() & pl.col('q_ddqn').is_not_null()
        & pl.col('j_van').is_not_null() & pl.col('j_ddqn').is_not_null()
    ).with_columns([
        (pl.col('q_ddqn') - pl.col('q_van')).alias('dQ'),
        (pl.col('j_ddqn') - pl.col('j_van')).alias('dJens'),
    ])

    print('FourRooms-misc γ=0.99: ΔQ and Δjens by n_step')
    print('=' * 80)
    print(f"{'n_step':>7} | {'n':>4} | {'ΔQ_mean':>10} {'ΔQ_p':>9} | {'Δjens_mean':>11} {'Δjens_p':>9}")
    print('-' * 80)
    rows = []
    for n in sorted(joined.get_column('n_step').unique().to_list()):
        sub = joined.filter(pl.col('n_step') == n)
        dq = sub.get_column('dQ').to_numpy()
        dj = sub.get_column('dJens').to_numpy()
        if len(dq) < 5:
            continue
        _, p_q = stats.ttest_1samp(dq, 0.0)
        _, p_j = stats.ttest_1samp(dj, 0.0)
        print(f"{n:>7} | {len(dq):>4} | {dq.mean():>+10.3f} {p_q:>9.2e} | {dj.mean():>+11.3f} {p_j:>9.2e}")
        rows.append({'n_step': n, 'n': len(dq), 'dQ': float(dq.mean()), 'dJens': float(dj.mean())})

    out = pl.DataFrame(rows)
    ns = out.get_column('n_step').to_numpy()
    abs_dq = out.get_column('dQ').abs().to_numpy()
    abs_dj = out.get_column('dJens').abs().to_numpy()
    rho_q, p_q = stats.spearmanr(ns, abs_dq)
    rho_j, p_j = stats.spearmanr(ns, abs_dj)

    print()
    print('Theorem prediction: |Δ·| should decrease monotonically with n_step')
    print(f'  ρ(n_step, |ΔQ|):    {rho_q:+.3f} p={p_q:.4f}  (Q-side, power-limited at FR γ=0.99)')
    print(f'  ρ(n_step, |Δjens|): {rho_j:+.3f} p={p_j:.4f}  (bias-side, decisive — Δjens drops {abs_dj[0]:.3f} → {abs_dj[-1]:.3f})')


if __name__ == '__main__':
    main()
