"""Observed n-step attenuation on FourRooms γ=0.99 — and why it's
NOT decisive evidence for the integrated-clip theorem.

Initial framing: longer n-step shrinks bootstrap weight γⁿ, so
the integrated structural clip during training should shrink, so
|Δjens| should shrink monotonically.

The data confirms the monotonicity (ρ(n_step, |Δjens|) = −1.000)
but this is observationally equivalent to a trivial alternative:
at n=∞, training targets ARE Monte-Carlo, so Q → MC in both arms
by construction, so jens = max(0, Q−MC) → 0 in both arms
mechanically. Δjens shrinking to zero is forced by the training
setup, not by the clip story.

The falsifiable quantity is **relative attenuation**:
`|Δjens| / jens_van`. The theorem predicts THIS also shrinks
(less bootstrap weight → less DDQN distinctiveness). Empirically
it does NOT — it sits at ~50-65% across all n_step values.

So the n-step axis on this corpus cannot distinguish the
integrated-clip theory from "Q is trained on something closer to
MC at high n". This script reports the result for completeness;
do not cite as evidence for the theorem.

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
    print('Absolute attenuation (NOT decisive — see below):')
    print(f'  ρ(n_step, |ΔQ|):    {rho_q:+.3f} p={p_q:.4f}')
    print(f'  ρ(n_step, |Δjens|): {rho_j:+.3f} p={p_j:.4f}')

    # Per-arm levels at each n_step — the diagnostic that shows jens_van
    # itself shrinks toward zero (training-target → MC at high n).
    df_summary = df.group_by(['n_step', 'arm']).agg([
        pl.col('jensen_gap').mean().alias('jens_mean'),
        pl.len().alias('n'),
    ]).sort(['n_step', 'arm'])
    print()
    print('Relative attenuation |Δjens| / jens_van — the falsifiable quantity:')
    print(f"{'n_step':>7} | {'jens_van':>10} {'jens_ddqn':>10} | {'rel_atten':>10}")
    print('-' * 50)
    for n in sorted(df.get_column('n_step').unique().to_list()):
        v = float(df_summary.filter((pl.col('n_step') == n) & (pl.col('arm') == 'vanilla')).get_column('jens_mean')[0])
        d = float(df_summary.filter((pl.col('n_step') == n) & (pl.col('arm') == 'ddqn')).get_column('jens_mean')[0])
        rel = abs(d - v) / v if v > 0 else float('nan')
        print(f"{n:>7} | {v:>10.4f} {d:>10.4f} | {rel:>10.1%}")
    print()
    print('Conclusion: vanilla jens shrinks 65× from n=1 to n=10 — Q-trained-on-MC')
    print('drives the absolute |Δjens| → 0, NOT the clip story. Relative attenuation')
    print('stays ~50-65% across n_step, contradicting the theorem\'s strict prediction')
    print('that relative effect should also shrink. The n-step axis here is not')
    print('a falsifying test for the integrated-clip framing.')


if __name__ == '__main__':
    main()
