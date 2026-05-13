"""Polyak-τ intervention test of the integrated-clip framing.

The clean intervention the n-step axis couldn't be: Polyak-τ
controls Q_target ↔ Q_online divergence without changing what Q
is trained on. Vanilla's jens isn't dragged toward zero by the
intervention, so the relative-attenuation prediction
|Δjens|/jens_van is resolvable.

Predicted (integrated-clip theory, strict form):
- τ=1.0: target ≡ online every step ⇒ clip = 0 by construction ⇒
  DDQN ≡ vanilla algorithmically ⇒ Δjens = 0 (theorem's
  necessary condition).
- τ < 1.0: smaller τ ⇒ higher target staleness ⇒ larger per-step
  clip ⇒ larger relative attenuation.

Empirical result on `polyak_tau_fr_postfix` (FR γ=0.99,
n=30/τ paired):
  τ=1.0   : Δjens = +0.0000 EXACT, rel_atten = 0.0%
  τ=0.1   : Δjens = -0.0475***,    rel_atten = 28.0%
  τ=0.01  : Δjens = -0.0411*,      rel_atten = 30.4%
  τ=0.001 : Δjens = -0.1462***,    rel_atten = 73.0%
  ρ(1/τ, rel_atten) = +1.000 (monotone, all 4 points).

Run: `uv run python scripts/verify_polyak_relative_attenuation.py`."""
from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats


CACHE_PATH = 'experiments/data/cache/ddqn.parquet'


def main() -> None:
    df = pl.scan_parquet(CACHE_PATH).filter(
        pl.col('target_sync.tau').is_not_null()
    ).select(['arm_key', 'jensen_gap', 'q_late_mean', 'target_sync.tau', 'seed']).collect()

    df = df.with_columns(
        pl.when(pl.col('arm_key') == 'baseline').then(pl.lit('vanilla'))
        .when(pl.col('arm_key').str.contains('Claim:double_greedify')).then(pl.lit('ddqn'))
        .otherwise(pl.lit('other')).alias('arm')
    )
    df = df.filter(pl.col('jensen_gap').is_finite() & pl.col('arm').is_in(['vanilla', 'ddqn']))

    paired = (
        df.pivot(values='jensen_gap', index=['target_sync.tau', 'seed'], on='arm', aggregate_function='mean')
        .rename({'vanilla': 'jens_van', 'ddqn': 'jens_ddqn'})
        .filter(pl.col('jens_van').is_not_null() & pl.col('jens_ddqn').is_not_null())
        .with_columns((pl.col('jens_ddqn') - pl.col('jens_van')).alias('dJens'))
    )

    print('Polyak-τ on FourRooms γ=0.99 (post-buffer-fix, n=30/τ paired):')
    print('=' * 85)
    print(f"{'τ':>8} | {'n':>4} | {'jens_van':>10} {'jens_ddqn':>11} | {'Δjens':>9} {'p':>9} | {'rel_atten':>10}")
    print('-' * 85)
    rows = []
    for tau in sorted(paired.get_column('target_sync.tau').unique().to_list()):
        sub = paired.filter(pl.col('target_sync.tau') == tau)
        v = sub.get_column('jens_van').to_numpy()
        d = sub.get_column('jens_ddqn').to_numpy()
        dj = sub.get_column('dJens').to_numpy()
        p_j = stats.ttest_1samp(dj, 0.0).pvalue if dj.std(ddof=1) > 0 else float('nan')
        rel = abs(dj.mean()) / v.mean() if v.mean() > 0 else float('nan')
        rows.append({'tau': tau, 'n': len(dj), 'rel_atten': rel, 'dJens': float(dj.mean())})
        print(f"{tau:>8.4f} | {len(dj):>4} | {v.mean():>10.4f} {d.mean():>11.4f} | "
              f"{dj.mean():>+9.4f} {p_j:>9.2e} | {rel:>10.1%}")

    out = pl.DataFrame(rows).sort('tau')
    inv_tau = 1.0 / out.get_column('tau').to_numpy()
    rel = out.get_column('rel_atten').to_numpy()
    abs_dj = out.get_column('dJens').abs().to_numpy()
    rho_rel, p_rel = stats.spearmanr(inv_tau, rel)
    rho_abs, p_abs = stats.spearmanr(inv_tau, abs_dj)

    print()
    print('Theorem prediction: relative attenuation grows monotonically with 1/τ')
    print(f'  ρ(1/τ, |Δjens|)            = {rho_abs:+.3f}, p={p_abs:.4f}')
    print(f'  ρ(1/τ, |Δjens|/jens_van)   = {rho_rel:+.3f}, p={p_rel:.4f}  <-- the falsifiable quantity')
    print()
    print('Note: τ=1.0 gives Δjens = +0.0000 EXACT — at τ=1, target=online every step,')
    print('so DDQN and vanilla compute IDENTICAL bootstrap targets and are algorithmically')
    print('equivalent. This is the theorem\'s necessary condition manifest in the data.')


if __name__ == '__main__':
    main()
