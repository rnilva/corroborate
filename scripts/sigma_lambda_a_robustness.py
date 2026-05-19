"""Robustness audit for σ_Λ_a → d_out cross-env moderation (n=8).

The framework-typed bridge `sigma_lambda_a_moderates_ddqn_outcome__
cross_env_g0999` fires POWER_INSUFFICIENT at the current cache
state: ρ=−0.619, p=0.102, n_strata=8 (below the p≤0.05 +
|ρ|≥0.6 calibration). This script characterises three aspects
of the n=8 finding's robustness without new data:

1. **Jackknife**: leave-one-env-out — recompute ρ on the
   remaining 7 envs, 8 times. Identifies single-env load-bearing
   observations and surfaces "without env X" sensitivity.

2. **Permutation null**: 1000 random shuffles of (σ_Λ_a, d_out)
   pairings, empirical fraction with |ρ_shuffled| ≥ 0.619.
   Empirical FPR for the observed magnitude under H0.

3. **Bootstrap CI**: 1000 resamples-with-replacement of the n=8
   panel, 2.5/97.5 percentiles of ρ. 95% CI on the population
   ρ given the n=8 sample.

Output: stderr + JSON to `/tmp/sigma_lambda_a_robustness.json`
for downstream commit-time reference.
"""
from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import polars as pl
from scipy.stats import spearmanr


def main() -> int:
    cache = Path('experiments/data/cache/ddqn_sweeps.parquet')
    df = pl.read_parquet(cache)
    panel = df.filter(
        (pl.col('gamma') == 0.999)
        & pl.col('lambda_a_late').is_finite()
        & pl.col('eval_best_burst_raw_mean').is_finite()
    )
    vanilla = panel.filter(pl.col('arm_key') == 'baseline')
    ddqn = panel.filter(pl.col('arm_key') != 'baseline')

    v_agg = (
        vanilla.group_by('env_name')
        .agg([
            pl.col('lambda_a_late').std().alias('sigma_la'),
            pl.col('eval_best_burst_raw_mean').mean().alias('v_out'),
        ])
        .sort('env_name')
    )
    d_agg = (
        ddqn.group_by('env_name')
        .agg([pl.col('eval_best_burst_raw_mean').mean().alias('d_out_mean')])
        .sort('env_name')
    )
    panel_per_env = v_agg.join(d_agg, on='env_name').with_columns(
        (pl.col('d_out_mean') - pl.col('v_out')).alias('delta_out')
    )
    envs: tuple[str, ...] = tuple(panel_per_env.get_column('env_name').to_list())
    sigma_la_arr = panel_per_env.get_column('sigma_la').to_numpy()
    delta_out_arr = panel_per_env.get_column('delta_out').to_numpy()
    n = len(envs)

    point_estimate = spearmanr(sigma_la_arr, delta_out_arr)
    rho_point = float(point_estimate.statistic)
    p_point = float(point_estimate.pvalue)

    print(f'Panel: {n} envs', file=sys.stderr)
    print(f'Point estimate: ρ={rho_point:+.3f} p={p_point:.3g}', file=sys.stderr)
    print(file=sys.stderr)

    # ---- Jackknife ----
    print('=== Jackknife (leave-one-env-out) ===', file=sys.stderr)
    jackknife: list[dict[str, object]] = []
    for i, dropped in enumerate(envs):
        mask = [j for j in range(n) if j != i]
        xs = sigma_la_arr[mask]
        ys = delta_out_arr[mask]
        r = spearmanr(xs, ys)
        rho_i = float(r.statistic)
        p_i = float(r.pvalue)
        jackknife.append({'dropped': dropped, 'rho': rho_i, 'p': p_i})
        print(
            f'  drop {dropped:<25}  ρ={rho_i:+.3f}  p={p_i:.3g}',
            file=sys.stderr,
        )
    rho_jk = [j['rho'] for j in jackknife]
    rho_jk_min = min(rho_jk, key=lambda x: float(x))
    rho_jk_max = max(rho_jk, key=lambda x: float(x))
    print(f'  range: [{rho_jk_min:+.3f}, {rho_jk_max:+.3f}]', file=sys.stderr)
    print(file=sys.stderr)

    # ---- Permutation null ----
    print('=== Permutation null (K=10000) ===', file=sys.stderr)
    K = 10_000
    rng = random.Random(20260519)
    abs_obs = abs(rho_point)
    n_ge = 0
    null_rhos: list[float] = []
    for _ in range(K):
        ys_shuf = list(delta_out_arr)
        rng.shuffle(ys_shuf)
        r_shuf = spearmanr(sigma_la_arr, ys_shuf)
        rho_shuf = float(r_shuf.statistic)
        if math.isnan(rho_shuf):
            continue
        null_rhos.append(rho_shuf)
        if abs(rho_shuf) >= abs_obs:
            n_ge += 1
    perm_p = n_ge / max(len(null_rhos), 1)
    print(
        f'  P(|ρ_null| ≥ {abs_obs:.3f}) = {n_ge}/{len(null_rhos)} = {perm_p:.4f}',
        file=sys.stderr,
    )
    print(f'  null SD: {pl.Series(null_rhos).std():.3f}', file=sys.stderr)
    print(file=sys.stderr)

    # ---- Bootstrap CI ----
    print('=== Bootstrap (K=10000) ===', file=sys.stderr)
    rng2 = random.Random(20260520)
    boot_rhos: list[float] = []
    for _ in range(K):
        idx = [rng2.randint(0, n - 1) for _ in range(n)]
        xs_b = sigma_la_arr[idx]
        ys_b = delta_out_arr[idx]
        r_b = spearmanr(xs_b, ys_b)
        rho_b = float(r_b.statistic)
        if not math.isnan(rho_b):
            boot_rhos.append(rho_b)
    boot_rhos.sort()
    ci_lo = boot_rhos[int(0.025 * len(boot_rhos))]
    ci_hi = boot_rhos[int(0.975 * len(boot_rhos))]
    boot_mean = sum(boot_rhos) / len(boot_rhos)
    print(
        f'  ρ̂ = {boot_mean:+.3f}  95% CI = [{ci_lo:+.3f}, {ci_hi:+.3f}]',
        file=sys.stderr,
    )
    print(file=sys.stderr)

    # ---- Persist ----
    out = {
        'panel_n': n,
        'envs': list(envs),
        'sigma_lambda_a': list(sigma_la_arr),
        'delta_out': list(delta_out_arr),
        'point_estimate': {'rho': rho_point, 'p': p_point},
        'jackknife': jackknife,
        'permutation_null': {
            'K': K, 'observed_abs_rho': abs_obs,
            'p_empirical': perm_p, 'null_sd': float(pl.Series(null_rhos).std()),
        },
        'bootstrap': {
            'K': K, 'mean': boot_mean,
            'ci_low': ci_lo, 'ci_high': ci_hi,
        },
    }
    out_path = Path('/tmp/sigma_lambda_a_robustness.json')
    out_path.write_text(json.dumps(out, indent=2))
    print(f'Wrote {out_path}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
