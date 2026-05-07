"""Proper causal test of the Q-regime mechanism story.

Mechanism claim:
    r_min  →  sign(Q̄_late_vanilla)  →  direction(Hasselt bias)
                                      →  sign(ATE(stale → Δ_o))

Two formal tests:

1. **Per-stratum ATE under DoWhy backdoor + refutations**.
   Stratify pairs by sign(q_late_mean); run backdoor_ate per
   stratum. Predicts: positive Q-regime ATE > 0; negative
   Q-regime ATE < 0. Refutations validate per stratum.

2. **Interaction-term regression**.
   Δ_outcome = β₀ + β_S·stale + β_Q·q + β_int·(stale × q) + ε
   The interaction coefficient `β_int` tests whether staleness's
   effect on Δ_outcome depends on Q-regime. Predicts: β_int ≠ 0
   AND signed positive (because flipping q from negative to
   positive flips the staleness slope from negative to positive).

3. **Per-stratum partial Spearman**.
   ρ(stale, Δ_o | env) per Q-regime stratum, Fisher-z pooled.
   The framework's `stratified_partial_spearman_rho` is the JCI
   form. Predicts: ρ_strat > 0 for q_pos-regime; ρ_strat < 0 for
   q_neg-regime.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import t as t_dist

import corroborate_rl.dqn.measurables  # register
from corroborate.analyses.dowhy import (
    backdoor_ate, placebo_refutation, random_common_cause_refutation,
)
from corroborate.graph.discovery import stratified_partial_spearman_rho

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'


def _build_pairs() -> list[dict]:
    """Pair (DDQN, baseline) on (env, gamma, sync, total_steps,
    seed, target_sync.tau). Each pair carries: env, q_late_mean
    (from baseline), stale (baseline staleness), Δ_o."""
    rows: list[dict] = []
    for sweep_dir in (
        'polyak_tau_intervention', 'polyak_tau_asterix',
    ):
        base = Path(f'experiments/data/{sweep_dir}')
        for sub in sorted(base.iterdir()):
            if not sub.is_dir() or not (sub / 'runs.parquet').exists():
                continue
            runs = pl.read_parquet(sub / 'runs.parquet')
            ms_path = sub / 'measurements.parquet'
            if ms_path.exists():
                ms = pl.read_parquet(ms_path)
                collide = [c for c in ms.columns if c in runs.columns and c != 'id']
                if collide:
                    runs = runs.drop(collide)
                df = runs.join(ms, on='id', how='left')
            else:
                continue  # skip τ=1.0 sub-sweep without traces
            keyed: dict[
                tuple, dict[str, dict[str, object]],
            ] = defaultdict(dict)
            for cell in df.iter_rows(named=True):
                arm = cell['arm_key']
                if arm not in ('baseline', DDQN):
                    continue
                key = (
                    cell.get('env_name'), cell.get('gamma'),
                    cell.get('sync_period'), cell.get('total_steps'),
                    cell.get('seed'), cell.get('target_sync.tau'),
                )
                keyed[key][arm] = cell
            for k, arms in keyed.items():
                v = arms.get('baseline')
                d = arms.get(DDQN)
                if v is None or d is None:
                    continue
                env = v['env_name']
                stale = v.get('target_staleness_late')
                q = v.get('q_late_mean')
                ot = d.get('eval_best_burst_mean')
                ob = v.get('eval_best_burst_mean')
                if any(x is None for x in (stale, q, ot, ob)):
                    continue
                if any(
                    not isinstance(x, (int, float)) or math.isnan(x)
                    for x in (stale, q, ot, ob)
                ):
                    continue
                rows.append({
                    'env': env,
                    'tau': float(k[5]) if k[5] is not None else float('nan'),
                    'stale': float(stale),
                    'q_late_mean': float(q),
                    'delta_outcome': float(ot) - float(ob),
                })
    return rows


def main() -> None:
    rows = _build_pairs()
    print(f'paired rows: {len(rows)}')
    df = pl.DataFrame(rows)
    print(df.group_by('env').agg(
        pl.len().alias('n'),
        pl.col('q_late_mean').mean().alias('q_mean'),
    ).sort('q_mean'))

    # Tag regime
    df = df.with_columns(
        (pl.col('q_late_mean') > 0).alias('q_pos'),
    )

    # ============================================================
    # CHECK 1: per-stratum DoWhy backdoor ATE
    # ============================================================
    print()
    print('=== CHECK 1: per-Q-regime backdoor ATE(stale → Δ_o) + refutations ===\n')
    print(f'{"regime":<10} {"n":>4} {"ATE":>10} {"placebo":>10} {"rcc_drift":>10} {"sign":<10}')
    print('-' * 65)
    panel_per_regime = []
    for regime_label, mask in [
        ('Q > 0', df['q_pos']),
        ('Q < 0', ~df['q_pos']),
    ]:
        sub = df.filter(mask)
        if sub.height < 10:
            continue
        cells = sub.select(['stale', 'delta_outcome']).to_dicts()
        dag = [('stale', 'delta_outcome')]
        try:
            r = backdoor_ate.fn(
                cells, treatment='stale', outcome='delta_outcome',
                dag=dag, method_name='backdoor.linear_regression',
            )
            pl_r = placebo_refutation.fn(
                cells, treatment='stale', outcome='delta_outcome',
                dag=dag, method_name='backdoor.linear_regression',
            )
            rcc_r = random_common_cause_refutation.fn(
                cells, treatment='stale', outcome='delta_outcome',
                dag=dag, method_name='backdoor.linear_regression',
            )
            sign = 'POS' if r.ate > 0 else 'NEG'
            print(
                f'{regime_label:<10} {sub.height:>4d} {r.ate:>+10.3f} '
                f'{pl_r.refuted_ate:>+10.3f} {rcc_r.drift:>10.4f} {sign:<10}',
                flush=True,
            )
            panel_per_regime.append({
                'regime': regime_label, 'n': sub.height,
                'ate': r.ate, 'placebo_refuted_ate': pl_r.refuted_ate,
                'rcc_drift': rcc_r.drift, 'identified': r.identified,
            })
        except Exception as e:
            print(f'{regime_label}: ERROR {type(e).__name__}: {e}')

    # ============================================================
    # CHECK 2: interaction-term regression
    #   Δ_o = β0 + β_S·stale + β_Q·q + β_int·(stale × q) + ε
    # ============================================================
    print()
    print('=== CHECK 2: interaction term `stale × q_late_mean` ===\n')

    n = df.height
    stale = df['stale'].to_numpy()
    q = df['q_late_mean'].to_numpy()
    do = df['delta_outcome'].to_numpy()

    X = np.column_stack([
        np.ones(n),
        stale,
        q,
        stale * q,
    ])
    beta, _, rank, _ = np.linalg.lstsq(X, do, rcond=None)
    y_pred = X @ beta
    resid = do - y_pred
    df_resid = n - rank
    sigma2 = float((resid ** 2).sum() / df_resid)
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    t_stats = beta / se
    p_vals = 2 * (1.0 - t_dist.cdf(np.abs(t_stats), df=df_resid))

    names = ['intercept', 'stale', 'q_late_mean', 'stale × q_late_mean']
    print(f'{"term":<22} {"β":>12} {"SE":>10} {"t":>7} {"p":>10}')
    print('-' * 70)
    for nm, b, s, t, p in zip(names, beta, se, t_stats, p_vals):
        print(f'{nm:<22} {b:>+12.4f} {s:>10.4f} {t:>+7.2f} {p:>10.4g}')

    int_t = float(t_stats[3])
    int_p = float(p_vals[3])
    if abs(int_t) >= 2.0 and int_p < 0.05:
        print()
        print(
            '  → INTERACTION SIGNIFICANT: staleness\'s effect on Δ_o '
            'depends on Q-regime.',
            flush=True,
        )

    # ============================================================
    # CHECK 3: stratified partial Spearman ρ(stale, Δ_o | env)
    # per Q-regime, Fisher-z pooled.
    # ============================================================
    print()
    print('=== CHECK 3: stratified partial Spearman per Q-regime ===\n')
    print(f'{"regime":<10} {"ρ_strat":>10} {"p":>10} {"n_strata":>10}')
    print('-' * 50)

    panel_strat = []
    for regime_label, mask in [
        ('Q > 0', df['q_pos']),
        ('Q < 0', ~df['q_pos']),
    ]:
        sub = df.filter(mask)
        if sub.height < 10:
            continue
        x = sub['stale'].to_numpy()
        y = sub['delta_outcome'].to_numpy()
        z = sub['q_late_mean'].to_numpy()
        strata = sub['env'].to_list()
        rho, p = stratified_partial_spearman_rho(
            x, y, z, strata, min_stratum_size=5,
        )
        print(f'{regime_label:<10} {rho:>+10.4f} {p:>10.4g} {len(set(strata)):>10}')
        panel_strat.append({
            'regime': regime_label, 'rho_strat_partial': float(rho),
            'p': float(p), 'n_strata': len(set(strata)),
        })

    # ============================================================
    # Save
    # ============================================================
    out = {
        'per_regime_ate': panel_per_regime,
        'interaction_test': {
            'beta_intercept': float(beta[0]),
            'beta_stale': float(beta[1]),
            'beta_q_late_mean': float(beta[2]),
            'beta_interaction': float(beta[3]),
            't_interaction': int_t,
            'p_interaction': int_p,
            'n_obs': n,
            'r_squared_total': float(
                1.0 - resid.var(ddof=0) / do.var(ddof=0)
            ) if do.var() > 0 else float('nan'),
        },
        'stratified_partial_spearman': panel_strat,
    }
    out_path = Path(
        'experiments/findings/sync_curve_breakout/q_regime_interaction_test.json'
    )
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f'\nwrote: {out_path}', flush=True)


if __name__ == '__main__':
    main()
