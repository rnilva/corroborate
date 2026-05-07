"""Polyak-τ causal analysis using framework primitives (replaces the
naive meta-regression which conflated descriptive trend with causal
evidence).

Three primitives applied to the polyak sweep within each env:

1. **`backdoor_ate`** with DAG `tau → delta_outcome`. τ is exogenous
   (intervened by sweep design), so the backdoor ATE is just a
   linear-regression slope of Δ_outcome on log τ. The framework
   wraps this in DoWhy + refutation tests.

2. **`placebo_refutation`** — shuffle τ within env, the ATE should
   collapse. If it doesn't, the original ATE was an artifact of
   randomness (small-sample noise).

3. **`random_common_cause_refutation`** — add a synthetic confounder
   between τ and outcome. The real ATE should be robust.

The pair-level `Δ_outcome = DDQN_outcome − baseline_outcome at fixed
seed and τ` is the unit of analysis. By construction, τ varies across
sub-sweeps but the (seed, env) pair is fixed; treating Δ_outcome as
the outcome and τ as the treatment under do(τ) puts us at Pearl
rung-2 with no need for confounder adjustment. The refutations are
therefore the meaningful tests — they validate that the
log-τ → Δ_outcome relationship isn't driven by chance.

Output: per-env table of (ATE, p_value, placebo, RCC) values.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from pathlib import Path

import numpy as np
import polars as pl

import corroborate_rl.dqn.measurables  # register
from corroborate.analyses.dowhy import (
    backdoor_ate, placebo_refutation, random_common_cause_refutation,
)
from corroborate.runner.runner import _join_required_traces, _measurable_signature
from corroborate.corpus.measurements import build_measurements, load_measurements
from corroborate.measurables import get_registered

DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'


def _load_polyak_pairs() -> pl.DataFrame:
    """Load polyak sub-sweeps, pair (DDQN, baseline) on (env, seed,
    τ), compute Δ_outcome and Δ_staleness. Returns flat
    DataFrame ready for causal analysis: one row per
    (env, seed, τ) pair."""
    base = Path('experiments/data/polyak_tau_intervention')
    required = (
        'target_staleness_late', 'eval_best_burst_mean',
    )
    trace_reads = set()
    for n in required:
        m = get_registered(n)
        if m and hasattr(m, 'reads'):
            trace_reads.update(m.reads)

    cells: list[pl.DataFrame] = []
    for sub in sorted(base.iterdir()):
        if not sub.is_dir():
            continue
        runs_path = sub / 'runs.parquet'
        if not runs_path.exists():
            continue
        runs = pl.read_parquet(runs_path)
        traces_path = sub / 'traces.parquet'
        if traces_path.exists():
            df = _join_required_traces(runs, traces_path, frozenset(trace_reads))
            build_measurements(
                sub, required=required, runs_df=df,
                measurable_signature_fn=_measurable_signature,
            )
            loaded = load_measurements(sub, columns=list(required))
            present = [c for c in required if c in loaded.columns]
            collide = [c for c in present if c in runs.columns]
            if collide:
                runs = runs.drop(collide)
            df_full = runs.join(
                loaded.select(['id', *present]), on='id', how='left',
            )
        else:
            df_full = runs  # τ=1.0 has no traces; staleness will be null
        tau_v = float(df_full['target_sync.tau'][0])
        df_full = df_full.with_columns(pl.lit(tau_v).alias('tau'))
        cells.append(df_full)

    df = pl.concat(cells, how='diagonal_relaxed')

    # Pair on (env_name, seed, tau)
    pair_keys = ['env_name', 'seed', 'tau']
    v = df.filter(pl.col('arm_key') == 'baseline').select(
        pair_keys + ['eval_best_burst_mean', 'target_staleness_late']
    ).rename({
        'eval_best_burst_mean': 'outcome_v',
        'target_staleness_late': 'staleness_v',
    })
    d = df.filter(pl.col('arm_key') == DDQN).select(
        pair_keys + ['eval_best_burst_mean', 'target_staleness_late']
    ).rename({
        'eval_best_burst_mean': 'outcome_d',
        'target_staleness_late': 'staleness_d',
    })
    paired = v.join(d, on=pair_keys, how='inner').filter(
        pl.col('outcome_v').is_finite() & pl.col('outcome_d').is_finite()
    ).with_columns([
        (pl.col('outcome_d') - pl.col('outcome_v')).alias('delta_outcome'),
        pl.col('tau').log10().alias('log_tau'),
    ])
    return paired


def main() -> None:
    paired = _load_polyak_pairs()
    print(f'paired cells: {paired.height} across '
          f'{paired["env_name"].n_unique()} envs × '
          f'{paired["tau"].n_unique()} τ values', flush=True)

    print()
    print('=== Per-env do(τ) → Δ_outcome ATE with DoWhy refutations ===\n')
    print(f'{"env":<24} {"ATE":>10} {"p":>10} {"placebo_β":>11} '
          f'{"RCC_drift":>10}', flush=True)
    print('-' * 80)

    panel = []
    for env in sorted(paired['env_name'].unique()):
        env_pairs = paired.filter(pl.col('env_name') == env)
        if env_pairs.height < 30:
            print(f'{env:<24} insufficient pairs ({env_pairs.height})', flush=True)
            continue

        # DAG: log_tau is the do-treatment (exogenous), delta_outcome
        # the outcome. No confounders since τ is intervened by the
        # sweep design.
        dag_edges: list[tuple[str, str]] = [
            ('log_tau', 'delta_outcome'),
        ]

        cells_for_dowhy = env_pairs.select(['log_tau', 'delta_outcome']).to_dicts()

        try:
            ate_result = backdoor_ate.fn(
                cells_for_dowhy,
                treatment='log_tau',
                outcome='delta_outcome',
                dag=dag_edges,
                method_name='backdoor.linear_regression',
            )
            placebo_result = placebo_refutation.fn(
                cells_for_dowhy,
                treatment='log_tau',
                outcome='delta_outcome',
                dag=dag_edges,
                method_name='backdoor.linear_regression',
            )
            rcc_result = random_common_cause_refutation.fn(
                cells_for_dowhy,
                treatment='log_tau',
                outcome='delta_outcome',
                dag=dag_edges,
                method_name='backdoor.linear_regression',
            )
        except Exception as e:
            print(f'{env:<24} ERROR: {type(e).__name__}: {e}', flush=True)
            continue

        # BackdoorResult schema: ate, identified, ...
        # RefutationResult schema: real_ate, refuted_ate, drift, ...
        ate = ate_result.ate
        identified = ate_result.identified
        placebo_refuted = placebo_result.refuted_ate
        placebo_drift = placebo_result.drift
        rcc_refuted = rcc_result.refuted_ate
        rcc_drift = rcc_result.drift

        # Compute a per-pair t-statistic on the slope as proxy for p-value.
        # OLS slope's SE = sigma / sqrt(SSx) where sigma = std(residuals).
        x = env_pairs['log_tau'].to_numpy()
        y = env_pairs['delta_outcome'].to_numpy()
        n = len(x)
        x_c = x - x.mean()
        y_pred = ate * x_c + y.mean()
        resid = y - y_pred
        sigma = float(resid.std(ddof=2))
        ss_x = float((x_c ** 2).sum())
        se_ate = sigma / math.sqrt(ss_x) if ss_x > 0 else float('nan')
        t_stat = ate / se_ate if se_ate > 0 else float('nan')
        # Two-sided p (Gaussian approx; df = n-2 large)
        from scipy.stats import t as t_dist
        p = 2 * (1 - t_dist.cdf(abs(t_stat), df=n - 2)) if not math.isnan(t_stat) else float('nan')

        print(
            f'{env:<24} '
            f'{ate:>+10.4f} {p:>10.4g} {placebo_refuted:>+11.4f} {rcc_drift:>+10.4f} '
            f'(id={identified})',
            flush=True,
        )
        panel.append({
            'env': env,
            'n_pairs': env_pairs.height,
            'ate_log_tau_to_delta_outcome': float(ate),
            'ate_se': float(se_ate) if not math.isnan(se_ate) else None,
            'ate_t': float(t_stat) if not math.isnan(t_stat) else None,
            'ate_p_value': float(p) if not math.isnan(p) else None,
            'identified': bool(identified),
            'placebo_refuted_ate': float(placebo_refuted),
            'placebo_drift_from_real': float(placebo_drift),
            'rcc_refuted_ate': float(rcc_refuted),
            'rcc_drift_from_real': float(rcc_drift),
        })

    print()
    print('Reading: ATE = β(log τ → Δ_outcome). Negative ATE means', flush=True)
    print('  DDQN-vs-baseline gain shrinks as τ↑ (less staleness ⇒', flush=True)
    print('  less DDQN benefit). Placebo β should be ~0; RCC drift', flush=True)
    print('  should be small. The combination is the rung-2 evidence.', flush=True)

    out = Path('experiments/findings/sync_curve_breakout/polyak_causal_panel.json')
    out.write_text(json.dumps(panel, indent=2, default=str))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
