"""Target staleness as the upstream mediator of sync→outcome.

Hypothesis: sync_period controls target_staleness (online−target Q gap),
which controls how much DDQN's bootstrap rule diverges from vanilla's,
which determines the late-Q amplification we already showed mediates
outcome harm.

Two operationalizations of target staleness, computed from
`online_max_q_per_step` and `target_max_q_per_step`:

1. **Absolute value gap** ``|online_max_q − target_max_q|`` — per-step
   raw difference in the max-Q estimate between the two networks.

2. **Relative value gap** ``|gap| / (max(|online|,|target|, eps))`` —
   normalized so it isn't dominated by absolute Q magnitude
   (which itself blows up with sync via Q-explosion at low sync).

Both binned per burst (50k steps), then tested:

- **Cross-sync correlation** ρ(log_sync, target_staleness) — must be
  monotone if staleness is the upstream variable.
- **Within-seed mediation**: does target_staleness predict
  Δ_q_b19 (the within-seed late-Q amplification mediator) within
  each sync stratum?
- **Meta-regression**: add target_staleness covariate alongside
  log_sync and q_b0_ratio. Does it reduce the q_b0_ratio coefficient
  (suggesting staleness is upstream)?
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from pathlib import Path
from typing import cast

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from corroborate.analyses.paired.paired_g import paired_g
from corroborate.corpus.schema import StratumG
from corroborate.graph.discovery import (
    partial_spearman_rho,
    partial_spearman_rho_multi,
    stratified_partial_spearman_rho,
)
from corroborate.stats import meta_regress_panel

ENV = 'Breakout-MinAtar'
DATA = Path('experiments/data')
TREATMENT_ARM = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASELINE_ARM = 'baseline'
N_BURSTS, STEPS_PER_BURST = 20, 50_000


def _per_cell(corpus: str, sync: int, *, traces_paths: list[Path] | None = None) -> pl.DataFrame:
    runs = pl.read_parquet(
        DATA / corpus / 'runs.parquet',
        columns=['id', 'env_name', 'arm_key', 'seed'],
    ).filter(pl.col('env_name') == ENV)
    trace_cols = [
        'id', 'mc_return', 'predicted_q_at_start',
        'online_max_q_per_step', 'target_max_q_per_step',
    ]
    if traces_paths is None:
        traces_paths = [DATA / corpus / 'traces.parquet']
    trace_dfs = [pl.read_parquet(p, columns=trace_cols) for p in traces_paths]
    traces = pl.concat(trace_dfs, how='vertical_relaxed') if len(trace_dfs) > 1 else trace_dfs[0]
    bk = runs.join(traces, on='id', how='inner')
    rows: list[dict[str, object]] = []
    for r in bk.iter_rows(named=True):
        mc_2d = np.asarray(r['mc_return'], dtype=np.float64)  # (20, 5)
        qs_2d = np.asarray(r['predicted_q_at_start'], dtype=np.float64)
        if mc_2d.ndim != 2 or qs_2d.ndim != 2:
            continue
        mc_pb = mc_2d.mean(axis=1)
        qs_pb = qs_2d.mean(axis=1)
        omax = np.asarray(r['online_max_q_per_step'], dtype=np.float64)[: N_BURSTS * STEPS_PER_BURST]
        tmax = np.asarray(r['target_max_q_per_step'], dtype=np.float64)[: N_BURSTS * STEPS_PER_BURST]
        omax_b = omax.reshape(N_BURSTS, STEPS_PER_BURST)
        tmax_b = tmax.reshape(N_BURSTS, STEPS_PER_BURST)
        abs_gap = np.abs(omax_b - tmax_b)  # (20, 50000)
        denom = np.maximum(np.maximum(np.abs(omax_b), np.abs(tmax_b)), 1e-6)
        rel_gap = abs_gap / denom
        # Per-burst means
        abs_gap_pb = abs_gap.mean(axis=1)
        rel_gap_pb = rel_gap.mean(axis=1)
        rows.append({
            'sync_period': sync,
            'log_sync': math.log(sync),
            'arm_key': r['arm_key'],
            'seed': int(r['seed']),
            'env_name': r['env_name'],
            'mc_b0': float(mc_pb[0]),
            'mc_b19': float(mc_pb[19]),
            'mc_late': float(mc_pb[-5:].mean()),
            'q_b0': float(qs_pb[0]),
            'q_b19': float(qs_pb[19]),
            'staleness_abs_b0': float(abs_gap_pb[0]),
            'staleness_abs_b8': float(abs_gap_pb[8]),
            'staleness_abs_b19': float(abs_gap_pb[19]),
            'staleness_abs_late': float(abs_gap_pb[-5:].mean()),
            'staleness_rel_b0': float(rel_gap_pb[0]),
            'staleness_rel_b8': float(rel_gap_pb[8]),
            'staleness_rel_b19': float(rel_gap_pb[19]),
            'staleness_rel_late': float(rel_gap_pb[-5:].mean()),
            'staleness_abs_mean': float(abs_gap.mean()),
            'staleness_rel_mean': float(rel_gap.mean()),
        })
    return pl.DataFrame(rows)


def _build_panel() -> pl.DataFrame:
    sync100_shards = [
        DATA / 'minatar_1M/tmp/arm002__Breakout-MinAtar__vanilla_dqn__traces.parquet',
        DATA / 'minatar_1M/tmp/arm003__Breakout-MinAtar__ddqn__traces.parquet',
    ]
    parts = [
        _per_cell('minatar_1M', 100, traces_paths=sync100_shards),
        _per_cell('minatar_sync_curve/ddqn_sync1k', 1000),
        _per_cell('minatar_sync_curve/ddqn_sync3k', 3000),
        _per_cell('minatar_sync_intervention', 10000),
    ]
    panel = pl.concat(parts, how='vertical_relaxed')
    panel = panel.unique(subset=['sync_period', 'arm_key', 'seed'], keep='first')
    return panel


def _paired_per_sync(panel: pl.DataFrame) -> pl.DataFrame:
    out_rows: list[dict[str, object]] = []
    for sync in sorted(panel['sync_period'].unique()):
        sub = panel.filter(pl.col('sync_period') == sync)
        van = sub.filter(pl.col('arm_key') == BASELINE_ARM)
        ddqn = sub.filter(pl.col('arm_key') == TREATMENT_ARM)
        seeds = sorted(set(van['seed']) & set(ddqn['seed']))
        van_by = {row['seed']: row for row in van.iter_rows(named=True)}
        ddqn_by = {row['seed']: row for row in ddqn.iter_rows(named=True)}
        for seed in seeds:
            v = van_by[seed]
            d = ddqn_by[seed]
            row: dict[str, object] = {
                'sync_period': sync, 'log_sync': v['log_sync'], 'seed': seed,
            }
            for col in ('mc_b0', 'mc_b19', 'mc_late', 'q_b0', 'q_b19',
                        'staleness_abs_b0', 'staleness_abs_b8',
                        'staleness_abs_b19', 'staleness_abs_late',
                        'staleness_rel_b0', 'staleness_rel_b8',
                        'staleness_rel_b19', 'staleness_rel_late',
                        'staleness_abs_mean', 'staleness_rel_mean'):
                row[f'{col}_van'] = v[col]
                row[f'{col}_ddqn'] = d[col]
                row[f'd_{col}'] = d[col] - v[col]
                row[f'mean_{col}'] = (v[col] + d[col]) / 2  # arm-mean for shared variables
            out_rows.append(row)
    return pl.DataFrame(out_rows)


def main() -> None:
    panel = _build_panel()
    pairs = _paired_per_sync(panel)
    pairs_pd = pairs.to_pandas()
    print(f'panel: {len(panel)} cells; {len(pairs)} pairs across {pairs["sync_period"].n_unique()} syncs')

    # Step 1: Sanity — staleness scales with sync and is ~symmetric across arms
    print()
    print('=' * 100)
    print('Step 1: Staleness scales with sync (per-arm mean per sync)?')
    print(f'  {"sync":>5} | {"arm":>8} | {"abs_b0":>8} {"abs_b19":>8} {"abs_late":>9} | {"rel_b0":>7} {"rel_b19":>7} {"rel_late":>8}')
    for sync in sorted(panel['sync_period'].unique()):
        for arm in (BASELINE_ARM, TREATMENT_ARM):
            sub = panel.filter((pl.col('sync_period') == sync) & (pl.col('arm_key') == arm))
            arm_short = 'baseline' if arm == BASELINE_ARM else 'ddqn'
            print(f'  {sync:>5} | {arm_short:>8} | '
                  f'{sub["staleness_abs_b0"].mean():>8.4f} {sub["staleness_abs_b19"].mean():>8.2f} {sub["staleness_abs_late"].mean():>9.2f} | '
                  f'{sub["staleness_rel_b0"].mean():>7.3f} {sub["staleness_rel_b19"].mean():>7.3f} {sub["staleness_rel_late"].mean():>8.3f}')

    # Step 2: Cross-cell correlation log_sync vs staleness (using all 240 cells)
    print()
    print('=' * 100)
    print('Step 2: Cross-cell ρ(log_sync, staleness measure) — pooled across arms.')
    panel_pd = panel.to_pandas()
    for col in ('staleness_abs_b0', 'staleness_abs_b19', 'staleness_abs_late',
                'staleness_rel_b0', 'staleness_rel_b19', 'staleness_rel_late',
                'staleness_abs_mean', 'staleness_rel_mean'):
        rho, p = spearmanr(panel_pd['log_sync'], panel_pd[col])
        print(f'  ρ(log_sync, {col:<20}) = {rho:>+.3f}, p = {p:.2e}')

    # Step 3: Within-seed correlation of mean_staleness vs Δ_q_b19 and Δ_outcome
    print()
    print('=' * 100)
    print('Step 3: stratified partial Spearman, strata=sync, n=120 pooled.')
    print('  3a. Marginal within-stratum: ρ(mean_staleness_late, Δ_mc_late | <none>)')
    rho_strat, p_strat = stratified_partial_spearman_rho(
        pairs_pd['mean_staleness_rel_late'].to_numpy(),
        pairs_pd['d_mc_late'].to_numpy(),
        # Need a Z; use a constant — falls back to NaN with the closed-form;
        # so do per-stratum marginal Spearman manually.
        np.zeros(len(pairs_pd)),
        list(pairs_pd['sync_period']),
    )
    # That's degenerate — instead do per-stratum marginal Spearman pooled:
    z_vals_marg, ws = [], []
    for sync in sorted(set(pairs_pd['sync_period'])):
        m = pairs_pd['sync_period'] == sync
        if m.sum() < 5:
            continue
        rho_k, _ = spearmanr(
            pairs_pd.loc[m, 'mean_staleness_rel_late'].to_numpy(),
            pairs_pd.loc[m, 'd_mc_late'].to_numpy(),
        )
        if not math.isfinite(rho_k):
            continue
        rho_clamp = max(-0.999999, min(0.999999, rho_k))
        z_vals_marg.append(0.5 * math.log((1 + rho_clamp) / (1 - rho_clamp)))
        ws.append(int(m.sum()) - 4)
    if z_vals_marg:
        from scipy.stats import norm as _norm
        z_pool = sum(w * z for w, z in zip(ws, z_vals_marg)) / sum(ws)
        rho_pool = math.tanh(z_pool)
        zstat = z_pool * math.sqrt(sum(ws))
        p_pool = 2 * (1 - float(_norm.cdf(abs(zstat))))
        print(f'  marginal pooled ρ(mean_staleness_rel_late, Δ_mc_late) = {rho_pool:+.3f}, p = {p_pool:.2e}')

    print()
    print('  3b. Pooled partial: ρ(mean_staleness_rel_late, Δ_mc_late | Δ_q_b19)')
    rho_p, p_p = stratified_partial_spearman_rho(
        pairs_pd['mean_staleness_rel_late'].to_numpy(),
        pairs_pd['d_mc_late'].to_numpy(),
        pairs_pd['d_q_b19'].to_numpy(),
        list(pairs_pd['sync_period']),
    )
    print(f'    ρ_pooled = {rho_p:+.3f}, p = {p_p:.2e}')

    print()
    print('  3c. Reverse: ρ(Δ_q_b19, Δ_mc_late | mean_staleness_rel_late)')
    rho_r, p_r = stratified_partial_spearman_rho(
        pairs_pd['d_q_b19'].to_numpy(),
        pairs_pd['d_mc_late'].to_numpy(),
        pairs_pd['mean_staleness_rel_late'].to_numpy(),
        list(pairs_pd['sync_period']),
    )
    print(f'    ρ_pooled = {rho_r:+.3f}, p = {p_r:.2e}')
    print('    (For comparison from previous run: ρ(Δ_q_b19, Δ_mc_late | Δ_q_b0) = -0.343, p=2.7e-4)')

    print()
    print('  3d. Pooled partial: ρ(log_sync, Δ_mc_late | mean_staleness_rel_late)')
    rho_a, p_a = partial_spearman_rho(
        pairs_pd['log_sync'].to_numpy(),
        pairs_pd['d_mc_late'].to_numpy(),
        pairs_pd['mean_staleness_rel_late'].to_numpy(),
    )
    print(f'    ρ_partial = {rho_a:+.3f}, p = {p_a:.2e}')
    print('    (Marginal ρ(log_sync, Δ_mc_late) was -0.128 ns; if staleness mediates, this should attenuate.)')

    # Step 4: Multi-Z partial: control for both Δ_q_b19 AND staleness
    print()
    print('  3e. Multi-Z: ρ(log_sync, Δ_mc_late | Δ_q_b19, mean_staleness_rel_late)')
    z_mat = pairs_pd[['d_q_b19', 'mean_staleness_rel_late']].to_numpy()
    rho_m, p_m = partial_spearman_rho_multi(
        pairs_pd['log_sync'].to_numpy(),
        pairs_pd['d_mc_late'].to_numpy(),
        z_mat,
    )
    print(f'    ρ_partial = {rho_m:+.3f}, p = {p_m:.2e}')

    # Step 5: Meta-regression with staleness as covariate
    print()
    print('=' * 100)
    print('Step 5: Meta-regression on per-sync paired g, with staleness covariate.')
    g_panel: dict[int, dict[str, float]] = {}
    sync_covs: dict[int, dict[str, float]] = {}
    for sync in sorted(panel['sync_period'].unique()):
        cells = list(panel.filter(pl.col('sync_period') == sync).iter_rows(named=True))
        r = paired_g.fn(
            cells, source='mc_late',
            treatment_arm=TREATMENT_ARM, baseline_arm=BASELINE_ARM,
            pair_by=('seed',),
        )
        g_panel[int(sync)] = {
            'g': r.g, 'se': r.se, 'n_pairs': r.n_pairs,
            'p_value': r.p_value,
        }
        sub = panel.filter(pl.col('sync_period') == sync)
        van = sub.filter(pl.col('arm_key') == BASELINE_ARM)
        ddqn = sub.filter(pl.col('arm_key') == TREATMENT_ARM)
        sync_covs[int(sync)] = {
            'log_sync': math.log(sync),
            'mean_staleness_rel_b0': float(sub['staleness_rel_b0'].mean()),
            'mean_staleness_rel_late': float(sub['staleness_rel_late'].mean()),
            'mean_staleness_rel_mean': float(sub['staleness_rel_mean'].mean()),
            'mean_q_b0_ratio': float(
                ddqn['q_b0'].mean() / max(van['q_b0'].mean(), 1e-12)
            ),
        }

    panel_points = [
        StratumG(stratum_id=str(s), g=r['g'], se=r['se'], n_pairs=r['n_pairs'])
        for s, r in g_panel.items()
    ]

    print(f'  {"sync":>5} | g, se      | ' + ' | '.join(
        f'{c:<24}' for c in ('log_sync', 'staleness_rel_b0', 'staleness_rel_late', 'staleness_rel_mean', 'q_b0_ratio')
    ))
    for sync in sorted(g_panel.keys()):
        cov = sync_covs[sync]
        print(f'  {sync:>5} | {g_panel[sync]["g"]:>+.3f} {g_panel[sync]["se"]:>.3f} | '
              + ' | '.join(f'{cov[c]:>24.4f}' for c in ('log_sync', 'mean_staleness_rel_b0', 'mean_staleness_rel_late', 'mean_staleness_rel_mean', 'mean_q_b0_ratio')))

    cov_specs = [
        ('staleness_rel_b0', {sync: {'mean_staleness_rel_b0': c['mean_staleness_rel_b0']} for sync, c in sync_covs.items()}),
        ('staleness_rel_late', {sync: {'mean_staleness_rel_late': c['mean_staleness_rel_late']} for sync, c in sync_covs.items()}),
        ('staleness_rel_mean', {sync: {'mean_staleness_rel_mean': c['mean_staleness_rel_mean']} for sync, c in sync_covs.items()}),
        ('q_b0_ratio + staleness_late',
         {sync: {'mean_q_b0_ratio': c['mean_q_b0_ratio'], 'mean_staleness_rel_late': c['mean_staleness_rel_late']}
          for sync, c in sync_covs.items()}),
    ]
    print()
    print('  Meta-regression coefficients:')
    meta_results: dict[str, dict[str, dict[str, float]]] = {}
    for name, cov_per_stratum in cov_specs:
        cov_str = {str(sync): cov for sync, cov in cov_per_stratum.items()}
        try:
            r = meta_regress_panel(panel_points, covariates_per_stratum=cov_str, pool='random')
        except Exception as e:
            print(f'    {name:<32} ERROR: {e}')
            continue
        rec: dict[str, dict[str, float]] = {}
        for coef in r.coefficients:
            print(f'    {name:<32} {coef.name:<28} = {coef.coefficient:>+8.3f} '
                  f'(95% CI [{coef.ci_lo:>+7.3f}, {coef.ci_hi:>+7.3f}]) p = {coef.p_value:.3e}')
            rec[coef.name] = {
                'coefficient': coef.coefficient,
                'ci_lo': coef.ci_lo, 'ci_hi': coef.ci_hi,
                'p_value': coef.p_value,
            }
        meta_results[name] = rec

    out = Path('experiments/findings/sync_curve_breakout/staleness_panel.json')
    out.write_text(json.dumps({
        'g_panel': g_panel,
        'sync_covariates': {str(k): v for k, v in sync_covs.items()},
        'meta_regression': meta_results,
        'partial_spearman': {
            'staleness_marg_pooled': {'rho': rho_pool, 'p': p_pool},
            'staleness_given_qb19': {'rho': rho_p, 'p': p_p},
            'qb19_given_staleness': {'rho': rho_r, 'p': p_r},
            'log_sync_given_staleness': {'rho': rho_a, 'p': p_a},
            'log_sync_given_qb19_and_staleness': {'rho': rho_m, 'p': p_m},
        },
    }, indent=2))
    print()
    print(f'Wrote: {out}')


if __name__ == '__main__':
    main()
