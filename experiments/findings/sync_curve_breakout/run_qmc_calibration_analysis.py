"""Q-MC calibration as candidate mediator of sync→outcome harm.

The Q-amplification mediator (Δ_q_b19) measures Q **magnitude** divergence;
Q-MC calibration measures **predictive validity**: Pearson(Q_at_start, MC_return)
across the 100 (burst × eval-episode) points per cell. If DDQN's Q at
sync=10k is poorly calibrated to its own returns, the policy's greedy
selection is essentially uninformed.

Cell-level mediator candidates:
- `pearson_q_mc_overall`: Pearson over 100 points (full trajectory)
- `pearson_q_mc_late`: Pearson over the last 5 bursts × 5 eps = 25 points
- `pearson_q_mc_early`: Pearson over the first 5 bursts × 5 eps = 25 points

Tested with the same per-pair framework (paired_g per sync,
proportion_mediated, stratified partial Spearman, meta-regression).
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import spearmanr

from corroborate.analyses.paired.paired_g import paired_g
from corroborate.corpus.schema import StratumG
from corroborate.graph.discovery import (
    partial_spearman_rho, partial_spearman_rho_multi,
    stratified_partial_spearman_rho,
)
from corroborate.stats import meta_regress_panel

ENV = 'Breakout-MinAtar'
DATA = Path('experiments/data')
TREATMENT_ARM = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASELINE_ARM = 'baseline'


def _pearson_safe(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float('nan')
    r = float(np.corrcoef(x, y)[0, 1])
    return r if math.isfinite(r) else float('nan')


def _per_cell(corpus: str, sync: int, *, traces_paths: list[Path] | None = None) -> pl.DataFrame:
    runs = pl.read_parquet(
        DATA / corpus / 'runs.parquet',
        columns=['id', 'env_name', 'arm_key', 'seed'],
    ).filter(pl.col('env_name') == ENV)
    trace_cols = ['id', 'mc_return', 'predicted_q_at_start']
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
        # Calibration Pearsons
        flat_mc = mc_2d.flatten()
        flat_q = qs_2d.flatten()
        r_overall = _pearson_safe(flat_q, flat_mc)
        r_early = _pearson_safe(qs_2d[:5].flatten(), mc_2d[:5].flatten())
        r_late = _pearson_safe(qs_2d[-5:].flatten(), mc_2d[-5:].flatten())
        # Per-burst-mean Pearson (20 paired (mean_Q, mean_MC) per cell)
        r_burst_mean = _pearson_safe(qs_pb, mc_pb)
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
            'pearson_q_mc_overall': r_overall,
            'pearson_q_mc_early': r_early,
            'pearson_q_mc_late': r_late,
            'pearson_q_mc_burst_mean': r_burst_mean,
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
                        'pearson_q_mc_overall', 'pearson_q_mc_early',
                        'pearson_q_mc_late', 'pearson_q_mc_burst_mean'):
                row[f'{col}_van'] = v[col]
                row[f'{col}_ddqn'] = d[col]
                row[f'd_{col}'] = (d[col] - v[col]) if v[col] is not None and d[col] is not None else float('nan')
                row[f'mean_{col}'] = (v[col] + d[col]) / 2 if v[col] is not None and d[col] is not None else float('nan')
            out_rows.append(row)
    return pl.DataFrame(out_rows)


def main() -> None:
    panel = _build_panel()
    pairs = _paired_per_sync(panel)
    pairs_pd = pairs.to_pandas()
    print(f'panel: {len(panel)} cells; {len(pairs)} pairs')

    # Step 1: Per-arm per-sync mean of each Pearson measure
    print()
    print('=' * 100)
    print('Step 1: Q-MC calibration Pearson, mean per (sync, arm).')
    print(f'  {"sync":>5} | {"arm":>8} | {"r_overall":>10} {"r_early":>10} {"r_late":>10} {"r_burst_mean":>14}')
    for sync in sorted(panel['sync_period'].unique()):
        for arm in (BASELINE_ARM, TREATMENT_ARM):
            sub = panel.filter((pl.col('sync_period') == sync) & (pl.col('arm_key') == arm))
            arm_short = 'baseline' if arm == BASELINE_ARM else 'ddqn'
            print(f'  {sync:>5} | {arm_short:>8} | '
                  f'{sub["pearson_q_mc_overall"].mean():>10.3f} '
                  f'{sub["pearson_q_mc_early"].mean():>10.3f} '
                  f'{sub["pearson_q_mc_late"].mean():>10.3f} '
                  f'{sub["pearson_q_mc_burst_mean"].mean():>14.3f}')

    # Step 2: Per-sync paired_g on calibration measures (does DDQN have worse calibration?)
    print()
    print('=' * 100)
    print('Step 2: paired_g per sync on each calibration measure.')
    print(f'  {"sync":>5} | ' + ' | '.join(f'g({col}):  est ± se   p   ' for col in ('overall','early','late','burst_mean')))
    g_per_measure: dict[int, dict[str, dict[str, float]]] = {}
    for sync in sorted(panel['sync_period'].unique()):
        cells = list(panel.filter(pl.col('sync_period') == sync).iter_rows(named=True))
        line = f'  {sync:>5} |'
        rec: dict[str, dict[str, float]] = {}
        for col in ('pearson_q_mc_overall', 'pearson_q_mc_early',
                    'pearson_q_mc_late', 'pearson_q_mc_burst_mean'):
            r = paired_g.fn(
                cells, source=col,
                treatment_arm=TREATMENT_ARM, baseline_arm=BASELINE_ARM,
                pair_by=('seed',),
            )
            sig = '*' if r.p_value < 0.05 else ' '
            line += f' {r.g:>+6.2f} ± {r.se:>4.2f} p={r.p_value:.2g}{sig} |'
            rec[col] = {
                'g': r.g, 'se': r.se, 'mean_diff': r.mean_diff,
                'p_value': r.p_value, 'n_pairs': r.n_pairs,
            }
        print(line)
        g_per_measure[int(sync)] = rec

    # Step 3: Stratified partial Spearman: does Δ_calibration mediate Δ_outcome?
    print()
    print('=' * 100)
    print('Step 3: Stratified partial Spearman, n=120 pooled.')

    print('  3a. Marginal: ρ(Δ_pearson_q_mc_overall, Δ_mc_late) per stratum, Fisher-z pooled')
    z_vals, ws = [], []
    for sync in sorted(set(pairs_pd['sync_period'])):
        m = pairs_pd['sync_period'] == sync
        if m.sum() < 5:
            continue
        rho_k, _ = spearmanr(
            pairs_pd.loc[m, 'd_pearson_q_mc_overall'].to_numpy(),
            pairs_pd.loc[m, 'd_mc_late'].to_numpy(),
        )
        if not math.isfinite(rho_k):
            continue
        rho_clamp = max(-0.999999, min(0.999999, rho_k))
        z_vals.append(0.5 * math.log((1 + rho_clamp) / (1 - rho_clamp)))
        ws.append(int(m.sum()) - 4)
    if z_vals:
        from scipy.stats import norm as _norm
        z_pool = sum(w * z for w, z in zip(ws, z_vals)) / sum(ws)
        rho_pool = math.tanh(z_pool)
        zstat = z_pool * math.sqrt(sum(ws))
        p_pool = 2 * (1 - float(_norm.cdf(abs(zstat))))
        print(f'    ρ_marginal_pooled = {rho_pool:+.3f}, p = {p_pool:.2e}')

    print()
    print('  3b. ρ(Δ_pearson_q_mc_<window>, Δ_mc_late | Δ_q_b19) — does calibration add over Q-amp?')
    for col in ('d_pearson_q_mc_overall', 'd_pearson_q_mc_early',
                'd_pearson_q_mc_late', 'd_pearson_q_mc_burst_mean'):
        x = pairs_pd[col].to_numpy()
        y = pairs_pd['d_mc_late'].to_numpy()
        z = pairs_pd['d_q_b19'].to_numpy()
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        rho_p, p_p = stratified_partial_spearman_rho(
            x[finite], y[finite], z[finite],
            list(pairs_pd.loc[finite, 'sync_period']),
        )
        print(f'    {col:<32} ρ_partial = {rho_p:+.3f}, p = {p_p:.2e}, n = {int(finite.sum())}')

    print()
    print('  3c. Reverse: ρ(Δ_q_b19, Δ_mc_late | Δ_pearson_q_mc_<window>)')
    for col in ('d_pearson_q_mc_overall', 'd_pearson_q_mc_early',
                'd_pearson_q_mc_late', 'd_pearson_q_mc_burst_mean'):
        x = pairs_pd['d_q_b19'].to_numpy()
        y = pairs_pd['d_mc_late'].to_numpy()
        z = pairs_pd[col].to_numpy()
        finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        rho_p, p_p = stratified_partial_spearman_rho(
            x[finite], y[finite], z[finite],
            list(pairs_pd.loc[finite, 'sync_period']),
        )
        print(f'    z = {col:<28}     ρ_partial = {rho_p:+.3f}, p = {p_p:.2e}, n = {int(finite.sum())}')

    # Step 4: Multi-Z partial — does sync survive controlling for both?
    print()
    print('  3d. Multi-Z: ρ(log_sync, Δ_mc_late | Δ_q_b19, Δ_pearson_q_mc_late)')
    z_mat = pairs_pd[['d_q_b19', 'd_pearson_q_mc_late']].to_numpy()
    finite = np.all(np.isfinite(z_mat), axis=1) & np.isfinite(pairs_pd['d_mc_late']) & np.isfinite(pairs_pd['log_sync'])
    rho_m, p_m = partial_spearman_rho_multi(
        pairs_pd.loc[finite, 'log_sync'].to_numpy(),
        pairs_pd.loc[finite, 'd_mc_late'].to_numpy(),
        z_mat[finite],
    )
    print(f'    ρ_partial = {rho_m:+.3f}, p = {p_m:.2e}, n = {int(finite.sum())}')

    # Step 5: Save
    out = Path('experiments/findings/sync_curve_breakout/qmc_calibration_panel.json')
    out.write_text(json.dumps({
        'g_per_measure_per_sync': g_per_measure,
        'description': 'paired_g(DDQN−vanilla) of Q-MC Pearson per sync; '
                       'partial Spearman tests follow.',
    }, indent=2))
    print()
    print(f'Wrote: {out}')


if __name__ == '__main__':
    main()
