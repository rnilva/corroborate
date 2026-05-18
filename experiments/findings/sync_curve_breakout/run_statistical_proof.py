"""Statistical proof of bootstrap-conservatism mediation on Breakout.

Tests the chain `sync → DDQN_early_Q_suppression → DDQN_late_outcome`
against the rival `sync → DDQN_late_Q_amplification → DDQN_late_outcome`
with framework primitives:

1. **Per-sync paired_g** on the outcome (sanity / power baseline)
2. **proportion_mediated** per sync × candidate mediator (linear Sobel-Baron)
3. **stratified_partial_spearman_rho (JCI form)** ρ(log_sync, Δ_outcome | Δ_q_b0)
   pooled across sync strata — does sync still affect outcome after the
   bootstrap-conservatism mediator is conditioned on?
4. **meta_regress_panel** g(sync) ~ log_sync + mean(q_b0_ratio) + mean(argmax_disagree_b0)
5. **Power analysis** — minimum detectable g per stratum at n=30, MDE for the
   meta-regression slope.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import norm

from corroborate.analyses.paired.paired_g import paired_g
from corroborate.analyses.paired.proportion_mediated import proportion_mediated
from corroborate.graph.discovery import (
    partial_spearman_rho,
    stratified_partial_spearman_rho,
)
from corroborate.corpus.schema import StratumG
from corroborate.stats import meta_regress_panel

ENV = 'Breakout-MinAtar'
DATA = Path('experiments/data')
TREATMENT_ARM = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
BASELINE_ARM = 'baseline'


def _per_burst_arrays(corpus: str, sync: int, *, traces_paths: list[Path] | None = None) -> pl.DataFrame:
    """Per-cell scalars (id, arm_key, seed, q_b0, q_b8, q_b19, mc_b0, mc_b19,
    mc_late, argmax_disagree_b0, argmax_disagree_late) on Breakout-MinAtar."""
    runs = pl.read_parquet(
        DATA / corpus / 'runs.parquet',
        columns=['id', 'env_name', 'arm_key', 'seed'],
    ).filter(pl.col('env_name') == ENV)
    trace_cols = [
        'id', 'mc_return', 'predicted_q_at_start',
        'online_max_q_per_step',
        'online_argmax_per_step', 'target_argmax_per_step',
    ]
    if traces_paths is None:
        traces_paths = [DATA / corpus / 'traces.parquet']
    trace_dfs = [pl.read_parquet(p, columns=trace_cols) for p in traces_paths]
    traces = pl.concat(trace_dfs, how='vertical_relaxed') if len(trace_dfs) > 1 else trace_dfs[0]
    bk = runs.join(traces, on='id', how='inner')

    rows: list[dict[str, object]] = []
    for r in bk.iter_rows(named=True):
        mc = np.asarray(r['mc_return'], dtype=np.float64)  # (20, 5)
        qs = np.asarray(r['predicted_q_at_start'], dtype=np.float64)  # (20, 5)
        if mc.ndim != 2 or qs.ndim != 2:
            continue
        mc_pb = mc.mean(axis=1)
        qs_pb = qs.mean(axis=1)
        online_max = np.asarray(r['online_max_q_per_step'], dtype=np.float64)
        online_amax = np.asarray(r['online_argmax_per_step'], dtype=np.int64)
        target_amax = np.asarray(r['target_argmax_per_step'], dtype=np.int64)
        # Bin step diagnostics into 20 bursts of 50k steps
        n_bursts, sps = 20, 50_000
        ol = online_max[: n_bursts * sps].reshape(n_bursts, sps)
        oa = online_amax[: n_bursts * sps].reshape(n_bursts, sps)
        ta = target_amax[: n_bursts * sps].reshape(n_bursts, sps)
        online_max_pb = ol.mean(axis=1)
        disagree_pb = (oa != ta).mean(axis=1)
        rows.append({
            'sync_period': sync,
            'log_sync': math.log(sync),
            'arm_key': r['arm_key'],
            'seed': int(r['seed']),
            'env_name': r['env_name'],
            'q_b0': float(qs_pb[0]),
            'q_b8': float(qs_pb[8]),
            'q_b19': float(qs_pb[19]),
            'q_late': float(qs_pb[-5:].mean()),
            'mc_b0': float(mc_pb[0]),
            'mc_b8': float(mc_pb[8]),
            'mc_b19': float(mc_pb[19]),
            'mc_late': float(mc_pb[-5:].mean()),
            'mc_mean': float(mc_pb.mean()),
            'online_max_q_b0': float(online_max_pb[0]),
            'online_max_q_b19': float(online_max_pb[19]),
            'argmax_disagree_b0': float(disagree_pb[0]),
            'argmax_disagree_late': float(disagree_pb[-5:].mean()),
        })
    return pl.DataFrame(rows)


def _build_panel() -> pl.DataFrame:
    sync100_shards = [
        DATA / 'minatar_1M/tmp/arm002__Breakout-MinAtar__vanilla_dqn__traces.parquet',
        DATA / 'minatar_1M/tmp/arm003__Breakout-MinAtar__ddqn__traces.parquet',
    ]
    parts = [
        _per_burst_arrays('minatar_1M', 100, traces_paths=sync100_shards),
        _per_burst_arrays('minatar_sync_curve/ddqn_sync1k', 1000),
        _per_burst_arrays('minatar_sync_curve/ddqn_sync3k', 3000),
        _per_burst_arrays('minatar_sync_intervention', 10000),
    ]
    panel = pl.concat(parts, how='vertical_relaxed')
    # Dedupe seeds — some sweeps recorded each seed twice (degenerate
    # vanilla_sync* hypothesis collision); take first occurrence per
    # (sync, arm, seed) so paired_g pairing is clean.
    panel = panel.unique(subset=['sync_period', 'arm_key', 'seed'], keep='first')
    return panel


def _paired_per_sync(panel: pl.DataFrame) -> pl.DataFrame:
    """Per (sync, seed) pair: vanilla and ddqn columns side-by-side
    with Δ_* derived columns. Drops seeds present in only one arm."""
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
            for col in ('q_b0', 'q_b8', 'q_b19', 'q_late',
                        'mc_b0', 'mc_b8', 'mc_b19', 'mc_late', 'mc_mean',
                        'online_max_q_b0', 'online_max_q_b19',
                        'argmax_disagree_b0', 'argmax_disagree_late'):
                row[f'{col}_van'] = v[col]
                row[f'{col}_ddqn'] = d[col]
                row[f'd_{col}'] = d[col] - v[col]
            out_rows.append(row)
    return pl.DataFrame(out_rows)


def _per_sync_paired_g_panel(
    panel: pl.DataFrame, target: str,
) -> dict[int, dict[str, float]]:
    """Per-sync paired g + SE on `target`. Used as the meta-regression
    panel and as the headline outcome table."""
    out: dict[int, dict[str, float]] = {}
    for sync in sorted(panel['sync_period'].unique()):
        cells = [
            r for r in panel.filter(pl.col('sync_period') == sync).iter_rows(named=True)
        ]
        result = paired_g.fn(
            cells, source=target,
            treatment_arm=TREATMENT_ARM, baseline_arm=BASELINE_ARM,
            pair_by=('seed',),
        )
        out[int(sync)] = {
            'g': result.g, 'se': result.se,
            'mean_diff': result.mean_diff,
            'mean_diff_se': result.mean_diff_se,
            'n_pairs': result.n_pairs,
            'p_value': result.p_value,
            'mean_diff_p_value': result.mean_diff_p_value,
        }
    return out


def _proportion_mediated_per_sync(
    panel: pl.DataFrame, target: str, mediator: str,
) -> dict[int, dict[str, float | bool]]:
    out: dict[int, dict[str, float | bool]] = {}
    for sync in sorted(panel['sync_period'].unique()):
        cells = [
            r for r in panel.filter(pl.col('sync_period') == sync).iter_rows(named=True)
        ]
        result = proportion_mediated.fn(
            cells, target=target, mediator=mediator,
            treatment_arm=TREATMENT_ARM, baseline_arm=BASELINE_ARM,
            pair_by=('seed',),
        )
        out[int(sync)] = {
            'proportion': result.proportion,
            'total': result.total,
            'direct': result.direct,
            'indirect': result.indirect,
            'slope_y_on_m': result.slope_y_on_m,
            'in_unit_interval': result.in_unit_interval,
            'n_pairs': result.n_pairs,
        }
    return out


def _power_paired_g(n_pairs: int, alpha: float = 0.05) -> dict[str, float]:
    """Minimum-detectable Cohen's d for paired g at n_pairs (two-sided
    z-test approximation, normal SE = sqrt(2/n + g²/(2n)) ≈ sqrt(2/n) for
    moderate g). Returns the d at power=0.5/0.8/0.95."""
    z_a = norm.ppf(1 - alpha / 2)
    se_approx = math.sqrt(2.0 / n_pairs)  # ignores second-order g²/(2n)
    out: dict[str, float] = {}
    for power in (0.5, 0.8, 0.95):
        z_b = norm.ppf(power)
        out[f'mde_g_at_{int(power*100):02d}'] = float((z_a + z_b) * se_approx)
    out['se_g_approx'] = float(se_approx)
    return out


def main() -> None:
    panel = _build_panel()
    pairs = _paired_per_sync(panel)
    print(f'panel: {len(panel)} cells, {len(pairs)} pairs across {pairs["sync_period"].n_unique()} sync values')

    # Step 1: Per-sync paired g on outcome (sanity, headline, power baseline)
    print()
    print('=' * 100)
    print('Step 1: Per-sync paired_g on outcome (mc_late = mean of last 5 bursts).')
    g_panel = _per_sync_paired_g_panel(panel, 'mc_late')
    print(f'  {"sync":>5} | {"g":>7} ± {"se":>5} | {"Δ_mc_late":>10} ± {"se_Δ":>6} | {"p":>9} | {"n":>3}')
    for sync, r in g_panel.items():
        sig = '*' if r['p_value'] < 0.05 else ' '
        print(f'  {sync:>5} | {r["g"]:>+7.3f} ± {r["se"]:>5.3f} | {r["mean_diff"]:>+10.4f} ± {r["mean_diff_se"]:>6.4f} | {r["p_value"]:>9.2e}{sig} | {r["n_pairs"]:>3}')

    pwr = _power_paired_g(30)
    print()
    print(f'  Power (paired-g, n_pairs=30, two-sided α=0.05):')
    print(f'    SE(g) ≈ {pwr["se_g_approx"]:.3f}; '
          f'MDE@power=0.50: g={pwr["mde_g_at_50"]:.2f}, '
          f'MDE@power=0.80: g={pwr["mde_g_at_80"]:.2f}, '
          f'MDE@power=0.95: g={pwr["mde_g_at_95"]:.2f}')

    # Step 2: Per-sync proportion_mediated for candidate mediators
    print()
    print('=' * 100)
    print('Step 2: proportion_mediated, per-sync, target=mc_late, four candidate mediators.')
    mediators = ['q_b0', 'mc_b0', 'online_max_q_b0', 'q_b19']
    mediator_panels = {
        m: _proportion_mediated_per_sync(panel, 'mc_late', m) for m in mediators
    }
    print(f'  {"sync":>5} | ' + ' | '.join(f'{m+"  proportion (indir/total)":>34}' for m in mediators))
    for sync in sorted(g_panel.keys()):
        line = f'  {sync:>5} |'
        for m in mediators:
            r = mediator_panels[m][sync]
            valid = r['in_unit_interval']
            tag = '✓' if valid else '✗'
            line += f' {r["proportion"]:>+8.3f} (Δ={r["indirect"]:>+7.4f}/{r["total"]:>+7.4f}) {tag} |'
        print(line)
    print()
    print('  ✓ = proportion ∈ [0,1] (linear mediation assumptions hold);')
    print('  ✗ = outside unit interval → linear mediation breaks (escalate to counterfactual).')

    # Step 3: Stratified partial Spearman (JCI form)
    print()
    print('=' * 100)
    print('Step 3: stratified_partial_spearman_rho (JCI form), strata=sync_period.')
    print('  Tests whether sync still predicts Δ_outcome AFTER conditioning on a mediator.')
    print('  Both X and Y are at the per-pair level; stratification removes pure log_sync')
    print("  effect; remaining variation tests whether 'within-sync' the mediator chain holds.")
    pairs_pd = pairs.to_pandas()
    print()
    print('  3a. Marginal: ρ(log_sync, Δ_mc_late)')
    rho, p = partial_spearman_rho(
        pairs_pd['log_sync'].to_numpy(),
        pairs_pd['d_mc_late'].to_numpy(),
        # Use a constant z (length n) → degenerate, falls back to marginal Spearman
        np.ones(len(pairs_pd)),
    )
    # The above will return NaN (z constant); use marginal directly.
    from scipy.stats import spearmanr
    rho_marg, p_marg = spearmanr(pairs_pd['log_sync'].to_numpy(), pairs_pd['d_mc_late'].to_numpy())
    print(f'    ρ_marginal = {rho_marg:+.3f}, p = {p_marg:.2e}, n = {len(pairs_pd)}')

    print()
    print('  3b. Partial: ρ(log_sync, Δ_mc_late | candidate mediator)')
    cand_mediators = ['d_q_b0', 'd_mc_b0', 'd_online_max_q_b0', 'd_q_b19']
    for m in cand_mediators:
        rho_p, p_p = partial_spearman_rho(
            pairs_pd['log_sync'].to_numpy(),
            pairs_pd['d_mc_late'].to_numpy(),
            pairs_pd[m].to_numpy(),
        )
        attenuation = 1.0 - abs(rho_p) / abs(rho_marg) if rho_marg != 0 else float('nan')
        print(f'    | {m:>20}: ρ_partial = {rho_p:+.3f}, p = {p_p:.2e}, '
              f'attenuation = {attenuation:+.1%}')

    print()
    print('  3c. Stratified-pooled-partial: per-sync ρ(Δ_q_b0, Δ_mc_late | Δ_q_b19),')
    print('      Fisher-z pooled across syncs (within-sync dose-response after late-Q control).')
    rho_strat, p_strat = stratified_partial_spearman_rho(
        pairs_pd['d_q_b0'].to_numpy(),
        pairs_pd['d_mc_late'].to_numpy(),
        pairs_pd['d_q_b19'].to_numpy(),
        list(pairs_pd['sync_period']),
    )
    print(f'    ρ_pooled = {rho_strat:+.3f}, p = {p_strat:.2e}')
    print('    Same with mediators reversed: ρ(Δ_q_b19, Δ_mc_late | Δ_q_b0)')
    rho_strat2, p_strat2 = stratified_partial_spearman_rho(
        pairs_pd['d_q_b19'].to_numpy(),
        pairs_pd['d_mc_late'].to_numpy(),
        pairs_pd['d_q_b0'].to_numpy(),
        list(pairs_pd['sync_period']),
    )
    print(f'    ρ_pooled = {rho_strat2:+.3f}, p = {p_strat2:.2e}')

    # Step 4: Meta-regression on per-sync panel
    print()
    print('=' * 100)
    print('Step 4: Meta-regression of g(sync) on stratum-level covariates.')
    print('  Random-effects pooling; treats each sync as a stratum (n_strata=4).')
    panel_points: list[StratumG[str]] = []
    sync_covariates: dict[int, dict[str, float]] = {}
    for sync, r in g_panel.items():
        sub = panel.filter(pl.col('sync_period') == sync)
        van = sub.filter(pl.col('arm_key') == BASELINE_ARM)
        ddqn = sub.filter(pl.col('arm_key') == TREATMENT_ARM)
        # Stratum-level covariates
        sync_covariates[sync] = {
            'log_sync': math.log(sync),
            'mean_argmax_disagree_b0': float(
                pl.concat([van, ddqn])['argmax_disagree_b0'].mean()
            ),
            'mean_q_b0_ratio': float(
                ddqn['online_max_q_b0'].mean() / max(van['online_max_q_b0'].mean(), 1e-12)
            ),
        }
        panel_points.append(StratumG(
            stratum_id=str(sync), g=r['g'], se=r['se'],
            n_pairs=r['n_pairs'],
        ))

    print('  Stratum-level covariates:')
    print(f'  {"sync":>5} | {"g":>7} {"se":>5} | {"log_sync":>9} {"argmax_disagree_b0":>20} {"q_b0_ratio":>12}')
    for sync in sorted(g_panel.keys()):
        cov = sync_covariates[sync]
        print(f'  {sync:>5} | {g_panel[sync]["g"]:>+7.3f} {g_panel[sync]["se"]:>5.3f} | '
              f'{cov["log_sync"]:>9.3f} {cov["mean_argmax_disagree_b0"]:>20.3f} {cov["mean_q_b0_ratio"]:>12.3f}')

    # Run meta-regression with each candidate covariate (and intercept-only)
    cov_specs = [
        ('intercept_only', {}),
        ('log_sync', {sync: {'log_sync': c['log_sync']} for sync, c in sync_covariates.items()}),
        ('mean_argmax_disagree_b0', {sync: {'mean_argmax_disagree_b0': c['mean_argmax_disagree_b0']} for sync, c in sync_covariates.items()}),
        ('mean_q_b0_ratio', {sync: {'mean_q_b0_ratio': c['mean_q_b0_ratio']} for sync, c in sync_covariates.items()}),
        ('log_sync + q_b0_ratio',
         {sync: {'log_sync': c['log_sync'], 'mean_q_b0_ratio': c['mean_q_b0_ratio']} for sync, c in sync_covariates.items()}),
    ]
    print()
    print('  Meta-regression results (slope ± SE, p):')
    meta_results: dict[str, dict[str, dict[str, float]]] = {}
    for name, cov_per_stratum in cov_specs:
        # Build a dict from stratum_str → {coef: value}
        cov_str_keyed: dict[str, dict[str, float]] = {
            str(sync): cov for sync, cov in cov_per_stratum.items()
        } if cov_per_stratum else {}
        try:
            r = meta_regress_panel(
                panel_points,
                covariates_per_stratum=cov_str_keyed,
                pool='random',
            )
        except Exception as e:
            print(f'    {name:<32} ERROR: {e}')
            continue
        rec: dict[str, dict[str, float]] = {}
        for coef in r.coefficients:
            se_implied = (coef.ci_hi - coef.ci_lo) / (2 * 1.96)
            print(f'    {name:<32} {coef.name:<25} = {coef.coefficient:>+8.3f} '
                  f'(95% CI [{coef.ci_lo:>+7.3f}, {coef.ci_hi:>+7.3f}]) p = {coef.p_value:.3e}')
            rec[coef.name] = {
                'coefficient': coef.coefficient, 'se_implied': se_implied,
                'p_value': coef.p_value,
                'ci_lo': coef.ci_lo, 'ci_hi': coef.ci_hi,
            }
        meta_results[name] = rec

    # Step 5: Save everything
    out = Path('experiments/findings/sync_curve_breakout/statistical_proof.json')
    out.write_text(json.dumps({
        'per_sync_paired_g': g_panel,
        'proportion_mediated': mediator_panels,
        'partial_spearman': {
            'marginal': {'rho': float(rho_marg), 'p': float(p_marg), 'n': len(pairs_pd)},
            'stratified_q_b0_given_q_b19': {'rho': rho_strat, 'p': p_strat},
            'stratified_q_b19_given_q_b0': {'rho': rho_strat2, 'p': p_strat2},
        },
        'meta_regression': meta_results,
        'power_n30': pwr,
        'sync_covariates': {str(k): v for k, v in sync_covariates.items()},
    }, indent=2))
    print()
    print(f'Wrote: {out}')


if __name__ == '__main__':
    main()
