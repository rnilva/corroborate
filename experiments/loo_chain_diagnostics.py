"""Diagnostic cross-checks on the rev 10 chain regression.

Two methodological concerns to test:

1. **Per-burst noise structure.** Random-effects meta-regression
   pools across (env, burst) strata assuming burst-independence;
   bursts within an env are training-time-autocorrelated. Cross-
   check: per-class median SE; within-env burst-to-burst
   autocorrelation of g_link.

2. **Trajectory quality.** Unsolved / absent-threshold envs may
   be high-noise carriers that inflate the β(bootstrap_fraction)
   coefficient via leverage. Cross-check: re-run the joint
   regression on convergence-conditioned subsets (solved,
   solved+partial, with-thresholds-only, absent-only).

Reads experiments/data/ddqn/{runs,traces}.parquet."""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from corroborate.stats import (
    StratumObservation, meta_regression,
)
from corroborate.corpus.persistence import read_runrows
from corroborate_rl.convergence import classify_envs
from corroborate_rl.env_catalogue import get as _get_env_spec
from corroborate.stats import hedges_g_paired


_RUNS = Path('experiments/data/ddqn/runs.parquet')
_TRACES = Path('experiments/data/ddqn/traces.parquet')
_TOTAL_STEPS = 200000
_TREATMENT = 'ddqn'
_BASELINE = 'vanilla_dqn'
_COVARS = (
    'log_action_dim', 'log_obs_dim', 'log_horizon',
    'empirical_reward_density', 'bootstrap_fraction',
)


def _per_seed_arrays(
    runs_df: pl.DataFrame, traces_df: pl.DataFrame, env: str,
    intervention: str,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    sub = runs_df.filter(
        (pl.col('env_name') == env)
        & (pl.col('intervention_name') == intervention)
        & (pl.col('total_steps') == _TOTAL_STEPS)
    )
    if sub.height == 0:
        return {}, {}
    ids = sub['id'].to_list()
    seeds = sub['seed'].to_list()
    cell_traces = traces_df.filter(pl.col('id').is_in(ids))
    by_id = {row['id']: row for row in cell_traces.iter_rows(named=True)}
    bias_by_seed: dict[int, np.ndarray] = {}
    return_by_seed: dict[int, np.ndarray] = {}
    for cell_id, seed in zip(ids, seeds):
        trace = by_id.get(cell_id)
        if trace is None:
            continue
        mc = np.asarray(trace.get('mc_return'), dtype=np.float64)
        pq = np.asarray(trace.get('predicted_q_at_start'), dtype=np.float64)
        if mc.ndim != 2 or pq.ndim != 2 or mc.shape != pq.shape:
            continue
        bias = (pq - mc).mean(axis=1)
        ret = mc.mean(axis=1)
        bias_by_seed[int(seed)] = bias
        return_by_seed[int(seed)] = ret
    return bias_by_seed, return_by_seed


def _paired_arrays(
    runs_df: pl.DataFrame, traces_df: pl.DataFrame, env: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    t_bias, t_ret = _per_seed_arrays(runs_df, traces_df, env, _TREATMENT)
    b_bias, b_ret = _per_seed_arrays(runs_df, traces_df, env, _BASELINE)
    common_seeds = sorted(set(t_bias) & set(b_bias))
    if len(common_seeds) < 5:
        return None
    n_bursts = min(t_bias[s].shape[0] for s in common_seeds)
    if n_bursts < 2:
        return None
    delta_bias = np.array([
        t_bias[s][:n_bursts] - b_bias[s][:n_bursts] for s in common_seeds
    ])
    delta_ret = np.array([
        t_ret[s][:n_bursts] - b_ret[s][:n_bursts] for s in common_seeds
    ])
    return delta_bias, delta_ret


def _env_features(env: str) -> dict[str, float]:
    try:
        spec = _get_env_spec(env)
    except KeyError:
        return {}
    n_a = int(spec.n_actions)
    obs = spec.observation_shape
    obs_n = int(np.prod(np.asarray(obs))) if obs else 1
    horizon = float(spec.horizon) if spec.horizon else 1000.0
    return {
        'log_action_dim': math.log(max(n_a, 2)),
        'log_obs_dim': math.log(max(obs_n, 1)),
        'log_horizon': math.log(max(horizon, 1.0)),
    }


def _empirical_reward_features(
    traces_df: pl.DataFrame, runs_df: pl.DataFrame, env: str,
) -> tuple[float, float]:
    sub_ids = runs_df.filter(pl.col('env_name') == env)['id'].to_list()
    if not sub_ids:
        return float('nan'), float('nan')
    cell_traces = traces_df.filter(pl.col('id').is_in(sub_ids))
    nz: list[float] = []
    boot: list[float] = []
    for row in cell_traces.iter_rows(named=True):
        rwd = np.asarray(row['reward'], dtype=np.float64)
        done = np.asarray(row['done'], dtype=np.float64)
        if rwd.size > 0:
            nz.append(float((rwd != 0.0).mean()))
        if done.size > 0:
            boot.append(float(1.0 - done.mean()))
    if not nz or not boot:
        return float('nan'), float('nan')
    return float(np.mean(nz)), float(np.mean(boot))


def _build_panel(
    runs_df: pl.DataFrame, traces_df: pl.DataFrame, envs: tuple[str, ...],
) -> tuple[
    dict[tuple[str, int], tuple[float, float, float, float]],
    dict[tuple[str, int], dict[str, float]],
]:
    panel: dict[tuple[str, int], tuple[float, float, float, float]] = {}
    covars: dict[tuple[str, int], dict[str, float]] = {}
    for env in envs:
        ef = _env_features(env)
        if not ef:
            continue
        arrs = _paired_arrays(runs_df, traces_df, env)
        if arrs is None:
            continue
        delta_bias, delta_ret = arrs
        emp_nz, emp_boot = _empirical_reward_features(
            traces_df, runs_df, env,
        )
        n_pairs, n_bursts = delta_ret.shape
        for b in range(n_bursts):
            dr = delta_ret[:, b].astype(float).tolist()
            db = delta_bias[:, b].astype(float).tolist()
            gl, sl = hedges_g_paired(dr)
            gm, sm = hedges_g_paired(db)
            if not all(math.isfinite(v) for v in (gl, sl, gm, sm)):
                continue
            if sl <= 0 or sm <= 0:
                continue
            panel[(env, b)] = (gl, sl, gm, sm)
            covars[(env, b)] = {
                **ef,
                'empirical_reward_density': emp_nz,
                'bootstrap_fraction': emp_boot,
            }
    return panel, covars


def _fit_subset(
    panel: Mapping[tuple[str, int], tuple[float, float, float, float]],
    covars: Mapping[tuple[str, int], dict[str, float]],
    env_subset: set[str],
    target: str,
) -> tuple[float, float, int, int]:
    """Returns (β_bootstrap_fraction, p, n_strata, n_envs)."""
    obs: list[StratumObservation] = []
    g_idx, se_idx = (0, 1) if target == 'g_link' else (2, 3)
    envs_used: set[str] = set()
    for key, stats in panel.items():
        env_name, _burst = key
        if env_name not in env_subset:
            continue
        g, se = stats[g_idx], stats[se_idx]
        if not (math.isfinite(g) and math.isfinite(se) and se > 0):
            continue
        cov = covars[key]
        if not all(math.isfinite(cov[c]) for c in _COVARS):
            continue
        envs_used.add(env_name)
        obs.append(StratumObservation(
            stratum_id=key, g=g, se=se,
            covariates={c: cov[c] for c in _COVARS},
        ))
    if len(obs) < len(_COVARS) + 2:
        return float('nan'), float('nan'), len(obs), len(envs_used)
    res = meta_regression(obs)
    boot = next(
        (c for c in res.coefficients if c.name == 'bootstrap_fraction'),
        None,
    )
    if boot is None:
        return float('nan'), float('nan'), len(obs), len(envs_used)
    return boot.coefficient, boot.p_value, res.n_strata, len(envs_used)


def _per_class_se_distribution(
    panel: Mapping[tuple[str, int], tuple[float, float, float, float]],
    classes: Mapping[str, set[str]],
) -> None:
    """Median + max of per-burst SE per convergence class."""
    print('\nPer-burst SE distribution by convergence class:')
    print(f'  {"class":<12} {"n_strata":>9} {"median se_link":>15}'
          f' {"max se_link":>13} {"median se_mech":>15}')
    print(f'  {"-"*12} {"-"*9} {"-"*15} {"-"*13} {"-"*15}')
    for cls, env_set in classes.items():
        ses_link: list[float] = []
        ses_mech: list[float] = []
        for (env, _b), stats in panel.items():
            if env in env_set:
                ses_link.append(stats[1])
                ses_mech.append(stats[3])
        if not ses_link:
            continue
        med_link = float(np.median(ses_link))
        max_link = float(np.max(ses_link))
        med_mech = float(np.median(ses_mech))
        print(f'  {cls:<12} {len(ses_link):>9} {med_link:>15.4f}'
              f' {max_link:>13.4f} {med_mech:>15.4f}')


def _within_env_autocorr(
    panel: Mapping[tuple[str, int], tuple[float, float, float, float]],
) -> None:
    """Lag-1 autocorrelation of g_link within each env (across
    consecutive bursts). Random-effects pool assumes
    burst-independence; high autocorrelation flags violation."""
    by_env: dict[str, list[tuple[int, float]]] = {}
    for (env, b), stats in panel.items():
        by_env.setdefault(env, []).append((b, stats[0]))
    print('\nWithin-env lag-1 autocorrelation of g_link:')
    print(f'  {"env":<28} {"n_bursts":>9} {"r(g_b, g_b+1)":>14}')
    print(f'  {"-"*28} {"-"*9} {"-"*14}')
    for env, vals in sorted(by_env.items()):
        vals.sort()
        gs = [g for _b, g in vals]
        if len(gs) < 4:
            continue
        a = np.asarray(gs[:-1])
        b = np.asarray(gs[1:])
        if float(a.std()) == 0.0 or float(b.std()) == 0.0:
            r = float('nan')
        else:
            r = float(np.corrcoef(a, b)[0, 1])
        print(f'  {env:<28} {len(gs):>9} {r:>14.3f}')


def main() -> None:
    print('Loading runs + traces...')
    runs_df = pl.read_parquet(
        str(_RUNS),
        columns=[
            'id', 'env_name', 'intervention_name', 'seed',
            'total_steps', 'eval_best_burst_mean',
            'eval_final_mean',
        ],
    )
    traces_df = pl.read_parquet(
        str(_TRACES),
        columns=['id', 'mc_return', 'predicted_q_at_start', 'reward', 'done'],
    )
    envs = tuple(sorted(runs_df['env_name'].unique().to_list()))
    print(f'envs: {len(envs)}; building panel...')
    panel, covars = _build_panel(runs_df, traces_df, envs)
    panel_envs = sorted({k[0] for k in panel})
    print(f'panel strata: {len(panel)}; envs: {len(panel_envs)}')

    # Convergence-class assignment via existing classifier on the
    # baseline arm at 200k.
    runs_obj = read_runrows(str(_RUNS))
    baseline_runs = [
        r for r in runs_obj
        if r.measurements.get('total_steps') == _TOTAL_STEPS
        and r.measurements.get('intervention_name') == _BASELINE
    ]
    classes_full = classify_envs(baseline_runs)
    classes: dict[str, set[str]] = {
        'solved': set(), 'partial': set(),
        'unsolved': set(), 'absent': set(),
    }
    for env, ec in classes_full.items():
        classes[ec.classification].add(env)
    # The "absent" class includes envs with no defensible
    # threshold; baseline_runs may not cover all 18 envs, so
    # collect remaining as absent.
    panel_env_set = set(panel_envs)
    for env in panel_env_set:
        if not any(env in s for s in classes.values()):
            classes['absent'].add(env)

    print('\nConvergence-class membership:')
    for cls, env_set in classes.items():
        in_panel = env_set & panel_env_set
        print(f'  {cls:<10} ({len(in_panel)} in panel): '
              f'{sorted(in_panel)}')

    # Diagnostic 1: per-class SE distribution.
    _per_class_se_distribution(panel, classes)

    # Diagnostic 2: within-env burst autocorrelation.
    _within_env_autocorr(panel)

    # Diagnostic 3: re-run regression per convergence-class
    # subset.
    print('\nβ(bootstrap_fraction) per convergence subset:')
    print(f'  {"subset":<35} {"target":<8} {"β":>9} {"p":>8} '
          f'{"n_strata":>9} {"n_envs":>7}')
    print(f'  {"-"*35} {"-"*8} {"-"*9} {"-"*8} {"-"*9} {"-"*7}')
    subsets: list[tuple[str, set[str]]] = [
        ('full panel', set(panel_envs)),
        ('solved only', classes['solved'] & panel_env_set),
        ('solved + partial', (classes['solved'] | classes['partial'])
            & panel_env_set),
        ('with-thresholds (s+p+u)',
         (classes['solved'] | classes['partial'] | classes['unsolved'])
             & panel_env_set),
        ('absent only', classes['absent'] & panel_env_set),
        ('full minus FourRooms',
         set(panel_envs) - {'FourRooms-misc'}),
        ('absent minus FourRooms',
         classes['absent'] & panel_env_set - {'FourRooms-misc'}),
    ]
    for label, env_set in subsets:
        for target in ('g_link', 'g_mech'):
            beta, p, n, ne = _fit_subset(panel, covars, env_set, target)
            print(f'  {label:<35} {target:<8} {beta:>+9.3f} {p:>8.4f} '
                  f'{n:>9d} {ne:>7d}')


if __name__ == '__main__':
    main()
