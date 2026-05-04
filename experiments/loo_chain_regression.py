"""Leave-one-out chain meta-regression on the rev 10 panel.

Test: is the residual `bootstrap_fraction → g_link | g_mech`
(ATE=+0.88) FourRooms-driven? Reproduce the 5-covariate joint
g_link / g_mech meta-regressions on the full 18-env DDQN 200k
panel; then iterate, holding one env out at a time, and report
β(bootstrap_fraction → g_link) plus its p-value.

Reading: if FourRooms-out crashes the residual to ≈0 while the
other leave-one-out fits keep it intact, the rev 10 panel-level
+0.88 was the FourRooms attenuation pulling the panel average.
If the residual survives every leave-one-out, bootstrap_fraction
is genuine — would warrant a designed intervention.

Usage:
  uv run python experiments/loo_chain_regression.py

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
from corroborate.rl.env_catalogue import get as _get_env_spec
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
    by_id_traces = {row['id']: row for row in cell_traces.iter_rows(named=True)}
    bias_by_seed: dict[int, np.ndarray] = {}
    return_by_seed: dict[int, np.ndarray] = {}
    for cell_id, seed in zip(ids, seeds):
        trace = by_id_traces.get(cell_id)
        if trace is None:
            continue
        mc = np.asarray(trace.get('mc_return'), dtype=np.float64)
        pq = np.asarray(trace.get('predicted_q_at_start'), dtype=np.float64)
        if mc.ndim != 2 or pq.ndim != 2 or mc.shape != pq.shape:
            continue
        bias = (pq - mc).mean(axis=1)  # (n_bursts,)
        ret = mc.mean(axis=1)  # (n_bursts,)
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
    """Per-(env, burst) panel: (g_link, se_link, g_mech, se_mech)
    + covariates."""
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


def _fit(
    panel: Mapping[tuple[str, int], tuple[float, float, float, float]],
    covars: Mapping[tuple[str, int], dict[str, float]],
    target: str,
    holdout_env: str | None,
) -> tuple[float, float, int]:
    """Returns (β_bootstrap_fraction, p_value, n_strata)."""
    obs: list[StratumObservation] = []
    g_idx, se_idx = (0, 1) if target == 'g_link' else (2, 3)
    for key, stats in panel.items():
        env_name, burst = key
        if holdout_env is not None and env_name == holdout_env:
            continue
        g, se = stats[g_idx], stats[se_idx]
        if not (math.isfinite(g) and math.isfinite(se) and se > 0):
            continue
        cov = covars[key]
        if not all(math.isfinite(cov[c]) for c in _COVARS):
            continue
        obs.append(StratumObservation(
            stratum_id=key, g=g, se=se,
            covariates={c: cov[c] for c in _COVARS},
        ))
    if len(obs) < len(_COVARS) + 2:
        return float('nan'), float('nan'), len(obs)
    res = meta_regression(obs)
    boot_coef = next(
        (c for c in res.coefficients if c.name == 'bootstrap_fraction'),
        None,
    )
    if boot_coef is None:
        return float('nan'), float('nan'), len(obs)
    return boot_coef.coefficient, boot_coef.p_value, res.n_strata


def main() -> None:
    print('Loading runs + traces...')
    runs_df = pl.read_parquet(
        str(_RUNS),
        columns=['id', 'env_name', 'intervention_name', 'seed', 'total_steps'],
    )
    traces_df = pl.read_parquet(
        str(_TRACES),
        columns=['id', 'mc_return', 'predicted_q_at_start', 'reward', 'done'],
    )
    envs = tuple(sorted(runs_df['env_name'].unique().to_list()))
    print(f'envs: {len(envs)}; building panel...')
    panel, covars = _build_panel(runs_df, traces_df, envs)
    panel_envs = tuple(sorted({k[0] for k in panel}))
    print(f'panel strata: {len(panel)}; envs in panel: {len(panel_envs)}')
    print()

    print('Full-panel joint regression (rev 10 reproduction):')
    for target in ('g_link', 'g_mech'):
        beta, p, n = _fit(panel, covars, target, holdout_env=None)
        print(f'  {target}: n={n}, β(bootstrap_fraction)={beta:+.3f}, p={p:.4f}')
    print()

    print('Leave-one-out — β(bootstrap_fraction) per holdout:')
    print(f'  {"holdout":<28} {"g_link β":>10} {"p":>8} {"n":>4}  '
          f'{"g_mech β":>10} {"p":>8} {"n":>4}')
    print(f'  {"-"*28} {"-"*10} {"-"*8} {"-"*4}  '
          f'{"-"*10} {"-"*8} {"-"*4}')
    for env in panel_envs:
        b_link, p_link, n_link = _fit(panel, covars, 'g_link', env)
        b_mech, p_mech, n_mech = _fit(panel, covars, 'g_mech', env)
        print(f'  {env:<28} {b_link:>+10.3f} {p_link:>8.4f} {n_link:>4d}  '
              f'{b_mech:>+10.3f} {p_mech:>8.4f} {n_mech:>4d}')


if __name__ == '__main__':
    main()
