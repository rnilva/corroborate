"""Per-(env, burst) Hedges' g + meta-regression on time-series.

Two-stage analysis on a corpus's per-burst trajectories:

Stage 1 — per-stratum Hedges' g
  For each (env, burst): treat the per-pair Δret_burst as the
  paired-Δ vector; compute hedges_g_paired + SE. Output: a table
  of g_link(env, burst) values revealing how DDQN's standardized
  outcome advantage varies across training phase per env.

Stage 2 — meta-regression
  Treat each (env, burst) as a stratum. Covariates per stratum:
    - log_action_dim   (env-property; constant across bursts of one env)
    - burst_index      (training-phase covariate)
  Inverse-variance-weighted OLS via `corroborate.meta_regression`
  produces β coefficients on each covariate. Significant β on
  burst_index → DDQN's effect varies across training phase.
  Significant β on log_action_dim → action_dim modulates the
  effect (replicates / extends the env-level finding).

Usage:
  uv run python experiments/analyze_per_burst_meta_regression.py
  uv run python experiments/analyze_per_burst_meta_regression.py --corpus action_dim_wide
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import polars as pl

from corroborate.meta_regression import (
    StratumObservation, meta_regression,
)
from corroborate.rl.env_catalogue import get
from corroborate.statistics import hedges_g_paired


def _load_arrays(
    corpus: str, env: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Per-pair Δbias and Δret (shape (n_pairs, n_bursts)) for env."""
    base = Path('experiments/data') / corpus
    runs_path = base / 'runs.parquet'
    if not runs_path.exists():
        runs_path = base / 'runs_with_mediators.parquet'
    if not runs_path.exists():
        return None
    traces_path = base / 'traces.parquet'
    if not traces_path.exists():
        return None
    df = pl.read_parquet(str(runs_path)).filter(
        pl.col('env_name') == env
    )
    if df.height == 0:
        return None
    ddqn_ids = df.filter(
        pl.col('intervention_name') == 'ddqn'
    ).select(['id', 'seed']).to_dicts()
    van_ids = df.filter(
        pl.col('intervention_name') == 'vanilla_dqn'
    ).select(['id', 'seed']).to_dicts()
    if not ddqn_ids or not van_ids:
        return None
    all_ids = [d['id'] for d in ddqn_ids] + [d['id'] for d in van_ids]
    trace_df = pl.read_parquet(
        str(traces_path),
        columns=['id', 'predicted_q_at_start', 'mc_return'],
    ).filter(pl.col('id').is_in(all_ids))
    bias_by_id: dict[str, np.ndarray] = {}
    return_by_id: dict[str, np.ndarray] = {}
    for row in trace_df.iter_rows(named=True):
        cid = row['id']
        pred = np.asarray(row['predicted_q_at_start'], dtype=np.float64)
        actual = np.asarray(row['mc_return'], dtype=np.float64)
        if pred.ndim != 2 or actual.ndim != 2 or pred.shape != actual.shape:
            continue
        bias_by_id[cid] = (pred - actual).mean(axis=-1)
        return_by_id[cid] = actual.mean(axis=-1)
    seed_to_van = {d['seed']: d['id'] for d in van_ids}
    seed_to_ddqn = {d['seed']: d['id'] for d in ddqn_ids}
    common = sorted(set(seed_to_van) & set(seed_to_ddqn))
    common = [
        s for s in common
        if seed_to_van[s] in bias_by_id and seed_to_ddqn[s] in bias_by_id
    ]
    if len(common) < 4:
        return None
    van_bias = np.stack([bias_by_id[seed_to_van[s]] for s in common])
    van_ret = np.stack([return_by_id[seed_to_van[s]] for s in common])
    ddqn_bias = np.stack([bias_by_id[seed_to_ddqn[s]] for s in common])
    ddqn_ret = np.stack([return_by_id[seed_to_ddqn[s]] for s in common])
    return ddqn_bias - van_bias, ddqn_ret - van_ret


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='ddqn')
    parser.add_argument('--total-steps', type=int, default=200000)
    args = parser.parse_args()
    corpus: str = args.corpus

    runs_path = Path('experiments/data') / corpus / 'runs.parquet'
    if not runs_path.exists():
        runs_path = (
            Path('experiments/data') / corpus / 'runs_with_mediators.parquet'
        )
    df = pl.read_parquet(str(runs_path)).filter(
        pl.col('total_steps') == args.total_steps,
    )
    envs = sorted(df['env_name'].unique().to_list())

    print(f'corpus={corpus} envs={len(envs)} total_steps={args.total_steps}')
    print()

    # Stage 1: per-(env, burst) Hedges' g on Δret.
    print("Stage 1 — per-(env, burst) Hedges' g on Δret")
    print(f'  {"env":<25} {"|A|":>4} ' + ''.join(f'b{b:<5}' for b in range(10)))
    print('-' * 100)

    # Reward-regime ordinal: 0=terminal_only (sparsest), 1=event/shaped,
    # 2=per_step (densest). Captures bootstrap-signal density.
    _reward_density_map: dict[str, float] = {
        'terminal_only': 0.0,
        'shaped': 1.0,
        'event_triggered': 1.0,
        'per_step': 2.0,
    }

    obs_list: list[StratumObservation] = []
    saturated: list[str] = []
    for env in envs:
        try:
            spec = get(env)
            n_a = spec.n_actions
            obs_n = 1
            for d in spec.observation_shape:
                obs_n *= int(d)
            horizon = float(spec.horizon) if spec.horizon else 1000.0
            reward_density = _reward_density_map.get(
                str(spec.reward_regime), 1.0,
            )
        except Exception:
            n_a, obs_n, horizon, reward_density = 0, 4, 1000.0, 1.0
        arrays = _load_arrays(corpus, env)
        if arrays is None:
            continue
        delta_bias, delta_ret = arrays
        n_pairs, n_bursts = delta_ret.shape
        gs_per_burst: list[float] = []
        env_obs: list[StratumObservation] = []
        for b in range(n_bursts):
            dr = list(map(float, delta_ret[:, b].tolist()))
            g, se = hedges_g_paired(dr)
            gs_per_burst.append(g)
            if (
                isinstance(g, float) and math.isfinite(g)
                and isinstance(se, float) and math.isfinite(se) and se > 0.0
            ):
                mean_dbias = float(delta_bias[:, b].mean())
                env_obs.append(StratumObservation(
                    stratum_id=(env, b),
                    g=g, se=se,
                    covariates={
                        'log_action_dim': math.log(max(n_a, 2)),
                        'burst_index': float(b),
                        'mean_dbias': mean_dbias,
                        'log_obs_dim': math.log(max(obs_n, 1)),
                        'log_horizon': math.log(max(horizon, 1.0)),
                        'reward_density': reward_density,
                    },
                ))
        # Filter env if all per-burst g values are exactly zero
        # (saturated outcome, no signal at any burst).
        if env_obs and all(abs(o.g) < 1e-9 for o in env_obs):
            saturated.append(env)
        else:
            obs_list.extend(env_obs)
        gs_str = ''.join(
            f'{g:>+6.2f}' if isinstance(g, float) and math.isfinite(g) else '   nan'
            for g in gs_per_burst
        )
        print(f'  {env:<25} {n_a:>4} {gs_str}')

    if saturated:
        print()
        print(f'Filtered (g=0 across all bursts, no signal): {sorted(saturated)}')

    print()
    print(f'Stage 2 — meta-regression on g_(env, burst)')
    print(f'  n_strata: {len(obs_list)}')
    if len(obs_list) < 4:
        print('  (too few strata to fit)')
        return

    def _project(
        observations: list[StratumObservation], covs: tuple[str, ...],
        *, extra: dict[tuple, dict[str, float]] | None = None,
    ) -> list[StratumObservation]:
        """Project obs onto a covariate subset, optionally injecting
        author-computed extras (e.g. interaction terms)."""
        out: list[StratumObservation] = []
        for o in observations:
            base = {c: o.covariates[c] for c in covs}
            if extra is not None:
                base.update(extra.get(o.stratum_id, {}))
            out.append(StratumObservation(
                stratum_id=o.stratum_id, g=o.g, se=o.se,
                covariates=base,
            ))
        return out

    def _print_result(label: str, res) -> None:
        print()
        print(f'  --- {label} ---')
        print(f'    n={res.n_strata} R²={res.r_squared:+.3f} '
              f'intercept={res.intercept:+.3f}')
        for c in res.coefficients:
            sig = '✓ SIGNIFICANT' if c.is_significant else ' '
            print(
                f'    {c.name:<22} β={c.coefficient:+.4f}  '
                f'CI=[{c.ci_lo:+.4f}, {c.ci_hi:+.4f}]  '
                f'p={c.p_value:.4f}  {sig}'
            )

    # Singletons.
    for cov in ('burst_index', 'log_action_dim', 'mean_dbias'):
        try:
            res = meta_regression(_project(obs_list, (cov,)))
            _print_result(f'g ~ {cov}', res)
        except ValueError as e:
            print(f'  --- g ~ {cov} skipped: {e} ---')

    # Joint (burst + action_dim).
    try:
        res = meta_regression(_project(obs_list, ('burst_index', 'log_action_dim')))
        _print_result('g ~ burst_index + log_action_dim', res)
    except ValueError as e:
        print(f'  joint skipped: {e}')

    # Joint (burst + action_dim + mean_dbias). Tests whether
    # action_dim's effect survives controlling for the actual
    # bias-reduction observed at that (env, burst).
    try:
        res = meta_regression(_project(
            obs_list, ('burst_index', 'log_action_dim', 'mean_dbias'),
        ))
        _print_result('g ~ burst_index + log_action_dim + mean_dbias', res)
    except ValueError as e:
        print(f'  3-cov skipped: {e}')

    # With interaction term (action_dim × burst_index).
    interaction: dict[tuple, dict[str, float]] = {
        o.stratum_id: {
            'log_action_dim_x_burst': (
                o.covariates['log_action_dim']
                * o.covariates['burst_index']
            ),
        }
        for o in obs_list
    }
    try:
        res = meta_regression(_project(
            obs_list,
            ('burst_index', 'log_action_dim', 'mean_dbias'),
            extra=interaction,
        ))
        _print_result(
            'g ~ burst_index + log_action_dim + mean_dbias + (log_action_dim × burst_index)',
            res,
        )
    except ValueError as e:
        print(f'  interaction skipped: {e}')

    # Confound test: does log_action_dim survive when we add
    # plausible confounds (log_obs_dim, log_horizon, reward_density)?
    # If another covariate absorbs action_dim, then action_dim was
    # a proxy for that variable.
    confound_sets: tuple[tuple[str, ...], ...] = (
        ('log_action_dim', 'log_obs_dim'),
        ('log_action_dim', 'log_horizon'),
        ('log_action_dim', 'reward_density'),
        ('log_action_dim', 'log_obs_dim', 'log_horizon', 'reward_density'),
        ('log_action_dim', 'log_obs_dim', 'log_horizon', 'reward_density',
         'mean_dbias', 'burst_index'),
    )
    for cset in confound_sets:
        try:
            res = meta_regression(_project(obs_list, cset))
            label = ' + '.join(cset)
            _print_result(f'g ~ {label}', res)
        except ValueError as e:
            print(f'  confound-set {cset} skipped: {e}')


if __name__ == '__main__':
    main()
