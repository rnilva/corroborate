"""Causal discovery on the per-(env, burst) link-moderator panel.

After the meta-regression deconfounding (5c447f2 → 4661c24)
identified four candidate moderators of g_link
(bootstrap_fraction, log_horizon, empirical_reward_density,
log_obs_dim), this script applies the framework's
causal-discovery primitives:

  1. Conservative-PC adjacency (`discover_adjacency`) at depth ≤ 2
     over (covariates ∪ {g_link}). Identifies which moderators
     remain edge-adjacent to g_link after conditioning on
     candidate separators.

  2. DoWhy backdoor + refutation triple on the strongest
     surviving edge: causal ATE estimate at rung-2-conditional-
     on-DAG.

These are **observational** primitives — the env-features are
predetermined by the env, not interventionally manipulated. The
discovery's value is reducing the ambiguity about which
moderators are direct-link contributors vs which are
correlated-but-screened-off.

Usage:
  uv run python experiments/causal_discovery_link_moderators.py
  uv run python experiments/causal_discovery_link_moderators.py \
    --corpus ddqn_effective_cohort --total-steps 200000
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from corroborate.bridges_dowhy import (
    backdoor_ate, placebo_refutation, random_common_cause_refutation,
)
from corroborate.causal_discovery import discover_adjacency
from corroborate.rl.env_catalogue import get
from corroborate.statistics import hedges_g_paired


# Reuse the helpers from analyze_per_burst_meta_regression.
import sys
sys.path.insert(0, 'experiments')
from analyze_per_burst_meta_regression import (  # type: ignore[import-untyped]
    _empirical_reward_features, _load_arrays,
)


_REWARD_DENSITY_MAP: dict[str, float] = {
    'terminal_only': 0.0,
    'shaped': 1.0,
    'event_triggered': 1.0,
    'per_step': 2.0,
}


def _per_burst_mediator_deltas(
    corpus: str, env: str, total_steps: int, eval_every: int,
) -> dict[str, np.ndarray] | None:
    """Per-(pair, burst) DDQN−vanilla deltas of candidate non-bias
    mediators, sliced from the flat per-step trace columns by eval
    burst window of length `eval_every`.

    Returns dict with keys: action_margin, argmax_disagreement,
    state_coverage, vanilla_q_spread (per-burst mean of vanilla's
    Q-vector max−min — a σ-proxy for the extreme-value max bias
    that XQL targets, not a delta), delta_q_spread, delta_q_lower
    (DDQN−vanilla on the mean−min lower tail). Each value is
    shape (n_pairs, n_bursts), or None if data is missing."""
    base = Path('experiments/data') / corpus
    runs_path = base / 'runs.parquet'
    if not runs_path.exists():
        runs_path = base / 'runs_with_mediators.parquet'
    runs_df = pl.read_parquet(str(runs_path)).filter(
        (pl.col('env_name') == env)
        & (pl.col('total_steps') == total_steps)
    )
    if runs_df.height == 0:
        return None
    ddqn_ids = runs_df.filter(
        pl.col('intervention_name') == 'ddqn'
    ).select(['id', 'seed']).to_dicts()
    van_ids = runs_df.filter(
        pl.col('intervention_name') == 'vanilla_dqn'
    ).select(['id', 'seed']).to_dicts()
    if not ddqn_ids or not van_ids:
        return None

    all_ids = [d['id'] for d in ddqn_ids] + [d['id'] for d in van_ids]
    cols = [
        'id',
        'online_max_q_per_step', 'online_mean_q_per_step',
        'online_min_q_per_step',
        'online_argmax_per_step', 'target_argmax_per_step',
        'state_hash',
    ]
    trace_df = pl.read_parquet(
        str(base / 'traces.parquet'), columns=cols,
    ).filter(pl.col('id').is_in(all_ids))

    n_bursts = total_steps // eval_every
    by_id_action_margin: dict[str, np.ndarray] = {}
    by_id_disagreement: dict[str, np.ndarray] = {}
    by_id_state_cov: dict[str, np.ndarray] = {}
    by_id_q_spread: dict[str, np.ndarray] = {}
    by_id_q_lower: dict[str, np.ndarray] = {}
    for row in trace_df.iter_rows(named=True):
        cid = row['id']
        omax = np.asarray(row['online_max_q_per_step'], dtype=np.float64)
        omean = np.asarray(row['online_mean_q_per_step'], dtype=np.float64)
        omin = np.asarray(row['online_min_q_per_step'], dtype=np.float64)
        oarg = np.asarray(row['online_argmax_per_step'], dtype=np.int64)
        targ = np.asarray(row['target_argmax_per_step'], dtype=np.int64)
        sh = np.asarray(row['state_hash'], dtype=np.int64)
        if min(len(omax), len(omean), len(omin), len(oarg), len(targ), len(sh)) < total_steps:
            continue
        am = np.zeros(n_bursts)
        dg = np.zeros(n_bursts)
        sc = np.zeros(n_bursts)
        sp = np.zeros(n_bursts)
        lo = np.zeros(n_bursts)
        for b in range(n_bursts):
            s, e = b * eval_every, (b + 1) * eval_every
            am[b] = float((omax[s:e] - omean[s:e]).mean())
            dg[b] = float((oarg[s:e] != targ[s:e]).mean())
            sc[b] = float(len(np.unique(sh[s:e])) / max(len(sh[s:e]), 1))
            sp[b] = float((omax[s:e] - omin[s:e]).mean())
            lo[b] = float((omean[s:e] - omin[s:e]).mean())
        by_id_action_margin[cid] = am
        by_id_disagreement[cid] = dg
        by_id_state_cov[cid] = sc
        by_id_q_spread[cid] = sp
        by_id_q_lower[cid] = lo

    seed_to_van = {d['seed']: d['id'] for d in van_ids}
    seed_to_ddqn = {d['seed']: d['id'] for d in ddqn_ids}
    common = sorted(
        s for s in (set(seed_to_van) & set(seed_to_ddqn))
        if seed_to_van[s] in by_id_action_margin
        and seed_to_ddqn[s] in by_id_action_margin
    )
    if len(common) < 4:
        return None

    def _stack_delta(
        by_id: dict[str, np.ndarray],
    ) -> np.ndarray:
        van = np.stack([by_id[seed_to_van[s]] for s in common])
        ddqn = np.stack([by_id[seed_to_ddqn[s]] for s in common])
        return ddqn - van

    def _stack_vanilla_only(
        by_id: dict[str, np.ndarray],
    ) -> np.ndarray:
        return np.stack([by_id[seed_to_van[s]] for s in common])

    return {
        'action_margin': _stack_delta(by_id_action_margin),
        'argmax_disagreement': _stack_delta(by_id_disagreement),
        'state_coverage': _stack_delta(by_id_state_cov),
        'delta_q_spread': _stack_delta(by_id_q_spread),
        'delta_q_lower': _stack_delta(by_id_q_lower),
        # Vanilla's q_spread is the structural σ-proxy: per the
        # XQL paper, the residual max-bias is σ-proportional even
        # after DDQN's action-noise fix. Sparse-reward envs should
        # have larger vanilla_q_spread, and that should explain
        # the residual sparse-reward → outcome edge.
        'vanilla_q_spread': _stack_vanilla_only(by_id_q_spread),
    }


def _build_panel(corpus: str, total_steps: int, *, include_env: bool = False) -> pl.DataFrame:
    runs_path = Path('experiments/data') / corpus / 'runs.parquet'
    if not runs_path.exists():
        runs_path = (
            Path('experiments/data') / corpus / 'runs_with_mediators.parquet'
        )
    df = pl.read_parquet(str(runs_path)).filter(
        pl.col('total_steps') == total_steps,
    )
    eval_every_unique = df['eval_every'].unique().to_list()
    if len(eval_every_unique) != 1:
        raise ValueError(
            f'eval_every not uniform across corpus: {eval_every_unique}',
        )
    eval_every = int(eval_every_unique[0])
    envs = sorted(df['env_name'].unique().to_list())
    rows: list[dict[str, float]] = []
    for env in envs:
        spec = get(env)
        n_a = spec.n_actions
        obs_n = 1
        for d in spec.observation_shape:
            obs_n *= int(d)
        horizon = float(spec.horizon) if spec.horizon else 1000.0
        empirical = _empirical_reward_features(corpus, env)
        if empirical is None:
            continue
        nonzero_reward_frac, bootstrap_fraction = empirical
        arrays = _load_arrays(corpus, env)
        if arrays is None:
            continue
        delta_bias, delta_ret = arrays
        mediators = _per_burst_mediator_deltas(
            corpus, env, total_steps, eval_every,
        )
        n_pairs, n_bursts = delta_ret.shape
        for b in range(n_bursts):
            dr = list(map(float, delta_ret[:, b].tolist()))
            db = list(map(float, delta_bias[:, b].tolist()))
            g_link, se_link = hedges_g_paired(dr)
            g_mech, se_mech = hedges_g_paired(db)
            if not (
                isinstance(g_link, float) and math.isfinite(g_link)
                and isinstance(se_link, float) and math.isfinite(se_link)
                and se_link > 0.0
            ):
                continue
            if not (
                isinstance(g_mech, float) and math.isfinite(g_mech)
                and isinstance(se_mech, float) and math.isfinite(se_mech)
                and se_mech > 0.0
            ):
                continue
            row: dict[str, object] = {
                'g_link': float(g_link),
                'g_mech': float(g_mech),
                'log_action_dim': math.log(max(n_a, 2)),
                'log_obs_dim': math.log(max(obs_n, 1)),
                'log_horizon': math.log(max(horizon, 1.0)),
                'empirical_reward_density': float(nonzero_reward_frac),
                'bootstrap_fraction': float(bootstrap_fraction),
                'burst_index': float(b),
                'mean_dbias': float(delta_bias[:, b].mean()),
                'd_action_margin': (
                    float(mediators['action_margin'][:, b].mean())
                    if mediators is not None else float('nan')
                ),
                'd_argmax_disagreement': (
                    float(mediators['argmax_disagreement'][:, b].mean())
                    if mediators is not None else float('nan')
                ),
                'd_state_coverage': (
                    float(mediators['state_coverage'][:, b].mean())
                    if mediators is not None else float('nan')
                ),
                'd_q_spread': (
                    float(mediators['delta_q_spread'][:, b].mean())
                    if mediators is not None else float('nan')
                ),
                'd_q_lower': (
                    float(mediators['delta_q_lower'][:, b].mean())
                    if mediators is not None else float('nan')
                ),
                'vanilla_q_spread': (
                    float(mediators['vanilla_q_spread'][:, b].mean())
                    if mediators is not None else float('nan')
                ),
            }
            if include_env:
                row['env_name'] = env
            rows.append(row)
    return pl.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', default='ddqn')
    parser.add_argument('--total-steps', type=int, default=200_000)
    args = parser.parse_args()
    corpus: str = args.corpus
    total_steps: int = args.total_steps

    print('=' * 100)
    print(f'Causal discovery on per-(env, burst) link moderator panel '
          f'[corpus={corpus}, total_steps={total_steps}]')
    print('=' * 100)
    panel = _build_panel(corpus, total_steps, include_env=True)
    # Drop rows with NaN in any mediator (envs missing per-step
    # trace columns); PC's CI test can't handle NaN.
    mediator_cols = (
        'd_action_margin', 'd_argmax_disagreement', 'd_state_coverage',
        'd_q_spread', 'd_q_lower', 'vanilla_q_spread',
    )
    panel = panel.drop_nulls(subset=list(mediator_cols)).filter(
        ~pl.any_horizontal(pl.col(c).is_nan() for c in mediator_cols)
    )
    print(f'  n_strata={panel.height}  n_features={len(panel.columns)}')

    variables = (
        'g_link',
        'g_mech',
        'bootstrap_fraction',
        'log_horizon',
        'log_obs_dim',
        'empirical_reward_density',
        'log_action_dim',
        'mean_dbias',
        'd_action_margin',
        'd_argmax_disagreement',
        'd_state_coverage',
        'd_q_spread',
        'd_q_lower',
        'vanilla_q_spread',
    )

    # Stage 1a: Conservative-PC adjacency at depth ≤ 2 — no JCI.
    print()
    print('Stage 1a — Conservative-PC adjacency (depth ≤ 2, no JCI)')
    adj = discover_adjacency(
        panel, variables=variables,
        alpha=0.05, max_conditioning=2,
    )
    print()
    print('Surviving edges:')
    for edge in sorted(
        adj.edges, key=lambda e: tuple(sorted(e)),
    ):
        a, b = sorted(edge)
        print(f'  {a:<28} ⟷ {b}')
    print()
    print('Edges removed (with separating sets):')
    for edge, seps in sorted(
        adj.separating_sets.items(),
        key=lambda kv: tuple(sorted(kv[0])),
    ):
        a, b = sorted(edge)
        sep_str = ', '.join(
            '{}'.format(', '.join(sorted(s))) if s else '∅'
            for s in sorted(seps, key=len)
        )
        print(f'  {a:<28} ⊥ {b:<28}  | {{ {sep_str} }}')

    # Stage 1b: PC with JCI — within-env-stratified CI tests pooled
    # via Fisher z. Addresses the within-env-correlation concern
    # (149 strata aren't iid; bursts within an env share cells).
    print()
    print('Stage 1b — Conservative-PC with JCI (stratify_by=env_name)')
    adj_jci = discover_adjacency(
        panel, variables=variables,
        alpha=0.05, max_conditioning=2,
        stratify_by='env_name',
    )
    print()
    print('Surviving edges (JCI):')
    for edge in sorted(adj_jci.edges, key=lambda e: tuple(sorted(e))):
        a, b = sorted(edge)
        print(f'  {a:<28} ⟷ {b}')

    # Markov-blanket extraction for both endpoints.
    g_link_neighbors = sorted(
        nb for edge in adj.edges for nb in edge
        if 'g_link' in edge and nb != 'g_link'
    )
    g_mech_neighbors = sorted(
        nb for edge in adj.edges for nb in edge
        if 'g_mech' in edge and nb != 'g_mech'
    )
    print()
    print(f'g_link neighbors: {g_link_neighbors}')
    print(f'g_mech neighbors: {g_mech_neighbors}')
    chain_edge = frozenset({'g_link', 'g_mech'}) in adj.edges
    print(f'g_link ⟷ g_mech direct edge: {chain_edge}')

    # Stage 2: DoWhy backdoor on the surviving direct-edge candidate
    # for g_link. Use bootstrap_fraction → g_link with the rest as
    # confounders (those that are NOT screened off from g_link).

    if not g_link_neighbors:
        print('  no surviving edges from g_link; skipping DoWhy.')
        return

    treatment = 'bootstrap_fraction' if (
        'bootstrap_fraction' in g_link_neighbors
    ) else g_link_neighbors[0]
    print()
    print(f'Stage 2 — DoWhy backdoor: {treatment} → g_link | confounders')

    confounders = [v for v in g_link_neighbors if v != treatment]
    dag: list[tuple[str, str]] = [
        *((c, treatment) for c in confounders),
        *((c, 'g_link') for c in confounders),
        (treatment, 'g_link'),
    ]
    record: Mapping[str, np.ndarray] = {
        col: np.asarray(panel[col].to_list(), dtype=np.float64)
        for col in panel.columns
        if col != 'env_name'  # exclude string column from numeric record
    }
    triple = (
        ('backdoor_ate', backdoor_ate(
            treatment, 'g_link', graph=dag,
            expected_sign=+1, threshold=0.05,
        )),
        ('placebo', placebo_refutation(
            treatment, 'g_link', graph=dag, tolerance=0.1,
        )),
        ('random_common_cause', random_common_cause_refutation(
            treatment, 'g_link', graph=dag, tolerance=0.1,
        )),
    )
    print(f'  DAG: {len(confounders)} confounders → {treatment} → g_link')
    for label, bridge in triple:
        r = bridge(record)  # type: ignore[arg-type]
        keystat = ''
        if 'ate' in r.stats:
            ate = r.stats['ate']
            keystat = (
                f'ATE={float(ate):+.4f}'
                if isinstance(ate, (int, float)) else 'ATE=?'
            )
        elif 'placebo_ate' in r.stats:
            p = r.stats['placebo_ate']
            real = r.stats.get('real_ate', float('nan'))
            keystat = (
                f'placebo={float(p):+.4f} real={float(real):+.4f}'
                if isinstance(p, (int, float))
                and isinstance(real, (int, float))
                else 'placebo=?'
            )
        elif 'drift' in r.stats:
            d = r.stats['drift']
            real = r.stats.get('real_ate', float('nan'))
            keystat = (
                f'drift={float(d):.4f} real={float(real):+.4f}'
                if isinstance(d, (int, float))
                and isinstance(real, (int, float))
                else 'drift=?'
            )
        print(f'  {label:<22} verdict={r.verdict.value:<22} {keystat}')


if __name__ == '__main__':
    main()
