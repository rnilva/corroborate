"""Canonical chain-decomposition analysis for any DDQN-shaped corpus.

Replaces the per-study analyze_*.py scripts. One script, parameterised
by the (corpus, treatment, baseline, mediator_path, outcome_path) tuple.
Composes existing primitives — does NOT introduce new ones:

  - per-pair Δ extraction (inline; see `_paired_arrays`)
  - `hedges_g_paired` for per-stratum effect size
  - `meta_regression` for moderator coefficients
  - `discover_adjacency` for PC adjacency
  - `backdoor_ate` + refutations for rung-2-conditional ATEs

Stages (run all by default; select with --stages):
  paired_g  — per-(env, burst) Hedges' g for g_link (Δoutcome) and
              g_mech (Δmediator); cell means; within-pair r table.
  lag       — cross-burst lag correlation diagnostic (τ ∈ [-3..+3]).
  meta_reg  — random-effects meta-regression of g_link and g_mech on
              env-feature covariates; confound-deconfounding sets.
  pc        — PC adjacency on the panel (with optional JCI via
              stratify_by=env_name).
  dowhy     — DoWhy backdoor + placebo + RCC on the strongest residual
              edge from PC.

Cross-corpus 2×2 factorial via repeated --pair:

  uv run python experiments/analyze.py \\
    --pair vanilla_3step nstep_vanilla_arms vanilla_1step nstep_vanilla_arms \\
    --pair ddqn_3step    nstep_intervention   ddqn_1step    nstep_intervention \\
    --pair ddqn_1step    nstep_intervention   vanilla_1step nstep_vanilla_arms \\
    --pair ddqn_3step    nstep_intervention   vanilla_3step nstep_vanilla_arms

Each --pair is `treatment_arm treatment_corpus baseline_arm baseline_corpus`.
Seed-aligned across corpora so paired Hedges' g is admissible.

Single-corpus default usage:

  uv run python experiments/analyze.py --corpus ddqn \\
    --treatment-arm ddqn --baseline-arm vanilla_dqn

Output: stdout summary + (optional) structured parquet under
`<corpus>/analysis/`. The latter is a follow-up; for now stdout
matches what the consolidated pipeline produces.
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import scipy.stats as ss

from corroborate.analyses.dowhy import (
    backdoor_ate as backdoor_ate_analysis,
    placebo_refutation as placebo_refutation_analysis,
    random_common_cause_refutation as random_common_cause_refutation_analysis,
)
from corroborate.graph.discovery import discover_adjacency
from corroborate.stats import (
    StratumObservation, meta_regression,
)
from corroborate_rl.env_catalogue import get as _get_env_spec
from corroborate.stats import hedges_g_paired


# ============ Pair specification ============

@dataclass(frozen=True, slots=True)
class Pair:
    """One paired comparison: (treatment_arm, treatment_corpus,
    baseline_arm, baseline_corpus). Seeds align across corpora when
    consistent HPs were used; pairing fails (yields no pairs) on
    seed mismatch."""
    name: str
    treatment_arm: str
    treatment_corpus: str
    baseline_arm: str
    baseline_corpus: str


# ============ Primitive: per-seed (bias, return) arrays ============

def _per_seed_arrays(
    corpus: str, env: str, intervention: str, total_steps: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Per-seed arrays at native (n_bursts,) shape: (bias, return).
    Reads `predicted_q_at_start - mc_return` for bias; `mc_return`
    for return. Both shaped `(n_bursts, K)` and burst-mean'd."""
    base = Path('experiments/data') / corpus
    if not (base / 'runs.parquet').exists():
        return {}, {}
    runs = pl.read_parquet(str(base / 'runs.parquet')).filter(
        (pl.col('env_name') == env)
        & (pl.col('intervention_name') == intervention)
        & (pl.col('total_steps') == total_steps)
    )
    if runs.height == 0:
        return {}, {}
    rows = runs.select(['id', 'seed']).to_dicts()
    ids = [r['id'] for r in rows]
    traces = pl.read_parquet(
        str(base / 'traces.parquet'),
        columns=['id', 'predicted_q_at_start', 'mc_return'],
    ).filter(pl.col('id').is_in(ids))
    bias_by_id: dict[str, np.ndarray] = {}
    ret_by_id: dict[str, np.ndarray] = {}
    for row in traces.iter_rows(named=True):
        pred = np.asarray(row['predicted_q_at_start'], dtype=np.float64)
        actual = np.asarray(row['mc_return'], dtype=np.float64)
        if pred.ndim != 2 or actual.ndim != 2 or pred.shape != actual.shape:
            continue
        bias_by_id[row['id']] = (pred - actual).mean(axis=-1)
        ret_by_id[row['id']] = actual.mean(axis=-1)
    seed_to_id = {r['seed']: r['id'] for r in rows}
    bias = {s: bias_by_id[i] for s, i in seed_to_id.items() if i in bias_by_id}
    ret = {s: ret_by_id[i] for s, i in seed_to_id.items() if i in ret_by_id}
    return bias, ret


def _paired_arrays(
    pair: Pair, env: str, total_steps: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Δbias, Δret as (n_pairs, n_bursts) arrays for one (pair, env).
    Returns None if fewer than 4 seeds align across corpora."""
    t_bias, t_ret = _per_seed_arrays(
        pair.treatment_corpus, env, pair.treatment_arm, total_steps,
    )
    b_bias, b_ret = _per_seed_arrays(
        pair.baseline_corpus, env, pair.baseline_arm, total_steps,
    )
    common = sorted(set(t_bias) & set(b_bias))
    if len(common) < 4:
        return None
    delta_bias = np.stack([t_bias[s] - b_bias[s] for s in common])
    delta_ret = np.stack([t_ret[s] - b_ret[s] for s in common])
    return delta_bias, delta_ret


# ============ Env-feature panel ============

def _env_features(env: str) -> dict[str, float]:
    """Standard env-feature covariates for meta-regression / PC."""
    spec = _get_env_spec(env)
    n_a = spec.n_actions
    obs_n = 1
    for d in spec.observation_shape:
        obs_n *= int(d)
    horizon = float(spec.horizon) if spec.horizon else 1000.0
    return {
        'log_action_dim': math.log(max(n_a, 2)),
        'log_obs_dim': math.log(max(obs_n, 1)),
        'log_horizon': math.log(max(horizon, 1.0)),
    }


def _empirical_reward_features(
    corpus: str, env: str,
) -> tuple[float, float] | None:
    """Per-env mean (nonzero_reward_frac, bootstrap_fraction)
    computed inline from the persisted raw trace."""
    base = Path('experiments/data') / corpus
    if not ((base / 'runs.parquet').exists()
            and (base / 'traces.parquet').exists()):
        return None
    runs = pl.read_parquet(str(base / 'runs.parquet')).filter(
        pl.col('env_name') == env
    )
    if runs.height == 0:
        return None
    ids = runs['id'].to_list()
    traces = pl.read_parquet(
        str(base / 'traces.parquet'), columns=['id', 'reward', 'done'],
    ).filter(pl.col('id').is_in(ids))
    nz: list[float] = []
    boot: list[float] = []
    for row in traces.iter_rows(named=True):
        rwd = np.asarray(row['reward'], dtype=np.float64)
        done = np.asarray(row['done'], dtype=np.float64)
        if rwd.size > 0:
            nz.append(float((rwd != 0.0).mean()))
        if done.size > 0:
            boot.append(float(1.0 - done.mean()))
    if not nz or not boot:
        return None
    return float(np.mean(nz)), float(np.mean(boot))


# ============ Stage: paired g panel ============

def _build_panel(
    pairs: tuple[Pair, ...], envs: tuple[str, ...], total_steps: int,
) -> tuple[
    dict[tuple[str, str, int], tuple[float, float, float, float, int]],
    dict[tuple[str, str, int], dict[str, float]],
]:
    """Build (env, pair_name, burst) → (g_link, se_link, g_mech,
    se_mech, n_pairs) and (env, pair_name, burst) → covariates dict.

    n_pairs is the number of paired seeds; the paired Hedges' g uses
    a (seed,) → scalar reduction (mean across burst-K samples) per
    cell, then per-burst over the seeds."""
    panel: dict[tuple[str, str, int], tuple[float, float, float, float, int]] = {}
    covars: dict[tuple[str, str, int], dict[str, float]] = {}
    for env in envs:
        env_feat = _env_features(env)
        for pair in pairs:
            arr = _paired_arrays(pair, env, total_steps)
            if arr is None:
                continue
            delta_bias, delta_ret = arr
            n_pairs, n_bursts = delta_ret.shape
            # Per-corpus empirical reward features (use treatment
            # corpus as the source; for cross-corpus pairs they should
            # match in HP regime).
            emp = _empirical_reward_features(pair.treatment_corpus, env)
            if emp is None:
                emp_nz, emp_boot = float('nan'), float('nan')
            else:
                emp_nz, emp_boot = emp
            for b in range(n_bursts):
                dr = delta_ret[:, b].astype(float).tolist()
                db = delta_bias[:, b].astype(float).tolist()
                gl, sl = hedges_g_paired(dr)
                gm, sm = hedges_g_paired(db)
                key = (env, pair.name, b)
                panel[key] = (
                    float(gl), float(sl), float(gm), float(sm), n_pairs,
                )
                covars[key] = {
                    **env_feat,
                    'empirical_reward_density': emp_nz,
                    'bootstrap_fraction': emp_boot,
                    'burst_index': float(b),
                    'mean_dbias': float(delta_bias[:, b].mean()),
                }
    return panel, covars


def _print_paired_g(
    panel: Mapping[tuple[str, str, int], tuple[float, float, float, float, int]],
    pairs: tuple[Pair, ...], envs: tuple[str, ...], n_bursts: int,
) -> None:
    """Per-(env, pair) header + per-burst g_link table; per-(env, pair)
    cell mc_return + bias means."""
    print('=' * 110)
    print('Per-(env, pair, burst) Hedges\' g — g_link is on Δoutcome, g_mech on Δbias')
    print('=' * 110)
    for pair in pairs:
        any_for_pair = any(
            (env, pair.name, b) in panel for env in envs for b in range(n_bursts)
        )
        if not any_for_pair:
            continue
        print()
        print(f'pair: {pair.name}  '
              f'({pair.treatment_arm}@{pair.treatment_corpus} − '
              f'{pair.baseline_arm}@{pair.baseline_corpus})')
        print(f'  {"env":<26}  ' + ' '.join(f'b{b:<3}' for b in range(n_bursts))
              + f'  {"|n|":>4}')
        print('  ' + '-' * 100)
        for env in envs:
            cells = [
                panel.get((env, pair.name, b)) for b in range(n_bursts)
            ]
            if not any(c is not None for c in cells):
                continue
            row_link: list[str] = []
            n_obs = 0
            for c in cells:
                if c is None:
                    row_link.append('  nan')
                else:
                    gl, _sl, _gm, _sm, n = c
                    n_obs = n
                    row_link.append(
                        f'{gl:>+5.2f}'
                        if math.isfinite(gl) else '  nan'
                    )
            print(f'  {env:<26}  ' + ' '.join(row_link) + f'  {n_obs:>4}')


# ============ Stage: cross-burst lag ============

def _print_lag_correlation(
    pairs: tuple[Pair, ...], envs: tuple[str, ...], total_steps: int,
    taus: tuple[int, ...] = (-3, -2, -1, 0, 1, 2, 3),
) -> None:
    """For each env+pair, Pearson r between Δbias[k] and Δret[k+τ]
    pooled across (seed, k) — diagnostic for causal precedence."""
    print()
    print('=' * 110)
    print('Cross-burst lag correlation r(Δbias[k], Δret[k+τ]) — forward asymmetry = causal')
    print('=' * 110)
    for pair in pairs:
        any_data = False
        rows: list[tuple[str, list[str]]] = []
        for env in envs:
            arr = _paired_arrays(pair, env, total_steps)
            if arr is None:
                continue
            any_data = True
            delta_bias, delta_ret = arr
            n_seeds, n_bursts = delta_ret.shape
            cells: list[str] = []
            for tau in taus:
                xs: list[float] = []
                ys: list[float] = []
                for k in range(n_bursts):
                    k2 = k + tau
                    if k2 < 0 or k2 >= n_bursts:
                        continue
                    xs.extend(delta_bias[:, k].astype(float).tolist())
                    ys.extend(delta_ret[:, k2].astype(float).tolist())
                if len(xs) < 4 or float(np.std(xs)) == 0 or float(np.std(ys)) == 0:
                    cells.append('   nan')
                    continue
                r = ss.pearsonr(np.asarray(xs), np.asarray(ys))
                cells.append(f'{float(r.statistic):>+5.2f}')
            rows.append((env, cells))
        if not any_data:
            continue
        print()
        print(f'pair: {pair.name}')
        print(f'  {"env":<26}  ' + ' '.join(f'τ={t:>+2}' for t in taus))
        print('  ' + '-' * 100)
        for env, cells in rows:
            print(f'  {env:<26}  ' + ' '.join(f'{c:>5}' for c in cells))


# ============ Stage: meta-regression ============

_DEFAULT_COVARIATE_SETS: tuple[tuple[str, ...], ...] = (
    ('log_action_dim',),
    ('bootstrap_fraction',),
    ('log_horizon',),
    ('log_obs_dim',),
    ('empirical_reward_density',),
    ('mean_dbias',),
    ('log_action_dim', 'log_obs_dim', 'log_horizon',
     'empirical_reward_density', 'bootstrap_fraction'),
    ('log_action_dim', 'log_obs_dim', 'log_horizon',
     'empirical_reward_density', 'bootstrap_fraction',
     'mean_dbias', 'burst_index'),
)


def _print_meta_regression(
    panel: Mapping[tuple[str, str, int], tuple[float, float, float, float, int]],
    covars: Mapping[tuple[str, str, int], dict[str, float]],
    pairs: tuple[Pair, ...], target: str = 'g_link',
) -> None:
    """Random-effects meta-regression on env-feature covariates;
    one regression set per pair × covariate menu. `target` ∈
    {'g_link', 'g_mech'}."""
    print()
    print('=' * 110)
    print(f'Meta-regression on {target} (random-effects DerSimonian-Laird)')
    print('=' * 110)
    g_idx = 0 if target == 'g_link' else 2
    se_idx = 1 if target == 'g_link' else 3
    for pair in pairs:
        obs: list[StratumObservation] = []
        for (env, pname, b), stats in panel.items():
            if pname != pair.name:
                continue
            g, _, gm, _, _ = stats
            se = stats[se_idx]
            g_val = stats[g_idx]
            if not (math.isfinite(g_val) and math.isfinite(se) and se > 0):
                continue
            obs.append(StratumObservation(
                stratum_id=(env, b), g=g_val, se=se,
                covariates=dict(covars[(env, pname, b)]),
            ))
        if not obs:
            continue
        print()
        print(f'pair: {pair.name}  (n_strata={len(obs)})')
        for cset in _DEFAULT_COVARIATE_SETS:
            try:
                sub = [
                    StratumObservation(
                        stratum_id=o.stratum_id, g=o.g, se=o.se,
                        covariates={c: o.covariates[c] for c in cset},
                    )
                    for o in obs
                ]
                res = meta_regression(sub)
            except (ValueError, KeyError):
                continue
            label = ' + '.join(cset)
            print(f'  --- {target} ~ {label} ---')
            print(f'    n={res.n_strata} R²={res.r_squared:+.3f} '
                  f'intercept={res.intercept:+.3f}')
            for c in res.coefficients:
                sig = '✓' if c.is_significant else ' '
                print(f'    {c.name:<26} β={c.coefficient:+.4f}  '
                      f'CI=[{c.ci_lo:+.4f}, {c.ci_hi:+.4f}]  '
                      f'p={c.p_value:.4f}  {sig}')


# ============ Stage: PC adjacency ============

def _print_pc_adjacency(
    panel: Mapping[tuple[str, str, int], tuple[float, float, float, float, int]],
    covars: Mapping[tuple[str, str, int], dict[str, float]],
    pairs: tuple[Pair, ...],
) -> None:
    """PC adjacency on the (g_link, g_mech, ...covariates...) panel
    per pair, plus JCI variant stratify_by=env."""
    print()
    print('=' * 110)
    print('PC adjacency on per-(env, burst) panel')
    print('=' * 110)
    for pair in pairs:
        rows: list[dict[str, float | str]] = []
        for (env, pname, b), stats in panel.items():
            if pname != pair.name:
                continue
            g_link, _, g_mech, _, _ = stats
            if not (math.isfinite(g_link) and math.isfinite(g_mech)):
                continue
            row: dict[str, float | str] = {
                'g_link': g_link, 'g_mech': g_mech, 'env_name': env,
                **{k: v for k, v in covars[(env, pname, b)].items()
                   if math.isfinite(v)},
            }
            rows.append(row)
        if not rows:
            continue
        df = pl.DataFrame(rows)
        if df.height < 8:
            continue
        variables = tuple(c for c in df.columns if c != 'env_name')
        print()
        print(f'pair: {pair.name}  (n_strata={df.height}, vars={len(variables)})')
        for label, kwargs in [
            ('cross-env (no JCI)', {}),
            ('within-env (JCI, stratify_by=env_name)',
             {'stratify_by': 'env_name'}),
        ]:
            try:
                adj = discover_adjacency(
                    df, variables=variables,
                    alpha=0.05, max_conditioning=2, **kwargs,  # type: ignore[arg-type]
                )
            except (ValueError, KeyError) as e:
                print(f'  {label}: skipped ({e})')
                continue
            print(f'  {label}: {len(adj.edges)} surviving edges')
            for edge in sorted(adj.edges, key=lambda e: tuple(sorted(e))):
                a, b = sorted(edge)
                print(f'    {a:<28} ⟷ {b}')


# ============ Stage: DoWhy backdoor ============

def _print_dowhy_backdoor(
    panel: Mapping[tuple[str, str, int], tuple[float, float, float, float, int]],
    covars: Mapping[tuple[str, str, int], dict[str, float]],
    pairs: tuple[Pair, ...],
) -> None:
    """DoWhy backdoor + placebo + RCC on the strongest residual edge
    for g_link from PC. Hard-codes treatment='bootstrap_fraction'
    (the leading moderator from the 200k corpus); future iterations
    parameterise this."""
    print()
    print('=' * 110)
    print('DoWhy backdoor on bootstrap_fraction → g_link | g_mech')
    print('=' * 110)
    for pair in pairs:
        rows: list[dict[str, float]] = []
        for (env, pname, b), stats in panel.items():
            if pname != pair.name:
                continue
            g_link, _, g_mech, _, _ = stats
            if not (math.isfinite(g_link) and math.isfinite(g_mech)):
                continue
            row = {
                'g_link': g_link, 'g_mech': g_mech,
                **{k: v for k, v in covars[(env, pname, b)].items()
                   if math.isfinite(v)},
            }
            rows.append(row)
        if len(rows) < 8:
            continue
        keys = ('bootstrap_fraction', 'g_mech', 'g_link')
        if not all(k in rows[0] for k in keys):
            continue
        dag = [('g_mech', 'bootstrap_fraction'),
               ('g_mech', 'g_link'),
               ('bootstrap_fraction', 'g_link')]
        # `analyses/dowhy.py` `@analysis` versions consume cells
        # directly (each cell a Mapping[str, object]); rows fits the
        # shape verbatim. The framework no longer wraps these in
        # per-record Bridges (Phase 4F deleted the per-record Bridge[R]
        # channel); call the analysis bodies directly and decide the
        # verdict here.
        cells: list[Mapping[str, object]] = list(rows)
        backdoor_r = backdoor_ate_analysis.fn(
            cells=cells,
            treatment='bootstrap_fraction',
            outcome='g_link',
            dag=dag,
        )
        placebo_r = placebo_refutation_analysis.fn(
            cells=cells,
            treatment='bootstrap_fraction',
            outcome='g_link',
            dag=dag,
        )
        rcc_r = random_common_cause_refutation_analysis.fn(
            cells=cells,
            treatment='bootstrap_fraction',
            outcome='g_link',
            dag=dag,
        )
        print()
        print(f'pair: {pair.name}  (n={len(rows)})')
        ate_held = (
            backdoor_r.identified
            and abs(backdoor_r.ate) >= 0.05
            and backdoor_r.ate > 0  # expected_sign=+1
        )
        print(
            f'  backdoor_ate           '
            f'verdict={"held" if ate_held else "no_effect":<22} '
            f'ATE={backdoor_r.ate:+.4f}',
        )
        placebo_held = placebo_r.drift <= 0.1
        print(
            f'  placebo                '
            f'verdict={"held" if placebo_held else "no_effect":<22} '
            f'placebo={placebo_r.refuted_ate:+.4f} '
            f'real={placebo_r.real_ate:+.4f}',
        )
        rcc_held = rcc_r.drift <= 0.1
        print(
            f'  random_common_cause    '
            f'verdict={"held" if rcc_held else "no_effect":<22} '
            f'drift={rcc_r.drift:.4f} real={rcc_r.real_ate:+.4f}',
        )


# ============ CLI ============

_ALL_STAGES: tuple[str, ...] = ('paired_g', 'lag', 'meta_reg', 'pc', 'dowhy')


def _resolve_pairs(args: argparse.Namespace) -> tuple[Pair, ...]:
    """Either a single pair from --treatment-arm/--baseline-arm
    on --corpus, or N pairs from --pair (treatment, treatment_corpus,
    baseline, baseline_corpus) repeated."""
    if getattr(args, 'pair', None):
        out: list[Pair] = []
        for tup in args.pair:
            if len(tup) != 4:
                raise SystemExit(
                    f'--pair takes 4 values (treatment_arm '
                    f'treatment_corpus baseline_arm baseline_corpus); '
                    f'got {len(tup)}: {tup}'
                )
            t_arm, t_cor, b_arm, b_cor = tup
            out.append(Pair(
                name=f'{t_arm}_vs_{b_arm}',
                treatment_arm=t_arm, treatment_corpus=t_cor,
                baseline_arm=b_arm, baseline_corpus=b_cor,
            ))
        return tuple(out)
    if not (args.corpus and args.treatment_arm and args.baseline_arm):
        raise SystemExit(
            'either --corpus + --treatment-arm + --baseline-arm OR '
            '--pair (repeatable, 4 values each) must be supplied'
        )
    return (Pair(
        name=f'{args.treatment_arm}_vs_{args.baseline_arm}',
        treatment_arm=args.treatment_arm,
        treatment_corpus=args.corpus,
        baseline_arm=args.baseline_arm,
        baseline_corpus=args.corpus,
    ),)


def _resolve_envs(pairs: tuple[Pair, ...], total_steps: int) -> tuple[str, ...]:
    """Envs present in BOTH treatment and baseline corpora of every
    requested pair."""
    env_sets: list[set[str]] = []
    for p in pairs:
        for corpus in (p.treatment_corpus, p.baseline_corpus):
            base = Path('experiments/data') / corpus / 'runs.parquet'
            if not base.exists():
                env_sets.append(set())
                continue
            df = pl.read_parquet(str(base)).filter(
                pl.col('total_steps') == total_steps,
            )
            env_sets.append(set(df['env_name'].unique().to_list()))
    if not env_sets:
        return ()
    common = set.intersection(*env_sets)
    return tuple(sorted(common))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus')
    parser.add_argument('--treatment-arm')
    parser.add_argument('--baseline-arm')
    parser.add_argument(
        '--pair', nargs=4, action='append',
        metavar=('TARM', 'TCORPUS', 'BARM', 'BCORPUS'),
        help='cross-corpus pair; repeat for 2×2 factorial',
    )
    parser.add_argument('--total-steps', type=int, default=200_000)
    parser.add_argument(
        '--stages', default=','.join(_ALL_STAGES),
        help=f'comma-separated subset of {_ALL_STAGES}',
    )
    args = parser.parse_args()
    total_steps: int = int(args.total_steps)
    stages_raw: str = str(args.stages)

    pairs = _resolve_pairs(args)
    envs = _resolve_envs(pairs, total_steps)
    if not envs:
        raise SystemExit('no envs in common across the requested corpora')

    stages = tuple(s.strip() for s in stages_raw.split(',') if s.strip())
    print('=' * 110)
    print(f'analyze.py — total_steps={total_steps}, '
          f'envs={len(envs)}, pairs={len(pairs)}, '
          f'stages={stages}')
    for p in pairs:
        print(f'  {p.name}: {p.treatment_arm}@{p.treatment_corpus} − '
              f'{p.baseline_arm}@{p.baseline_corpus}')
    print('=' * 110)

    panel, covars = _build_panel(pairs, envs, total_steps)
    n_bursts = max((b for _, _, b in panel), default=-1) + 1

    if 'paired_g' in stages:
        _print_paired_g(panel, pairs, envs, n_bursts)
    if 'lag' in stages:
        _print_lag_correlation(pairs, envs, total_steps)
    if 'meta_reg' in stages:
        _print_meta_regression(panel, covars, pairs, target='g_link')
        _print_meta_regression(panel, covars, pairs, target='g_mech')
    if 'pc' in stages:
        _print_pc_adjacency(panel, covars, pairs)
    if 'dowhy' in stages:
        _print_dowhy_backdoor(panel, covars, pairs)


if __name__ == '__main__':
    main()
