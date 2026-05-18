"""Tautology audit on the canonical ddqn corpus's full mediator panel.

The panel includes the candidate mediators we've been chasing
(target_staleness, eff_h, bf, q_max_growth, etc.) plus the
suspected outcome-tautological candidates (env_reward_polarity,
learning_curve_auc, return_at_25pct_steps, q_mc_calibration_pearson)
plus the mech step (jensen_gap).

Three audit checks per measurable:
  1. Structural jaccard with outcome reads
  2. HP-R²: deterministic on any HP axis (here: total_steps, env_idx)
  3. Stratified ρ: marginal correlation with outcome within env strata.
     If |ρ| < 0.1 and p > 0.05 → HP-SHADOW (correlation is env-mediated).

Stratifying by env_idx tests within-env signal — if a "mediator"
shows a strong marginal correlation with outcome only because envs
differ systematically (and the within-env relationship is null or
sign-flipping), it's flagged as HP-SHADOW.

Predicted outcomes:
  - eff_h, bf: HP-SHADOW (within-env r flips sign by polarity, pools to ~0)
  - target_staleness_late: CLEAN structurally; stratified ρ marginal
    (within-env r is polarity-orthogonal, mixed signs across envs)
  - jensen_gap: OUTCOME (jaccard=0.5 with mc_return)
  - learning_curve_auc, return_at_25pct_steps: OUTCOME (jaccard=1.0)
  - q_mc_calibration_pearson, env_reward_polarity: OUTCOME (jaccard=0.5)
  - q_max_growth, v_vs_max_delta_late, td_residual_late, greedy_match_late:
    CLEAN structurally; stratified ρ depends on within-env signal
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
from pathlib import Path

import polars as pl
from corroborate_rl.dqn import measurables as _dqn_measurables  # noqa: F401
from corroborate.analyses.diagnostic.tautology_audit import tautology_audit as _ta
from corroborate.measurables import get_registered

tautology_audit = _ta.fn

CORPUS_DIR = Path('experiments/data/ddqn')
DDQN = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'

OUTCOME = 'eval_best_burst_mean'
OUTCOME_READS = ('mc_return',)

CANDIDATES = [
    # core mediators of interest
    'target_staleness_late',
    'target_staleness_early',
    'effective_horizon',
    'bootstrap_fraction',
    'jensen_gap',
    # polarity itself, as a candidate
    'env_reward_polarity',
    # outcome-tautological candidates (negative controls)
    'learning_curve_auc',
    'return_at_25pct_steps',
    'q_mc_calibration_pearson',
    # additional mechanism candidates
    'q_max_growth',
    'v_vs_max_delta_late',
    'td_residual_late',
    'greedy_match_late',
]


def main() -> None:
    runs = pl.read_parquet(CORPUS_DIR / 'runs.parquet', columns=[
        'id', 'env_name', 'arm_key', 'seed', 'total_steps',
        'gamma', 'sync_period', 'timestamp', 'verdict',
    ])
    ms = pl.read_parquet(CORPUS_DIR / 'measurements.parquet')
    df = runs.join(ms, on='id', how='inner')
    print(f'ddqn corpus: {len(df)} cells across {df["env_name"].n_unique()} envs', flush=True)

    # Encode env_name as integer index for HP-stratification
    # (audit's HP-axis check requires numeric values).
    env_names = sorted(df['env_name'].unique())
    env_to_idx = {n: i for i, n in enumerate(env_names)}
    df = df.with_columns(pl.col('env_name').replace_strict(env_to_idx, return_dtype=pl.Int64).alias('env_idx'))

    # Build mediator specs
    measurable_specs = []
    bare_paths = {}
    for name in CANDIDATES:
        m = get_registered(name)
        if m is None:
            print(f'  SKIP {name!r}: not registered', flush=True)
            continue
        reads = tuple(m.reads) if hasattr(m, 'reads') else ()
        measurable_specs.append({'name': name, 'reads': reads})
        bare_paths[name] = name  # cells expose at bare names

    # Use both arms for the audit (DDQN+baseline) — we're testing the
    # measurable's structural / HP-shadow properties, not arm contrast.
    cells = df.filter(pl.col('arm_key').is_in(['baseline', DDQN])).to_dicts()
    print(f'cells (baseline+DDQN): {len(cells)}', flush=True)

    print()
    print('Running tautology_audit (env_idx as stratum)...', flush=True)
    result = tautology_audit(
        cells=cells,
        measurables=measurable_specs,
        outcome_path=OUTCOME,
        outcome_reads=OUTCOME_READS,
        hp_axes=('total_steps', 'env_idx'),
        hp_stratum_axis='env_idx',  # stratify within env
        mediator_path_for=bare_paths,
        outcome_jaccard_threshold=0.5,
        hp_r_squared_threshold=0.95,
        stratified_rho_threshold=0.1,
        stratified_alpha=0.05,
    )

    print()
    print('=== Tautology audit results (ddqn corpus, env-stratified) ===\n')
    hdr = ('mediator', 'jaccard', 'hp_r²(env)', 'hp_r²(steps)', 'strat_ρ', 'strat_p', 'tags')
    fmt = '{:<28} {:>9} {:>11} {:>13} {:>9} {:>10} {:>30}'
    print(fmt.format(*hdr))
    print('-' * 120)

    summary = []
    for r in result.reports:
        jac = getattr(r, 'outcome_jaccard', float('nan'))
        hp_rs = getattr(r, 'hp_r_squared', {})
        env_r2 = hp_rs.get('env_idx', float('nan')) if isinstance(hp_rs, dict) else float('nan')
        ts_r2 = hp_rs.get('total_steps', float('nan')) if isinstance(hp_rs, dict) else float('nan')
        strat_rho = getattr(r, 'outcome_stratified_rho', float('nan'))
        strat_p = getattr(r, 'outcome_stratified_p', float('nan'))

        tag_list = []
        if getattr(r, 'flagged_outcome', False):
            tag_list.append('OUTCOME')
        flagged_hp = getattr(r, 'flagged_hp', ())
        if flagged_hp:
            tag_list.append(f'HP({"+".join(str(a) for a in flagged_hp)})')
        if getattr(r, 'flagged_no_residual_signal', False):
            tag_list.append('SHADOW')
        tags_str = '+'.join(tag_list) or 'CLEAN'

        print(fmt.format(
            r.measurable_name,
            f'{jac:.3f}' if isinstance(jac, float) else str(jac),
            f'{env_r2:.3f}' if not math.isnan(env_r2) else 'nan',
            f'{ts_r2:.3f}' if not math.isnan(ts_r2) else 'nan',
            f'{strat_rho:+.3f}' if not math.isnan(strat_rho) else 'nan',
            f'{strat_p:.3g}' if not math.isnan(strat_p) else 'nan',
            tags_str,
        ), flush=True)
        summary.append({
            'mediator': r.measurable_name,
            'jaccard': jac, 'env_r2': env_r2, 'total_steps_r2': ts_r2,
            'stratified_rho': strat_rho, 'stratified_p': strat_p,
            'tags': tag_list, 'is_clean': r.is_clean,
        })

    print()
    print(f'CLEAN mediators: {result.clean_names}', flush=True)

    out = Path('experiments/findings/sync_curve_breakout/tautology_audit_ddqn_corpus.json')
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
