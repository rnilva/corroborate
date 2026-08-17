"""Analyse the SB3 runs — corroborate's side of the boundary.

Run from the repo root, **after** `train.py` has produced
`examples/sb3_demo/runs/`. Reading SB3's artifacts needs the
DQN constructor's signature (that is where the configuration
registry comes from), so run with stable-baselines3 available:

    uv run --with 'stable-baselines3>=2.3' examples/sb3_demo/analyze.py

Three steps: read SB3's own artifacts (checkpoint zips +
EvalCallback logs) into a DataFrame, explore it with plain
polars, then evaluate the executable claim test in
``sb3_claim.py`` against it.

The record is live: run more seeds, re-load, and the same claim
test recomputes — a verdict that moves with the evidence is the
system working. (Runs logged your own way? A directory of plain
JSON records loads via ``corroborate.data.load_runs``, and any
DataFrame evaluates directly.)
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
from stable_baselines3 import DQN

from corroborate.analyses.paired.paired_directional import (
    PairedDirectionalResult,
)
from corroborate.bridge.bridge import evaluate
from corroborate_rl.sb3 import load_sb3_runs, sb3_config_columns
from sb3_claim import higher_gamma_improves_return

RUNS = Path(__file__).parent / 'runs'

# ── 1. load: SB3's own artifacts in, one row per run out ────────
# Configuration is recovered from each checkpoint's `data` record
# intersected with DQN's constructor signature; evaluations come
# from EvalCallback's evaluations.npz. The checkpoint doesn't
# record which environment it trained on, so the analyst states
# that known context in plain polars.
df = load_sb3_runs(RUNS, DQN).with_columns(
    pl.lit('CartPole-v1').alias('env_id'),
)
print(f'loaded: {df.height} runs × {df.width} columns')
print(df.select('id', 'seed', 'gamma', 'return_mean').sort('gamma', 'seed'))

# ── 2. explore, in plain polars ─────────────────────────────────
# One `return_mean_at_<step>` column per evaluation point makes
# the trajectory — not just the final mean — explorable.
curve_cols = sorted(
    (c for c in df.columns if c.startswith('return_mean_at_')),
    key=lambda c: int(c.rsplit('_', 1)[-1]),
)
print('\nmean return per checkpoint (seeds pooled per condition):')
print(df.group_by('gamma').agg(
    pl.col(c).mean().round(1).alias(c.removeprefix('return_mean_at_'))
    for c in curve_cols
).sort('gamma'))

# The same contrast per seed pair, descriptively.
wide = df.pivot('gamma', index='seed', values='return_mean')
paired = wide.with_columns(
    (pl.col('0.99') - pl.col('0.8')).alias('delta'),
).sort('seed')
print('\nΔ(return_mean) per seed (gamma 0.99 − 0.80):')
print(paired)

# ── 3. evaluate the authored claim test ─────────────────────────
# The claim module owns the estimand and verdict rule; the data
# side registers which columns were configuration (recovered from
# the checkpoints themselves). Conditions derive from the gamma
# column's scoped values, and the admission gates check contrast
# presence, isolation, and pair completeness over exactly the
# cells the claim admits.
evaluation = evaluate(
    higher_gamma_improves_return,
    df,
    leaves=sb3_config_columns(RUNS, DQN),
)
for warning in evaluation.warnings:
    print(f'\nwarning [{warning.gate_name}]: {warning.message}')
result_obj = evaluation.analysis_results.get('paired_directional')
if not isinstance(result_obj, PairedDirectionalResult):
    raise RuntimeError('claim did not produce paired_directional evidence')
result = result_obj
print(f'\nclaim: gamma 0.99 > gamma 0.80 on {result.measurable}')
print(f'  n_pairs={result.n_pairs}  mean_diff={result.mean_diff:+.1f} '
      f'(CI {result.mean_diff_ci[0]:+.1f}..{result.mean_diff_ci[1]:+.1f})')
print(f'  dz={result.dz:+.2f}  p={result.p_value:.4f}')
print(f'  verdict: {evaluation.verdict.name}'
      + (
          f' ({evaluation.refutation_class.name})'
          if evaluation.refutation_class is not None else ''
      ))
