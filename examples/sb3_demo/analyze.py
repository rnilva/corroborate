"""Analyse the SB3 runs — corroborate's side of the boundary.

Run from the repo root, **after** `train.py` has produced
`examples/sb3_demo/runs/`. No SB3 or torch needed here:

    uv run python examples/sb3_demo/analyze.py

Three steps: load the producer's files into a DataFrame, explore
it with plain polars, then evaluate the executable claim test in
``sb3_claim.py`` against it.

The record is live: run more seeds, re-load, and the same claim
test recomputes — a verdict that moves with the evidence is the
system working. (Already have your own logs? Skip the loader:
``pl.read_csv(...)`` and go straight to evaluating.)
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from corroborate.analyses.paired.paired_directional import (
    PairedDirectionalResult,
)
from corroborate.bridge.bridge import evaluate
from corroborate.data import config_columns, load_runs
from sb3_claim import higher_gamma_improves_return

RUNS = Path(__file__).parent / 'runs'

# ── 1. load: producer files in, one row per run out ─────────────
df = load_runs(RUNS)
print(f'loaded: {df.height} runs × {df.width} columns')
print(df.select('id', 'seed', 'gamma', 'return_mean').sort('gamma', 'seed'))

# ── 2. explore, in plain polars ─────────────────────────────────
# The loader derives one `return_mean_at_<step>` column per
# evaluation checkpoint, so the trajectory — not just the final
# mean — is explorable.
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
# side registers which columns were configuration (derived from
# the run directory's config files — the producer's own record of
# what was assigned). Conditions derive from the gamma column's
# scoped values, and the admission gates check contrast presence,
# isolation, and pair completeness over exactly the cells the
# claim admits.
evaluation = evaluate(
    higher_gamma_improves_return, df, leaves=config_columns(RUNS),
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
