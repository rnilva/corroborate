"""Analyse the SB3 runs — corroborate's side of the boundary.

Run from the repo root, **after** `train.py` has produced
`examples/sb3_demo/runs/`. The reader consumes SB3 Logger output
directly (`progress.json`), so nothing beyond corroborate itself
is needed:

    uv run examples/sb3_demo/analyze.py

Three steps: read the Logger's own files into a Panel, explore
its rows with plain polars, then evaluate the executable claim
test in ``sb3_claim.py`` against the Panel.

The record is live: run more seeds, re-load, and the same claim
test recomputes — a verdict that moves with the evidence is the
system working.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from corroborate.analyses.paired.paired_directional import (
    PairedDirectionalResult,
)
from corroborate.bridge import evaluate
from corroborate_rl.sb3_logger import load_sb3_logger_runs
from sb3_claim import higher_gamma_improves_return

RUNS = Path(__file__).parent / 'runs'

# ── 1. load: the Logger's own files in, one Panel out ───────────
# One row per progress file. `config/*` keys — the producer's own
# statement of what was configured — become scalar columns and the
# registry; every other key stays a record: an array column with
# one slot per dump, aligned by `time/total_timesteps`. Nothing is
# renamed, reduced, or classified at load time.
panel = load_sb3_logger_runs(RUNS)
df = panel.df
print(f'loaded: {df.height} runs × {df.width} columns')
print(f'registry: {", ".join(sorted(panel.leaves or ()))}')

# ── 2. explore, in plain polars ─────────────────────────────────
# The record arrays explode into an ordinary long frame; the eval
# trajectory — not just any single point — is explorable directly.
curve = (
    df.select('config/gamma', 'time/total_timesteps', 'eval/mean_reward')
    .filter(pl.col('eval/mean_reward').is_not_null())
    .explode('time/total_timesteps', 'eval/mean_reward')
    .filter(pl.col('eval/mean_reward').is_not_nan())
    .sort('time/total_timesteps')
    .group_by('config/gamma', 'time/total_timesteps', maintain_order=True)
    .agg(pl.col('eval/mean_reward').mean().round(1))
    .pivot(
        'time/total_timesteps',
        index='config/gamma',
        values='eval/mean_reward',
    )
    .sort('config/gamma')
)
print('\nmean eval reward per evaluation point (seeds pooled per condition):')
print(curve)

# The claim's horizon column, materialised here only for the
# exploration pivot below — `evaluate` resolves registered
# measurables per row on its own ("everything is measurables").
panel = panel.with_measurables(('eval_reward_at_25k',))
paired = (
    panel.df
    .pivot('config/gamma', index='config/seed', values='eval_reward_at_25k')
    .with_columns((pl.col('0.99') - pl.col('0.8')).alias('delta'))
    .sort('config/seed')
)
print('\nΔ(eval reward at 25k) per seed (gamma 0.99 − 0.80):')
print(paired)

# ── 3. evaluate the authored claim test ─────────────────────────
# The claim module owns the estimand and verdict rule: the declared
# DoEffect maps gamma=0.80 and 0.99 to symbolic baseline/treatment
# identities — never inferred from observed support. The Panel
# carries which columns were configuration (the `config/*` keys the
# producer recorded); the gates use that registry to verify the
# declared source is a knob and that no other knob moves with the
# contrast inside a seed pair. Assignment itself is the one thing
# no external record can prove.
evaluation = evaluate(higher_gamma_improves_return, panel)
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
