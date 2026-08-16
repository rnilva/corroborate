"""Analyse the SB3 bundle — corroborate's side of the boundary.

Run from the repo root, **after** `train.py` has produced
`examples/sb3_demo/bundle/`. No SB3 or torch needed here:

    uv run python examples/sb3_demo/analyze.py

Four steps: seal, adapt (verify + normalise, receipt printed),
explore the run set as a Panel (trajectory + descriptive probe),
then evaluate the executable claim test in ``sb3_claim.py``.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from corroborate.analyses.paired.paired_directional import (
    PairedDirectionalResult,
)
from corroborate.analyses.paired.paired_g import paired_g
from corroborate.bridge.bridge import evaluate
from corroborate.data import adapt_study, seal_bundle
from sb3_claim import higher_gamma_improves_return

BUNDLE = Path(__file__).parent / 'bundle'

# ── 1. seal: content-address the record ─────────────────────────
if not (BUNDLE / 'manifest.json').exists():
    seal_bundle(BUNDLE)
    print(f'sealed: {BUNDLE / "manifest.json"}')

# ── 2. adapt: verify + normalise, fail-closed ───────────────────
study = adapt_study(BUNDLE)
print(f'\nadmissible: {study.receipt.admissible}')
for check in study.receipt.checks:
    print(f'  [{check.status.name:12s}] {check.code}: {check.message}')

# ── 3. explore the run set as a Panel ───────────────────────────
# `panel.cells` is a polars DataFrame; registered analyses accept
# it directly. This is descriptive exploration, not the authored
# claim test.
panel = study.to_panel()
print(f'\npanel: {panel.cells.height} seeded runs × '
      f'{panel.cells.width} columns')
print(panel.cells.select(
    'id', 'arm_key', 'seed', 'gamma', 'return_mean',
).sort('arm_key', 'seed'))

# How the contrast evolves over training: the adapter derives one
# `return_mean_at_<step>` column per evaluation checkpoint, so the
# trajectory — not just the final mean — is explorable.
curve_cols = sorted(
    (
        c for c in panel.cells.columns
        if c.startswith('return_mean_at_')
        and c.rsplit('_', 1)[-1].isdigit()
    ),
    key=lambda c: int(c.rsplit('_', 1)[-1]),
)
print('\nmean return per checkpoint (seeds pooled per condition):')
print(panel.cells.group_by('arm_key').agg(
    pl.col(c).mean().round(1).alias(c.removeprefix('return_mean_at_'))
    for c in curve_cols
).sort('arm_key'))

# A descriptive paired probe of the same contrast: direction and
# magnitude while looking around.
probe = paired_g(
    panel.cells,
    source='return_mean',
    treatment_arm=study.contrast.treatment_key,
    baseline_arm=study.contrast.baseline_key,
    pair_by=('seed',),
)
print(f'\nprobe: Δ(return_mean) = {probe.mean_diff:+.1f} '
      f'± {probe.mean_diff_se:.1f}  g={probe.g:+.2f}  '
      f'pairs helped: {probe.helped_fraction:.0%} of {probe.n_pairs}')

# ── 4. evaluate the authored claim test ─────────────────────────
# The claim module owns the edge, scope, predicted direction,
# statistical configuration, and verdict rule. The verified record
# supplies only its producer-specific arm labels at evaluation time.
evaluation = evaluate(
    higher_gamma_improves_return,
    panel.cells,
    recorded_contrast=study.contrast,
)
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
