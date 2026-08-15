"""Analyse the SB3 bundle — corroborate's side of the boundary.

Run from the repo root, **after** `train.py` has produced
`examples/sb3_demo/bundle/`. No SB3 or torch needed here:

    uv run python examples/sb3_demo/analyze.py

Four steps: seal, adapt (verify + normalise, receipt printed),
Panel, and a pre-registered directional test of "higher gamma
improves CartPole return".
"""
from __future__ import annotations

from pathlib import Path

from corroborate.analyses.paired.paired_directional import (
    DirectionalDesign, paired_directional, paired_directional_verdict,
)
from corroborate.data import adapt_study, seal_bundle

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

# ── 3. the run set as a Panel ───────────────────────────────────
panel = study.to_panel()
print(f'\npanel: {panel.cells.height} seeded runs × {panel.cells.width} columns')
print(panel.cells.select(
    'id', 'arm_key', 'seed', 'gamma', 'return_mean',
).sort('arm_key', 'seed'))

# ── 4. pre-registered directional claim ─────────────────────────
# The design is declared before looking at outcomes: one-sided
# ("treatment > baseline"), alpha 0.05, smallest effect size of
# interest dz = 0.5, and the number of pairs we planned to train.
design = DirectionalDesign(
    alternative='greater', alpha=0.05, sesoi_dz=0.5,
    minimum_pairs=3, planned_pairs=3,
)
result = paired_directional(
    panel.cells.to_dicts(),
    source='return_mean',
    treatment_arm=study.contrast.treatment_key,
    baseline_arm=study.contrast.baseline_key,
    pair_by=('seed',),
    design=design,
)
print(f'\nclaim: gamma 0.99 > gamma 0.80 on {result.measurable}')
print(f'  n_pairs={result.n_pairs}  mean_diff={result.mean_diff:+.1f} '
      f'(CI {result.mean_diff_ci[0]:+.1f}..{result.mean_diff_ci[1]:+.1f})')
print(f'  dz={result.dz:+.2f}  p={result.p_value:.4f}')
verdict, refutation = paired_directional_verdict(result)
print(f'  verdict: {verdict.name}'
      + (f' ({refutation.name})' if refutation is not None else ''))
