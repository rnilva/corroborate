"""Analyse the SB3 bundle — corroborate's side of the boundary.

Run from the repo root, **after** `train.py` has produced
`examples/sb3_demo/bundle/`. No SB3 or torch needed here:

    uv run python examples/sb3_demo/analyze.py

Four steps: seal, adapt (verify + normalise, receipt printed),
explore the run set as a Panel (trajectory + design-free probe),
then a directional test under a declared design. The receipt
records the claim's register: no prospective protocol is sealed
in this bundle, so the design is admitted retrospectively — the
ordinary, exploratory case.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from corroborate.analyses.paired.paired_directional import (
    DirectionalDesign, paired_directional, paired_directional_verdict,
)
from corroborate.analyses.paired.paired_g import paired_g
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

# ── 3. explore the run set as a Panel ───────────────────────────
# `panel.cells` is a polars DataFrame; registered analyses accept
# it directly. Nothing here requires a committed design — this is
# the exploratory register, and the receipt above says so.
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
    (c for c in panel.cells.columns if c.startswith('return_mean_at_')),
    key=lambda c: int(c.rsplit('_', 1)[-1]),
)
print('\nmean return per checkpoint (seeds pooled per condition):')
print(panel.cells.group_by('arm_key').agg(
    pl.col(c).mean().round(1).alias(c.removeprefix('return_mean_at_'))
    for c in curve_cols
).sort('arm_key'))

# A design-free paired probe of the same contrast: direction and
# magnitude, no design declared while looking around.
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

# ── 4. a directional claim under a declared design ──────────────
# The design is declared here, at analysis time: one-sided
# ("treatment > baseline"), alpha 0.05, smallest effect size of
# interest dz = 0.5, and the number of pairs trained. The receipt
# already recorded the register — no sealed protocol, so this is a
# retrospective claim; sealing a `prospective_protocol` before the
# confirmation run is what earns the VERIFIED prospective mark.
design = DirectionalDesign(
    alternative='greater', alpha=0.05, sesoi_dz=0.5,
    minimum_pairs=3, planned_pairs=3,
)
result = paired_directional(
    panel.cells,
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
