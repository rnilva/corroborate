"""Per-sync outcome analysis on Breakout-MinAtar.

Companion to `run_analysis.py`. The per-burst link analysis showed
plc(sync_period) is monotone increasing on Breakout. This script
asks the orthogonal question: at sync values where the link is
active, does DDQN actually IMPROVE the outcome?

Three outcome reductions per cell (all derived from mc_per_burst):
- `outcome_mean`        — mean across all 20 bursts (trajectory mean)
- `outcome_best_burst`  — max burst mean (Hasselt convention)
- `outcome_late`        — mean of last 5 bursts (steady-state)

Plus per-burst paired_g on `mc_per_burst` to see *which burst* the
DDQN advantage materializes at.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from corroborate.analyses.paired_g import paired_g
from corroborate.analyses.paired_g_per_burst import paired_g_per_burst
from corroborate.measurables.reductions import from_key

# Reuse the cell loaders from the link analysis.
from experiments.findings.sync_curve_breakout.run_analysis import (
    BASELINE_ARM, ENV, TREATMENT_ARM,
    _load_sync_100_breakout, _load_sync_from_bridge_cache,
    _load_sync_from_traces,
)


def _add_scalar_outcomes(
    cells: list[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Add three scalar outcome fields derived from mc_per_burst."""
    enriched: list[dict[str, object]] = []
    for c in cells:
        mc = np.asarray(c['mc_per_burst'], dtype=np.float64)
        d = dict(c)
        d['outcome_mean'] = float(mc.mean()) if mc.size else float('nan')
        d['outcome_best_burst'] = float(mc.max()) if mc.size else float('nan')
        d['outcome_late'] = (
            float(mc[-5:].mean()) if mc.size >= 5 else float('nan')
        )
        enriched.append(d)
    return enriched


def main() -> None:
    print(f'Outcome paired_g per sync on {ENV}...')
    cohorts = {
        100:   _add_scalar_outcomes(_load_sync_100_breakout()),
        1000:  _add_scalar_outcomes(_load_sync_from_traces(
            'minatar_sync_curve/ddqn_sync1k', 1000,
        )),
        3000:  _add_scalar_outcomes(_load_sync_from_traces(
            'minatar_sync_curve/ddqn_sync3k', 3000,
        )),
        10000: _add_scalar_outcomes(_load_sync_from_bridge_cache(
            'minatar_sync_intervention', 10000,
        )),
    }

    print()
    print('Scalar outcome (paired Hedges g, DDQN − vanilla, paired by seed):')
    print(f'  {"sync":>5} | {"outcome_mean":>30} | {"best_burst":>30} | {"late_quarter":>30}')
    summary: dict[int, dict[str, object]] = {}
    for sync, cells in cohorts.items():
        rec: dict[str, object] = {}
        line = f'  {sync:>5} |'
        for col in ('outcome_mean', 'outcome_best_burst', 'outcome_late'):
            r = paired_g.fn(
                cells,
                source=col,
                treatment_arm=TREATMENT_ARM,
                baseline_arm=BASELINE_ARM,
                pair_by=('seed',),
            )
            rec[col] = {
                'g': r.g, 'se': r.se,
                'mean_diff': r.mean_diff, 'mean_diff_se': r.mean_diff_se,
                'n_pairs': r.n_pairs, 'helped_fraction': r.helped_fraction,
                'p_value': r.p_value, 'mean_diff_p_value': r.mean_diff_p_value,
            }
            sig = '*' if r.p_value < 0.05 else ' '
            line += f' g={r.g:+.2f}±{r.se:.2f} Δ={r.mean_diff:+.3f} p={r.p_value:.2g}{sig} |'
        print(line)
        summary[sync] = rec
    print()
    print('  * = two-sided g p < 0.05; n_pairs=30 each')

    print()
    print(f'Per-burst Δ outcome (paired_g_per_burst on mc_per_burst):')
    print(f'  {"burst":>5} | ' + ' | '.join(f'sync={s:>5}: g(Δ_mc)' for s in cohorts))
    per_burst: dict[int, list[dict[str, object]]] = {}
    for sync, cells in cohorts.items():
        result = paired_g_per_burst.fn(
            cells,
            treatment_arm=TREATMENT_ARM,
            baseline_arm=BASELINE_ARM,
            source=from_key('mc_per_burst'),
            pair_by=('seed',),
            env_name=ENV,
        )
        per_burst[sync] = [
            {
                'burst': s.burst_index, 'g': s.g, 'se': s.se,
                'helped_fraction': s.helped_fraction, 'n_pairs': s.n_pairs,
            }
            for s in result.strata
        ]
    for b in range(20):
        line = f'  {b:>5} |'
        for sync in cohorts:
            stratum = next(
                (s for s in per_burst[sync] if s['burst'] == b),
                None,
            )
            if stratum is None:
                line += '          -          |'
                continue
            g_val = float(stratum['g'])  # type: ignore[arg-type]
            se_val = float(stratum['se'])  # type: ignore[arg-type]
            sig = '*' if abs(g_val) > 1.96 * se_val else ' '
            line += f' g={g_val:+.2f}±{se_val:.2f}{sig} |'
        print(line)
    print()
    print('  * = |g/se| > 1.96 (two-sided z); + means DDQN > vanilla')

    out = Path('experiments/findings/sync_curve_breakout/outcome_panel.json')
    out.write_text(json.dumps(
        {'scalar': summary, 'per_burst': per_burst}, indent=2,
    ))
    print()
    print(f'Wrote: {out}')


if __name__ == '__main__':
    main()
