"""Three-check tautology audit on the FourRooms mediator panel.

Tests whether the candidate mediators (especially target_staleness_late
which HELD at 27% mediation) are real or false-positives via:

  1. Structural check: jaccard of mediator's `reads` with outcome's
     reads. High overlap (≥0.5) = mediator restates outcome.
  2. HP-R² check: per-HP-axis OLS R². High R² (≥0.95) on any axis
     = mediator is HP-deterministic, shadows the HP itself.
  3. Stratified ρ check: Spearman ρ pooled WITHIN HP strata.
     |ρ| < 0.1 with p > 0.05 = HP-mediated, no residual signal.

For FourRooms capacity_sweep, the varying HP is `replay.capacity`
(10k, 20k, 50k). All other HPs fixed.
"""
from __future__ import annotations

import os
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import json
import math
import time
from pathlib import Path

import numpy as np
import polars as pl
from corroborate_rl.dqn import measurables as _dqn_measurables  # noqa: F401
from corroborate.analyses.tautology_audit import tautology_audit as _ta
from corroborate.measurables import get_registered

tautology_audit = _ta.fn

# Reuse the same data path / loaders from the mediator audit.
import sys
sys.path.insert(
    0, str(Path('experiments/findings/sync_curve_breakout').resolve())
)
from run_mediator_audit_fourrooms import (
    CORPUS_DIR, DDQN, PER_STEP_CANDIDATES, SHORT_LIST_CANDIDATES,
    SCALAR_TARGETS, SHORT_TRACE_COLS,
    compute_per_step_measurables, compute_short_list_measurables,
)

OUTCOME = 'eval_best_burst_mean'
OUTCOME_READS = ('mc_return',)  # see @measurable(reads=...)
HP_AXES = ('replay.capacity',)  # the only varying HP
PAIR_BY = ('gamma', 'total_steps', 'seed', 'replay.capacity')


def main() -> None:
    t0 = time.monotonic()

    runs_full = pl.read_parquet(CORPUS_DIR / 'runs.parquet')
    # RunRow.from_row_dict requires id, timestamp, verdict; keep them.
    keep_cols = list(dict.fromkeys(
        ['id', 'arm_key', 'timestamp', 'verdict', 'parent_id', 'cycle_id']
        + list(PAIR_BY) + list(HP_AXES)
    ))
    runs = runs_full.select([c for c in keep_cols if c in runs_full.columns])
    print(f'[{time.monotonic()-t0:.1f}s] runs: {len(runs)}, cols: {runs.columns}', flush=True)

    print('Computing per-step measurables...', flush=True)
    per_step = compute_per_step_measurables(runs)
    print('Computing short-list measurables...', flush=True)
    short = compute_short_list_measurables(runs)

    # Cells expose mediator scalars under bare names (e.g.
    # `target_staleness_late`, not `mediator.target_staleness_late`).
    # Pass `mediator_path_for` to the audit so it reads them at the
    # bare key rather than synthesising a prefixed lookup path.
    extra: list[pl.Series] = []
    for k, v in per_step.items():
        extra.append(pl.Series(name=k, values=v.tolist()))
    for k, v in short.items():
        extra.append(pl.Series(name=k, values=v))
    small = runs.with_columns(extra)
    cells = small.to_dicts()
    print(f'cells: {len(cells)} (sample keys: {sorted(cells[0].keys())[:6]}...)', flush=True)

    # Build measurable specs (name + reads)
    candidates = list(per_step.keys()) + list(short.keys())
    # Skip jensen_gap and env_reward_polarity — they're upstream/scope, not mediator candidates
    candidates = [c for c in candidates if c not in ('jensen_gap', 'env_reward_polarity', 'eval_best_burst_mean')]
    measurable_specs = []
    for name in candidates:
        m = get_registered(name)
        if m is None:
            print(f'  WARN: {name} not registered, skipping', flush=True)
            continue
        reads = tuple(m.reads) if hasattr(m, 'reads') else ()
        measurable_specs.append({'name': name, 'reads': reads})
        print(f'  {name}: reads={reads}', flush=True)

    print()
    print('Running tautology_audit (DDQN arm only)...', flush=True)
    # Audit on the DDQN arm — the mediator's behavior under the
    # treatment is what bridges claim about.
    # Bare-name mediator keys (vs the audit's default
    # `mediator.<name>`); pass mediator_path_for so the audit
    # reads them in place.
    bare_paths = {spec['name']: spec['name'] for spec in measurable_specs}
    result = tautology_audit(
        cells=cells,
        measurables=measurable_specs,
        outcome_path=OUTCOME,
        outcome_reads=OUTCOME_READS,
        hp_axes=HP_AXES,
        hp_stratum_axis=HP_AXES[0],
        arm_filter=DDQN,
        mediator_path_for=bare_paths,
    )

    print()
    print('=== Tautology audit results (DDQN arm, FourRooms) ===\n', flush=True)
    hdr = ('mediator', 'jaccard', 'hp_r²', 'strat_ρ', 'strat_p', 'tags')
    fmt = '{:<28} {:>9} {:>9} {:>9} {:>10} {:>30}'
    print(fmt.format(*hdr))
    print('-' * 100, flush=True)

    summary = []
    for r in result.reports:
        # The TautologyReport has structural_jaccard, hp_r_squared (per-axis dict), stratified_*
        # Let me handle robustly
        jac = getattr(r, 'outcome_jaccard', getattr(r, 'structural_jaccard', float('nan')))
        hp_rs = getattr(r, 'hp_r_squared', None)
        if isinstance(hp_rs, dict):
            max_r2 = max(hp_rs.values()) if hp_rs else float('nan')
        else:
            max_r2 = float('nan')
        strat_rho = getattr(r, 'outcome_stratified_rho', float('nan'))
        strat_p = getattr(r, 'outcome_stratified_p', float('nan'))
        # Build tags from the boolean flags
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
            f'{max_r2:.3f}' if not math.isnan(max_r2) else 'nan',
            f'{strat_rho:+.3f}' if isinstance(strat_rho, float) and not math.isnan(strat_rho) else 'nan',
            f'{strat_p:.3g}' if isinstance(strat_p, float) and not math.isnan(strat_p) else 'nan',
            tags_str,
        ), flush=True)
        summary.append({
            'mediator': r.measurable_name,
            'jaccard': jac if isinstance(jac, float) else None,
            'hp_r_squared': hp_rs if isinstance(hp_rs, dict) else None,
            'stratified_rho': strat_rho,
            'stratified_p': strat_p,
            'tags': tag_list,
            'is_clean': r.is_clean if hasattr(r, 'is_clean') else None,
        })

    print()
    if hasattr(result, 'clean_names'):
        print(f'CLEAN mediators (passed all 3 checks): {result.clean_names}', flush=True)

    out = Path('experiments/findings/sync_curve_breakout/tautology_audit_fourrooms.json')
    out.write_text(json.dumps(summary, indent=2, default=str))
    print(f'\nwrote: {out}', flush=True)


if __name__ == '__main__':
    main()
