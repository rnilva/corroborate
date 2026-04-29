"""Within-env mediator search on FourRooms-misc.

The action_dim_wide corpus has FourRooms as the lone DDQN-positive
env: link g=+0.71 on outcome.eval_final_mean (HELD), but mechanism
g=+0.13 (sign reversed). Whatever drives DDQN's outcome benefit
on FourRooms isn't Jensen-bias reduction. Question: which
substrate-measurable mediator's per-pair Δ predicts the outcome's
per-pair Δ?

Method:
  1. Read FourRooms cells from action_dim_wide/runs.parquet.
  2. Stream FourRooms traces (per-arm parquets in tmp/) through
     `compute_mediator_panel` to evaluate the substrate's
     candidate mediators per cell.
  3. Pair DDQN/vanilla by seed via `paired_deltas_from_runs`.
  4. For each candidate mediator, scipy.pearsonr between
     Δ_outcome and Δ_mediator across the 60 pairs. Rank by |r|.

All glue lives in framework primitives; this script is the call
site.

Usage:
  uv run python experiments/analyze_fourrooms_mediators.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import scipy.stats as ss

from corroborate._polars_boundary import to_dicts as _to_dicts
from corroborate.aggregate import paired_deltas_from_runs
from corroborate.causal_discovery import partial_spearman_rho
from corroborate.rl.dqn.compute_mediators import (
    DEFAULT_PANEL, compute_mediator_panel,
)
from corroborate.schema import RunRow


# Outcome-tautological mediators (reads ⊇ outcome reads). Filtered
# from the mediator search since their high correlation with
# Δ_outcome is just a re-encoding, not a causal mediator. Same
# four flagged by `audit_mediator_panel` as `flagged_outcome=True`.
_OUTCOME_TAUTOLOGICAL: frozenset[str] = frozenset({
    'mediator.learning_curve_auc',
    'mediator.plateau_slope_late',
    'mediator.return_at_25pct_steps',
    'mediator.time_to_threshold',
})


_RUNS = Path('experiments/data/action_dim_wide/runs.parquet')
_TRACES_TMP = Path('experiments/data/action_dim_wide/tmp')


def main() -> None:
    runs_df = pl.read_parquet(_RUNS).filter(
        pl.col('env_name') == 'FourRooms-misc'
    )
    runs = [RunRow.from_row_dict(d) for d in _to_dicts(runs_df)]
    print(f'FourRooms cells: {len(runs)}')

    trace_paths = sorted(_TRACES_TMP.glob('*FourRooms*__traces.parquet'))
    print(f'trace parquets: {len(trace_paths)}')

    enriched = compute_mediator_panel(runs, trace_paths)
    print(f'enriched: {len(enriched)} cells')

    ddqn = [r for r in enriched if r.measurements.get('intervention_name') == 'ddqn']
    vanilla = [r for r in enriched if r.measurements.get('intervention_name') == 'vanilla_dqn']
    print(f'ddqn: {len(ddqn)}  vanilla: {len(vanilla)}')

    # Per-pair Δ on outcome + each candidate mediator.
    mediator_paths = [f'mediator.{m.name}' for m in DEFAULT_PANEL]
    paths = ['outcome.eval_final_mean', 'outcome.eval_best_burst_mean',
             'mechanism.jensen_gap', *mediator_paths]
    deltas = paired_deltas_from_runs(
        ddqn, vanilla, paths=paths, pair_by=('seed',),
    )

    # Anchor on Δ outcome (final-mean and best-burst); rank candidates
    # by Pearson r against each.
    print()
    print(f'{"candidate path":<40} '
          f'{"r vs Δ_final":>14} {"p":>6} '
          f'{"r vs Δ_best":>14} {"p":>6}')
    print('-' * 100)
    final = deltas['outcome.eval_final_mean']
    best = deltas['outcome.eval_best_burst_mean']
    print(f'{"  Δ outcome.eval_final_mean":<40} {"-":>14} {"-":>6} '
          f'{ss.pearsonr(final, best).statistic:>+14.3f} '
          f'{ss.pearsonr(final, best).pvalue:>6.3f}')
    print()

    # Build a list (path, r_final, p_final, r_best, p_best) and rank by
    # max |r|.
    rows: list[tuple[str, float, float, float, float]] = []
    for path in [*mediator_paths, 'mechanism.jensen_gap']:
        d_med = deltas[path]
        # Align: deltas may have different length across paths if some
        # pairs had non-finite values for a path. Use the intersection
        # by index — paired_deltas_from_runs preserves the original
        # paired_keys order, so paths agree on which pair index they
        # represent (just with some pairs dropped per path).
        n_min = min(len(final), len(d_med))
        if n_min < 3:
            rows.append((path, float('nan'), float('nan'), float('nan'), float('nan')))
            continue
        rf = ss.pearsonr(final[:n_min], d_med[:n_min])
        rb = ss.pearsonr(best[:n_min], d_med[:n_min])
        rows.append((path, rf.statistic, rf.pvalue, rb.statistic, rb.pvalue))

    # Sort by max |r| across the two outcome paths.
    rows.sort(key=lambda x: max(abs(x[1]), abs(x[3])), reverse=True)
    for path, rf, pf, rb, pb in rows:
        sig_f = '✓' if pf < 0.05 else ' '
        sig_b = '✓' if pb < 0.05 else ' '
        flag = ' [TAUT]' if path in _OUTCOME_TAUTOLOGICAL else ''
        print(
            f'{path:<40} '
            f'{rf:>+14.3f} {pf:>6.3f}{sig_f} '
            f'{rb:>+14.3f} {pb:>6.3f}{sig_b}'
            f'{flag}'
        )

    # ============ Partial-ρ analysis: mediator | jensen_gap ============
    # For each non-tautological mediator, compute partial Spearman
    # ρ(Δ_outcome, Δ_mediator | Δ_jensen_gap). If significant, the
    # mediator carries outcome-predictive info BEYOND jensen_gap —
    # candidate intermediate variable. If null after controlling for
    # jensen_gap, the mediator is collinear with (or downstream of)
    # the gap-reduction signal.
    print()
    print('Partial ρ(Δ_outcome, Δ_mediator | Δ_jensen_gap):')
    print(f'  {"candidate":<40} {"partial_ρ_final":>15} {"p":>7} '
          f'{"partial_ρ_best":>15} {"p":>7}')
    print('-' * 90)
    delta_jensen = np.asarray(
        deltas['mechanism.jensen_gap'], dtype=np.float64,
    )
    delta_final = np.asarray(
        deltas['outcome.eval_final_mean'], dtype=np.float64,
    )
    delta_best = np.asarray(
        deltas['outcome.eval_best_burst_mean'], dtype=np.float64,
    )
    partial_rows: list[tuple[str, float, float, float, float]] = []
    for path in mediator_paths:
        if path in _OUTCOME_TAUTOLOGICAL:
            continue
        delta_med = np.asarray(deltas[path], dtype=np.float64)
        if not np.all(np.isfinite(delta_med)):
            partial_rows.append((path, float('nan'), float('nan'), float('nan'), float('nan')))
            continue
        n = min(len(delta_final), len(delta_med), len(delta_jensen))
        rf, pf = partial_spearman_rho(
            delta_final[:n], delta_med[:n], delta_jensen[:n],
        )
        rb, pb = partial_spearman_rho(
            delta_best[:n], delta_med[:n], delta_jensen[:n],
        )
        partial_rows.append((path, rf, pf, rb, pb))

    partial_rows.sort(
        key=lambda x: max(
            abs(x[1]) if x[1] == x[1] else 0.0,
            abs(x[3]) if x[3] == x[3] else 0.0,
        ),
        reverse=True,
    )
    for path, rf, pf, rb, pb in partial_rows:
        rf_s = f'{rf:+.3f}' if rf == rf else '   nan'
        pf_s = f'{pf:.3f}' if pf == pf else '  nan'
        rb_s = f'{rb:+.3f}' if rb == rb else '   nan'
        pb_s = f'{pb:.3f}' if pb == pb else '  nan'
        sig_f = '✓' if pf == pf and pf < 0.05 else ' '
        sig_b = '✓' if pb == pb and pb < 0.05 else ' '
        print(
            f'  {path:<40} {rf_s:>15} {pf_s:>7}{sig_f} '
            f'{rb_s:>15} {pb_s:>7}{sig_b}'
        )


if __name__ == '__main__':
    main()
