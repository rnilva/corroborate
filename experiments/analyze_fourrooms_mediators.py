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

import polars as pl
import scipy.stats as ss

from corroborate._polars_boundary import to_dicts as _to_dicts
from corroborate.aggregate import paired_deltas_from_runs
from corroborate.rl.dqn.compute_mediators import (
    DEFAULT_PANEL, compute_mediator_panel,
)
from corroborate.schema import RunRow


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
        print(
            f'{path:<40} '
            f'{rf:>+14.3f} {pf:>6.3f}{sig_f} '
            f'{rb:>+14.3f} {pb:>6.3f}{sig_b}'
        )


if __name__ == '__main__':
    main()
