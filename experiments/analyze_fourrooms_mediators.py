"""Within-env mediator search + time-series probe on a target env.

Pipeline:
  1. Read target-env cells from runs.parquet.
  2. Stream traces from per-arm parquets through
     `compute_mediator_panel` to evaluate substrate mediators.
  3. Pair DDQN/vanilla by seed (inline `_paired_deltas_by_seed`).
  4. For each candidate, scipy.pearsonr between Δ_outcome and
     Δ_mediator across paired pairs. Filter outcome-tautological.
  5. Partial ρ controlling for Δ_jensen_gap (separates
     independent mediators from collinear ones).
  6. Per-burst trajectory (predicted_q_at_start − mc_return)
     and per-burst Pearson r(Δbias, Δret) — surfaces phase-
     dependent effects scalar reductions hide.

Defaults to FourRooms-misc on action_dim_wide. Override via CLI.

Usage:
  uv run python experiments/analyze_fourrooms_mediators.py
  uv run python experiments/analyze_fourrooms_mediators.py \\
      --env Acrobot-v1
  uv run python experiments/analyze_fourrooms_mediators.py \\
      --env CartPole-v1 --corpus action_dim_sweep
"""
from __future__ import annotations

import argparse
import math
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import polars as pl
import scipy.stats as ss

from corroborate._internals.polars_boundary import to_dicts as _to_dicts
from corroborate.graph.discovery import partial_spearman_rho
from corroborate.rl.dqn.compute_mediators import (
    DEFAULT_PANEL, compute_mediator_panel,
)
from corroborate.schema import RunRow


def _paired_deltas_by_seed(
    treatment_runs: Sequence[RunRow],
    baseline_runs: Sequence[RunRow],
    *,
    paths: Sequence[str],
) -> dict[str, list[float]]:
    """Per-pair Δ (treatment − baseline) for each path. Pairs by
    `seed`; drops pairs whose source value is non-finite or
    missing per path, so different paths can have different
    survivor counts. Caller index-aligns by truncation."""
    by_seed_t = {r.measurements.get('seed'): r for r in treatment_runs}
    by_seed_b = {r.measurements.get('seed'): r for r in baseline_runs}
    paired_seeds = sorted(by_seed_t.keys() & by_seed_b.keys(), key=repr)
    out: dict[str, list[float]] = {}
    for path in paths:
        deltas: list[float] = []
        for seed in paired_seeds:
            t = by_seed_t[seed].measurements.get(path)
            b = by_seed_b[seed].measurements.get(path)
            if not isinstance(t, (int, float)) or isinstance(t, bool):
                continue
            if not isinstance(b, (int, float)) or isinstance(b, bool):
                continue
            tf, bf = float(t), float(b)
            if not (math.isfinite(tf) and math.isfinite(bf)):
                continue
            deltas.append(tf - bf)
        out[path] = deltas
    return out


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


def _arg_paths(corpus: str, env: str) -> tuple[Path, list[Path]]:
    """Locate runs.parquet + trace parquets for `corpus`, restricted
    to `env` if per-arm parquets are available. Falls back to the
    corpus's single merged traces.parquet (e.g., older corpora that
    don't keep per-arm tmp/ around)."""
    base = Path('experiments/data') / corpus
    runs = base / 'runs.parquet'
    if not runs.exists():
        # `runs_with_mediators.parquet` is the older corpus naming.
        alt = base / 'runs_with_mediators.parquet'
        if alt.exists():
            runs = alt
        else:
            raise SystemExit(f'runs.parquet not found at {runs}')
    env_token = env.replace('/', '_')
    tmp = base / 'tmp'
    if tmp.exists():
        traces = sorted(tmp.glob(f'*{env_token}*__traces.parquet'))
        if traces:
            return runs, traces
    # Fallback: single merged traces.parquet at corpus root.
    merged = base / 'traces.parquet'
    if merged.exists():
        return runs, [merged]
    return runs, []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env', default='FourRooms-misc')
    parser.add_argument('--corpus', default='action_dim_wide')
    args = parser.parse_args()
    env_name: str = args.env
    corpus: str = args.corpus

    runs_path, trace_paths = _arg_paths(corpus, env_name)
    print(f'corpus={corpus}  env={env_name}')
    runs_df = pl.read_parquet(runs_path).filter(
        pl.col('env_name') == env_name
    )
    runs = [RunRow.from_row_dict(d) for d in _to_dicts(runs_df)]
    print(f'{env_name} cells: {len(runs)}')
    print(f'trace parquets: {len(trace_paths)}')

    enriched = compute_mediator_panel(runs, trace_paths)
    print(f'enriched: {len(enriched)} cells')

    ddqn = [r for r in enriched if r.measurements.get('intervention_name') == 'ddqn']
    vanilla = [r for r in enriched if r.measurements.get('intervention_name') == 'vanilla_dqn']
    print(f'ddqn: {len(ddqn)}  vanilla: {len(vanilla)}')

    # Per-pair Δ on outcome + each candidate mediator.
    mediator_paths = [f'mediator.{m.name}' for m in DEFAULT_PANEL]
    paths = ['eval_final_mean', 'eval_best_burst_mean',
             'jensen_gap', *mediator_paths]
    deltas = _paired_deltas_by_seed(ddqn, vanilla, paths=paths)

    # Anchor on Δ outcome (final-mean and best-burst); rank candidates
    # by Pearson r against each.
    print()
    print(f'{"candidate path":<40} '
          f'{"r vs Δ_final":>14} {"p":>6} '
          f'{"r vs Δ_best":>14} {"p":>6}')
    print('-' * 100)
    final = deltas['eval_final_mean']
    best = deltas['eval_best_burst_mean']
    print(f'{"  Δ eval_final_mean":<40} {"-":>14} {"-":>6} '
          f'{ss.pearsonr(final, best).statistic:>+14.3f} '
          f'{ss.pearsonr(final, best).pvalue:>6.3f}')
    print()

    # Build a list (path, r_final, p_final, r_best, p_best) and rank by
    # max |r|.
    rows: list[tuple[str, float, float, float, float]] = []
    for path in [*mediator_paths, 'jensen_gap']:
        d_med = deltas[path]
        # Align: deltas may have different length across paths if some
        # pairs had non-finite values for a path. Use the intersection
        # by index — _paired_deltas_by_seed preserves the original
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
        deltas['jensen_gap'], dtype=np.float64,
    )
    delta_final = np.asarray(
        deltas['eval_final_mean'], dtype=np.float64,
    )
    delta_best = np.asarray(
        deltas['eval_best_burst_mean'], dtype=np.float64,
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

    # ============ Time-series analysis (per-burst trajectories) ============
    # Scalar mediators collapse training to a single number per cell,
    # losing the time dimension. The persisted raw trace columns
    # (`predicted_q_at_start`, `mc_return`, both shape (n_bursts, K))
    # let us recover per-burst trajectories. Inline analysis: not a
    # new Measurable.

    print()
    print('=' * 100)
    print(f'Time-series analysis — per-burst trajectories on {env_name}')
    print('=' * 100)

    # Pair runs by id → seed; read trace records by id.
    runs_by_id = {r.id: r for r in enriched}
    seed_by_id = {
        r.id: int(r.measurements['seed'])
        for r in enriched
        if isinstance(r.measurements.get('seed'), int)
    }
    intervention_by_id = {
        r.id: r.measurements['intervention_name']
        for r in enriched
    }

    # Read per-burst arrays from each FourRooms trace parquet.
    bias_by_id: dict[str, np.ndarray] = {}    # (n_bursts,)
    return_by_id: dict[str, np.ndarray] = {}  # (n_bursts,)
    for tp in trace_paths:
        df_t = pl.read_parquet(str(tp), columns=[
            'id', 'predicted_q_at_start', 'mc_return',
        ])
        for row in df_t.iter_rows(named=True):
            cell_id = row['id']
            if cell_id not in runs_by_id:
                continue
            pred = np.asarray(row['predicted_q_at_start'], dtype=np.float64)
            actual = np.asarray(row['mc_return'], dtype=np.float64)
            if pred.ndim != 2 or actual.ndim != 2:
                continue
            bias_by_id[cell_id] = (pred - actual).mean(axis=-1)
            return_by_id[cell_id] = actual.mean(axis=-1)

    # Stack per arm: (n_cells, n_bursts).
    def _stack(intervention: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
        ids = sorted(
            (i for i in bias_by_id if intervention_by_id.get(i) == intervention),
            key=lambda i: seed_by_id.get(i, -1),
        )
        bias = np.stack([bias_by_id[i] for i in ids], axis=0)
        ret = np.stack([return_by_id[i] for i in ids], axis=0)
        seeds = [seed_by_id[i] for i in ids]
        return bias, ret, seeds

    van_bias, van_ret, van_seeds = _stack('vanilla_dqn')
    ddqn_bias, ddqn_ret, ddqn_seeds = _stack('ddqn')
    if van_seeds != ddqn_seeds:
        # Realign by intersection.
        common = sorted(set(van_seeds) & set(ddqn_seeds))
        van_idx = [van_seeds.index(s) for s in common]
        ddqn_idx = [ddqn_seeds.index(s) for s in common]
        van_bias, van_ret = van_bias[van_idx], van_ret[van_idx]
        ddqn_bias, ddqn_ret = ddqn_bias[ddqn_idx], ddqn_ret[ddqn_idx]
    n_pairs, n_bursts = van_bias.shape
    print(f'  paired pairs={n_pairs}, n_bursts={n_bursts}')

    delta_bias = ddqn_bias - van_bias       # (n_pairs, n_bursts)
    delta_ret = ddqn_ret - van_ret           # (n_pairs, n_bursts)

    # Per-burst summary table.
    print()
    print(f'  {"burst":>5} '
          f'{"van_bias(μ)":>12} {"ddqn_bias(μ)":>13} {"Δbias(μ)":>10} {"Δbias_se":>9} '
          f'{"van_ret(μ)":>11} {"ddqn_ret(μ)":>12} {"Δret(μ)":>9} '
          f'{"r(Δbias,Δret)":>13} {"p":>6}')
    print('-' * 130)
    for b in range(n_bursts):
        db = delta_bias[:, b]
        dr = delta_ret[:, b]
        # Skip degenerate bursts (constant outputs across pairs)
        if float(db.std()) == 0.0 or float(dr.std()) == 0.0:
            r, p = float('nan'), float('nan')
        else:
            r_obj = ss.pearsonr(db, dr)
            r, p = r_obj.statistic, r_obj.pvalue
        sig = '✓' if p == p and p < 0.05 else ' '
        print(
            f'  {b:>5} '
            f'{float(van_bias[:, b].mean()):>+12.3f} '
            f'{float(ddqn_bias[:, b].mean()):>+13.3f} '
            f'{float(db.mean()):>+10.3f} '
            f'{float(db.std() / np.sqrt(n_pairs)):>9.3f} '
            f'{float(van_ret[:, b].mean()):>+11.3f} '
            f'{float(ddqn_ret[:, b].mean()):>+12.3f} '
            f'{float(dr.mean()):>+9.3f} '
            f'{r:>+13.3f} {p:>6.3f}{sig}'
        )


if __name__ == '__main__':
    main()
