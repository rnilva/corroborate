"""Three-check tautology audit on the DDQN 200k corpus.

The methodology that survived the CartPole-HP side-quest (reads-set
jaccard + HP R² + within-stratum Spearman ρ) is now applied to the
corpus that carries the actual DDQN claim. The question:

    Which of the panel's mediators carry within-env outcome signal
    that isn't a structural restatement of the outcome or a
    deterministic relabeling of the experimental design?

The DDQN corpus differs from the CartPole HP corpus in axes:
  * fixed numerical HPs (capacity=10000, batch=32, lr=1e-3, sync=100)
  * varies on: intervention ∈ {ddqn, vanilla_dqn}, env (18 levels),
    total_steps ∈ {50k, 200k}, seed
  * no within-HP grid to stratify by — the relevant stratum is env.

The audit's `hp_stratum_axis` is plumbed with an integer env-id we
inject into `RunRow.measurements` (`_env_id`), which puts the
within-env Spearman check at full power: 30 seeds × 2 budgets = 60
cells per env, 18 envs pooled via Fisher-z.

Note: `jensen_gap` reads `(predicted_q_at_start, mc_return)`
and the outcome reads `mc_return` — jaccard = 1/2 = 0.5, exactly at
the default threshold. Reported either way; the analyst judges
whether a residual-on-outcome mediator counts as "outcome-tautological"
(0.5 says yes, but the semantics — `gap = predicted - actual` —
say no).

Usage:
    uv run python experiments/audit_ddqn_panel.py
    uv run python experiments/audit_ddqn_panel.py --intervention vanilla_dqn
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from corroborate.measurables import measurable
from corroborate._internals.polars import to_dicts as _to_dicts
from corroborate.measurables.redundancy_check import (
    TautologyReport, audit_mediator_panel,
)
from corroborate.rl.dqn.measurables import (
    fill_ratio_late,
    greedy_match_late,
    learning_curve_auc,
    plateau_slope_late,
    q_gap_growth,
    q_gap_late,
    q_max_growth,
    return_at_25pct_steps,
    state_coverage_kl_uniform_late,
    state_visit_entropy_late,
    td_residual_late,
    td_within_batch_var_late,
    time_to_threshold,
    v_vs_max_delta_late,
)
from corroborate.corpus.schema import RunRow

_RUNS = Path('experiments/data/ddqn/runs_with_mediators.parquet')
_OUTCOME_PATH = 'eval_best_burst_mean'
_OUTCOME_READS: frozenset[str] = frozenset({'mc_return'})


# A surrogate Measurable for the Jensen overestimation gap to
# expose its `reads` set to the audit. The actual computation
# happens in `rl/dqn/invariants.jensen_overestimation_gap`; the
# audit only needs (.name, .reads).
@measurable(reads=('predicted_q_at_start', 'mc_return'))
def jensen_gap(record):  # type: ignore[no-untyped-def]
    del record
    return 0.0


_PANEL: tuple[object, ...] = (
    jensen_gap,
    q_gap_late, q_gap_growth, q_max_growth,
    v_vs_max_delta_late,
    td_residual_late, td_within_batch_var_late,
    greedy_match_late, fill_ratio_late,
    learning_curve_auc, time_to_threshold, return_at_25pct_steps,
    plateau_slope_late,
    state_visit_entropy_late, state_coverage_kl_uniform_late,
)
_MEDIATOR_PATH_FOR: dict[str, str] = {
    'jensen_gap': 'jensen_gap',
    # all others default to `mediator.{name}`
}


def _load_runs(intervention: str) -> tuple[Sequence[RunRow], dict[str, int]]:
    """Read corpus, filter to one intervention, inject `_env_id`
    into `RunRow.measurements`. Returns (rows, env_id_lookup)."""
    if not _RUNS.exists():
        raise SystemExit(f'corpus not found: {_RUNS}')
    df = pl.read_parquet(_RUNS).filter(pl.col('intervention_name') == intervention)
    if df.height == 0:
        raise SystemExit(
            f'no rows for intervention={intervention!r}'
        )
    envs = sorted(df['env_name'].unique().to_list())
    env_id = {e: i for i, e in enumerate(envs)}

    runs_with_env_id = df.with_columns(
        pl.col('env_name').replace_strict(env_id).alias('_env_id'),
    )
    rows = [
        RunRow.from_row_dict(d) for d in _to_dicts(runs_with_env_id)
    ]
    return rows, env_id


def _print_report(r: TautologyReport) -> None:
    flag_o = '×' if r.flagged_outcome else ' '
    flag_h = ','.join(r.flagged_hp) if r.flagged_hp else '—'
    flag_n = '×' if r.flagged_no_residual_signal else ' '
    clean = '✓' if r.is_clean else ' '
    rho = r.outcome_stratified_rho
    p = r.outcome_stratified_p
    rho_str = f'{rho:+.3f}' if rho == rho else '   nan'
    p_str = f'{p:.3f}' if p == p else '  nan'
    r2_total = r.hp_r_squared.get('total_steps', float('nan'))
    r2_env = r.hp_r_squared.get('_env_id', float('nan'))
    r2_str = f'R²(steps)={r2_total:.2f} R²(env)={r2_env:.2f}'
    print(
        f'  {r.measurable_name:<35} '
        f'jacc={r.outcome_jaccard:.2f}[{flag_o}] '
        f'hp[{flag_h}] '
        f'rho={rho_str} p={p_str} [{flag_n}] '
        f'{r2_str}  {clean}'
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--intervention', default='ddqn',
        choices=('ddqn', 'vanilla_dqn'),
    )
    parser.add_argument(
        '--outcome-jaccard-threshold', type=float, default=0.5,
        help='reads-jaccard threshold for outcome tautology (default 0.5)',
    )
    args = parser.parse_args()

    runs, env_id = _load_runs(args.intervention)
    print('=' * 100)
    print(f'DDQN panel audit  — intervention={args.intervention}')
    print(f'  corpus: {_RUNS}')
    print(f'  outcome: {_OUTCOME_PATH}  outcome_reads={set(_OUTCOME_READS)}')
    print(f'  n_cells={len(runs)}  n_envs={len(env_id)}')
    print('=' * 100)
    reports = audit_mediator_panel(
        _PANEL,  # type: ignore[arg-type]
        runs,
        outcome_reads=_OUTCOME_READS,
        hp_axes=('total_steps', '_env_id'),
        outcome_path=_OUTCOME_PATH,
        hp_stratum_axis='_env_id',
        mediator_path_for=_MEDIATOR_PATH_FOR,
        outcome_jaccard_threshold=args.outcome_jaccard_threshold,
    )
    print(
        f'  {"name":<35} {"jaccard":<8}  {"hp":<14} '
        f'{"rho":<7} {"p":<6}     {"R²":<22}  clean'
    )
    print('-' * 100)
    for r in reports:
        _print_report(r)
    print()
    n_clean = sum(1 for r in reports if r.is_clean)
    print(f'clean: {n_clean}/{len(reports)} mediators survive all three checks.')


if __name__ == '__main__':
    main()
