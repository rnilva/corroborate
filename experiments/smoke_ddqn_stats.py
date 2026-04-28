"""Step 5 acceptance smoke: paired-by-seed DDQN-vs-vanilla
ComparisonRows on the full DDQN sweep.

Consumes `experiments/data/ddqn/runs.parquet` (produced by
`collect_ddqn_runs.py`), groups by env, builds one
`paired_comparison_from_runs(treatment=ddqn, baseline=vanilla)`
per env, prints the verdict / RefutationClass / g / SE /
adequately_powered.

This is the v0 acceptance test for the §3 outcome layer:
mechanism ↛ outcome ↛ link separation comes later (the link
bridge across envs); here we land per-env outcome verdicts with
real Hedges' g + MDE + power gating.

Run: `uv run python experiments/smoke_ddqn_stats.py`."""
from __future__ import annotations

import math
from pathlib import Path

from corroborate.aggregate import paired_comparison_from_runs
from corroborate.persistence import read_runrows
from corroborate.schema import RunRow


_DATA_PATH = Path(__file__).parent / 'data' / 'ddqn' / 'runs.parquet'


def _by_env_intervention(
    rows: list[RunRow],
) -> dict[str, dict[str, list[RunRow]]]:
    """Index runs by env_name → intervention_name → [runs]."""
    out: dict[str, dict[str, list[RunRow]]] = {}
    for r in rows:
        env_name = r.measurements.get('env_name')
        intervention_name = r.measurements.get('intervention_name')
        if not isinstance(env_name, str) or not isinstance(intervention_name, str):
            continue
        out.setdefault(env_name, {}).setdefault(intervention_name, []).append(r)
    return out


def _f(x: float | int | str | bool | None, prec: int = 3) -> str:
    """Format an optional float for display, NaN-safe."""
    if x is None:
        return 'None'
    if not isinstance(x, (int, float)):
        return repr(x)
    if math.isnan(x):
        return 'NaN'
    return f'{x:.{prec}f}'


_OUTCOME_PATHS = (
    'outcome.late_window_mean',
    'outcome.eval_final_mean',
    'outcome.eval_best_burst_mean',
)


def _print_table_header() -> None:
    print(f'{"env":<22} {"outcome":<32} {"verdict":<22} {"g":>7} '
          f'{"se":>7} {"q":>7} {"powered":>8} {"n":>3}')
    print('─' * 110)


def _maybe_compare(
    env: str, outcome_path: str,
    treatment_runs: list[RunRow], baseline_runs: list[RunRow],
) -> None:
    """Try to build a paired comparison; print one row of the
    table. Skip silently if the outcome path is missing on any
    run (older sweeps don't carry the eval-based reductions)."""
    if any(outcome_path not in r.measurements for r in treatment_runs):
        return
    if any(outcome_path not in r.measurements for r in baseline_runs):
        return

    cmp = paired_comparison_from_runs(
        treatment_runs=treatment_runs,
        baseline_runs=baseline_runs,
        predicted_direction='a_gt_b',
        outcome_path=outcome_path,
    )
    m = cmp.measurements
    g_v = m.get(f'{outcome_path}.effect_size_g')
    se_v = m.get(f'{outcome_path}.se')
    q_v = m.get(f'{outcome_path}.derived_q')
    n_pairs_v = m.get('n_pairs')
    n_pairs = int(n_pairs_v) if isinstance(n_pairs_v, int) else 0

    print(
        f'{env:<22} {outcome_path:<32} {cmp.verdict.value:<22} '
        f'{_f(g_v):>7} {_f(se_v):>7} {_f(q_v):>7} '
        f'{str(cmp.adequately_powered):>8} {n_pairs:>3}'
    )


def main() -> None:
    rows = read_runrows(_DATA_PATH)
    print(f'loaded {len(rows)} RunRows from {_DATA_PATH.name}')
    print()

    grouped = _by_env_intervention(rows)
    envs = sorted(grouped.keys())
    print(f'envs: {envs}')
    print()

    _print_table_header()

    for env in envs:
        intervention_groups = grouped[env]
        if 'ddqn' not in intervention_groups:
            print(f'{env:<22} (no ddqn runs)')
            continue
        if 'vanilla_dqn' not in intervention_groups:
            print(f'{env:<22} (no vanilla_dqn runs)')
            continue
        for outcome_path in _OUTCOME_PATHS:
            _maybe_compare(
                env, outcome_path,
                intervention_groups['ddqn'],
                intervention_groups['vanilla_dqn'],
            )
        print()  # spacer between envs

    print('All comparisons rendered.')


if __name__ == '__main__':
    main()
