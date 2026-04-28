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


def main() -> None:
    rows = read_runrows(_DATA_PATH)
    print(f'loaded {len(rows)} RunRows from {_DATA_PATH.name}')
    print()

    grouped = _by_env_intervention(rows)
    envs = sorted(grouped.keys())
    print(f'envs: {envs}')
    print()

    print(f'{"env":<22} {"verdict":<22} {"refutation":<22} '
          f'{"g":>7} {"se":>7} {"q":>7} {"ΔI":>7} {"powered":>8} {"n":>3}')
    print('─' * 120)

    for env in envs:
        intervention_groups = grouped[env]
        if 'ddqn' not in intervention_groups:
            print(f'{env:<22} (no ddqn runs)')
            continue
        if 'vanilla_dqn' not in intervention_groups:
            print(f'{env:<22} (no vanilla_dqn runs)')
            continue

        cmp = paired_comparison_from_runs(
            treatment_runs=intervention_groups['ddqn'],
            baseline_runs=intervention_groups['vanilla_dqn'],
            predicted_direction='a_gt_b',  # DDQN should reduce overestimation
        )
        m = cmp.measurements

        # Format and print.
        rc = (
            cmp.refutation_class.value
            if cmp.refutation_class is not None
            else '—'
        )
        g_v = m.get('outcome.late_window_mean.effect_size_g')
        se_v = m.get('outcome.late_window_mean.se')
        q_v = m.get('outcome.late_window_mean.derived_q')
        di_v = m.get('outcome.late_window_mean.delta_i_population')
        n_pairs_v = m.get('n_pairs')
        n_pairs = int(n_pairs_v) if isinstance(n_pairs_v, int) else 0

        print(
            f'{env:<22} {cmp.verdict.value:<22} {rc:<22} '
            f'{_f(g_v):>7} {_f(se_v):>7} {_f(q_v):>7} {_f(di_v):>7} '
            f'{str(cmp.adequately_powered):>8} {n_pairs:>3}'
        )

    print()
    print('All comparisons rendered.')


if __name__ == '__main__':
    main()
