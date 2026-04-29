"""Convergence audit on the existing 200k DDQN corpus.

Calls into `rl.convergence` for the per-env classification (the
substrate now owns that logic) and uses the framework's verdict-
walk to compute a §3 pattern restricted to the converged subset.

Key finding the audit surfaces: the unrestricted §3 verdict
pattern is contaminated by underconverged envs at 200k. Once
restricted to the baseline-converged subset:
- DDQN's mechanism effect (Δ_jensen_gap) is much stronger
  (g≈-1) — the bias-reduction *does* activate.
- DDQN's outcome effect collapses to ≈0 — bias reduction does
  not propagate to return.

Usage:
    uv run python experiments/convergence_audit.py
    uv run python experiments/convergence_audit.py --total-steps 50000
"""
from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping
from functools import partial
from pathlib import Path

from corroborate.bridge import Bridge, BridgeResult, bridge as bridge_decorator
from corroborate.claimed_edge import (
    link_edge,
    mechanism_edge,
    outcome_edge,
)
from corroborate.hypothesis import Hypothesis
from corroborate.hypothesis_verdict import (
    HypothesisVerdict, hypothesis_subgraph_verdict,
)
from corroborate.intervention import Intervention
from corroborate.persistence import read_runrows
from corroborate.rl.convergence import (
    classify_envs, envs_in_class, filter_to_classes,
)
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.verdict import Verdict


_DEFAULT_RUNS = Path(
    '/workspace/corroborate/experiments/data/ddqn/'
    'runs_with_mediators.parquet'
)
_BEST = 'outcome.eval_best_burst_mean'
_FINAL = 'outcome.eval_final_mean'


def _path_finite_bridge(path: str) -> Bridge[Mapping[str, object]]:
    @bridge_decorator(targets=(path,), name=f'path_finite({path})')
    def _b(record: Mapping[str, object]) -> BridgeResult:
        v = record.get(path)
        finite = isinstance(v, (int, float)) and math.isfinite(float(v))
        return BridgeResult(
            verdict=Verdict.HELD if finite else Verdict.POWER_INSUFFICIENT,
            reason='', stats={},
            name=f'path_finite({path})', targets=(path,),
        )
    return _b


def _ddqn_hypothesis(outcome_path: str) -> Hypothesis[Mapping[str, object]]:
    return Hypothesis(
        name='ddqn',
        intervention={},
        intervention_arms=(
            Intervention(
                slot_path='bootstrap',
                replacement=partial(
                    bootstrap, greedification=double_greedify,
                ),
            ),
        ),
        edges=(
            mechanism_edge(
                target='mechanism.jensen_gap',
                predicted_direction='a_lt_b',
                bridge=_path_finite_bridge('mechanism.jensen_gap'),
            ),
            outcome_edge(
                target=outcome_path,
                predicted_direction='a_gt_b',
                bridge=_path_finite_bridge(outcome_path),
            ),
            link_edge(
                source='mechanism.jensen_gap',
                target=outcome_path,
                predicted_direction='a_gt_b',
                bridge=_path_finite_bridge(outcome_path),
            ),
        ),
    )


def _vanilla_hypothesis() -> Hypothesis[Mapping[str, object]]:
    return Hypothesis(
        name='vanilla_dqn', intervention={}, intervention_arms=(),
    )


def _print_verdict(
    label: str,
    verdict: HypothesisVerdict[Mapping[str, object]],
    hypothesis: Hypothesis[Mapping[str, object]],
) -> None:
    print(f'  §3 pattern (mechanism, outcome, link) — {label}: '
          f'{tuple(v.value for v in verdict.pattern())}')
    for edge in hypothesis.edges:
        if edge.target not in verdict.comparison_rows:
            continue
        row = verdict.comparison_rows[edge.target]
        g = (
            row.effect_size_g
            if row.effect_size_g is not None else float('nan')
        )
        i2 = row.pooled.I2 if row.pooled is not None else float('nan')
        print(
            f'    {edge.role:<10} → {edge.target!r:<35} '
            f'verdict={row.verdict.value:<22} '
            f'g={g:+.3f}  I²={i2:.3f}'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        '--runs-path', type=Path, default=_DEFAULT_RUNS,
    )
    _ = parser.add_argument(
        '--total-steps', type=int, default=200000,
    )
    args = parser.parse_args()
    runs_path = Path(args.runs_path)  # pyright: ignore[reportAny]
    total_steps = int(args.total_steps)  # pyright: ignore[reportAny]

    if not runs_path.exists():
        print(f'corpus not found: {runs_path}', file=sys.stderr)
        sys.exit(1)

    print('=' * 92)
    print(f'Convergence audit (corpus total_steps={total_steps})')
    print('=' * 92)
    print()

    # Read RunRows once; classify and filter through the substrate.
    all_runs = read_runrows(runs_path)
    ts_runs = [
        r for r in all_runs
        if r.measurements.get('total_steps') == total_steps
    ]
    baseline = [r for r in ts_runs if r.arm_key == 'baseline']
    print(f'baseline cells: {len(baseline)}')
    print()

    classifications = classify_envs(baseline)

    print(
        f'{"env":<28} {"thresh":>10} {"src":>26} '
        f'{"best_mean":>10} {"final_mean":>10} '
        f'{"best_solve":>11} {"final_solve":>12} {"class":>10}'
    )
    print('-' * 132)
    for env_name in sorted(classifications):
        c = classifications[env_name]
        spec = c.threshold
        thresh_str = (
            f'{spec.threshold:.3g}'
            if spec.threshold is not None else '—'
        )
        best_solve_str = (
            f'{c.best_solve_rate:.2f}'
            if c.best_solve_rate is not None else '—'
        )
        final_solve_str = (
            f'{c.final_solve_rate:.2f}'
            if c.final_solve_rate is not None else '—'
        )
        print(
            f'{env_name:<28} {thresh_str:>10} {spec.source:>26} '
            f'{c.best_mean:>10.3f} {c.final_mean:>10.3f} '
            f'{best_solve_str:>11} {final_solve_str:>12} '
            f'{c.classification:>10}'
        )

    solved = envs_in_class(classifications, 'solved')
    partial = envs_in_class(classifications, 'partial')
    unsolved = envs_in_class(classifications, 'unsolved')
    absent = envs_in_class(classifications, 'absent')
    print()
    print(f'  solved (final_solve ≥ 0.5): {len(solved)} envs — {list(solved)}')
    print(f'  partial: {len(partial)} envs — {list(partial)}')
    print(f'  unsolved: {len(unsolved)} envs — {list(unsolved)}')
    print(f'  absent: {len(absent)} envs — {list(absent)}')

    if not solved:
        print('\n  no envs reached the solved threshold — nothing to verdict.')
        return

    # ============ §3 verdict on solved subset ============
    print()
    print('=' * 92)
    print('§3 verdict pattern, restricted to SOLVED envs')
    print('=' * 92)
    converged_runs = filter_to_classes(
        ts_runs, classifications, ('solved',),
    )
    treatment_h = _ddqn_hypothesis(_BEST)
    baseline_h = _vanilla_hypothesis()
    treatment_arm_key = treatment_h.arm_key()
    treatment = [
        r for r in converged_runs if r.arm_key == treatment_arm_key
    ]
    baseline_solved = [
        r for r in converged_runs if r.arm_key == 'baseline'
    ]
    print(
        f'  envs={list(solved)!r}  '
        f'treatment={len(treatment)}  baseline={len(baseline_solved)}'
    )
    try:
        verdict = hypothesis_subgraph_verdict(
            treatment_h, treatment, baseline_solved,
            pair_by=('seed',), group_by='env_name',
            baseline_h=baseline_h,
        )
        _print_verdict('solved-only', verdict, treatment_h)
    except ValueError as e:
        print(f'  verdict-walk failed: {e}')
        return

    # ============ Comparison: unrestricted §3 ============
    print()
    print('=' * 92)
    print('Same pattern, all envs (for comparison)')
    print('=' * 92)
    treatment_all = [r for r in ts_runs if r.arm_key == treatment_arm_key]
    baseline_all = [r for r in ts_runs if r.arm_key == 'baseline']
    verdict_all = hypothesis_subgraph_verdict(
        treatment_h, treatment_all, baseline_all,
        pair_by=('seed',), group_by='env_name',
        baseline_h=baseline_h,
    )
    _print_verdict('all envs', verdict_all, treatment_h)


if __name__ == '__main__':
    main()
