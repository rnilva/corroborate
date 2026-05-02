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

from corroborate.causal_graph import Direction, Tier
from corroborate.claim_bridge import Bridge as ClaimBridge
from corroborate.hypothesis import Hypothesis
from corroborate.hypothesis_verdict import (
    HypothesisVerdict, hypothesis_subgraph_verdict,
)
from corroborate.intervention import DoEffect, Intervention
from corroborate.persistence import read_runrows
from corroborate.rl.convergence import (
    classify_envs, envs_in_class, filter_to_classes,
)
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.rl.dqn.measurables import dqn_default_measurables
from corroborate.verdict import Verdict


_DEFAULT_RUNS = Path(
    '/workspace/corroborate/experiments/data/ddqn/'
    'runs_with_mediators.parquet'
)
_BEST = 'eval_best_burst_mean'
_FINAL = 'eval_final_mean'
_MECHANISM = 'jensen_gap'
_DDQN_DO = DoEffect(treatment_arm='ddqn', baseline_arm='vanilla_dqn')


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
            ClaimBridge(
                name=f'ddqn_mechanism({_MECHANISM})',
                source=_DDQN_DO.node_key(),
                target=_MECHANISM,
                tier=Tier.INTERVENTIONAL,
                direction=Direction.DIRECT,
                intervention=_DDQN_DO,
                predicted_direction='a_lt_b',
            ),
            ClaimBridge(
                name=f'ddqn_outcome({outcome_path})',
                source=_DDQN_DO.node_key(),
                target=outcome_path,
                tier=Tier.INTERVENTIONAL,
                direction=Direction.DIRECT,
                intervention=_DDQN_DO,
                predicted_direction='a_gt_b',
            ),
            ClaimBridge(
                name=f'coupling({_MECHANISM}->{outcome_path})',
                source=_MECHANISM,
                target=outcome_path,
                tier=Tier.ASSOCIATIONAL,
                direction=Direction.DIRECT,
                predicted_direction='a_gt_b',
            ),
        ),
        measurables=dqn_default_measurables(),
    )


def _vanilla_hypothesis() -> Hypothesis[Mapping[str, object]]:
    return Hypothesis(
        name='vanilla_dqn', intervention={}, intervention_arms=(),
        measurables=dqn_default_measurables(),
    )


def _print_verdict(
    label: str,
    verdict: HypothesisVerdict[Mapping[str, object]],
    hypothesis: Hypothesis[Mapping[str, object]],
) -> None:
    # Substrate-side §3 narrative: the intervention edge with
    # target=mechanism path is "mechanism"; the intervention edge
    # with target=outcome path is "outcome"; the coupling edge is
    # the "link".
    intervention_targets = tuple(
        e.target for e in hypothesis.edges if e.intervention is not None
    )
    pattern_components: list[Verdict] = []
    for t in intervention_targets:
        pattern_components.append(verdict.verdict_at(t))
    coupling = next(
        (e for e in hypothesis.edges if e.intervention is None), None,
    )
    if coupling is not None:
        br = verdict.bridge_results.get((coupling.source, coupling.target))
        pattern_components.append(
            br.verdict if br is not None else Verdict.POWER_INSUFFICIENT
        )
    print(f'  §3 pattern (mechanism, outcome, coupling) — {label}: '
          f'{tuple(v.value for v in pattern_components)}')
    for edge in hypothesis.edges:
        if edge.target not in verdict.comparison_rows:
            continue
        if edge.intervention is None:
            continue
        row = verdict.comparison_rows[edge.target]
        g = (
            row.effect_size_g
            if row.effect_size_g is not None else float('nan')
        )
        i2 = row.pooled.I2 if row.pooled is not None else float('nan')
        role_label = (
            'mechanism' if edge.target == _MECHANISM else 'outcome'
        )
        print(
            f'    {role_label:<10} → {edge.target!r:<35} '
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
