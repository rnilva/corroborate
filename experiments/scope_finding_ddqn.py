"""DDQN scope-finding smoke — Phase 1 + Phase 2 end-to-end,
*gap-grounded*.

The framework's two-phase output run on the DDQN acceptance
corpus, with the cleavage covariate derived from the
**invariance gap** declared by Hasselt 2010's theorem:

- Theorem premise: vanilla DQN's Jensen-inequality bias `mean(Q̂
  − Q*) ≥ 0`. The gap is `max(0, mean(Q̂ − Q*))` per cell —
  exactly what `rl/dqn/invariants.py:jensen_overestimation_gap`
  computes.
- Substrate-level commitment: the author wraps the gap with
  `at_most(jensen_overestimation_gap(), threshold=None,
  of_claim=dqn)` — discovery mode (`threshold=None`), since
  this is the first pass; the threshold gets committed after
  Phase-1 cleavage shows where the regression line crosses zero.
- Phase 1: `build_scope` aggregates per-env baseline gap mean
  from `mechanism.jensen_gap` (which IS what the invariant
  bridge would record, in `invariant.<bridge_name>.stats.gap_value`
  — for the existing corpus we read the cell-runner's flat
  projection directly), regresses per-env outcome g on
  `log10(gap)`, returns a `Scope`.
- Phase 2: the typed CausalGraph from the verdict — `do(arm) →
  jensen_gap`, `do(arm) → outcome`, `jensen_gap → outcome`.

The Scope is gap-grounded by construction: `gap_name` is on the
record, and the cleavage axis is the gap (no other covariates
admitted). `Scope.threshold` is `None` until the author
commits scope.

Usage:
    uv run python experiments/scope_finding_ddqn.py
    uv run python experiments/scope_finding_ddqn.py --total-steps 200000
    uv run python experiments/scope_finding_ddqn.py --threshold 1.0
"""
from __future__ import annotations

import argparse
import math
import sys
from collections.abc import Mapping
from functools import partial
from pathlib import Path

import polars as pl

from corroborate.bridge import Bridge, BridgeResult, bridge as bridge_decorator
from corroborate.claimed_edge import (
    link_edge,
    mechanism_edge,
    outcome_edge,
)
from corroborate.hypothesis import Hypothesis
from corroborate.hypothesis_verdict import hypothesis_subgraph_verdict
from corroborate.intervention import Intervention
from corroborate.persistence import read_runrows
from corroborate.rl.dqn.claims.bootstrap import bootstrap, double_greedify
from corroborate.schema import RunRow
from corroborate.scope import Scope, build_scope
from corroborate.verdict import Verdict


_DEFAULT_RUNS = Path(
    '/workspace/corroborate/experiments/data/ddqn/'
    'runs_with_mediators.parquet'
)
_DEFAULT_OUTCOME = 'outcome.eval_best_burst_mean'
_MECHANISM = 'mechanism.jensen_gap'

# The Hasselt 2010 invariance gap — `jensen_overestimation_gap` from
# `rl/dqn/invariants.py`. The corpus has this scalar per cell at
# `mechanism.jensen_gap` (cell_runner's projection). Once
# `at_most(jensen_overestimation_gap(), threshold=None, of_claim=dqn)`
# is wired into the hypothesis at run time, the per-cell value lands
# at `invariant.at_most[...].stats.gap_value` instead — same number,
# different path.
_GAP_PATH = _MECHANISM
_GAP_NAME = 'jensen_overestimation_gap'


def _path_finite_bridge(path: str) -> Bridge[Mapping[str, object]]:
    """Stub: HELD iff the record carries a finite scalar at `path`.
    The cross-arm test happens at the `from_cells` layer reading
    these measurements directly from RunRows, not via per-cell
    bridge invocation."""
    @bridge_decorator(targets=(path,), name=f'path_finite({path})')
    def _b(record: Mapping[str, object]) -> BridgeResult:
        v = record.get(path)
        finite = isinstance(v, (int, float)) and math.isfinite(float(v))
        return BridgeResult(
            verdict=(
                Verdict.HELD if finite else Verdict.POWER_INSUFFICIENT
            ),
            reason='', stats={},
            name=f'path_finite({path})',
            targets=(path,),
        )
    return _b


def _ddqn_hypothesis(outcome_path: str) -> Hypothesis[Mapping[str, object]]:
    """Three-edge subgraph: do(arm) → jensen_gap, do(arm) →
    outcome, and the linking jensen_gap → outcome edge."""
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
                target=_MECHANISM,
                predicted_direction='a_lt_b',
                bridge=_path_finite_bridge(_MECHANISM),
            ),
            outcome_edge(
                target=outcome_path,
                predicted_direction='a_gt_b',
                bridge=_path_finite_bridge(outcome_path),
            ),
            link_edge(
                source=_MECHANISM,
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


def _print_scope(scope: Scope) -> None:
    print()
    print('=' * 92)
    regime = (
        'discovery (no threshold committed)'
        if scope.threshold is None
        else f'committed (threshold={scope.threshold:g})'
    )
    print(
        f'Scope[{scope.hypothesis_name}]  gap={scope.gap_name}  '
        f'alpha={scope.alpha}  regime={regime}'
    )
    print('=' * 92)
    print()
    print(
        f'  Phase 1 — cleavage on the gap '
        f'(n_strata={scope.cleavage.n_strata}, '
        f'R²={scope.cleavage.r_squared:.3f}, '
        f'intercept={scope.cleavage.intercept:+.3f})'
    )
    print(
        f'    {"covariate":<32} {"coef":>8} {"ci_lo":>8} '
        f'{"ci_hi":>8} {"p":>6} {"sig":>4}'
    )
    for c in scope.cleavage.coefficients:
        sig = '***' if c.is_significant else ''
        print(
            f'    {c.name:<32} {c.coefficient:>+8.3f} '
            f'{c.ci_lo:>+8.3f} {c.ci_hi:>+8.3f} '
            f'{c.p_value:>6.3f} {sig:>4}'
        )
    if scope.cleavage.cleavage_axes:
        print(
            f'    significant cleavage axes: '
            f'{list(scope.cleavage.cleavage_axes)!r}'
        )
    else:
        print(f'    no significant cleavage at alpha={scope.alpha}')

    print()
    print(f'  Phase 2 — chain ({len(scope.chain.edges)} edges)')
    for ge in scope.chain.edges:
        meta = ge.metadata
        print(
            f'    {ge.source!r} → {ge.target!r}  '
            f'tier={meta.tier.name:<14} '
            f'level={meta.evidentiary_level:<18} '
            f'name={meta.bridge_name}'
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        '--runs-path', type=Path, default=_DEFAULT_RUNS,
    )
    _ = parser.add_argument(
        '--total-steps', type=int, default=None,
        help='Filter to one total_steps grid point (50000 or 200000).',
    )
    _ = parser.add_argument(
        '--outcome-path', type=str, default=_DEFAULT_OUTCOME,
    )
    _ = parser.add_argument(
        '--alpha', type=float, default=0.05,
    )
    _ = parser.add_argument(
        '--threshold', type=float, default=None,
        help=(
            'Committed scope threshold on the per-cell gap. '
            'Default None (discovery mode); pass a float to record '
            'an author commitment after seeing Phase-1 cleavage.'
        ),
    )
    args = parser.parse_args()
    runs_path = Path(args.runs_path)
    total_steps_filter = args.total_steps
    outcome_path = str(args.outcome_path)
    alpha = float(args.alpha)
    threshold = (
        None if args.threshold is None else float(args.threshold)
    )

    if not runs_path.exists():
        print(f'corpus not found: {runs_path}', file=sys.stderr)
        sys.exit(1)

    df = pl.read_parquet(runs_path)
    if 'arm_key' not in df.columns:
        print(
            f'{runs_path.name} has no `arm_key` column — '
            f'run `migrate_runs_inject_arm_key.py` first',
            file=sys.stderr,
        )
        sys.exit(1)

    if total_steps_filter is not None:
        df = df.filter(pl.col('total_steps') == total_steps_filter)

    total_steps_grid: list[int] = sorted(
        df.select(pl.col('total_steps').unique()).to_series().to_list(),
    )
    print(f'corpus: {df.height} rows; total_steps grid: {total_steps_grid}')

    all_rows = read_runrows(runs_path)

    for total_steps in total_steps_grid:
        ts_rows = [
            r for r in all_rows
            if r.measurements.get('total_steps') == total_steps
        ]
        print()
        print('=' * 92)
        print(
            f'DDQN scope-finding (total_steps={total_steps}, '
            f'outcome={outcome_path!r})'
        )
        print('=' * 92)

        treatment_h = _ddqn_hypothesis(outcome_path)
        baseline_h = _vanilla_hypothesis()
        treatment_arm_key = treatment_h.arm_key()
        treatment = [r for r in ts_rows if r.arm_key == treatment_arm_key]
        baseline = [r for r in ts_rows if r.arm_key == 'baseline']
        print(f'  treatment={len(treatment)}  baseline={len(baseline)}')

        verdict = hypothesis_subgraph_verdict(
            treatment_h, treatment, baseline,
            pair_by=('seed',), group_by='env_name',
            baseline_h=baseline_h, alpha=alpha,
        )

        print(
            f'  §3 pattern (mechanism, outcome, link): '
            f'{tuple(v.value for v in verdict.pattern())}'
        )

        try:
            scope = build_scope(
                verdict, baseline,
                gap_path=_GAP_PATH,
                gap_name=_GAP_NAME,
                role='outcome', alpha=alpha,
                threshold=threshold,
                log_scale=True,
            )
        except ValueError as e:
            print(f'  build_scope failed: {e}')
            continue

        _print_scope(scope)

        if scope.threshold is not None:
            # Committed mode: report how many envs land in scope.
            in_scope_envs: list[str] = []
            out_scope_envs: list[str] = []
            for gs in verdict.comparison_rows[
                outcome_path
            ].per_group:
                env_str = str(gs.group_value)
                # Compute per-env baseline gap mean inline for the
                # report (matches what build_scope used internally).
                gaps = [
                    float(r.measurements[_GAP_PATH])
                    for r in baseline
                    if r.measurements.get('env_name') == env_str
                    and isinstance(r.measurements.get(_GAP_PATH), (int, float))
                ]
                if not gaps:
                    continue
                mean_gap = sum(gaps) / len(gaps)
                if scope.is_in_scope(mean_gap):
                    in_scope_envs.append(env_str)
                else:
                    out_scope_envs.append(env_str)
            print()
            print(
                f'  scope predicate (gap <= {scope.threshold:g}): '
                f'{len(in_scope_envs)} in scope, '
                f'{len(out_scope_envs)} out of scope'
            )
            if in_scope_envs:
                print(f'    in scope: {sorted(in_scope_envs)!r}')
            if out_scope_envs:
                print(f'    out of scope: {sorted(out_scope_envs)!r}')


if __name__ == '__main__':
    main()
