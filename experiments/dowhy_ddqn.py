"""DoWhy-backed corpus smoke on the DDQN acceptance data.

Composes the discovery findings from `paper_full_range.py` (§4-§6)
with interventional-tier estimation:

- Discovery told us `mechanism.jensen_gap → outcome` is depth-1
  spurious; `mediator.learning_curve_auc` is the strongest
  observational mediator. We test BOTH at INTERVENTIONAL tier
  here using `backdoor_ate` + 2 refuters, with a posited DAG.

- For each candidate edge, run the 1-estimate + 2-refuter triple.
  When all three HELD on the same (source, target), the
  `promote_bridged_evidence` post-pass upgrades to
  `'causal_bridged'`.

- Each variable is z-scored *per env* so within-env variation
  drives the regression — env-level confounding (different
  reward scales, different action dims) is stripped without
  needing to one-hot the 18-env factor.

Usage:
    uv run python experiments/dowhy_ddqn.py
    uv run python experiments/dowhy_ddqn.py --total-steps 200000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

from collections.abc import Mapping

from corroborate.bridge import Bridge, BridgeResult
from corroborate.bridges_dowhy import (
    backdoor_ate, placebo_refutation, random_common_cause_refutation,
)
from corroborate.causal_graph import (
    build_causal_graph, promote_bridged_evidence,
)


_DEFAULT_RUNS = Path(
    '/workspace/corroborate/experiments/data/ddqn/'
    'runs_with_mediators.parquet'
)
_OUTCOME = 'outcome.eval_best_burst_mean'
_GAP = 'mechanism.jensen_gap'
_LC_AUC = 'mediator.learning_curve_auc'
_LC_AUC_PEAK = 'mediator.learning_curve_auc_peak_truncated'


def _build_corpus_record(
    df: pl.DataFrame, columns: list[str],
) -> dict[str, np.ndarray]:
    """Z-score each variable PER ENV, stack across envs, return as
    `{column: ndarray[n_cells]}`. Strips env-level confounding so
    a pooled regression captures within-env effects only.

    Drops cells with NaN/null in any of the requested columns —
    dowhy's regression doesn't accept NaNs."""
    needed = ['env_name', 'arm_ddqn', *columns]
    df_clean = df.select(needed).drop_nulls()
    for c in columns + ['arm_ddqn']:
        df_clean = df_clean.filter(~pl.col(c).is_nan())
    print(f'  rows after NaN filter: {df_clean.height}')

    out: dict[str, list[float]] = {c: [] for c in ['arm_ddqn', *columns]}
    for env, sub in df_clean.group_by('env_name'):
        del env
        if sub.height < 10:
            continue
        for c in columns:
            # polars→numpy and numpy reduction returns are typed
            # loosely; coerce at the boundary.
            v: np.ndarray = np.asarray(sub[c].to_numpy(), dtype=np.float64)
            std = float(v.std())  # pyright: ignore[reportAny]
            if std == 0.0 or not np.isfinite(std):
                continue
            mean = float(v.mean())  # pyright: ignore[reportAny]
            out[c].extend(((v - mean) / std).tolist())  # pyright: ignore[reportAny]
        # arm_ddqn is binary; keep as-is.
        arm: np.ndarray = np.asarray(
            sub['arm_ddqn'].to_numpy(), dtype=np.float64,
        )
        out['arm_ddqn'].extend(arm.tolist())  # pyright: ignore[reportAny]
    # Equal-length sanity.
    lengths = {k: len(v) for k, v in out.items()}
    if len(set(lengths.values())) != 1:
        # Some columns dropped envs the others didn't; truncate to
        # the min length for safety.
        min_n = min(lengths.values())
        out = {k: v[:min_n] for k, v in out.items()}
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


def _run_triple(
    record: dict[str, np.ndarray],
    *,
    treatment: str,
    outcome: str,
    graph: list[tuple[str, str]],
    expected_sign: int,
    threshold: float,
    placebo_tol: float,
    rcc_tol: float,
) -> list[BridgeResult]:
    """Run the 1 estimate + 2 refuter triple. Print + return
    BridgeResults for build_causal_graph consumption."""
    print()
    print(f'  --- {treatment!r} → {outcome!r} ---')
    results: list[BridgeResult] = []
    bridge_factories: list[
        tuple[str, Bridge[Mapping[str, object]]]
    ] = [
        ('backdoor_ate', backdoor_ate(
            treatment, outcome, graph=graph,
            expected_sign=expected_sign, threshold=threshold,
        )),
        ('placebo_refutation', placebo_refutation(
            treatment, outcome, graph=graph, tolerance=placebo_tol,
        )),
        ('random_common_cause_refutation', random_common_cause_refutation(
            treatment, outcome, graph=graph, tolerance=rcc_tol,
        )),
    ]
    for label, bridge in bridge_factories:
        r = bridge(record)
        results.append(r)
        keystat: str
        if 'ate' in r.stats:
            ate_v = r.stats['ate']
            keystat = (
                f'ATE={float(ate_v):+.3f}'
                if isinstance(ate_v, (int, float)) else 'ATE=?'
            )
        elif 'placebo_ate' in r.stats:
            p = r.stats['placebo_ate']
            real = r.stats.get('real_ate', float('nan'))
            keystat = (
                f'placebo={float(p):+.3f} real={float(real):+.3f}'
                if isinstance(p, (int, float))
                and isinstance(real, (int, float))
                else 'placebo=? real=?'
            )
        elif 'drift' in r.stats:
            d = r.stats['drift']
            real = r.stats.get('real_ate', float('nan'))
            keystat = (
                f'drift={float(d):.4f} real={float(real):+.3f}'
                if isinstance(d, (int, float))
                and isinstance(real, (int, float))
                else 'drift=? real=?'
            )
        else:
            keystat = '—'
        print(
            f'    {label:<35} verdict={r.verdict.value:<22} {keystat}'
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument(
        '--runs-path', type=Path, default=_DEFAULT_RUNS,
    )
    _ = parser.add_argument(
        '--total-steps', type=int, default=200000,
    )
    args = parser.parse_args()
    # argparse types as Any — coerce at the boundary.
    runs_path = Path(args.runs_path)  # pyright: ignore[reportAny]
    total_steps = int(args.total_steps)  # pyright: ignore[reportAny]

    if not runs_path.exists():
        print(f'corpus not found: {runs_path}', file=sys.stderr)
        sys.exit(1)

    df = pl.read_parquet(runs_path)
    df = df.filter(pl.col('total_steps') == total_steps)
    df = df.with_columns(
        (pl.col('intervention_name') == 'ddqn').cast(pl.Int64).alias('arm_ddqn'),
    )
    print(f'corpus: {df.height} rows at total_steps={total_steps}')

    # Triple 1: arm_ddqn → outcome (the headline ATE).
    print()
    print('=' * 92)
    print('Triple 1 — arm_ddqn → outcome.eval_best_burst_mean')
    print('=' * 92)
    dag1 = [('arm_ddqn', _OUTCOME)]
    rec1 = _build_corpus_record(df, [_OUTCOME])
    res1 = _run_triple(
        rec1, treatment='arm_ddqn', outcome=_OUTCOME, graph=dag1,
        expected_sign=+1, threshold=0.05,
        placebo_tol=0.05, rcc_tol=0.05,
    )

    # Triple 2: mechanism.jensen_gap → outcome (the theorem-derived
    # mediator). Discovery said this was depth-1 spurious.
    print()
    print('=' * 92)
    print(f'Triple 2 — {_GAP} → {_OUTCOME}')
    print('=' * 92)
    dag2 = [(_GAP, _OUTCOME)]
    rec2 = _build_corpus_record(df, [_GAP, _OUTCOME])
    res2 = _run_triple(
        rec2, treatment=_GAP, outcome=_OUTCOME, graph=dag2,
        expected_sign=-1, threshold=0.05,  # Hasselt: smaller gap → larger return
        placebo_tol=0.05, rcc_tol=0.05,
    )

    # Triple 3: learning_curve_auc → outcome (the strongest data-
    # derived mediator from §4-§6 discovery).
    print()
    print('=' * 92)
    print(f'Triple 3 — {_LC_AUC} → {_OUTCOME}')
    print('=' * 92)
    dag3 = [(_LC_AUC, _OUTCOME)]
    rec3 = _build_corpus_record(df, [_LC_AUC, _OUTCOME])
    res3 = _run_triple(
        rec3, treatment=_LC_AUC, outcome=_OUTCOME, graph=dag3,
        expected_sign=+1, threshold=0.05,
        placebo_tol=0.05, rcc_tol=0.05,
    )

    # Build a unified CausalGraph + promote.
    print()
    print('=' * 92)
    print('CausalGraph after promote_bridged_evidence')
    print('=' * 92)
    g = build_causal_graph(res1 + res2 + res3)
    g = promote_bridged_evidence(g)
    print(f'  {len(g.nodes)} nodes, {len(g.edges)} edges')
    by_pair: dict[tuple[str, str], list[str]] = {}
    for e in g.edges:
        pair = (e.source, e.target)
        by_pair.setdefault(pair, []).append(
            f'{e.metadata.bridge_name} '
            f'[{e.metadata.tier.name}, '
            f'{e.metadata.evidentiary_level}]'
        )
    for pair, lines in sorted(by_pair.items()):
        print(f'  {pair[0]!r} → {pair[1]!r}:')
        for line in lines:
            print(f'    {line}')


if __name__ == '__main__':
    main()
