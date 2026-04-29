"""DoWhy backdoor on state_coverage_kl_uniform_late → outcome | HPs.

Tests whether `state_coverage_kl_uniform_late` — the only mediator
that survives all three tautology audits on the CartPole HP corpus
— remains a significant predictor of solving once the HPs are
explicitly backdoor-adjusted.

The DAG hypothesis (caller-posited):

    HP_capacity ──┐         ┌── HP_lr
                  ├──> SCV  ──> outcome.eval_final_mean
    HP_batch_size ┘         └── HP_sync_period
                            ↑
                            └── (and HPs directly)

Where `SCV = state_coverage_kl_uniform_late`. The HPs affect both
SCV (the treatment) and the outcome — they're confounders. Backdoor
adjustment conditions on the HPs to recover the SCV → outcome
direct effect.

Two outcomes for the framework's purposes:
- **Backdoor ATE near zero** (with placebo passing): the
  within-stratum ρ=+0.19 we observed was symptomatic — SCV is a
  downstream signature of solving, not an upstream cause. The
  candidate mediator dies cleanly even though the audit passed.
- **Backdoor ATE substantial + placebo + RCC HELD**: SCV is a
  rung-2-conditional-on-DAG mediator. The strongest non-
  interventional evidence we can extract from this corpus.

Usage:
    uv run python experiments/dowhy_state_coverage.py
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import polars as pl

from corroborate.bridge import Bridge, BridgeResult
from corroborate.bridges_dowhy import (
    backdoor_ate,
    placebo_refutation,
    random_common_cause_refutation,
)
from corroborate.causal_graph import (
    build_causal_graph,
    promote_bridged_evidence,
)


_RUNS = Path(
    'experiments/data/cartpole_hp_v3/runs_with_mediators.parquet'
)
_TREATMENT = 'state_coverage_kl'
_OUTCOME = 'outcome'
_HP_AXES: tuple[str, ...] = (
    'capacity', 'batch_size', 'lr', 'sync_period',
)

# Posited DAG: every HP edges to both SCV and outcome (confounders).
# SCV → outcome is the hypothesized causal edge we're testing.
_DAG: list[tuple[str, str]] = [
    *((hp, _TREATMENT) for hp in _HP_AXES),
    *((hp, _OUTCOME) for hp in _HP_AXES),
    (_TREATMENT, _OUTCOME),
]


def _build_corpus_record() -> dict[str, np.ndarray]:
    """Project the CartPole HP corpus to per-cell columns matching
    the DAG nodes. dowhy needs a 1-D record (1 row per observation),
    not a per-step record."""
    df = pl.read_parquet(_RUNS).select([
        'mediator.state_coverage_kl_uniform_late',
        'outcome.eval_final_mean',
        'replay.capacity',
        'replay.batch_size',
        'optimizer.inner.lr',
        'sync_period',
    ]).drop_nulls()

    # Drop rows with non-finite SCV or outcome.
    df = df.filter(
        ~pl.col('mediator.state_coverage_kl_uniform_late').is_nan()
        & ~pl.col('outcome.eval_final_mean').is_nan()
    )

    arr = lambda col: np.asarray(  # noqa: E731 — terse local lambda
        df[col].to_numpy(), dtype=np.float64,
    )
    return {
        _TREATMENT: arr('mediator.state_coverage_kl_uniform_late'),
        _OUTCOME: arr('outcome.eval_final_mean'),
        'capacity': arr('replay.capacity'),
        'batch_size': arr('replay.batch_size'),
        'lr': arr('optimizer.inner.lr'),
        'sync_period': arr('sync_period'),
    }


def _print_result(label: str, r: BridgeResult) -> None:
    keystat = ''
    if 'ate' in r.stats:
        ate = r.stats['ate']
        keystat = (
            f'ATE={float(ate):+.4f}'
            if isinstance(ate, (int, float)) else 'ATE=?'
        )
    elif 'placebo_ate' in r.stats:
        p = r.stats['placebo_ate']
        real = r.stats.get('real_ate', float('nan'))
        keystat = (
            f'placebo={float(p):+.4f} real={float(real):+.4f}'
            if isinstance(p, (int, float))
            and isinstance(real, (int, float))
            else 'placebo=?'
        )
    elif 'drift' in r.stats:
        d = r.stats['drift']
        real = r.stats.get('real_ate', float('nan'))
        keystat = (
            f'drift={float(d):.4f} real={float(real):+.4f}'
            if isinstance(d, (int, float))
            and isinstance(real, (int, float))
            else 'drift=?'
        )
    print(f'  {label:<35} verdict={r.verdict.value:<22} {keystat}')


def main() -> None:
    print('=' * 92)
    print(
        'DoWhy backdoor: state_coverage_kl_uniform_late → '
        'outcome.eval_final_mean | hp_axes'
    )
    print('=' * 92)

    record = _build_corpus_record()
    n = len(record[_TREATMENT])
    print(f'  n_cells={n}')
    print(
        f'  treatment range: '
        f'[{record[_TREATMENT].min():.3f}, '
        f'{record[_TREATMENT].max():.3f}]'
    )
    print(
        f'  outcome range: '
        f'[{record[_OUTCOME].min():.3f}, '
        f'{record[_OUTCOME].max():.3f}]'
    )
    print()

    rec_typed: Mapping[str, object] = record  # type: ignore[assignment]

    # Predicted direction: positive — higher state-coverage KL
    # (more concentrated visit distribution) predicts better
    # solving. Effect-size threshold 0.5 corresponds to a 1-unit
    # change in SCV moving the discounted-return outcome by ~0.5
    # units (~0.5% of the 0-99 outcome range; small but
    # meaningful). Refuter tolerances 1.0 ≈ 10% of a typical real
    # ATE on this corpus (~8-10 outcome units per SCV unit) — the
    # absolute "noise floor" the estimator can recover on permuted
    # / synthetic-confounded data.
    triple: list[tuple[str, Bridge[Mapping[str, object]]]] = [
        ('backdoor_ate', backdoor_ate(
            _TREATMENT, _OUTCOME, graph=_DAG,
            expected_sign=+1, threshold=0.5,
        )),
        ('placebo_refutation', placebo_refutation(
            _TREATMENT, _OUTCOME, graph=_DAG, tolerance=1.0,
        )),
        ('random_common_cause_refutation',
         random_common_cause_refutation(
            _TREATMENT, _OUTCOME, graph=_DAG, tolerance=1.0,
        )),
    ]
    results: list[BridgeResult] = []
    for label, bridge in triple:
        r = bridge(rec_typed)
        results.append(r)
        _print_result(label, r)

    # Build typed CausalGraph + promotion pass.
    print()
    print('CausalGraph after promote_bridged_evidence:')
    g = build_causal_graph(results)
    g = promote_bridged_evidence(g)
    pair = (_TREATMENT, _OUTCOME)
    edges = [
        e.metadata for e in g.edges
        if (e.source, e.target) == pair
    ]
    if not edges:
        print(f'  (no edges for {pair})')
    for md in edges:
        print(
            f'  {md.bridge_name:<50} '
            f'tier={md.tier.name:<14} '
            f'level={md.evidentiary_level}'
        )


if __name__ == '__main__':
    main()
