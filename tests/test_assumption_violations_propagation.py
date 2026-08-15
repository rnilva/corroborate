"""End-to-end regression test for `assumption_violations`
propagation: analysis Result → BridgeEvaluation → BridgeReportEntry.

The chain:

    1. `paired_g.fn` populates `PairedGResult.assumption_violations`
       from heuristics calibrated against the empirical bias map
       at `tests/analytic/robustness/test_paired_g_skew_robustness.py`.
    2. `evaluate(bridge, cells)` reads `.assumption_violations` off
       each fixture's result and prefixes each string with the
       fixture name → `BridgeEvaluation.assumption_violations`.
    3. The runner's `_build_bridge_entry` copies through to
       `BridgeReportEntry.assumption_violations`.
    4. Eventually surfaces in `experiments/findings/<short>.run.json`
       so downstream consumers (drift sentinel, audit dashboards)
       see the flags alongside the verdict.

This test exercises steps 1-3. Step 4 (parquet round-trip into
the run.json schema) is covered separately by the snapshot
regression test once snapshots get regenerated.
"""
from __future__ import annotations

from collections.abc import Mapping

import polars as pl

import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.paired.paired_g import PairedGResult
from corroborate.bridge.bridge import (
    Direction, Tier, claim_bridge, evaluate,
)
from corroborate.bridge.verdict import Verdict


def _skewed_paired_cells(
    n_seeds: int, mu_log: float = 0.0, sigma_log: float = 0.7,
) -> list[Mapping[str, object]]:
    """Construct paired (treatment, baseline) cells where Δ is
    log-normally distributed (skew ≈ 1.86 at σ_log=0.7). The
    framework's paired_g heuristic should flag this.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    deltas = rng.lognormal(mu_log, sigma_log, n_seeds)
    cells: list[Mapping[str, object]] = []
    for s, d in enumerate(deltas):
        cells.append({
            'arm_key': 'T', 'seed': s, 'env_name': 'X',
            'value': float(d),
        })
        cells.append({
            'arm_key': 'B', 'seed': s, 'env_name': 'X',
            'value': 0.0,
        })
    return cells


@claim_bridge(
    source='value',
    target='value',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
)
def skew_bridge(
    paired_g: PairedGResult,
    *,
    treatment_arm: str = 'T',
    baseline_arm: str = 'B',
) -> Verdict:
    """Bridge that consumes paired_g and HOLDs when g > 0."""
    del treatment_arm, baseline_arm
    if paired_g.n_pairs < 2:
        return Verdict.POWER_INSUFFICIENT
    return Verdict.HELD if paired_g.g > 0 else Verdict.NO_EFFECT


def test_skewed_delta_assumption_violation_propagates_to_bridge_evaluation() -> None:
    """**Headline propagation test**: a bridge consuming paired_g
    on log-normal Δ surfaces the skew_bias_likely flag on
    `BridgeEvaluation.assumption_violations` — prefixed with the
    fixture name `paired_g:`.

    The framework's contract: analysis-level flags propagate to
    the bridge audit trail; implementation author don't have to
    inspect each fixture result individually."""
    cells = _skewed_paired_cells(n_seeds=30)
    out = evaluate(skew_bridge, cells)
    assert any(
        'skew_bias_likely' in flag
        for flag in out.assumption_violations
    ), (
        f'expected skew_bias_likely on log-normal Δ at n=30; '
        f'got assumption_violations = {out.assumption_violations}'
    )
    # Each flag is prefixed with the fixture name so the audit
    # reader can trace which fixture surfaced it.
    assert all(
        flag.startswith('paired_g:')
        for flag in out.assumption_violations
    ), (
        f'flags should be prefixed with fixture name; got '
        f'{out.assumption_violations}'
    )


def test_normal_delta_no_assumption_violations() -> None:
    """**Negative control**: under normal Δ, no flags fire.
    Validates the propagation doesn't over-fire on calibrated
    inputs.

    Uses n=200 for stable sample-skew estimation (sample skew SE
    is √(6/n); at n=30 a 3σ outlier sample like -1.35 trips the
    threshold even on true-normal data — that's a property of
    the heuristic, not a failure of the propagation. n=200 brings
    sample-skew SE to 0.17, so the heuristic doesn't fire at the
    1.0 threshold absent real population skew)."""
    import numpy as np
    rng = np.random.default_rng(1)
    deltas = rng.normal(1.0, 2.0, 200)
    cells: list[Mapping[str, object]] = []
    for s, d in enumerate(deltas):
        cells.append({
            'arm_key': 'T', 'seed': s, 'env_name': 'X',
            'value': float(d),
        })
        cells.append({
            'arm_key': 'B', 'seed': s, 'env_name': 'X',
            'value': 0.0,
        })

    out = evaluate(skew_bridge, cells)
    assert out.assumption_violations == (), (
        f'normal Δ at n=30 should produce no violations; '
        f'got {out.assumption_violations}'
    )


def test_assumption_violations_propagate_through_bridge_report_entry() -> None:
    """Step 3 of the propagation chain: `_build_bridge_entry`
    copies `BridgeEvaluation.assumption_violations` through to
    `BridgeReportEntry.assumption_violations`. Verifies the
    runner-side wiring."""
    # `_build_bridge_entry` is module-private; the regression test
    # exercises it directly to verify the runner-side wiring
    # without spinning up the full `run()` pipeline (which would
    # require a hypothesis module + cache writes + report writes).
    from corroborate.runner.report import _build_bridge_entry

    cells = _skewed_paired_cells(n_seeds=30)
    evaluation = evaluate(skew_bridge, cells)
    n_cells_total = pl.DataFrame(list(cells)).height
    entry = _build_bridge_entry(skew_bridge, evaluation, n_cells_total)
    assert entry.assumption_violations == evaluation.assumption_violations, (
        f'BridgeReportEntry.assumption_violations = '
        f'{entry.assumption_violations}; expected to match '
        f'BridgeEvaluation.assumption_violations = '
        f'{evaluation.assumption_violations}'
    )
    # And the contents are non-trivial — we DID fire a flag.
    assert any('skew_bias_likely' in f for f in entry.assumption_violations)
