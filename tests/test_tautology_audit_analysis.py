"""Smoke for the `tautology_audit` analysis. Synthetic corpus
covers the structural and stratified-ρ checks; the real-corpus
smoke confirms the analysis runs against a published corpus and
the jaccard reproduces FINDINGS revision 5's number for
jensen_gap (jaccard = 0.50)."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.tautology_audit import (
    AuditResult, tautology_audit,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
ACTION_DIM_PARQUET = (
    REPO_ROOT / 'experiments' / 'data' / 'action_dim_sweep'
    / 'runs.parquet'
)


def _synthetic_audit_cells() -> list[dict[str, object]]:
    """20 cells, 2 HPs (capacity, lr), 2 mediators:
    - `clean_mediator`: no overlap with outcome; varies with HP.
    - `outcome_shadow`: high jaccard with outcome (reads the
      same column).
    """
    cells: list[dict[str, object]] = []
    for i in range(20):
        cap = 10000 if i < 10 else 50000
        cells.append({
            'id': f'c{i}', 'parent_id': None, 'cycle_id': None,
            'timestamp': '2026-04-30T00:00:00',
            'verdict': 'held',
            'arm_key': 'baseline',
            'arm_key': 'arm',
            'capacity': cap,
            'lr': 0.001,
            'eval_final_mean': 0.5 + 0.01 * i,
            'mediator.clean': 0.1 * i + 0.2,
            'mediator.shadow': 0.5 + 0.01 * i,  # shadows outcome
        })
    return cells


def test_synthetic_audit_runs_and_returns_per_measurable() -> None:
    cells = _synthetic_audit_cells()
    result = tautology_audit.fn(
        cells,
        measurables=[
            {'name': 'mediator.clean', 'reads': ('clean_input',)},
            {
                'name': 'mediator.shadow',
                'reads': ('mc_return',),  # same as outcome's reads
            },
        ],
        outcome_path='eval_final_mean',
        outcome_reads=('mc_return',),
        hp_axes=('capacity', 'lr'),
        hp_stratum_axis='capacity',
    )
    assert isinstance(result, AuditResult)
    assert len(result.reports) == 2

    clean = result.by_name('mediator.clean')
    shadow = result.by_name('mediator.shadow')
    assert clean is not None
    assert shadow is not None
    # `mediator.shadow` reads the same column as the outcome →
    # jaccard = 1.0 → flagged_outcome.
    assert shadow.outcome_jaccard == 1.0
    assert shadow.flagged_outcome
    # `mediator.clean` reads a different column → jaccard = 0.
    assert clean.outcome_jaccard == 0.0
    assert not clean.flagged_outcome


def test_audit_registered() -> None:
    """`tautology_audit` populates the registry on import."""
    from corroborate.bridge.analysis import get_registered
    assert get_registered('tautology_audit') is not None


# ============ Real-corpus smoke ============

@pytest.fixture(scope='module')
def action_dim_cells() -> list[dict[str, object]]:
    if not ACTION_DIM_PARQUET.exists():
        pytest.skip('action_dim_sweep corpus not available')
    return list(
        pl.read_parquet(ACTION_DIM_PARQUET).iter_rows(named=True),
    )


def test_jensen_gap_jaccard_matches_findings_revision_5(
    action_dim_cells: list[dict[str, object]],
) -> None:
    """FINDINGS revision 5: jensen_gap reads
    `(predicted_q_at_start, mc_return)`; outcome reads
    `mc_return` → jaccard = 1/2 = 0.50.

    The number is structural (independent of corpus); this smoke
    just confirms the analysis computes it."""
    result = tautology_audit.fn(
        action_dim_cells,
        measurables=[
            {
                'name': 'jensen_gap',
                'reads': ('predicted_q_at_start', 'mc_return'),
            },
        ],
        outcome_path='eval_best_burst_mean',
        outcome_reads=('mc_return',),
        hp_axes=('total_steps',),
        arm_filter='ddqn',
    )
    report = result.by_name('jensen_gap')
    assert report is not None
    assert report.outcome_jaccard == 0.5  # matches FINDINGS rev 5
