"""Real-corpus smoke for `tautology_audit` + registry probe.

The synthetic-input audit test previously living here was
superseded by the closed-form analytic substrate at
`tests/analytic/lg_scm/test_tautology_audit.py`, which
constructs four mediator candidates (outcome_shadow, hp_shadow,
hp_correlated, clean) on real LG-SCM cells and asserts each of
the three audit checks (structural jaccard, HP-R², stratified-ρ)
fires (or doesn't) per closed-form expectation. The HP-shadow
false-positive case is the headline test that the
`findings_tautology_audit` auto-memory calls out.

What stays here: registry probe + the FINDINGS-rev-5 jaccard
smoke against a published corpus."""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

import corroborate.analyses  # noqa: F401  # pyright: ignore[reportUnusedImport]

from corroborate.analyses.tautology_audit import tautology_audit


REPO_ROOT = Path(__file__).resolve().parent.parent
ACTION_DIM_PARQUET = (
    REPO_ROOT / 'experiments' / 'data' / 'action_dim_sweep'
    / 'runs.parquet'
)


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
