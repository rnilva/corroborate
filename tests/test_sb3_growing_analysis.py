"""A growing external record reaches the native hypothesis path intact.

Use real producer logs, and remove one declared horizon as a negative
control. No extra seeds or verdicts are manufactured for these tests.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import polars as pl
import pytest

from corroborate.analyses.paired.paired_directional import PairedDirectionalResult
from corroborate.bridge import Verdict
from corroborate.data import concat_panels
from corroborate.runner import run
from corroborate_rl.sb3_logger import load_sb3_logger_runs

from examples.sb3_demo import hypothesis


_RUNS = Path(__file__).parents[1] / 'examples' / 'sb3_demo' / 'runs'


def test_same_hypothesis_accepts_later_batch_without_reauthoring() -> None:
    records = load_sb3_logger_runs(_RUNS)
    first = records.narrow(pl.col('config/seed') < 2)
    second = records.narrow(pl.col('config/seed') >= 2)
    initial = run(hypothesis, data=first, use_cache=False)
    combined = concat_panels((first, second))
    updated = run(hypothesis, data=combined, use_cache=False)
    direct = run(hypothesis, data=records, use_cache=False)

    assert combined.leaves == records.leaves
    for name in initial:
        early = initial[name].analysis_results['paired_directional']
        later = updated[name].analysis_results['paired_directional']
        expected = direct[name].analysis_results['paired_directional']
        assert isinstance(early, PairedDirectionalResult)
        assert isinstance(later, PairedDirectionalResult)
        assert isinstance(expected, PairedDirectionalResult)
        assert early.n_pairs == 2 and later.n_pairs == 3
        assert early.mean_diff != pytest.approx(later.mean_diff)
        assert later.mean_diff == pytest.approx(expected.mean_diff)
        assert updated[name].verdict is direct[name].verdict
        assert updated[name].blocked_by is None
        assert updated[name].warnings == ()


def test_missing_early_horizon_does_not_rebase_or_change_late_claim() -> None:
    records = load_sb3_logger_runs(_RUNS)
    original = run(hypothesis, data=records, use_cache=False)
    # A record without the declared 15k evaluation cannot substitute a
    # nearby point. The 25k observations remain exactly the same.
    missing_early = replace(records, df=records.df.with_columns(
        pl.col('time/total_timesteps').list.eval(
            pl.when(pl.element() == 15_000)
            .then(14_999)
            .otherwise(pl.element()),
        ),
    ))
    reevaluated = run(hypothesis, data=missing_early, use_cache=False)
    early = reevaluated['higher_gamma_improves_early_return']
    assert early.verdict is Verdict.POWER_INSUFFICIENT
    evidence = early.analysis_results['paired_directional']
    assert isinstance(evidence, PairedDirectionalResult)
    assert evidence.n_pairs == 0

    name = 'higher_gamma_improves_return'
    before = original[name].analysis_results['paired_directional']
    after = reevaluated[name].analysis_results['paired_directional']
    assert isinstance(before, PairedDirectionalResult)
    assert isinstance(after, PairedDirectionalResult)
    assert after.n_pairs == before.n_pairs == 3
    assert after.mean_diff == before.mean_diff
    assert reevaluated[name].verdict is original[name].verdict
