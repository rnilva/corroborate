"""Evaluate the same authored graph as compatible observations arrive.

Run ``python -m examples.sb3_demo.growing_analysis`` from the repo root.
This example splits the committed real records into two disjoint batches
to demonstrate accumulation. It never launches or modifies experiments.
"""
from __future__ import annotations

from pathlib import Path

import polars as pl

from corroborate.analyses.paired.paired_directional import PairedDirectionalResult
from corroborate.data import concat_panels
from corroborate.runner import run
from corroborate_rl.sb3_logger import load_sb3_logger_runs

from examples.sb3_demo import hypothesis


def main() -> None:
    records = load_sb3_logger_runs(Path(__file__).parent / 'runs')
    first = records.narrow(pl.col('config/seed') < 2)
    second = records.narrow(pl.col('config/seed') >= 2)
    for label, panel in (
        ('First two seeds', first),
        ('After the third seed arrives', concat_panels((first, second))),
    ):
        evaluations = run(hypothesis, data=panel, use_cache=False)
        print(f'\n{label}:')
        for name, evaluation in evaluations.items():
            evidence = evaluation.analysis_results.get('paired_directional')
            if isinstance(evidence, PairedDirectionalResult):
                print(f'  {name}: {evaluation.verdict.name}; '
                      f'n={evidence.n_pairs}, Δ={evidence.mean_diff:+.2f}')
            else:
                print(f'  {name}: {evaluation.verdict.name}; '
                      f'blocked by {evaluation.blocked_by}')


if __name__ == '__main__':
    main()
