"""Lemma 4 (Bellman-MSE SGD gradient is B-invariant in expectation)
empirical hypothesis panel — pre-registered.

One bridge + one Finding testing Corollary 4.1's prediction that
vanilla `jensen_gap` is approximately B-invariant at FR γ=0.999 ×
1M training across B ∈ {32, 128, 512, 2048}.

Pre-registration: THEORY note §12 committed at `b416432`. Refutation
criterion: |ρ(B, jens)| > 0.5 with p < 0.05.

Data source: `experiments/probes/fr_batch_size_sweep/` (in flight
as of 2026-05-18). B=32 anchor cells come from the canonical FR
γ=0.999 cache. Bridge resolves once `batch_2048` completes and the
top-level merge produces the canonical corpus.

Cache: `experiments/data/cache/lemma4_batch_invariance.parquet`
(materialises once ingest target lands).
"""
from __future__ import annotations

import polars as pl

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate analysis registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate measurable registry

from corroborate_rl.dqn.dqn import dqn
from experiments.findings.ddqn_three_conditions._arms import (
    INTERVENTION as INTERVENTION,
)
from experiments.findings.lemma4_batch_invariance import (
    finding_alpha_drives_jens_pre_registered,
    finding_b_invariance_pre_registered,
    finding_mechanism_corroborated_per_seed,
)
from experiments.findings.lemma4_batch_invariance.bridges import (
    lemma4_b_invariance__fr_g999_vanilla,
    lr_drives_jens_up__fr_b128_g999_vanilla,
    mechanism_jens_predicts_outcome_within_high_B__fr_g999_vanilla,
)


CLAIM = dqn


MODULE_SCOPE = (
    (pl.col('env_name') == 'FourRooms-misc')
    & (pl.col('gamma') == 0.999)
)


BRIDGES = (
    lemma4_b_invariance__fr_g999_vanilla,
    mechanism_jens_predicts_outcome_within_high_B__fr_g999_vanilla,
    lr_drives_jens_up__fr_b128_g999_vanilla,
)


FINDINGS = (
    finding_b_invariance_pre_registered,
    finding_mechanism_corroborated_per_seed,
    finding_alpha_drives_jens_pre_registered,
)


REQUIRED_MEASURABLES: tuple[str, ...] = (
    'jensen_gap',
    'eval_best_burst_mean',
)


__all__ = [
    'BRIDGES',
    'CLAIM',
    'FINDINGS',
    'INTERVENTION',
    'MODULE_SCOPE',
    'REQUIRED_MEASURABLES',
]
