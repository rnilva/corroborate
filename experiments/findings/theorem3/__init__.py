"""Theorem 3 (DDQN-clip argmax preservation; THEORY note §6.1)
empirical hypothesis panel.

Two bridges + two Findings tested on Breakout γ ∈ {0.95, 0.99,
0.999} from the `minatar_gamma_sweep_k2` corpus (effective K=6,
60 cells/env). The bridges consume cell-level scalars derived
from the new `q_lambda_a_per_burst` measurable in
`corroborate_rl.dqn.measurables`.

What this module tests (Phase 1 scope, Breakout-only pilot):

- **(A4'a) magnitude alignment in converged tail.** The
  `a4a_tail_cv_invariant_across_gamma__minatar_gamma_sweep` bridge
  computes Spearman ρ(γ, q_lambda_a_tail_cv) on the
  baseline-arm cells. Predicted NULL (|ρ| < 0.3) — converged-tail
  CV is γ-invariant and small in absolute terms (the (A4'a)
  prediction). Finding `finding_a4a_holds_in_converged_tail`.

- **Geometric-series gap γ-scaling.** The
  `geometric_gap_scales_with_gamma__minatar_gamma_sweep` bridge computes
  Spearman ρ(γ, q_lambda_a_growth_ratio). Predicted POSITIVE
  (ρ > +0.3) — the init-to-converged drift scales with γ. The
  THEORY §6.1 open limitation (parallel to §9.3's Robbins-Monro
  gap) is real but γ-bounded. Finding
  `finding_geometric_gap_scales_with_gamma`.

What this module does NOT test (Phase 2, deferred):

- (A4'a) at the harm-anchor env (Asterix γ=0.999) — traces are
  cloud-only at the canonical 1M corpus, restoration deferred.
- The full Theorem 3 corruption-side chain σ_Λa^env → d_out;
  that lives in `findings_theorem3_sigma_clip_validation`
  (memory) and was committed to the THEORY note at `b416432`.

Cache population canonical via `corroborate hypothesis
experiments.findings.theorem3 --ingest minatar_gamma_sweep_k2`.
"""
from __future__ import annotations

import polars as pl

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate analysis registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate measurable registry

from corroborate_rl.dqn.dqn import dqn
from experiments.findings.ddqn_three_conditions._arms import (
    INTERVENTION as INTERVENTION,
)
from experiments.findings.theorem3 import (
    finding_a4a_invariance_refuted,
    finding_geometric_gap_scales_with_gamma,
)
from experiments.findings.theorem3.bridges import (
    a4a_tail_cv_invariant_across_gamma__minatar_gamma_sweep,
    geometric_gap_scales_with_gamma__minatar_gamma_sweep,
)


CLAIM = dqn


MODULE_SCOPE = pl.col('env_name').is_in([
    'Breakout-MinAtar', 'Asterix-MinAtar',
    'Freeway-MinAtar', 'SpaceInvaders-MinAtar',
])


BRIDGES = (
    a4a_tail_cv_invariant_across_gamma__minatar_gamma_sweep,
    geometric_gap_scales_with_gamma__minatar_gamma_sweep,
)


FINDINGS = (
    finding_a4a_invariance_refuted,
    finding_geometric_gap_scales_with_gamma,
)


REQUIRED_MEASURABLES: tuple[str, ...] = (
    'q_lambda_a_per_burst',
    'q_lambda_a_tail_cv',
    'q_lambda_a_tail_mean',
    'q_lambda_a_init_mean',
    'q_lambda_a_growth_ratio',
)


__all__ = [
    'BRIDGES',
    'CLAIM',
    'FINDINGS',
    'INTERVENTION',
    'MODULE_SCOPE',
    'REQUIRED_MEASURABLES',
]
