"""Theorem 3 (THEORY note §6.1) empirical bridges — tests the
(A4'a) magnitude-alignment assumption and visualises the
geometric-series argmax-accumulation gap noted as an open
limitation parallel to §9.3's Robbins-Monro gap.

Both bridges operate on Breakout + Asterix γ ∈ {0.95, 0.99, 0.999}
from the `minatar_gamma_sweep_k2` corpus (effective K=6, 60
cells/env, 6 sub-corpora = 360 cells). The two cell-level
measurables `q_lambda_a_tail_cv` and `q_lambda_a_growth_ratio`
(`corroborate_rl/dqn/measurables.py`) materialise the per-burst
σ_Λa trajectory's converged-tail stability and init-to-converged
drift. Bridges stratify by env_name and pool ρ via Fisher-z.

Bridge 1 — `a4a_tail_cv_invariant_across_gamma__minatar_gamma_sweep`:
  Tests whether converged-tail CV of σ_Λa is small and γ-invariant.
  Predicted: NULL (|ρ| < 0.3) — (A4'a) holds across γ.

Bridge 2 — `geometric_gap_scales_with_gamma__minatar_gamma_sweep`:
  Tests whether init-to-converged growth ratio scales with γ.
  Predicted: DIRECT (positive ρ) — the open limitation γ-scales.

Both bridges are vanilla-arm-restricted (consistent with σ_Λa^env
measurement convention in `findings_theorem3_sigma_clip_validation`).
The (A4'a) test does not depend on arm — it characterises within-cell
trajectory stability."""
from __future__ import annotations

import polars as pl

from corroborate.analyses.spearman.partial_spearman import (
    PartialSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.predicates import finite
from corroborate.bridge.verdict import Verdict
from experiments.findings.ddqn_three_conditions._verdicts import (
    spearman_rho_verdict,
)


_MINATAR_GAMMA_SCOPE = (
    pl.col('env_name').is_in(['Breakout-MinAtar', 'Asterix-MinAtar'])
    & pl.col('gamma').is_in([0.95, 0.99, 0.999])
    & (pl.col('arm_key') == 'baseline')
    & finite(pl.col('q_lambda_a_tail_cv'))
    & finite(pl.col('q_lambda_a_growth_ratio'))
)


@claim_bridge(
    source='gamma',
    target='q_lambda_a_tail_cv',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_MINATAR_GAMMA_SCOPE,
    predicted_direction='null',
)
def a4a_tail_cv_invariant_across_gamma__minatar_gamma_sweep(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'gamma',
    y: str = 'q_lambda_a_tail_cv',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 30,
    rho_threshold: float = 0.3,
) -> Verdict:
    """At Breakout + Asterix × baseline × γ ∈ {0.95, 0.99, 0.999}
    (n=180 cells in `minatar_gamma_sweep_k2`, stratified by env),
    Spearman ρ(γ, q_lambda_a_tail_cv) Fisher-z-pooled across envs
    tests whether the converged-tail CV of σ_Λa is γ-invariant.

    (A4'a) prediction (THEORY §6.1): one-step ≈ converged σ_clip
    alignment HOLDS in the converged-iterate regime → tail CV is
    small and γ-invariant. HELD iff |ρ| ≤ 0.3 AND p ≥ 0.05.

    Refutations:
    - NO_EFFECT (significant positive ρ): tail CV GROWS with γ →
      (A4'a) violated at high γ (the regime Theorem 3 targets).
    - NO_EFFECT (significant negative ρ): tail CV SHRINKS with γ
      — alignment improves with discount; doesn't refute (A4'a)
      but motivates a different bridge framing.

    Pilot ad-hoc analysis (`findings_theorem3_a4a_empirical_test`)
    showed CV ∈ [3.3%, 13.6%] across γ — small in absolute terms;
    γ-trend modest. Expected verdict: HELD."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return spearman_rho_verdict(
        partial_spearman,
        sign=0,
        threshold=rho_threshold,
    )


@claim_bridge(
    source='gamma',
    target='q_lambda_a_growth_ratio',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_MINATAR_GAMMA_SCOPE,
    predicted_direction='a_gt_b',
)
def geometric_gap_scales_with_gamma__minatar_gamma_sweep(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'gamma',
    y: str = 'q_lambda_a_growth_ratio',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 30,
    rho_threshold: float = 0.3,
) -> Verdict:
    """At Breakout + Asterix × baseline × γ ∈ {0.95, 0.99, 0.999}, Spearman
    ρ(γ, q_lambda_a_growth_ratio) tests whether init-to-converged
    σ_Λa drift scales with γ — the geometric-series
    argmax-accumulation gap (THEORY §6.1 open limitation, parallel
    to §9.3's Robbins-Monro gap for Theorem 1).

    Predicted: ρ > +0.3 — at higher γ the bias-amplification
    mechanism produces larger accumulated drift; the open
    limitation is γ-scaled, not γ-invariant.

    Refutations:
    - NULL (|ρ| < 0.3): growth ratio is γ-invariant → the
      geometric-series gap is a fixed multiplicative correction,
      not a γ-amplified one.
    - NO_EFFECT (significant negative ρ): growth shrinks with γ
      — surprising; would suggest σ_Λa stabilises faster at
      higher γ, contradicting the pilot reading.

    Pilot showed growth 2.4× (γ=0.95) → 4.6× (γ=0.99 and γ=0.999).
    Expected verdict: HELD with positive ρ."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return spearman_rho_verdict(
        partial_spearman,
        sign=+1,
        threshold=rho_threshold,
    )
