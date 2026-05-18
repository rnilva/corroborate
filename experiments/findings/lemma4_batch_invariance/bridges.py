"""Pre-registered bridge: Lemma 4 → Corollary 4.1 predicts vanilla
`jensen_gap` is approximately B-invariant on the FR γ=0.999 ×
batch_size ∈ {32, 128, 512, 2048} panel.

Theory (THEORY note §7, eq. 11):
  E[∇L] = E[(s,a,y)~D ∇(Q(s,a) − y)²] — independent of B.
  Var[∇L] = O(1/B).

Corollary 4.1: Theorem 1's Λ_m doesn't depend on B → the expected
regime classification is B-invariant. Empirically, at the
EXPECTED Bellman fixed point, jens magnitude is determined by
σ_action × √(2 ln K) × γ/(1-γ) and the FA-truncation floor —
neither depends on B.

Caveat (THEORY note §7): finite-T SGD trajectories can escape
unfavorable attractors via lucky high-variance steps at small B.
So while expected divergence direction is B-invariant, the
*probability of escape* from bias-attraction during finite-T
training MAY depend on B. The empirical sweep tests whether
this caveat bites at FR γ=0.999 × 1M-step training.

Pre-registered refutation criterion (THEORY note §12):
  significant negative trend ρ(B, jens) ≤ −0.5.

Pre-registered prediction:
  HELD iff |ρ| ≤ 0.5 AND p ≥ 0.05 (B-invariance approximately
  holds).
  NO_EFFECT iff p < 0.05 AND |ρ| > 0.5 (finite-T escape probability
  significantly B-dependent).

Honest prior (not formal prediction):
  HELD likely. Lemma 4 is a textbook SGD result; the variance
  asymmetry would need to drive a >0.5 Spearman ρ to refute. At
  FR γ=0.999 in 1M steps the algorithm doesn't reach the exact
  bias-equilibrium (q_late,V ≈ 8 vs Lemma-2 analytic 18.4),
  leaving room for finite-T effects. But the magnitude needed
  for refutation (|ρ| > 0.5 across 4 B-levels with n=30 each)
  is substantial.

Data source: `experiments/probes/fr_batch_size_sweep/` (in flight
as of 2026-05-18 — bridge will resolve once batch_2048 completes
and the top-level merge produces the canonical corpus). The B=32
anchor cells come from the canonical FR γ=0.999 cache."""
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


_FR_VANILLA_GAMMA999_SCOPE = (
    (pl.col('env_name') == 'FourRooms-misc')
    & (pl.col('gamma') == 0.999)
    & (pl.col('arm_key') == 'baseline')
    & (pl.col('total_steps') == 1000000)
    & pl.col('replay.batch_size').is_in([32, 128, 512, 2048])
    & finite(pl.col('jensen_gap'))
    & finite(pl.col('replay.batch_size'))
)


@claim_bridge(
    source='replay.batch_size',
    target='jensen_gap',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_FR_VANILLA_GAMMA999_SCOPE,
    predicted_direction='null',
)
def lemma4_b_invariance__fr_g999_vanilla(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'replay.batch_size',
    y: str = 'jensen_gap',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 30,
    rho_threshold: float = 0.5,
) -> Verdict:
    """Pre-registered bridge: Spearman ρ(batch_size, jensen_gap)
    at FR γ=0.999 × MLP × unshaped × baseline arm × B ∈ {32, 128,
    512, 2048} × n=30 seeds each.

    HELD iff |ρ| ≤ 0.5 AND p ≥ 0.05 — Lemma 4 → Corollary 4.1's
    B-invariance prediction is approximately confirmed at finite
    (1M-step) training.

    NO_EFFECT iff p < 0.05 AND |ρ| > 0.5 — finite-T escape
    probability significantly B-dependent. The §7 caveat bites:
    Theorem 1 covers expected-fixed-point regime classification,
    not per-trajectory escape from bias-attraction in finite T.

    Pre-registration: THEORY note §12 (committed at `b416432`)
    states this refutation criterion. The bridge encodes the
    falsifiable prediction in framework primitives."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return spearman_rho_verdict(
        partial_spearman,
        sign=0,
        threshold=rho_threshold,
    )
