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


_FR_VANILLA_HIGH_B_SCOPE = (
    (pl.col('env_name') == 'FourRooms-misc')
    & (pl.col('gamma') == 0.999)
    & (pl.col('arm_key') == 'baseline')
    & (pl.col('total_steps') == 1000000)
    & pl.col('replay.batch_size').is_in([512, 2048])  # high-B where escape variance is detectable
    & finite(pl.col('jensen_gap'))
    & finite(pl.col('eval_best_burst_mean'))
)


_FR_LR_SWEEP_SCOPE = (
    (pl.col('env_name') == 'FourRooms-misc')
    & (pl.col('gamma') == 0.999)
    & (pl.col('arm_key') == 'baseline')
    & (pl.col('total_steps') == 1000000)
    & (pl.col('replay.batch_size') == 128)
    & pl.col('optimizer.inner.lr').is_in([2.5e-5, 1e-4, 2e-4])
    & finite(pl.col('jensen_gap'))
    & finite(pl.col('optimizer.inner.lr'))
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


@claim_bridge(
    source='jensen_gap',
    target='eval_best_burst_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_FR_VANILLA_HIGH_B_SCOPE,
    predicted_direction='a_lt_b',
)
def mechanism_jens_predicts_outcome_within_high_B__fr_g999_vanilla(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'jensen_gap',
    y: str = 'eval_best_burst_mean',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'replay.batch_size',
    min_stratum_size: int = 30,
    rho_threshold: float = 0.4,
) -> Verdict:
    """At FR γ=0.999 × vanilla × B ∈ {512, 2048} (n=60), Spearman
    ρ(jensen_gap, eval_best_burst_mean) stratified by batch_size
    tests the Theorem 1 mechanism chain at the per-seed level:
    bias-attraction (high jens) → trapped at zero reward;
    escape (low jens) → policy succeeds.

    HELD iff ρ ≤ −0.4 (significantly negative) AND p < 0.05.

    Why B ∈ {512, 2048} only: at B=128 effectively all 30 seeds
    are stuck in the bias-attraction basin (median outcome=0,
    jens uniformly high). Within-B variance is dominated by
    measurement noise rather than mechanism, so the per-seed
    chain is undetectable. At B=512/2048, the escape fraction
    (17-20%) produces a bimodal distribution where the within-B
    mechanism chain emerges (escapees: jens ≈ 1.2-1.9; stuck:
    jens ≈ 1.9-2.7).

    Pilot per-B Spearman ρ (analysis 2026-05-19):
        B=128:  ρ = +0.21 NS (no escape variance to correlate)
        B=512:  ρ = −0.48 p=0.008
        B=2048: ρ = −0.67 p<0.001

    Pooled across B ∈ {512, 2048} via Fisher-z, expect ρ ≤ −0.5
    with strong significance. This corroborates Theorem 1's chain
    at the per-seed level: the bias-attraction regime is real,
    AND the predicted mechanism (high bias → poor policy) holds
    within-stratum once there's outcome variance to correlate."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return spearman_rho_verdict(
        partial_spearman,
        sign=-1,
        threshold=rho_threshold,
    )


@claim_bridge(
    source='optimizer.inner.lr',
    target='jensen_gap',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_FR_LR_SWEEP_SCOPE,
    predicted_direction='a_gt_b',
)
def lr_drives_jens_up__fr_b128_g999_vanilla(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'optimizer.inner.lr',
    y: str = 'jensen_gap',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 30,
    rho_threshold: float = 0.3,
) -> Verdict:
    """Pre-registered (YAML sweep `fr_lr_sweep_at_b128`):
    Spearman ρ(α, jens) at FR γ=0.999 × MLP[64,64] × B=128 × baseline
    × α ∈ {2.5e-5, 1e-4, 2e-4} × n=30 seeds each = 90 cells.

    Tests the α/√B effective-noise hypothesis: if higher α (at
    fixed B) inflates effective gradient noise, the bias chain
    amplifier (Lemma 2) compounds more aggressively → higher jens.

    HELD iff ρ ≥ +0.3 AND p < 0.05. Predicted direction: a_gt_b
    (positive correlation, larger α → larger jens).

    α values chosen to span the canonical α/√B reference grid:
    α=2.5e-5 → α/√B = 2.21e-6 (matches canonical B=2048)
    α=1e-4   → α/√B = 8.84e-6 (matches canonical B=128 lemma4-ref)
    α=2e-4   → α/√B = 1.77e-5 (matches canonical B=32 default)

    Resolution:
    - HELD: α drives jens at fixed B → the Lemma 4 refutation's
      mechanism is effective-noise, not pure-B. The expectation-
      invariance holds at constant α/√B.
    - NO_EFFECT (significant negative): higher α → lower jens?
      Would suggest the mechanism is something other than
      bias chain amplification (counterexample to Lemma 2's
      reading).
    - POWER_INSUFFICIENT: 3 α levels with 30 seeds each may not
      have enough across-stratum variance to detect.

    Sibling to `lemma4_b_invariance__fr_g999_vanilla` (the B-sweep
    pre-registration). Together they disentangle Lemma 4's
    refutation mechanism."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return spearman_rho_verdict(
        partial_spearman,
        sign=+1,
        threshold=rho_threshold,
    )
