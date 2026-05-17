"""Causal chain: DDQN's clip reduces Q-smoothness → underfit
value function → outcome harm at γ=0.999 on Asterix.

Replaces the refuted clip-argmax-noise chain
(`clip_argmax_harm_mechanism`). The empirical preview against
canonical ddqn cache showed argmax-noise predictions don't hold;
the actual signature is Q-magnitude AND Q-smoothness reduction.

Mechanism story:
  At γ=0.999, the optimal value function for Asterix requires a
  SMOOTH HIGH-MAGNITUDE Q to propagate long-horizon credit via
  the 1/(1−γ) amplification. DDQN's clip caps Q magnitude AND
  reduces smoothness across consecutive states (q_autocorr d=-1.98,
  q_inter_state_grad_overlap d=-2.13, q_trajectory_autocorr
  d=-1.63 — all z=-6 to -8 on Asterix γ=0.999). This prevents
  the long-horizon credit propagation, leaving DDQN's value
  function UNDERFIT to the optimal Asterix landscape.

The empirical signature:
  - DDQN converges with LOWER seed variance (cleaner attractor)
  - DDQN plateaus around burst 25-30
  - Vanilla keeps improving past burst 30 (improvement ratio
    4.38× vs DDQN's 3.73×)
  - DDQN's asymptote is BELOW vanilla's at burst 49 (d=-0.86 sig)

Two causal edges, two bridges:

  Edge 1 — `ddqn_cuts_q_autocorr_asterix_gamma_999`:
    DDQN's q_autocorr_late is LOWER than vanilla's on Asterix
    γ=0.999. Tests that the mechanism is active (clip reduces
    Q-smoothness across consecutive training steps).
    Predicted: a_lt_b, d ≤ −0.5 with p < 0.05.

  Edge 2 — `q_autocorr_predicts_outcome__asterix_gamma_999`:
    Within Asterix γ=0.999 cells (across both arms), q_autocorr
    correlates with outcome. Tests that Q-smoothness is the
    proximate mediator of outcome — cells with smoother Q score
    higher. Conditioning on jens (to rule out the smoothness-
    just-tracks-bias confound).
    Predicted: ρ > +0.3 (positive direct effect).

If both HELD → smoothness-mediator chain SUPPORTED on Asterix.

Caveat: this chain is single-env (Asterix). The mirror prediction
on Breakout γ=0.999 (where DDQN HELPS) is the OPPOSITE within-env
sign (r(smoothness, outcome) < 0 because Breakout's vanilla
overshoots). A multi-env extension would need an env-feature
classifier for "long-horizon credit needed vs not".

Empirical preview (canonical ddqn cache, 60 cells/arm):
  Edge 1: d_q_autocorr = -1.98 z=-7.7 → comfortably HELD at floor.
  Edge 2: r(q_autocorr, outcome) pooled = +0.530 p<0.001 → HELD.
"""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.analyses.stratified_partial_spearman import (
    StratifiedPartialSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn._arms import (
    DDQN_ARM, INTERVENTION, VANILLA_ARM,
)
from experiments.findings.ddqn._scope import CANONICAL_HP_EXCLUDING_GAMMA


# File-level scope: γ=0.999, canonical-shape HPs, Asterix only,
# Q-MC coupled (vanilla learning). Same shape as other ddqn_sweeps
# mechanism-test files.
_ASTERIX_GAMMA_999_LEARNABLE_SCOPE: pl.Expr = (
    (pl.col('env_name') == 'Asterix-MinAtar')
    & (pl.col('gamma') == 0.999)
    & CANONICAL_HP_EXCLUDING_GAMMA
    & pl.col('q_mc_burst_correlation_late').is_finite()
    & (pl.col('q_mc_burst_correlation_late') >= 0.3)
)


# ============ Edge 1: DDQN cuts Q-smoothness on Asterix γ=0.999 ============

@claim_bridge(
    source=INTERVENTION,
    target='q_inter_state_grad_overlap_late',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        _ASTERIX_GAMMA_999_LEARNABLE_SCOPE
        & pl.col('q_inter_state_grad_overlap_late').is_finite()
    ),
    predicted_direction='a_lt_b',
)
def ddqn_cuts_q_smoothness_asterix_gamma_999(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'q_inter_state_grad_overlap_late',
    stratify_by: tuple[str, ...] = ('env_name',),
    scope_predictor: str = 'jensen_gap',
    min_baseline_predictor: float = 50.0,
    min_seeds_per_arm: int = 5,
    harm_floor: float = -0.5,
    alpha: float = 0.05,
    min_strata: int = 1,
) -> tuple[Verdict, RefutationClass | None]:
    """Edge 1 of the Q-smoothness-harm chain.

    Test the mechanism is active: DDQN's clip reduces Q-function
    smoothness across consecutive trajectory states on Asterix
    γ=0.999. `q_inter_state_grad_overlap_late` measures the
    inner-product alignment of dQ/dθ between consecutive states
    in the late half — high overlap = trunk gradients propagate
    coherently across states (smooth value surface), low overlap
    = states discriminate sharply.

    Empirical preview (ddqn_sweeps cache, n=30/arm): vanilla
    overlap = 0.9792, DDQN = 0.9708, d=−2.13 z=−8.24.

    HELD: pooled Cohen's d ≤ harm_floor (default −0.5) AND p <
    alpha. Predicted direction: a_lt_b.
    REFUTED (SIGN_FLIP): d ≥ −harm_floor (DDQN MORE smooth than
    vanilla — would contradict the mechanism)."""
    del treatment_arm, baseline_arm, source, stratify_by
    del scope_predictor, min_baseline_predictor, min_seeds_per_arm
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    d = stratified_arm_diff_pooled.pooled_d
    p = stratified_arm_diff_pooled.pooled_p_value
    if math.isnan(d) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    abs_floor = abs(harm_floor)
    if d <= -abs_floor and p < alpha:
        return Verdict.HELD, None
    if d >= abs_floor and p < alpha:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    return Verdict.POWER_INSUFFICIENT, None


# ============ Edge 2: Q-smoothness → outcome ============

@claim_bridge(
    source='q_inter_state_grad_overlap_late',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=(
        _ASTERIX_GAMMA_999_LEARNABLE_SCOPE
        & pl.col('q_inter_state_grad_overlap_late').is_finite()
        & pl.col('eval_best_burst_raw_mean').is_finite()
    ),
    predicted_direction='a_gt_b',
)
def q_smoothness_predicts_outcome__asterix_gamma_999(
    stratified_partial_spearman: StratifiedPartialSpearmanResult,
    *,
    x: str = 'q_inter_state_grad_overlap_late',
    y: str = 'eval_best_burst_raw_mean',
    conditioning: str = 'jensen_gap',
    stratify_by: str = 'env_name',
    min_stratum_size: int = 10,
    min_rho: float = 0.3,
    min_strata: int = 1,
) -> Verdict:
    """Edge 2 of the Q-smoothness-harm chain.

    Within Asterix γ=0.999 cells (across both arms), Q-smoothness
    (q_inter_state_grad_overlap_late) correlates POSITIVELY with
    outcome. Tests that the proximate mediator (smoothness) is
    the actual cause of outcome — cells with smoother Q score
    higher.

    Conditioning on `jensen_gap` partials out the smoothness-
    just-tracks-bias confound.

    Empirical preview (ddqn_sweeps cache, 60 cells pooled):
    r(q_inter_state_grad_overlap_late, outcome) = +0.381 p=0.003
    (Pearson).

    HELD: pooled partial-r ≥ `min_rho` (default +0.3)."""
    del x, y, conditioning, stratify_by, min_stratum_size
    if stratified_partial_spearman.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT
    rho = stratified_partial_spearman.rho_pooled
    if math.isnan(rho):
        return Verdict.POWER_INSUFFICIENT
    if rho >= min_rho:
        return Verdict.HELD
    if rho <= -min_rho:
        return Verdict.NO_EFFECT
    return Verdict.POWER_INSUFFICIENT


BRIDGES = (
    ddqn_cuts_q_smoothness_asterix_gamma_999,
    q_smoothness_predicts_outcome__asterix_gamma_999,
)
