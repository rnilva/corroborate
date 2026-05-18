"""Polarity-conditional mediation of DDQN's outcome benefit by jens.

The substrate's "DDQN benefits flow through bias reduction" claim
admits a NUANCED disambiguation that the global pooled
`intervention_outcome_link_null__mech_conditioned` bridge collapses
into a single "null partial" verdict. The honest reading per
canonical re-analysis (2026-05-14):

- On REACH-polarity envs (sparse-terminal reward; pol < -0.3):
  Acrobot, MetaMaze, MountainCar, Snake. Marginal ρ(bg, outcome)
  per-env Fisher-z = **−0.30** (significant, p=0.0002).
  Partial ρ(bg, outcome | jens) per-env Fisher-z = **−0.10**
  (n.s., p=0.23). Partialling jens kills ~67% of the marginal
  signal → MEDIATION SUPPORTED on REACH.

- On SURVIVE-polarity envs (cumulative-positive reward;
  pol > +0.3): Asterix, Breakout, PacMan, SI. Marginal
  ρ(bg, outcome) = **−0.02** (null). Partial = +0.05. No
  marginal signal to mediate → NO LINK on SURVIVE.

Cluster shape: 3 bridges at polarity-disjoint scopes form a
Finding-level conjunction (`finding_mediation_polarity_conditional`)
that distinguishes full-mediation from no-link substantively —
something the global pooled bridge can't do because it averages
across cohorts.

Substrate distinction: REACH polarity envs are where DDQN
typically reduces bg (Hasselt's downward clip is operative on
the over-estimating regime); SURVIVE polarity envs are where DDQN
doesn't reduce bg (the per-step clip's propagation to trained-Q
can flip; see memory `findings_clip_to_trained_q_propagation`).
The mediation pattern follows: mediation HELDs where mech is
operative, no link where mech doesn't fire."""
from __future__ import annotations

import polars as pl

from corroborate.analyses.dowhy.mediation_dowhy import MediationResult
from corroborate.analyses.spearman.partial_spearman import (
    PartialSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.predicates import finite_gt, finite_lt
from corroborate.bridge.verdict import Verdict

from experiments.findings.ddqn._verdicts import (
    mediation_linearity_verdict,
    partial_spearman_null_verdict,
    partial_spearman_signed_verdict,
)


# DAG for bg → jens → outcome (single-mediator chain + direct
# path). Used by the linearity-diagnostic sibling bridge as the
# `mediation_dowhy` adjustment input.
_BG_JENS_OUTCOME_DAG: tuple[tuple[str, str], ...] = (
    ('bootstrap_gap_magnitude', 'jensen_gap'),
    ('jensen_gap', 'eval_best_burst_raw_mean'),
    ('bootstrap_gap_magnitude', 'eval_best_burst_raw_mean'),
)


_REACH_SCOPE: pl.Expr = (
    pl.col('eval_best_burst_raw_mean').is_finite()
    & pl.col('jensen_gap').is_finite()
    & pl.col('bootstrap_gap_magnitude').is_finite()
    & finite_lt('env_reward_polarity', -0.3)
)


_SURVIVE_SCOPE: pl.Expr = (
    pl.col('eval_best_burst_raw_mean').is_finite()
    & pl.col('jensen_gap').is_finite()
    & pl.col('bootstrap_gap_magnitude').is_finite()
    & finite_gt('env_reward_polarity', +0.3)
)


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.INVERSE,
    tier=Tier.ASSOCIATIONAL,
    scope=_REACH_SCOPE,
    predicted_direction='a_lt_b',
)
def bg_outcome_link_held_negative__reach_envs(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'bootstrap_gap_magnitude',
    y: str = 'eval_best_burst_raw_mean',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.2,
    min_strata: int = 3,
) -> Verdict:
    """Marginal ρ(bg, outcome) per-env Fisher-z pooled on REACH-
    polarity envs. Predicted-negative: DDQN reduces bg → outcome
    improves on sparse-terminal envs. HELD when ρ ≤ −threshold.

    Empirical: ρ_pool = −0.30, p=0.0002, n_strata=4 → HELD.

    This bridge establishes the EXISTENCE of a bg→outcome link on
    REACH envs. The mediation question (does it flow through
    jens?) is the sibling
    `bg_outcome_fully_mediated_by_jens__reach_envs`."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        partial_spearman,
        threshold=rho_threshold, sign=-1, min_strata=min_strata,
    )


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_REACH_SCOPE,
    predicted_direction='null',
)
def bg_outcome_fully_mediated_by_jens__reach_envs(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'bootstrap_gap_magnitude',
    y: str = 'eval_best_burst_raw_mean',
    conditioning: tuple[str, ...] = ('jensen_gap',),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    null_max_abs_rho: float = 0.2,
    min_strata: int = 3,
) -> Verdict:
    """Partial ρ(bg, outcome | jens) per-env Fisher-z pooled on
    REACH-polarity envs. Predicted NULL: the bg→outcome link
    documented by the sibling bridge flows ENTIRELY through jens
    (Hasselt's full-mediation reading). HELD when |ρ_partial| <
    null_max_abs_rho.

    Empirical: ρ_pool = −0.10, p=0.23, n_strata=4 → HELD as null.
    The marginal signal (−0.30) drops by ~67% under partialling
    jens — substantial mediation evidence.

    Together with the sibling, this bridge's HELD says:
    on REACH envs DDQN's outcome benefit IS mediated through jens
    (bias reduction)."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_null_verdict(
        partial_spearman,
        max_abs_rho=null_max_abs_rho, min_strata=min_strata,
    )


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_SURVIVE_SCOPE,
    predicted_direction='null',
)
def bg_outcome_link_null__survive_envs(
    partial_spearman: PartialSpearmanResult,
    *,
    x: str = 'bootstrap_gap_magnitude',
    y: str = 'eval_best_burst_raw_mean',
    conditioning: tuple[str, ...] = (),
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    null_max_abs_rho: float = 0.2,
    min_strata: int = 3,
) -> Verdict:
    """Marginal ρ(bg, outcome) per-env Fisher-z pooled on SURVIVE-
    polarity envs. Predicted NULL: on cumulative-positive reward
    envs, DDQN doesn't substantially reduce bg (Hasselt clip
    propagation flips per `findings_clip_to_trained_q_propagation`),
    so there's no marginal bg→outcome link to begin with. HELD
    when |ρ_marginal| < null_max_abs_rho.

    Empirical: ρ_pool = −0.02, p=0.79, n_strata=4 → HELD as null.
    No marginal link → mediation question is moot (no signal to
    mediate). Combined with the REACH bridges, this bridge says:
    on SURVIVE envs DDQN's mech doesn't fire AND there's no bg-
    mediated outcome benefit."""
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_null_verdict(
        partial_spearman,
        max_abs_rho=null_max_abs_rho, min_strata=min_strata,
    )


@claim_bridge(
    source='bootstrap_gap_magnitude',
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=_REACH_SCOPE,
    predicted_direction='null',
)
def bg_outcome_mediation_linearity_holds__reach_envs(
    mediation_dowhy: MediationResult,
    *,
    treatment: str = 'bootstrap_gap_magnitude',
    outcome: str = 'eval_best_burst_raw_mean',
    mediators: tuple[str, ...] = ('jensen_gap',),
    dag: tuple[tuple[str, str], ...] = _BG_JENS_OUTCOME_DAG,
) -> Verdict:
    """Linearity-diagnostic sibling of
    `bg_outcome_fully_mediated_by_jens__reach_envs`.

    The canonical bridge asserts mediation via rank-based partial
    Spearman (`ρ(bg, outcome | jens) ≈ 0` per-env Fisher-z
    pooled). This sibling asserts that the LINEAR mediation
    decomposition is ALSO coherent at this scope —
    `mediation_dowhy.linearity_status == RELIABLE` (direct/total
    same sign + indirect_proportion in [0, 1]).

    The pair forms a HYPOTHESIS_AS_GRAPH §3b scope-cluster
    Finding (`finding_bg_jens_mediation_robust__reach`): when
    BOTH the rank-based AND linear identifications admit, the
    mediation claim survives both methodological lenses → joint
    evidence stronger than partial_spearman alone.

    The sibling REFUTES when linearity_status is SIGN_FLIPPED
    or OUT_OF_BOUNDS — at that scope the linear assumption is
    broken (the v10 FR γ-WHY failure mode applies), and the
    canonical partial_spearman answer is the trustworthy one
    standing alone. POWER_INSUFFICIENT when DAG identification
    fails or OLS is rank-deficient."""
    del treatment, outcome, mediators, dag
    return mediation_linearity_verdict(mediation_dowhy)


BRIDGES = (
    bg_outcome_link_held_negative__reach_envs,
    bg_outcome_fully_mediated_by_jens__reach_envs,
    bg_outcome_mediation_linearity_holds__reach_envs,
    bg_outcome_link_null__survive_envs,
)
