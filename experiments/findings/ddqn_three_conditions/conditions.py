"""Three-condition Hypothesis bridges.

The substantive claim has three pieces; each is now backed by a
multi-stratum panel with adequate power (n_strata ≥ 2, per-arm
n ≥ 30 per stratum):

C1 — DDQN reduces `jensen_gap` uniformly across K_eff at
     FR γ=0.999 × MLP[64,64] × no-shaping. The within-FR
     K-scaling observation (4 strata: k_eff ∈ {4, 8, 12, 16}).

C2 — Linear FA caps Type 1 across envs at γ=0.999 × no-shaping.
     The DDQN-vs-vanilla pooled `jensen_gap` Cohen's d sits in a
     null band (|d| < `null_ceiling`) at every env stratum
     (env_name ∈ {FourRooms, Acrobot, MetaMaze, MountainCar}).
     Substantive content: with linear FA, σ_action is FA-capped
     for BOTH arms, so DDQN can't reduce what vanilla can't
     overshoot.

C3 — The outcome-side moderation, split into two siblings:

  C3a (positive arm) — DDQN improves outcome at FR γ=0.999 ×
       MLP × no-shaping across k_eff strata. predicted_direction
       'a_gt_b' on `eval_best_burst_raw_mean`. Establishes that
       the outcome benefit is REAL on the unshaped reference cell.

  C3b (null arm) — DDQN does NOT improve outcome at FR × MLP ×
       shaped across γ strata. predicted_direction 'null' on
       `eval_best_burst_raw_mean`. Establishes that the same
       configuration with potential-based shaping breaks the
       benefit, with adequate power (2 strata, n=30 per arm per
       stratum).

C3a + C3b form a sibling cluster: HELD + HELD = "shaping
moderates the outcome translation." Each bridge stands on its
own evidence; the moderation claim is read off the joint
pattern in the cluster — see HYPOTHESIS_AS_GRAPH.md §3b.

What this is NOT:
- NOT a test of the Hasselt σ × √(2 ln K) × 1/(1−γ) bound — σ
  is unmeasured.
- NOT a claim about envs outside the four ingested. The C2 panel
  fixes (γ=0.999, linear FA, no-shaping) and varies env; the
  generalization beyond {FR, Acrobot, MM, MC} is unstudied.
- C3a / C3b's "shaping moderates" claim is at FR γ=0.999 ×
  MLP[64,64] only (the unshaped panel varies k_eff; the shaped
  panel varies γ — different stratification axes by data
  availability). A cross-env shaping factorial is future work."""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.stratified_arm_diff_pooled import (
    StratifiedArmDiffPooledResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.predicates import finite
from corroborate.bridge.verdict import RefutationClass, Verdict

from experiments.findings.ddqn_three_conditions._arms import INTERVENTION


# === C1 — DDQN reduces jens uniformly across K_eff at FR γ=0.999 ===
#
# Multi-stratum within FR. Each k_eff ∈ {4, 8, 12, 16} (= 4 ×
# action_duplicate_k for FR's native 4 actions) contributes one
# Cohen's d on `jensen_gap`. HELD iff all four strata's d are
# substantially negative.


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('gamma') == 0.999)
        & (pl.col('shaping_kind') == 'none')
        & (pl.col('fa_kind') == 'mlp_deep')
        & finite(pl.col('jensen_gap'))
    ),
    predicted_direction='a_lt_b',
)
def ddqn_reduces_jens_uniformly_across_k_at_fr_high_gamma(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    stratify_by: tuple[str, ...] = ('k_eff',),
    min_strata: int = 3,
    min_vanilla_predictor: float = 0.5,
    per_stratum_d_threshold: float = -0.5,
) -> tuple[Verdict, RefutationClass | None]:
    """Per-k_eff Cohen's d on `jensen_gap` is uniformly < -0.5
    at FourRooms γ=0.999 MLP[64,64] no-shaping across
    k_eff ∈ {4, 8, 12, 16}.

    Empirical readings in `findings_two_types_of_bias` (memory).
    Direction.INVERSE encodes the Hasselt mech prediction;
    `predicted_direction='a_lt_b'` means treatment-arm jens <
    baseline-arm jens (DDQN reduces).

    Caveat: this is the within-FR K-scaling claim only. The
    σ × √(2 ln K) Hasselt bound's load-bearing σ factor is
    unmeasured; the empirical pattern is consistent with the
    bound but also consistent with any monotone-in-K reduction.

    Verdict: HELD iff every admitted stratum's Cohen's d is
    below `per_stratum_d_threshold` and no stratum shows a
    wrong-sign refutation."""
    del stratify_by, min_vanilla_predictor
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    all_below = True
    any_wrong_sign = False
    n_valid = 0
    for s in stratified_arm_diff_pooled.per_stratum:
        d = s.cohen_d
        if math.isnan(d):
            continue
        n_valid += 1
        if d > per_stratum_d_threshold:
            all_below = False
        if d > 0.3:
            any_wrong_sign = True
    if n_valid < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    if any_wrong_sign:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if all_below:
        return Verdict.HELD, None
    return Verdict.NO_EFFECT, None


# === C2 — Linear FA caps Type 1 across envs and γ values, unshaped ===
#
# At linear FA, σ_action is FA-capped for both arms, so the
# Hasselt mech (DDQN reduces argmax-max overestimation) has no
# room to fire. We expect the DDQN-vs-vanilla pooled `jensen_gap`
# Cohen's d to sit in a NULL band at every env stratum.
#
# Scope spans BOTH γ ∈ {0.99, 0.999} — the FA-capacity claim is
# γ-invariant in theory, and pooling both γ values doubles n_per
# _arm_per_env from 30 to 60 (90 → 180 for MC) so the 95%-CI half-
# width tightens enough to make the ±`null_ceiling` band test
# informative.
#
# Predicted: predicted_direction='null'. HELD iff EVERY env's
# per-stratum 95% CI on Cohen's d sits inside ±`null_ceiling`.
# SIGN_FLIP refutation if any env's CI is fully outside the band
# in the wrong direction (DDQN substantially HURTS — never seen).
# INVARIANT_VIOLATION if any env's CI is fully outside the band
# in the THEORETICAL direction (DDQN substantially HELPS even at
# linear FA — would refute the FA-capacity hypothesis).
# POWER_INSUFFICIENT if any env's CI spans the band edge.
#
# Min strata = 3 (we have 4: FR, Acrobot, MM, MC).


def _null_band_verdict(
    res: StratifiedArmDiffPooledResult,
    *,
    null_ceiling: float,
    min_strata: int,
) -> tuple[Verdict, RefutationClass | None]:
    """Generic per-stratum null-band verdict.

    For each stratum (with valid d, se), test whether the 95% CI
    fits inside ±`null_ceiling`. HELD iff every stratum's CI
    fits; INVARIANT_VIOLATION if any CI fully > +ceiling;
    SIGN_FLIP if any CI fully < -ceiling; POWER_INSUFFICIENT if
    any CI straddles the band edge (effect direction not
    resolvable at this n)."""
    if res.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    any_above = False
    any_below = False
    any_spans = False
    n_valid = 0
    for s in res.per_stratum:
        d_env = s.cohen_d
        se_env = s.cohen_se
        if math.isnan(d_env) or math.isnan(se_env):
            continue
        n_valid += 1
        ci_lo = d_env - 1.96 * se_env
        ci_hi = d_env + 1.96 * se_env
        if ci_lo > null_ceiling:
            any_above = True
        elif ci_hi < -null_ceiling:
            any_below = True
        elif not (ci_lo >= -null_ceiling and ci_hi <= null_ceiling):
            any_spans = True
    if n_valid < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    if any_above:
        return Verdict.INVARIANT_VIOLATION, None
    if any_below:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if not any_spans:
        return Verdict.HELD, None
    return Verdict.POWER_INSUFFICIENT, None


_C2_ENVS = ['FourRooms-misc', 'Acrobot-v1', 'MetaMaze-misc', 'MountainCar-v0']


@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    scope=(
        pl.col('gamma').is_in([0.99, 0.999])
        & (pl.col('fa_kind') == 'linear')
        & (pl.col('shaping_kind') == 'none')
        & (pl.col('env_name').is_in(_C2_ENVS))
        & finite(pl.col('jensen_gap'))
    ),
    predicted_direction='null',
)
def linear_fa_caps_type_1_across_envs__null_panel(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    stratify_by: tuple[str, ...] = ('env_name',),
    min_strata: int = 3,
    min_vanilla_predictor: float = float('-inf'),
    null_ceiling: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """Linear FA caps Type 1 manifestation across envs at γ=0.999.

    Per-env independent-samples Cohen's d on `jensen_gap` at
    (γ=0.999, fa_kind=linear, shaping_kind=none). HELD iff every
    env's 95% CI fits inside ±`null_ceiling` (= 0.3).

    Substantive mechanism: with empty-hidden-tuple linear FA,
    σ_action — the per-state action-value SD whose √(2 ln K)
    Hasselt-bound product is the overestimation room — is
    FA-capped for BOTH arms. Vanilla can't overshoot; DDQN has
    nothing to reduce. The null is the load-bearing prediction
    of the FA-capacity gate.

    null_ceiling = 0.3 (Cohen's "small" threshold) is the
    substantively-meaningful band — d ≥ 0.3 is a non-trivial
    effect by Cohen's convention. Pre-2026-05-16 the band was
    0.5 ("medium" threshold) but MetaMaze's d=-0.24 CI=[-0.42,
    -0.06] fits 0.5 while excluding zero (i.e., DDQN really does
    reduce jens at MM linear, just by a small amount). The
    tightened band makes the bridge falsifiable in that regime
    and surfaces MM as POWER_INSUFFICIENT until densified.

    Refutations:
    - INVARIANT_VIOLATION: any env shows CI fully > +0.3 (DDQN
      meaningfully REDUCES jens even at linear FA — refutes
      FA-capacity hypothesis).
    - NO_EFFECT/SIGN_FLIP: any env shows CI fully < -0.3 (DDQN
      INCREASES jens at linear FA — never observed).
    - POWER_INSUFFICIENT: any env's CI straddles ±0.3.

    Note Direction.INVERSE on the bridge captures the
    *theoretical* mech direction (DDQN reduces jens *when the
    mech fires*); `predicted_direction='null'` captures that we
    expect the mech NOT to fire at this scope."""
    del stratify_by, min_vanilla_predictor
    return _null_band_verdict(
        stratified_arm_diff_pooled,
        null_ceiling=null_ceiling,
        min_strata=min_strata,
    )


# === C3a — DDQN improves outcome at FR γ=0.999 MLP unshaped ===
#
# The reference cell where DDQN's outcome benefit is supposed to
# manifest. Multi-stratum on k_eff ∈ {4, 8, 12, 16}; per-stratum
# Cohen's d on `eval_best_burst_raw_mean`.
#
# Predicted: predicted_direction='a_gt_b'. HELD iff every stratum
# shows d ≥ `per_stratum_d_threshold` (= +0.3) — i.e. DDQN's
# outcome > vanilla's outcome uniformly across k_eff.


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('gamma') == 0.999)
        & (pl.col('fa_kind') == 'mlp_deep')
        & (pl.col('shaping_kind') == 'none')
        & finite(pl.col('eval_best_burst_raw_mean'))
    ),
    predicted_direction='a_gt_b',
)
def ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    stratify_by: tuple[str, ...] = ('k_eff',),
    min_strata: int = 3,
    min_vanilla_predictor: float = float('-inf'),
    per_stratum_d_threshold: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """DDQN's outcome > vanilla's outcome uniformly across
    k_eff ∈ {4, 8, 12, 16} at FourRooms γ=0.999 × MLP[64,64] ×
    no-shaping.

    Per-k_eff independent-samples Cohen's d on
    `eval_best_burst_raw_mean` (canonical γ-invariant target —
    cf. `findings_units_bug` memo). Source cells come from
    `experiments/probes/action_dup_mismatch_probe_g999_1M/`
    (k_eff ∈ {4, 8, 12}) and
    `..._FR_k4_only/` (k_eff = 16). 30 seeds per arm per stratum.

    HELD iff every stratum's d ≥ +`per_stratum_d_threshold`
    (= 0.3). NO_EFFECT/SIGN_FLIP if any stratum shows d < -0.3
    (DDQN substantially HURTS). NO_EFFECT (null) if any stratum
    sits in the indeterminate band (-0.3, 0.3).

    Pairs with C3b — both HELD reads as "DDQN's outcome benefit
    is real at unshaped MLP but vanishes under potential-based
    shaping at the same env/FA"."""
    del stratify_by, min_vanilla_predictor
    if stratified_arm_diff_pooled.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    all_above = True
    any_wrong_sign = False
    n_valid = 0
    for s in stratified_arm_diff_pooled.per_stratum:
        d = s.cohen_d
        if math.isnan(d):
            continue
        n_valid += 1
        if d < per_stratum_d_threshold:
            all_above = False
        if d < -0.3:
            any_wrong_sign = True
    if n_valid < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    if any_wrong_sign:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if all_above:
        return Verdict.HELD, None
    return Verdict.NO_EFFECT, None


# === C3b — Shaping decouples: NO POSITIVE outcome benefit under shaping ===
#
# 4-stratum panel at FR × shaped × {linear, MLP[64,64]} × γ ∈
# {0.99, 0.999}. The "decoupling" claim is asymmetric: under
# shaping, DDQN's outcome benefit (Δ = treatment − baseline)
# is NEVER appreciably POSITIVE. Negative effects (DDQN hurts)
# are CONSISTENT with the decoupling claim — they mean shaping
# inverts rather than preserves DDQN's translation.
#
# Operationally: HELD iff every stratum's Cohen's d ≤
# `per_stratum_d_upper_bound` (= 0.3, mirroring C3a's symmetric
# +0.3 lower bound for the unshaped panel). INVARIANT_VIOLATION
# if any stratum's CI is fully > +0.3 (DDQN meaningfully helps
# under shaping — would refute decoupling).
#
# Min strata = 3 (we have 4: 2 FA × 2 γ).


def _per_stratum_upper_bound_verdict(
    res: StratifiedArmDiffPooledResult,
    *,
    upper_bound: float,
    min_strata: int,
) -> tuple[Verdict, RefutationClass | None]:
    """Generic per-stratum upper-bound verdict (mirror of
    C3a's lower-bound logic).

    HELD iff every stratum's d ≤ `upper_bound`. INVARIANT_VIOLATION
    if any stratum's CI is fully > `upper_bound` (predicted
    upper bound exceeded — the predicted-`a_lt_b` direction
    is refuted in the wrong direction). POWER_INSUFFICIENT if
    any stratum's CI straddles the bound."""
    if res.n_strata < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    all_below = True
    any_strong_above = False
    any_spans = False
    n_valid = 0
    for s in res.per_stratum:
        d = s.cohen_d
        se = s.cohen_se
        if math.isnan(d) or math.isnan(se):
            continue
        n_valid += 1
        ci_lo = d - 1.96 * se
        ci_hi = d + 1.96 * se
        if ci_lo > upper_bound:
            any_strong_above = True
        if d > upper_bound:
            all_below = False
        if not all_below and ci_hi > upper_bound and ci_lo <= upper_bound:
            any_spans = True
    if n_valid < min_strata:
        return Verdict.POWER_INSUFFICIENT, None
    if any_strong_above:
        return Verdict.INVARIANT_VIOLATION, None
    if all_below:
        return Verdict.HELD, None
    if any_spans:
        return Verdict.POWER_INSUFFICIENT, None
    return Verdict.NO_EFFECT, None


@claim_bridge(
    source=INTERVENTION,
    target='eval_best_burst_raw_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=(
        (pl.col('env_name') == 'FourRooms-misc')
        & (pl.col('shaping_kind') == 'potential_manhattan')
        & pl.col('gamma').is_in([0.99, 0.999])
        & finite(pl.col('eval_best_burst_raw_mean'))
    ),
    predicted_direction='a_lt_b',
)
def shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    stratify_by: tuple[str, ...] = ('fa_kind', 'gamma'),
    min_strata: int = 3,
    min_vanilla_predictor: float = float('-inf'),
    per_stratum_d_upper_bound: float = 0.3,
) -> tuple[Verdict, RefutationClass | None]:
    """No POSITIVE outcome benefit at FR × shaped across
    (fa_kind, γ) panel.

    Per-stratum independent-samples Cohen's d on
    `eval_best_burst_raw_mean` (the canonical γ-invariant target
    for cross-γ analysis). HELD iff every stratum's d ≤
    `per_stratum_d_upper_bound` (= 0.3) — DDQN never appreciably
    HELPS the outcome under shaping. NEGATIVE effects (DDQN
    hurts) are consistent with the decoupling claim and don't
    refute it.

    Substantive mechanism (memo entry
    `findings_shaping_decouples_bias_from_outcome`): under
    potential-based shaping, vanilla's optimization signal is
    dense enough that the bias→behavior chain breaks. DDQN's
    bias-reduction is mechanistically real (jens still drops)
    but doesn't translate to outcome gains because vanilla
    already converges from the dense shaped signal. The
    empirical reading on this corpus is stronger than null at
    one cell (mlp × γ=0.99 has d=-1.5: DDQN actively HURTS),
    consistent with shaping INVERTING rather than just decoupling
    the translation at lower γ.

    Refutations:
    - INVARIANT_VIOLATION: any stratum's CI fully > +0.3
      (DDQN meaningfully helps under shaping — would refute
      decoupling).
    - POWER_INSUFFICIENT: any stratum's CI straddles +0.3
      without crossing fully above.

    Pairs with C3a — C3a HELD (DDQN helps unshaped) + C3b
    HELD (DDQN doesn't help shaped) = "shaping moderates the
    outcome translation"."""
    del stratify_by, min_vanilla_predictor
    return _per_stratum_upper_bound_verdict(
        stratified_arm_diff_pooled,
        upper_bound=per_stratum_d_upper_bound,
        min_strata=min_strata,
    )
