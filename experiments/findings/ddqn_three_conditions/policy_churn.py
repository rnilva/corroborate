"""Schaul-aligned policy-churn bridges at FR γ=0.999 + SI γ=0.999.

Schaul et al. 2022 ("The Phenomenon of Policy Churn", NeurIPS,
arXiv:2206.00730) defines per-state policy distance
`W(π, π'|s) = ½ Σ_a |π(a|s) − π'(a|s)|` between consecutive policy
snapshots; aggregating over states gives a "churn fraction" — for
deterministic policy, the fraction of state-time-pair samples where
the argmax flipped between snapshots. The lit's implicit prediction:
DDQN's clip reduces Q oscillation, so DDQN reduces policy churn vs
vanilla.

Our `policy_churn_late` measurable in `corroborate_rl.dqn.measurables`
is the trace-stream analog of Schaul's quantity: for each state that
recurs in the late 50% of training, fraction of consecutive same-state
appearances where the online argmax flipped. Pools weighted by
occurrence count. See THEORY_bootstrap_dominance.md §11 for the full
literature positioning.

Two scopes are pre-registered as direct tests of Schaul's prediction
at γ→1, where standard DDQN-vs-vanilla regimes are stressed:

1. **SI γ=0.999** — the "noisy-vanilla" regime where Q inflates with
   non-frozen policy. Schaul-aligned prediction: vanilla churn HIGH
   (Q oscillates → argmax flips), DDQN churn LOWER (clip stabilizes
   Q → fewer flips). Predicted direction: `a_lt_b` (DDQN < vanilla).

2. **FR γ=0.999** — the "stuck-vanilla" regime where vanilla policy
   freezes pre-anchoring. Schaul's published prediction was made for
   *healthy* DQN training; at this extreme γ→1 sparse-reward scope,
   vanilla churn may already be LOW (policy frozen → argmax stable
   even without DDQN). DDQN cells should ALSO be low-churn (policy
   anchored after rescue). Predicted direction: STILL `a_lt_b`
   (Schaul transfers) but expected magnitude smaller than SI; could
   be NO_EFFECT if both arms are floored.

If both HELD → Schaul's churn-reduction prediction transfers
cleanly to γ→1 sparse-reward envs.
If only SI HELDs → transfer fails in the stuck-vanilla regime
(consistent with vanilla churn floor being structurally low).
If neither → DDQN's effect on policy churn doesn't generalize
to γ→1 OR our state-conditional-revisit form isn't the right
operationalization of Schaul's between-snapshot form.

Pre-registration source-hash captures the predicted directions
before the corpus's `policy_churn_late` column existed; framework
drift detector catches if the materialized verdict diverges.
"""
from __future__ import annotations

import math

import polars as pl

from corroborate.analyses.paired.arm_mean_diff import ArmMeanDiffResult
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import RefutationClass, Verdict
from corroborate.bridge.predicates import finite

from experiments.findings.ddqn._arms import DDQN_ARM, INTERVENTION, VANILLA_ARM


# Scope predicates require `state_hash_n_unique_late > 1` to filter
# out cells whose `state_hash_per_step` is the degenerate constant-0
# default. FourRooms-misc cells collected BEFORE the `_FOURROOMS_HASH`
# registration (commit lands the hash in env_catalogue.py) have
# state_hash_per_step ≡ 0 — they will not admit. Re-running FR cells
# under the new substrate emits non-degenerate hashes and admits them.
# SI cells (registered `_SI_HASH`) admit at both old and new commits.
_FR_CANONICAL_G999_SCOPE: pl.Expr = (
    (pl.col('env_name') == 'FourRooms-misc')
    & (pl.col('gamma') == 0.999)
    & (pl.col('replay.batch_size') == 32)
    & (pl.col('fa_kind') == 'mlp_deep')
    & (pl.col('shaping_kind') == 'none')
    & (pl.col('total_steps') == 1000000)
    & finite(pl.col('state_hash_n_unique_late'))
    & (pl.col('state_hash_n_unique_late') > 1.5)
    & finite(pl.col('policy_churn_late'))
)


_SI_CANONICAL_G999_SCOPE: pl.Expr = (
    (pl.col('env_name') == 'SpaceInvaders-MinAtar')
    & (pl.col('gamma') == 0.999)
    & finite(pl.col('state_hash_n_unique_late'))
    & (pl.col('state_hash_n_unique_late') > 1.5)
    & finite(pl.col('policy_churn_late'))
)


def _signed_d_verdict_lt(
    result: ArmMeanDiffResult,
    *,
    d_floor: float,
    sign_flip_floor: float,
    null_band: float,
    alpha: float,
) -> tuple[Verdict, RefutationClass | None]:
    """HELD if Cohen's d ≤ -d_floor AND p < alpha (predicted negative —
    DDQN < vanilla).
    NO_EFFECT/SIGN_FLIP if d ≥ +sign_flip_floor with sig.
    NO_EFFECT/NULL if |d| < null_band.
    POWER_INSUFFICIENT otherwise."""
    d = result.standardized_effect
    p = result.mean_diff_p_value
    if math.isnan(d) or math.isnan(p):
        return Verdict.POWER_INSUFFICIENT, None
    if d <= -d_floor and p < alpha:
        return Verdict.HELD, None
    if d >= sign_flip_floor and p < alpha:
        return Verdict.NO_EFFECT, RefutationClass.SIGN_FLIP
    if abs(d) < null_band:
        return Verdict.NO_EFFECT, RefutationClass.NULL_EFFECT
    return Verdict.POWER_INSUFFICIENT, None


@claim_bridge(
    source=INTERVENTION,
    target='policy_churn_late',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=_SI_CANONICAL_G999_SCOPE,
    predicted_direction='a_lt_b',
)
def ddqn_reduces_policy_churn__si_g999(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'policy_churn_late',
    pair_by: tuple[str, ...] = ('seed',),
    d_floor: float = 0.6,
    sign_flip_floor: float = 0.3,
    null_band: float = 0.2,
    alpha: float = 0.05,
) -> tuple[Verdict, RefutationClass | None]:
    """At SI γ=0.999 × canonical, DDQN cells have LOWER state-
    conditional policy churn (Schaul 2022 `W(π,π')` analog on per-
    step argmax-at-recurring-state pairs) than vanilla cells.

    Schaul-aligned prediction: vanilla's inflated Q oscillates more
    than DDQN's clipped Q → vanilla argmax flips more frequently at
    revisited states → DDQN < vanilla churn. This is the lit's
    canonical "DDQN reduces churn" hypothesis tested at the γ→1
    extreme where Q magnitudes diverge most.

    Verdict matrix on Cohen's d (DDQN − vanilla):
      HELD              : d ≤ -0.6 AND p < 0.05 (DDQN reduces churn)
      NO_EFFECT (NULL)  : |d| < 0.2
      NO_EFFECT (SIGN_FLIP) : d ≥ +0.3 (DDQN INCREASES churn — refutes
                              Schaul's transfer to γ→1)
      POWER_INSUFFICIENT : otherwise

    Pre-registered direction: `a_lt_b`. Pre-registered verdict: HELD.
    Source-hash committed before `policy_churn_late` was populated."""
    del treatment_arm, baseline_arm, source, pair_by
    return _signed_d_verdict_lt(
        arm_mean_diff,
        d_floor=d_floor,
        sign_flip_floor=sign_flip_floor,
        null_band=null_band,
        alpha=alpha,
    )


@claim_bridge(
    source=INTERVENTION,
    target='policy_churn_late',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    scope=_FR_CANONICAL_G999_SCOPE,
    predicted_direction='a_lt_b',
)
def ddqn_reduces_policy_churn__fr_g999(
    arm_mean_diff: ArmMeanDiffResult,
    *,
    treatment_arm: str = DDQN_ARM,
    baseline_arm: str = VANILLA_ARM,
    source: str = 'policy_churn_late',
    pair_by: tuple[str, ...] = ('seed',),
    d_floor: float = 0.5,
    sign_flip_floor: float = 0.3,
    null_band: float = 0.2,
    alpha: float = 0.05,
) -> tuple[Verdict, RefutationClass | None]:
    """At FR γ=0.999 × MLP × unshaped × B=32, does Schaul's
    "DDQN reduces churn" prediction transfer to the stuck-vanilla
    regime?

    FR γ=0.999 vanilla cells are largely frozen pre-anchoring
    (`policy_growth_fraction ≈ 0`); their late-window policy
    churn might already be low because the policy doesn't change.
    DDQN cells anchor to sparse reward and stabilize. Both arms
    plausibly low-churn — testing whether DDQN's reduction effect
    survives in this regime.

    Sibling of `..._si_g999`. Together: cluster test of "Schaul's
    churn-reduction prediction transfers to γ→1 sparse-reward
    envs across both the noisy-vanilla (SI) and stuck-vanilla
    (FR) regimes."

    Pre-registered direction: `a_lt_b`. Pre-registered verdict:
    HELD (could also legitimately resolve as NO_EFFECT given the
    structural floor)."""
    del treatment_arm, baseline_arm, source, pair_by
    return _signed_d_verdict_lt(
        arm_mean_diff,
        d_floor=d_floor,
        sign_flip_floor=sign_flip_floor,
        null_band=null_band,
        alpha=alpha,
    )
