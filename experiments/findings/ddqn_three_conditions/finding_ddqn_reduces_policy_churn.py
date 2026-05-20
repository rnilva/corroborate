"""Folkloric "DDQN reduces churn" intuition fails at γ=0.999 sparse-
reward; DDQN's HIGHER churn here is consistent with Schaul 2022's
churn-as-implicit-exploration framing.

**What this Finding does and does not claim, after critic audit.**

A reviewer-grade audit (2026-05-20) corrected two overclaims in the
first version of this Finding:

1. **Schaul et al. 2022 ("The Phenomenon of Policy Churn", NeurIPS,
   arXiv:2206.00730) does NOT predict that DDQN reduces churn.**
   Schaul explicitly positions churn as "a beneficial but overlooked
   form of implicit exploration." The "DDQN reduces churn" framing
   was a folklore inference downstream of Schaul, NOT a Schaul claim.
   This Finding does not refute Schaul; it refutes the *folklore*
   that "stable policy = converged = good = low churn."

2. **FourRooms-misc was missing an env-specific `state_hash`.**
   The substrate previously fell back to `default_state_hash`
   (returns 0 for every obs). The FR `policy_churn_late`
   measurement therefore degenerated to a global consecutive-step
   argmax-flip rate. **Fixed** by registering `_FOURROOMS_HASH`
   in `env_catalogue.py` (no-bucket id over the 4-int obs
   `[agent_y, agent_x, goal_y, goal_x]` on the 13×13 grid;
   cardinality 13^4 = 28561). The bridge scope now requires
   `state_hash_n_unique_late > 1` to filter out cells with the
   pre-fix degenerate constant-0 hash. Pre-fix FR cells will not
   admit; the bridge enters `EMPTY_EXTENT` until FR cells are
   re-run under the new substrate. SI (already has `_SI_HASH`)
   continues to materialize.

The pre-registered prediction `predicted_direction='a_lt_b'` (DDQN
< vanilla) was the *folkloric* prediction — not a Schaul prediction.
Both bridges sign-flipped against that folklore.

**Materialized result (post-state-hash-fix):**

  ddqn_reduces_policy_churn__si_g999 (strict state-conditional churn):
    vanilla mean: 0.678
    DDQN mean:    0.708
    mean_diff:    +0.030
    Cohen's d:    +1.91
    p-value:      6.11e-9
    Verdict:      NO_EFFECT (SIGN_FLIP vs folkloric a_lt_b)

  ddqn_reduces_policy_churn__fr_g999:
    EMPTY_EXTENT until FR cells are re-run with `_FOURROOMS_HASH`.
    Pre-fix cells have `state_hash_n_unique_late = 1`; the scope
    predicate `state_hash_n_unique_late > 1` excludes them.

Composed cluster verdict: **REFUTED**. SI alone refutes the
folkloric a_lt_b prediction at adequate power (d=+1.91, p=6e-9);
FR is corroborative-pending-rerun, not a gate.

**Interpretive reading (hypothesis, not claim).** At γ→1 sparse-
reward, DDQN's higher argmax-flip rate is consistent with Schaul's
framing of churn as implicit exploration: DDQN's clip-mediated
Q-stabilization frees the policy to RESPOND to changing return
estimates rather than stay frozen on a bias-inflated argmax. Paired
with `finding_temporal_ordering_at_fr_g999` (SUPPORTED — DDQN's
`policy_growth_fraction` ≈ 0.80 vs vanilla ≈ 0):

|              | policy_growth_fraction | argmax-flip rate |
|--------------|------------------------|------------------|
| vanilla      | LOW (≈0)               | LOW (≈0.24 FR / 0.68 SI) |
| DDQN         | HIGH (≈0.80)           | HIGH (≈0.32 FR / 0.71 SI) |

At γ→1 sparse-reward, the actively-learning arm has BOTH higher
growth fraction AND higher flip rate. This is interpretive — the
joint shape could reflect (a) DDQN actively exploring while vanilla
freezes, OR (b) DDQN visiting a wider state distribution while
vanilla cycles through a narrow set. The two interpretations are
distinguishable via a per-cell state-visitation-diversity measure
(not yet implemented).

**Concerns for external citation (left open):**

- The FR bridge's "state-conditional" framing is misleading given
  the constant state_hash; rename or re-implement with a real
  `(pos, goal)`-based hash on FR before paper-level citation.
- "Pre-registration" was within-commit (measurable + bridges +
  Finding landed in commit `068bb30`, materialization ran from the
  same source). The framework's drift detector cannot fire on
  within-commit registration. Future churn-on-other-envs bridges
  should commit predicted_direction in a SEPARATE earlier commit
  for genuine pre-registration.
- The mechanism narrative (active-exploration vs frozen-policy) is
  post-hoc and not directly tested by these bridges.

Cross-refs:
- `corroborate_rl.dqn.measurables.policy_churn_late` — the Schaul-
  inspired measurable. Strict state-conditional form applies only
  when the env's `state_hash` is non-degenerate (SI, Asterix,
  Breakout, Freeway, classic-control envs have real hashes; FR-misc
  + MetaMaze + image-obs envs fall back to constant 0).
- `THEORY_bootstrap_dominance.md` §11 — literature positioning.
- `finding_temporal_ordering_at_fr_g999` — sibling on the hybrid
  axis-1+axis-3 (`policy_growth_fraction`).
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.policy_churn import (
    ddqn_reduces_policy_churn__fr_g999,
    ddqn_reduces_policy_churn__si_g999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


# FR re-run under the new `_FOURROOMS_HASH` substrate would
# corroborate the cross-env transfer at FR γ=0.999 but isn't a gate
# on the cluster verdict — SI alone refutes the folkloric a_lt_b
# prediction (state-conditional, d=+1.91, p=6e-9). BLOCKED_ON cleared
# to avoid CLAUDE.md "terminal-verdict-with-BLOCKED_ON" contradiction.
BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_reduces_policy_churn__si_g999,
    ddqn_reduces_policy_churn__fr_g999,
)
