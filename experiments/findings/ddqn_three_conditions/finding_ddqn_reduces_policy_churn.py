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

**Interpretive reading — partial resolution (2026-05-20).** A
state-visitation-diversity diagnostic (`ddqn_increases_state_diversity__si_g999`)
materialized HELD at SI γ=0.999: vanilla `state_hash_entropy_late`
4.13 → DDQN 4.61 (Cohen's d = +2.77, p = 4.9e-12). DDQN visits ~60%
more distinct state-hash buckets in the late window
(exp(4.13) ≈ 62 vs exp(4.61) ≈ 100).

This **weakens** the "DDQN actively learning at fixed states"
reading. The state-diversity effect size (d=+2.77) actually
EXCEEDS the churn effect size (d=+1.91), so the higher churn
could be fully explained — or even overshot — by state-distribution
drift alone. A pair (s_t, s_{t+k}) of same-hash appearances that
"flip" their argmax might actually be at semantically-distinct
states that collapsed into one hash bucket.

The honest joint reading at γ→1 sparse-reward:

|              | policy_growth_fraction | argmax-flip rate | state-visit entropy |
|--------------|------------------------|------------------|---------------------|
| vanilla      | LOW (≈0)               | LOW (≈0.24 FR / 0.68 SI) | LOW (4.13 SI) |
| DDQN         | HIGH (≈0.80)           | HIGH (≈0.32 FR / 0.71 SI) | HIGH (4.61 SI) |

DDQN's three coupled signatures at γ→1 sparse-reward: (1) higher
trajectory progress on the policy side, (2) more argmax flips
between consecutive state-revisits, (3) wider state distribution
visited. These cannot be cleanly disentangled into a "pure"
mechanism with the current measurables — they're all part of
"actively-engaged policy learning vs frozen-stuck policy."

To fully separate "wider state distribution" from "more argmax
flips at the SAME state" would require either (a) a state_hash
with much finer cardinality (so distinct semantic states don't
collapse), or (b) explicit substrate tracing of features +
Hussing-style representation rank.

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
