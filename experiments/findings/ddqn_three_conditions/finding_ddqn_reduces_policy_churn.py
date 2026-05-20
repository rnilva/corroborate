"""Schaul 2022 churn-reduction prediction REFUTED with sign-flip at
γ=0.999 sparse-reward envs.

Schaul et al. 2022 ("The Phenomenon of Policy Churn", NeurIPS,
arXiv:2206.00730) documents that DQN exhibits surprisingly high
argmax-flip rates between consecutive policy snapshots, and proposes
churn-reduction techniques (incl. DDQN-style decoupled targets) as
mitigation. The implicit prediction we tested: DDQN reduces policy
churn vs vanilla.

Pre-registered prediction (source-hash committed before
`policy_churn_late` was populated on the corpora): at both SI γ=0.999
and FR γ=0.999, DDQN < vanilla churn (`predicted_direction='a_lt_b'`).

**Materialized result — SIGN-FLIPPED at both scopes:**

  ddqn_reduces_policy_churn__si_g999:
    vanilla mean: 0.678
    DDQN mean:    0.708 (HIGHER, not lower)
    mean_diff:    +0.030
    Cohen's d:    +1.91
    p-value:      6.11e-9
    Verdict:      NO_EFFECT (SIGN_FLIP) — refutes a_lt_b

  ddqn_reduces_policy_churn__fr_g999:
    vanilla mean: 0.239
    DDQN mean:    0.321 (HIGHER, not lower)
    mean_diff:    +0.082
    Cohen's d:    +3.30
    p-value:      2.22e-16
    Verdict:      NO_EFFECT (SIGN_FLIP) — refutes a_lt_b

Composed cluster verdict: **REFUTED**. EXPECTED updated to REFUTED
to match the empirical state (per CLAUDE.md "EXPECTED pins the
EMPIRICAL state, not the theoretical claim").

**Mechanistic reading.** Schaul's prediction was made for *healthy*
DQN training where DDQN's clip stabilizes Q-oscillation → fewer
argmax flips. At γ→1 sparse-reward:

- **FR γ=0.999 vanilla** has LOW churn (0.24) because the policy is
  *frozen* — `policy_growth_fraction ≈ 0`, i.e. the network barely
  changes its argmax at any state because it's stuck in the
  bias-trap regime. Low churn here means stuck-bad, not converged.

- **FR γ=0.999 DDQN** has HIGHER churn (0.32) because the policy is
  ACTIVELY LEARNING — argmaxes shift as the policy anchors to
  sparse reward across late training. High churn here means
  productively-changing.

- **SI γ=0.999** shows the same direction with smaller magnitude:
  vanilla's bias-inflated Q has stable rankings (Q huge, ordering
  rigid); DDQN's clipped Q is more responsive to actual return
  signal and thus more flux.

**The "low churn" indicator is INVERTED in our regime.** At γ→1
sparse-reward, low churn signals stuck-frozen-bad-policy, not
converged-good-policy. Schaul's published prediction does not
transfer; our framework's pre-registration discipline caught the
sign-flip cleanly.

This sits in joint context with `finding_temporal_ordering_at_fr_g999`
(SUPPORTED — DDQN increases `policy_growth_fraction`). The two
findings reinforce: vanilla is stuck (low growth + low churn);
DDQN is actively-anchored (high growth + high churn). At γ→1
sparse-reward, BOTH growth fraction AND churn are HIGHER for the
healthier arm — opposite of what one might naively expect from
either lit alone.

Cross-refs:
- `corroborate_rl.dqn.measurables.policy_churn_late` — the
  Schaul-style state-conditional argmax-flip measurable on
  existing argmax + state-hash traces (no substrate change).
- `THEORY_bootstrap_dominance.md` §11 — literature positioning;
  this Finding contributes the policy-structure-axis refutation
  result (Schaul prediction doesn't transfer to γ→1).
- `finding_temporal_ordering_at_fr_g999` — sibling on the
  hybrid axis-1+axis-3 (`policy_growth_fraction`).
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.policy_churn import (
    ddqn_reduces_policy_churn__fr_g999,
    ddqn_reduces_policy_churn__si_g999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_reduces_policy_churn__si_g999,
    ddqn_reduces_policy_churn__fr_g999,
)
