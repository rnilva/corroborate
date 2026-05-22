"""DDQN's total effect on outcome — independent of the Hasselt
chain. Tests whether DDQN improves outcome consistently across
the canonical-dormancy panel (where the chain's mechanism is
broadly active).

# Why this is NOT a chain edge

Hasselt's theorem speaks to the mechanism: the σ-floor predicts
bias, DDQN reduces bias. The total effect of DDQN on outcome —
whether the intervention nets out positive at the agent's
actual return — is a structurally distinct claim. Pearl's
mediation framing:

  total = direct + indirect (via mediator)

The chain's `do(DDQN) → jens → outcome` quantifies the
INDIRECT path. The TOTAL effect captures all DDQN channels —
direct + indirect + any non-mediated paths the intervention
triggers (e.g., DDQN's effect on training stability, policy
churn, exploration).

Three Hasselt-chain edges all HELD (theorem + link + mech) is
compatible with B4 NULL: DDQN reduces bias as predicted, bias
predicts outcome as predicted, the indirect-through-bias path
is positive — but other DDQN channels may net out.

# Empirical result on the canonical-dormancy panel

  13 strata (9 γ=0.999 + 4 γ=0.99):
    helps:  FR γ=0.999 (+3.76), SI γ=0.999 (+2.16), Breakout γ=0.999 (+0.66),
            Snake γ=0.999 (+0.62), MetaMaze γ=0.99 (+0.40),
            LL γ=0.999 (+0.22), Acrobot γ=0.99 (+0.10)
    harms:  Asterix γ=0.999 (-0.80), MC γ=0.999 (-0.32),
            FR γ=0.99 (-0.20), MetaMaze γ=0.999 (-0.08),
            LL γ=0.99 (-0.02), Acrobot γ=0.999 (-0.007)

7 helps / 6 harm-null → sign-test p=0.5 → NO_EFFECT/NULL_EFFECT.
DDQN's total effect on outcome is genuinely env-and-γ-conditional;
no uniform improvement claim survives across the panel.

# Interpretation

The Hasselt chain holds as a mechanism (`hasselt_chain_explicit`
SUPPORTED at three edges). DDQN's outcome story is a separate
empirical question with a layered answer: the bias-reduction
pathway exists, but whether it translates to outcome improvement
depends on env/γ — likely on what OTHER channels DDQN operates
through (policy structure, training stability, exploration). A
future bridge cluster could decompose total → direct + indirect
via mediation analysis at the per-stratum level."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.hasselt_clean.outcome_consistency import (
    ddqn_helps_outcome__consistently_cross_env,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_helps_outcome__consistently_cross_env,
)
