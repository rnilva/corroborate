"""Asterix γ=0.999 harm mechanism: vanilla's anisotropic bias is policy-informative.

Single-bridge Finding pinning the substantive **per-env**
mechanism for the unique Asterix γ=0.999 DDQN-harms result:

  `asterix_vanilla_lambda_a_positively_predicts_outcome__g0999`:
    Within vanilla cells on Asterix-MinAtar γ=0.999 (n=30),
    per-cell Λ_a POSITIVELY predicts outcome after conditioning
    on jens. Empirical: ρ_partial = +0.352 p=0.061.

Cross-references:
- `finding_asterix_gamma_999_harm` (SUPPORTED) — the harm-side
  effect (d_out = −3.2 to −1.1 across outcome metrics).
- `finding_lambda_a_within_arm_asymmetry` (SUPPORTED) — the
  pooled cross-env within-vanilla finding (ρ=−0.17 averaged
  across 8 envs; the Type-A majority direction).
- `findings-asterix-breakout-channel-asymmetry-g999` (memory)
  — the Q-magnitude policy-informativeness asymmetry that
  unifies Asterix's vanilla-r-positive + DDQN-harms pair vs
  Breakout's vanilla-r-positive + DDQN-helps pair.
- `findings-within-vanilla-lambda-a-per-env-breakdown` (memory)
  — the per-env partial-r table showing Asterix as the
  standout positive-r + DDQN-harm env.

The substantive claim: at γ=0.999 on Asterix, vanilla's high-
magnitude Q with anisotropic cross-action variance carries
policy-information that the argmax leverages. DDQN's bootstrap
clip symmetrises the bias, destroying the
information-bearing asymmetry; per-cell outcome drops as Λ_a
"information-content" is removed."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.lambda_a_mediation import (
    asterix_vanilla_lambda_a_positively_predicts_outcome__g0999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    asterix_vanilla_lambda_a_positively_predicts_outcome__g0999,
)
