"""Shaping decouples DDQN's bias-reduction from outcome.

Two sibling bridges form the cluster:

- positive arm `ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel`:
  at FR γ=0.999 × MLP × unshaped (the "all Hasselt factors active"
  reference cell), DDQN improves `eval_best_burst_raw_mean` at
  every k_eff stratum.
- null arm `shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel`:
  at FR × shaped × {linear, MLP} × γ ∈ {0.99, 0.999}, DDQN's
  outcome effect is never positive. Shaping's dense Φ-gradient
  policy signal decouples bias-reduction from outcome
  improvement.

The cluster pattern positive (unshaped) + null (shaped) reads as
"shaping moderates the bias→outcome translation at FR ×
MLP × γ ∈ {0.99, 0.999}".

**What this Finding does NOT claim**:
- Cross-env shaping moderation. The null arm is scoped to FR
  only; no other env was run with PotentialReward shaping at
  this writing.
- That shaping necessarily INVERTS the translation (one cell
  has d ≈ −1.5 — DDQN actively hurts under MLP × γ=0.99 shaped
  — but the asymmetric `predicted_direction='a_lt_b'` is
  agnostic between "decouples" and "inverts")."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.outcome_translation import (
    ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel,
    shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel,
    shaping_decouples_outcome_benefit__fr_shaped_fa_x_gamma_panel,
)
