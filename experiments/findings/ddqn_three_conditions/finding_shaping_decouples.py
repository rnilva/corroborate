"""DDQN's outcome benefit has scope: helps at FR γ=0.999 × MLP ×
unshaped; never positively helps at FR × shaped. Cluster reads
as "the bias→outcome translation is scope-dependent at FR".

Two sibling bridges:

- positive arm `ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel`:
  at FR γ=0.999 × MLP × unshaped (the "all bias-reduction
  factors active" reference cell), DDQN improves
  `eval_best_burst_raw_mean` at every k_eff stratum.
- null arm `ddqn_no_positive_outcome_under_shaping__fr_shaped_fa_x_gamma_panel`:
  at FR × {linear, MLP} × {γ=0.99, 0.999} × PotentialReward,
  DDQN's outcome effect is never appreciably POSITIVE.

**What this Finding claims**: DDQN's outcome benefit appears
at the bias-active reference cell (FR γ=0.999 × MLP ×
unshaped) and does not appear in the shaped panel. Strict
empirical content: "helps at A, doesn't help anywhere in B"
on two non-overlapping scopes.

**What this Finding does NOT claim**:
- That the two bridges test matched scopes. Positive arm pins
  γ=0.999 + MLP and varies k_eff; null arm pools γ + fa_kind
  and fixes k_eff. The matched-scope comparison (positive vs
  null at FR γ=0.999 × MLP × {shaped, unshaped} × k_eff sweep)
  requires a shaped × k_eff sweep that does not yet exist.
- That shaping CAUSALLY DECOUPLES bias from outcome. The
  null arm's asymmetric `predicted_direction='a_lt_b'` admits
  both "decouples to ~0" and "inverts to ~−1.5" (one stratum
  at γ=0.99 × MLP shows DDQN actively HURTS with d ≈ −1.5).
  The substantive narrative in
  `findings_shaping_decouples_bias_from_outcome` (dense
  Φ-gradient swamps Q-noise on argmax) is plausible but not
  discriminated from "shaping × DDQN-clip distorts the optimal
  policy".
- Cross-env generalisation. Scope is FR-only; no other env has
  been run with PotentialReward shaping at this writing."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_three_conditions.outcome_translation import (
    ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel,
    ddqn_no_positive_outcome_under_shaping__fr_shaped_fa_x_gamma_panel,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_helps_outcome_at_fr_g999_mlp_unshaped__k_panel,
    ddqn_no_positive_outcome_under_shaping__fr_shaped_fa_x_gamma_panel,
)
