"""Cross-env mediation: DDQN's mech effect predicts its outcome effect.

This Finding tests the substantive bias-correction claim at the
cross-env arm-diff level, where the Q-MC algebraic identity that
pins per-cell ρ(jens, out_disc) DOESN'T apply (each env
contributes ONE arm-diff point; intervention assignment is the
causal cut).

Three sibling bridges using different mech predictors. The
cluster's substantive REFUTED arises from the bg-frac bridge
sign-flipping on the high-alignment cohort — the jens-based
bridges actually HELD:

  - `ddqn_outcome_scales_with_jens_reduction__xenv`: **HELD**
      (ρ_pool=−0.83, p=0.014, n_strata=6). Theory-canonical
      Hasselt-2016 predictor. Per-cell ρ(jens, out_disc) is
      Q-MC tautological; the cross-env arm-diff form is clean.
  - `ddqn_outcome_scales_with_jens_reduction__xenv_loo_robust`:
      **HELD** (LOO-robustness sibling — anchor isn't outlier-
      driven; signal survives every single-env removal).
  - `ddqn_outcome_scales_with_bg_frac_active__xenv`:
      **NO_EFFECT (sign_flip)** (p=0.34). The MC-free predictor
      (rate of online/target argmax disagreement) actually
      *INCREASES* on DDQN on this cohort's SURVIVE-polarity
      envs (PacMan, SpaceInvaders, Breakout) — opposite of the
      "DDQN reduces disagreement events" prediction. The
      cross-env scaling is not what the substrate-author's
      theory expected.

**Scope is `env_disc_raw_alignment > 0.7`** (the outcome-
translation alignment scope, per memory
`findings_bg_not_causally_manipulated_at_canonical`). This
filters to envs where the substrate's disc-MC and raw return
co-vary tightly — the precondition for the cross-env Δ_jens →
Δ_out_raw test to be substantively interpretable. The 6 envs
in scope: Acrobot, CartPole(no), MetaMaze, MountainCar,
PacMan, Snake, SlidingTilePuzzle, Breakout (modulo
CartPole-saturation exclusion). Without this scope, low-
alignment envs (Freeway 0.42, Asterix/SI 0.65, FourRooms-
sliced −0.85) inject γ^t reweighting noise that masked the
signal at n=12 (pre-scope ρ=−0.35).

Cluster shape (REFUTED): cluster verdict is AND-aggregate. The
two jens-based bridges HELD; the bg-frac-active sibling
NO_EFFECT(sign_flip) → cluster REFUTED. Substantively this is
the right reading: the substrate's full hypothesis was "both
the disc-space proxy (jens) AND the MC-free proxy
(bg_frac_active) predict cross-env outcome scaling." That
conjunction is REFUTED. The sub-claim "jens alone predicts
cross-env outcome on the alignment-scoped cohort" IS supported,
but the cluster-level test refuses to claim the weaker form
when the conjunction it explicitly enumerates fails.

The bg-frac-active sign-flip is itself substantively important
— it confirms `findings_bg_not_causally_manipulated_at_canonical`:
the wedge-frequency predictor doesn't carry the cross-env
mediation signal that jens does. If you want the SUPPORTED
sub-claim cleanly, factor a sibling Finding holding only the
two jens bridges (the LOO + anchor cluster would then be
SUPPORTED at ρ=−0.83). The current Finding documents the full
substrate hypothesis honestly: jens-only works, bg_frac_active
doesn't, conjunction REFUTED."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn.cross_env_mediation import (
    ddqn_outcome_scales_with_bg_frac_active__xenv,
    ddqn_outcome_scales_with_jens_reduction__xenv,
    ddqn_outcome_scales_with_jens_reduction__xenv_loo_robust,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_outcome_scales_with_jens_reduction__xenv,
    ddqn_outcome_scales_with_jens_reduction__xenv_loo_robust,
    ddqn_outcome_scales_with_bg_frac_active__xenv,
)
