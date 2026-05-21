"""DDQN's clip introduces argmax noise that propagates to outcome
harm at γ=0.999, on envs where vanilla's Q is uniformly inflated
(action ordering preserved).

This Finding aggregates the 3 causal-chain edges in
`clip_argmax_harm_mechanism`:

  Edge 1 — `ddqn_clip_increases_state_conditional_argmax_entropy`:
    Direct mechanism: DDQN's policy has HIGHER per-state argmax
    variability than vanilla. Uses H(argmax | state) so we
    isolate "noise within a state" from "state-discriminative
    policy" — `argmax_persistence_late` (across consecutive
    states) conflates the two, since a good state-discriminative
    policy will naturally have low persistence. Tests cause
    (mechanism active).

  Edge 2 — `mismatch_predicts_outcome_harm__within_ddqn`:
    Within DDQN cells at γ=0.999, more bootstrap-action mismatch
    correlates with worse outcome (after partialling out jens).
    Tests mediator-to-distal link.

  Edge 3 — `delta_h_cond_predicts_delta_outcome_xenv`:
    Cross-env dose-response: per-env Δ_H_cond (DDQN-vanilla,
    H(argmax|state)) correlates with per-env Δ_outcome.
    Tests effect at cohort: more per-state argmax noise →
    more outcome harm.

`composed_verdict` is AND-aggregate: SUPPORTED iff all 3 edges
HELD; REFUTED if any edge refutes.

Why a causal chain instead of a single regression: the prior
attempt (`finding_sigma_over_jens_regime_discriminator`) tested
the marginal cross-env correlate r(σ/jens, d_out) — a
descriptive predictor that doesn't isolate the mechanism. The
σ/jens predictor's REFUTATION at canonical data didn't tell us
whether (a) the mechanism is wrong or (b) σ/jens is a bad
operationalization. The chain isolates the mechanism: if Edge
1+2+3 ALL hold, the clip-argmax-harm story is supported even
if σ/jens is a poor marginal predictor.

Status (2026-05-17). Pending verification once the γ=0.999
cells in `ddqn_sweeps` cache get backfilled with
`argmax_persistence_late` and `q_argmax_margin_late` (currently
absent — only `bootstrap_action_mismatch_late` is populated for
the new MinAtar sweep cells).

Expected from prior analysis on canonical ddqn cache:
  Edge 1: HELD on Asterix γ=0.999 (DDQN's persistence < vanilla's
          per the canonical-verify data); cross-env d magnitude
          TBD.
  Edge 2: TBD — needs partial-spearman computation on the
          γ=0.999 DDQN cells.
  Edge 3: TBD — needs ≥4 envs with both arms at γ=0.999.

Setting EXPECTED to UNDERPOWERED + BLOCKED_ON to document the
backfill dependency."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.clip_argmax_harm_mechanism import (
    ddqn_clip_increases_state_conditional_argmax_entropy,
    delta_h_cond_predicts_delta_outcome_xenv,
    mismatch_predicts_outcome_harm__within_ddqn,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None

# Resolution (2026-05-21): backfill landed, chain fires REFUTED as
# the pre-registered prediction anticipated. The Asterix γ=0.999
# harm is real (see `finding_asterix_gamma_999_harm`) but the
# clip-induced per-state argmax-noise mechanism is REFUTED:
#   * Edge 1 H_cond gap tiny across MinAtar (d≈+0.15/+0.04, well
#     below the +0.2 harm floor).
#   * Edge 2 within-DDQN r(mismatch, outcome) = +0.40 at Asterix
#     (wrong sign — higher mismatch → BETTER outcome).
#   * Edge 3 cross-env panel direction inconsistent, n insufficient.
# The actual Asterix harm mechanism routes through Q-magnitude
# policy-content reduction (see memory
# `findings_asterix_breakout_channel_asymmetry_g999`), not
# clip-induced per-state argmax noise.


BRIDGES: tuple[Bridge, ...] = (
    ddqn_clip_increases_state_conditional_argmax_entropy,
    mismatch_predicts_outcome_harm__within_ddqn,
    delta_h_cond_predicts_delta_outcome_xenv,
)
