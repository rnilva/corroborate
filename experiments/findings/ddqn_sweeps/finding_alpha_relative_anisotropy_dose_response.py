"""DDQN's bg-wedge harm at Asterix γ=0.999 is mediated by
RELATIVE ANISOTROPY growth via the convex DDQN-vanilla
dose-response.

Pre-registered 2026-05-26. Five bridges form a scope cluster
spanning the chain:

  α (treatment) ─→ |Q| deflation ─→ relative anisotropy GROWS
                                  ─→ state-coverage SHRINKS
                                  ─→ eval_best DROPS

Each leg is a typed bridge with predicted direction; cluster
composes via `composed_verdict`. All HELD → mechanism
CORROBORATED end-to-end.

Individual bridges + failure mode by leg are documented in
`alpha_relative_anisotropy_dose_response.py`.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.alpha_relative_anisotropy_dose_response import (
    alpha_scales_q_magnitude_deflation__asterix_g0999,
    alpha_grows_relative_anisotropy__asterix_g0999,
    alpha_shrinks_state_coverage__asterix_g0999,
    alpha_scales_eval_harm__asterix_g0999,
    relative_anisotropy_mediates_alpha_harm__asterix_g0999,
)


# EMPIRICAL state — data landed 2026-05-26 (75-cell panel across
# canonical_n_eps20_ckpt + dampened_alpha_ckpt + dampened_alpha075_ckpt).
# Pre-reg outcome: 3/5 bridges HELD, 2/5 REFUTED → cluster REFUTED at
# the chain mediation step.
#
#   Bridge 1 (mech, ρ(α, q_late) = -0.959, p=10⁻⁴¹)             HELD
#   Bridge 2 (rel_aniso, ρ(α, rel_aniso) = +0.334)                HELD (weak)
#   Bridge 3 (state-coverage, ρ(α, n_states) = -0.003)            REFUTED (FLAT)
#   Bridge 4 (outcome, ρ(α, eval) = -0.467, p=10⁻⁵)               HELD
#   Bridge 5 (rel_aniso mediates α→eval, 2% absorbed)             REFUTED
#
# Per-burst Fisher-z pooled partial ρ(α, eval | mediator_per_burst[t])
# corroborates the refutation: rel_aniso_per_burst absorbs -2%,
# state_conditional_entSC absorbs 0%, unique_states absorbs +2%. The
# four candidate mediators (anisotropy + entSC + state coverage +
# marginal entropy) all sit OFF the channel at every burst. The
# non-tautological mediator is bg_per_burst (56% absorption) — the
# operator's own scaling variable. See
# `findings_dose_response_anisotropy_refuted.md` for the full
# breakdown.
EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


# The "DDQN induces relative anisotropy → harm" narrative is REFUTED
# at the per-burst within-corpus level. Cross-env V_entSC ρ=+0.833 is
# still an env-structural moderator; the within-corpus mediator
# remains unidentified non-tautologically.


BRIDGES: tuple[Bridge, ...] = (
    alpha_scales_q_magnitude_deflation__asterix_g0999,
    alpha_grows_relative_anisotropy__asterix_g0999,
    alpha_shrinks_state_coverage__asterix_g0999,
    alpha_scales_eval_harm__asterix_g0999,
    relative_anisotropy_mediates_alpha_harm__asterix_g0999,
)
