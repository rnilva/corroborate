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


# EMPIRICAL state pending — the α=0.25/0.5/0.75 cells are running.
# Set to UNDERPOWERED until ingest lands. BLOCKED_ON identifies the
# specific gap so the renderer flags `[blocked]`.
EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    "α=0.25/0.5 sweep in flight (`asterix_g0999_dampened_alpha_ckpt`, "
    "PID 3340856, ~2.5h GPU). α=0.75 queued behind it "
    "(`asterix_g0999_dampened_alpha075_ckpt`, ~75 min). Cluster "
    "verdict computable once both corpora ingest. The α=0 / α=1 "
    "endpoints are already in `asterix_g0999_canonical_n_eps20_ckpt` "
    "(n=30); intermediate arms complete the 5-point × 15-seed "
    "dose-response panel."
)


# Theoretical prediction (pre-registered): cluster SUPPORTED if
# Bridges 1+4 (mechanical + outcome) hold (replicates known V/D
# contrast continuously), Bridge 2 holds (NOVEL relative
# anisotropy growth), Bridge 3 holds (state-coverage lock-in),
# Bridge 5 holds (mediation closes the chain). Refuted at any
# leg → the chain's BROKEN where the refutation lands.


BRIDGES: tuple[Bridge, ...] = (
    alpha_scales_q_magnitude_deflation__asterix_g0999,
    alpha_grows_relative_anisotropy__asterix_g0999,
    alpha_shrinks_state_coverage__asterix_g0999,
    alpha_scales_eval_harm__asterix_g0999,
    relative_anisotropy_mediates_alpha_harm__asterix_g0999,
)
