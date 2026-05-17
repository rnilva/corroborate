"""Asterix γ=0.999 harm is mediated by DDQN's Q-smoothness reduction.

Replaces the refuted clip-argmax-noise chain
(`finding_ddqn_clip_argmax_harm_chain`). The empirical preview
against canonical ddqn cache showed argmax-noise predictions
don't hold; the actual mechanism is Q-magnitude AND Q-smoothness
reduction → underfit value function → outcome plateau below
vanilla's.

Two causal edges aggregated:

  Edge 1: `ddqn_cuts_q_smoothness_asterix_gamma_999`
    DDQN reduces Q-smoothness on Asterix γ=0.999.
    `q_inter_state_grad_overlap_late` is the inner-product
    alignment of dQ/dθ between consecutive trajectory states.
    Empirical: d = -2.13 z=-8.24 → comfortably HELD at -0.5.

  Edge 2: `q_smoothness_predicts_outcome__asterix_gamma_999`
    Within Asterix γ=0.999, smoother Q tracks higher outcome.
    Empirical: pooled r = +0.381 p=0.003 → HELD at +0.3 floor.

If both HELD → mechanism chain SUPPORTED. The framework's
`composed_verdict` AND-aggregates them."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.q_smoothness_harm_mechanism import (
    ddqn_cuts_q_smoothness_asterix_gamma_999,
    q_smoothness_predicts_outcome__cross_stratum,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'Edge 1 (single-env mechanism-active on Asterix γ=0.999) is '
    'HELD: d=-2.13 z=-8.24 on q_inter_state_grad_overlap_late. '
    'Edge 2 (cross-stratum partial-r) tests whether smoothness '
    'is independent of jens at the 6-stratum (env × γ) Δ panel. '
    'Empirical preview on ddqn_sweeps cache: partial ρ=-0.325 '
    'p=0.63 with n_strata=6 — UNDERPOWERED (n=6 needs ρ≥0.7 to '
    'reject at α=0.10). The 6-stratum panel suggests smoothness '
    'is NOT a universal positive mediator (sign trends wrong) '
    'but power is too low to refute decisively. k=2/k=4 sweep '
    'would push the panel to 12-18 strata and discriminate.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_cuts_q_smoothness_asterix_gamma_999,
    q_smoothness_predicts_outcome__cross_stratum,
)
