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
    q_smoothness_predicts_outcome__asterix_gamma_999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'Q-smoothness mechanism is real cross-arm (DDQN cuts '
    'q_inter_state_grad_overlap_late by d=-2.13 z=-8.24 on '
    'Asterix γ=0.999) but at single-env panel (n=29 after '
    'learnability + canonical-HP filters) we cannot distinguish '
    '"smoothness is independent mediator" from "smoothness is '
    'jens-shadow":\n'
    '  Edge 1 fires nan-d at n_strata=1 (DL random-effects pool '
    'requires ≥2 strata for tau estimation); needs multi-env '
    'panel.\n'
    '  Edge 2 partial-Spearman r(smoothness, outcome | jens) = '
    '-0.088 on the single Asterix stratum. The marginal Pearson '
    'r(smoothness, outcome) = +0.381 absorbs entirely into the '
    'jens conditioning — they are near-collinear (DDQN reduces '
    'both ~50% in lockstep).\n'
    'Disambiguation needs envs where DDQN affects smoothness and '
    'jens differently. Candidate panel: Breakout γ=0.999 (where '
    'DDQN HELPS outcome) + Freeway γ=0.999 + Asterix γ=0.999 + '
    'γ=0.95 controls. k=2 / k=4 amplification would also '
    'introduce within-env decoupling via the √(2 ln K) jens '
    'scaling.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_cuts_q_smoothness_asterix_gamma_999,
    q_smoothness_predicts_outcome__asterix_gamma_999,
)
