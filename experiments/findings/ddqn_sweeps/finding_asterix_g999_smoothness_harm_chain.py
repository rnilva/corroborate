"""Asterix γ=0.999 harm is mediated by DDQN's Q-smoothness reduction.

Replaces the refuted clip-argmax-noise chain
(`finding_ddqn_clip_argmax_harm_chain`). The empirical preview
against canonical ddqn cache showed argmax-noise predictions
don't hold; the actual mechanism is Q-magnitude AND Q-smoothness
reduction → underfit value function → outcome plateau below
vanilla's.

Two causal edges aggregated:

  Edge 1: `ddqn_cuts_q_autocorr_asterix_gamma_999`
    DDQN reduces Q-smoothness on Asterix γ=0.999.
    Empirical: d_q_autocorr = -1.98 z=-7.7 → comfortably HELD.

  Edge 2: `q_autocorr_predicts_outcome__asterix_gamma_999`
    Within Asterix γ=0.999, smoother Q tracks higher outcome.
    Empirical: pooled r(q_autocorr, outcome) = +0.530 p<0.001;
    holds within each arm (vanilla r=+0.486, DDQN r=+0.402).
    → HELD.

If both HELD → mechanism chain SUPPORTED. The framework's
`composed_verdict` AND-aggregates them.

Status (2026-05-17). Bridges ready; need backfill of q_autocorr_late
into ddqn_sweeps cache for Asterix γ=0.999 cells (currently
in canonical ddqn cache but not ddqn_sweeps). The Asterix γ=0.999
cells live in `experiments/data/minatar_gamma_sweep_k1/g0999_Asterix-MinAtar/`
— in the active sweep's out_dir, so backfill must wait for the
sweep to finish to avoid the race condition documented in
`feedback_no_ingest_during_sweep`.

EXPECTED stays UNDERPOWERED per CLAUDE.md "pin EMPIRICAL state"
discipline; the BLOCKED_ON gives the empirical preview that
predicts SUPPORTED post-backfill."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.q_smoothness_harm_mechanism import (
    ddqn_cuts_q_autocorr_asterix_gamma_999,
    q_autocorr_predicts_outcome__asterix_gamma_999,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'Bridges need q_autocorr_late + q_mc_burst_correlation_late + '
    'jensen_gap populated in ddqn_sweeps cache on Asterix γ=0.999 '
    'cells (live in minatar_gamma_sweep_k1/g0999_Asterix-MinAtar/). '
    'Currently those cells are in canonical ddqn cache only — '
    'backfill into ddqn_sweeps blocked until the active sweep '
    'finishes (avoids merge-cycle race per '
    'feedback_no_ingest_during_sweep). '
    'EMPIRICAL PREVIEW on canonical ddqn cache (n=30/arm) '
    'supports SUPPORTED verdict post-backfill:\n'
    '  Edge 1: pooled d on q_autocorr_late = -1.98 z=-7.7 → '
    'comfortably HELD at -0.5 floor.\n'
    '  Edge 2: pooled r(q_autocorr, outcome) = +0.530 p<0.001, '
    'holds within each arm separately (vanilla r=+0.486, DDQN '
    'r=+0.402) → HELD at +0.3 floor.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_cuts_q_autocorr_asterix_gamma_999,
    q_autocorr_predicts_outcome__asterix_gamma_999,
)
