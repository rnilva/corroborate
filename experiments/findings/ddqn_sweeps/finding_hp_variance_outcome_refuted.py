"""HP-variance dose-response on bias-correction → outcome is
REFUTED at ddqn_sweeps.

The salvaged 3-bridge DoWhy cluster (backdoor + placebo + RCC)
on per-stratum panel of vanilla bootstrap_gap_magnitude vs DDQN
Δ_outcome at HP-variance scope (n_strata=9, n_cells=2700) fires
REFUTED: backdoor ATE = -137.588 (sign FLIPS from predicted +);
placebo refuter clean (real=-137.6 vs refuted=0); RCC drift small
(-137.6 → -125.4, ~9%). The sign-flip is corroborated as a real
signal by clean refutations — not method noise.

Substantive: under HP-variance, configs with higher vanilla
bg_magnitude see SMALLER (not bigger) DDQN benefit on Δ_outcome.
The pre-canonical β=+244 diagnostic was confirmed to be a
HP/corpus-confound artifact — when the pool is properly
stratified with canonical-equivalent dimensions, the slope flips
to negative.

Pairs with `finding_reach_bias_link` at canonical (REFUTED on
env-variance via Spearman cluster). Together: the bias-correction
→ outcome translation claim is REFUTED at BOTH scope regimes —
env-axis variation (canonical) AND HP-axis variation (sweeps).
The practitioner-facing universal scaling claim has no surviving
form at canonical-equivalent stratification.

Memory: `findings_outcome_translation_refuted_cross_scope`."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.bias_correction_hp_variance import (
    bias_correction_clip_predicts_outcome__hp_variance__backdoor,
    bias_correction_clip_predicts_outcome__hp_variance__placebo,
    bias_correction_clip_predicts_outcome__hp_variance__rcc,
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    bias_correction_clip_predicts_outcome__hp_variance__backdoor,
    bias_correction_clip_predicts_outcome__hp_variance__placebo,
    bias_correction_clip_predicts_outcome__hp_variance__rcc,
)
