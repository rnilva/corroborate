"""Q-shape channel: per-burst Q-action-std / Q-argmax-margin
mediate the Q-channel residual beyond bg.

Companion to `bg_per_burst_link_to_outcome` (the bg-channel
per-burst link). The empirical mediator search at canonical
(`scripts/q_channel_mediator_search.py`, 680 cells × 12 envs)
finds that the Q-channel residual `ρ(q, mc | bg) = +0.579` is
heavily reduced by the within-cell Q-shape measures:

  q_argmax_margin_late  alone:  Δ = -0.352  (residual +0.227)
  q_action_std_late     alone:  Δ = -0.271  (residual +0.308)
  jens_dormancy_gap     alone:  Δ = -0.267  (residual +0.313)
  ALL-5 joint:                  Δ = -0.548  (residual +0.032)

Substantive: DDQN's outcome benefit operates through a bg-channel
(direct algorithmic clip) PLUS a Q-shape channel mediated by
within-cell action-margin / action-std. The latter is what the
script measures as the "Q-channel residual" — the chunk of Q→MC
coupling not explained by bg alone.

These bridges author the marginal per-burst link tests for the
Q-shape mediators (without partial-conditioning on bg, since the
framework's per-burst Spearman primitive doesn't take a
conditioning kwarg yet). They establish that q_action_std and
q_argmax_margin per-burst correlate with outcome above the null
band — corroborating the script's finding at the bridge level.

The cross-conditional partial-Spearman ρ(margin, outcome | bg)
form would test mediation strictly. The marginal form here tests
predictive contribution; pair with `bg_per_burst_link_to_outcome`
(which fires HELD as PREDICTED-NULL: bg-marginal cancels in pool)
to see the picture: bg pools to ≈0 (env-specific), but Q-shape
pools to consistent positive.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import numpy.typing as npt

from corroborate.analyses.per_burst_jci_spearman import (
    PerBurstJciSpearmanResult,
)
from corroborate.analyses.per_burst_partial_jci_spearman import (
    PerBurstPartialJciSpearmanResult,
)
from corroborate.bridge.bridge import Direction, Tier, claim_bridge
from corroborate.bridge.verdict import Verdict
from corroborate.measurables import Measurable
from corroborate_rl.dqn.measurables import (
    q_action_std_per_burst,  # pyright: ignore[reportUnknownVariableType]
    q_argmax_margin_per_burst,  # pyright: ignore[reportUnknownVariableType]
    q_per_burst,  # pyright: ignore[reportUnknownVariableType]
)

from experiments.findings.ddqn._common import MC_RETURN_RAW_PER_BURST_MEAN
from experiments.findings.ddqn._scope import DDQN_RELEVANT_SCOPE
from experiments.findings.ddqn._verdicts import (
    partial_spearman_signed_verdict,
)


type _PerBurstMeasurable = Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
]


_Q_ARGMAX_MARGIN_PER_BURST: _PerBurstMeasurable = cast(
    _PerBurstMeasurable, q_argmax_margin_per_burst,
)
_Q_ACTION_STD_PER_BURST: _PerBurstMeasurable = cast(
    _PerBurstMeasurable, q_action_std_per_burst,
)
_Q_PER_BURST: _PerBurstMeasurable = cast(
    _PerBurstMeasurable, q_per_burst,
)


@claim_bridge(
    source='q_action_std_per_burst',
    target='mc_return_raw_per_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='a_gt_b',
)
def q_action_std_per_burst_link_to_outcome(
    per_burst_jci_spearman: PerBurstJciSpearmanResult,
    *,
    x: _PerBurstMeasurable = _Q_ACTION_STD_PER_BURST,
    y: _PerBurstMeasurable = MC_RETURN_RAW_PER_BURST_MEAN,
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.2,
    min_strata: int = 5,
) -> Verdict:
    """Per-burst Q-action-std → outcome. Predicted positive: more
    Q-spread among actions → better policy discrimination → higher
    outcome. HELD when pooled Spearman ρ ≥ +`rho_threshold` across
    envs.

    Empirical at canonical: ρ_pooled = +0.249 (env-stratified
    Fisher-z pool across 10 envs). Above the +0.2 substantive
    threshold — Q-shape mediator (action-spread) correlates
    consistently with per-burst outcome.

    **Caveat — Q-MC tautology**: q_action_std scales with Q
    magnitude on positive-return envs (Q estimates the MC return,
    so high-Q-spread cells tend to have high MC by construction).
    The pooled HELD signal mixes (a) substantive Q-shape mediation
    with (b) trivial Q-IS-MC tautology. Per-env data partially
    contradicts naive tautology — Asterix (SURVIVE-polarity)
    shows ρ=-0.32, MountainCar (GOAL-polarity) shows ρ=+0.73 —
    so the signal isn't purely tautological, but partial
    contribution is plausible. Clean substantive test requires
    partialling on `q_per_burst` (Q-magnitude); the framework's
    per-burst Spearman primitive doesn't support a conditioning
    kwarg yet — deferred per
    `findings_q_shape_env_class_stratification`."""
    del x, y, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        per_burst_jci_spearman,
        threshold=rho_threshold, sign=+1, min_strata=min_strata,
    )


@claim_bridge(
    source='q_argmax_margin_per_burst',
    target='mc_return_raw_per_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='a_gt_b',
)
def q_argmax_margin_per_burst_link_to_outcome(
    per_burst_jci_spearman: PerBurstJciSpearmanResult,
    *,
    x: _PerBurstMeasurable = _Q_ARGMAX_MARGIN_PER_BURST,
    y: _PerBurstMeasurable = MC_RETURN_RAW_PER_BURST_MEAN,
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.2,
    min_strata: int = 5,
) -> Verdict:
    """Per-burst Q-argmax-margin → outcome. Predicted positive:
    larger Q-gap between top and second action → policy is more
    decisive → higher outcome. HELD when pooled Spearman
    ρ ≥ +`rho_threshold`.

    Empirical at canonical: ρ_pooled = +0.157 across 9 envs.
    Below the +0.2 substantive threshold — Q-argmax-margin has a
    weaker direct per-burst link than q_action_std. The mediator
    search shows q_argmax_margin's contribution lives at the
    Q→MC residual conditioning level, not as a marginal per-burst
    predictor."""
    del x, y, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        per_burst_jci_spearman,
        threshold=rho_threshold, sign=+1, min_strata=min_strata,
    )


# Stage 0' tautology baseline: Bellman-contraction edge q → mc.
@claim_bridge(
    source='q_per_burst',
    target='mc_return_raw_per_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='a_gt_b',
)
def q_to_mc_coupled__bellman_contraction_baseline(
    per_burst_jci_spearman: PerBurstJciSpearmanResult,
    *,
    x: _PerBurstMeasurable = _Q_PER_BURST,
    y: _PerBurstMeasurable = MC_RETURN_RAW_PER_BURST_MEAN,
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.2,
    min_strata: int = 5,
) -> Verdict:
    """Stage 0' tautology baseline (Bellman contraction): Q-late
    estimates MC return, so ρ(q_per_burst, mc_per_burst) should be
    POSITIVE within env. HELD when pooled within-env ρ ≥
    `rho_threshold`.

    This bridge documents the Q→MC structural coupling that other
    bridges (`q_action_std_per_burst_link_to_outcome__partial_q`)
    PARTIAL OUT explicitly. Analog of
    `mc_disc_raw_coupled__per_env_jci` (Stage 0 outcome-side
    coupling) but for the BIAS-side coupling driven by Bellman
    contraction.

    Empirical at canonical: ρ_pooled = +0.348 across 10 envs.
    Per-env strongly positive on most (Breakout +0.72, MountainCar
    +0.81, MountainCar +0.81, SI +0.43, PacMan +0.60, Freeway
    +0.60) — Bellman coupling HELDs. Strongly NEGATIVE on Asterix
    (-0.33) and Snake (-0.23); near-zero on SlidingTile (-0.04) —
    envs where training Q-magnitude doesn't track MC return,
    likely reflecting unconverged Q-estimation or env-specific
    structure (Asterix's per-step scoring dynamics could produce
    high Q at low-progress states; Snake's stochastic reward
    structure decouples Q from realized MC).

    The HELD-with-env-heterogeneity reading: Bellman contraction
    is the EXPECTED coupling but training imperfections produce
    env-specific deviations. The per-env structure documented in
    `findings_q_shape_env_class_stratification`."""
    del x, y, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        per_burst_jci_spearman,
        threshold=rho_threshold, sign=+1, min_strata=min_strata,
    )


@claim_bridge(
    source='q_action_std_per_burst',
    target='mc_return_raw_per_burst_mean',
    direction=Direction.DIRECT,
    tier=Tier.ASSOCIATIONAL,
    scope=DDQN_RELEVANT_SCOPE,
    predicted_direction='a_gt_b',
)
def q_action_std_per_burst_link_to_outcome__partial_q(
    per_burst_partial_jci_spearman: PerBurstPartialJciSpearmanResult,
    *,
    x: _PerBurstMeasurable = _Q_ACTION_STD_PER_BURST,
    y: _PerBurstMeasurable = MC_RETURN_RAW_PER_BURST_MEAN,
    conditioning: _PerBurstMeasurable = _Q_PER_BURST,
    stratify_by: str = 'env_name',
    min_stratum_size: int = 5,
    rho_threshold: float = 0.2,
    min_strata: int = 5,
) -> Verdict:
    """Substantively-clean form: per-burst partial Spearman
    ρ(q_action_std, mc | q_per_burst), env-stratified Fisher-z
    pooled. Partials Q-magnitude out so Q-IS-MC tautology no
    longer drives the signal — surviving correlation is the
    Q-SHAPE residual mediator beyond Q-magnitude.

    Predicted positive (Q-shape additionally predicts outcome
    beyond Q-magnitude). HELD when |ρ_partial| ≥ +`rho_threshold`.

    Sibling of `q_action_std_per_burst_link_to_outcome` (marginal
    form, ρ=+0.249 HELD with tautology caveat). The pair documents
    whether the marginal signal is substantive or tautology-driven.
    """
    del x, y, conditioning, stratify_by, min_stratum_size
    return partial_spearman_signed_verdict(
        per_burst_partial_jci_spearman,
        threshold=rho_threshold, sign=+1, min_strata=min_strata,
    )


BRIDGES = (
    q_to_mc_coupled__bellman_contraction_baseline,
    q_action_std_per_burst_link_to_outcome,
    q_action_std_per_burst_link_to_outcome__partial_q,
    q_argmax_margin_per_burst_link_to_outcome,
)
