"""Snake γ=0.99 fits a 4th DDQN-response regime: CLIP-RATCHET.

The cross-env outcome regime classification at γ=0.999 in
`finding_cross_env_outcome_regime_g999` enumerates three
distinct DDQN responses on MinAtar: HARM (Asterix), HELP
(Breakout/SI), NEUTRAL (Freeway). Snake γ=0.99 at canonical
HPs (n=60, `snake_1M` corpus) fits a structurally distinct
fourth regime — the "clip-ratchet" failure mode documented in
`findings_snake_ddqn_destabilizes_sparse_reward`.

Three bridges:

  1. `snake_arm_drives_temporal_cv` — PC's skeleton includes
     `arm — q_max_temporal_cv_late` (unique to Snake; absent
     from all 4 γ=0.999 MinAtar envs' PC skeletons). DDQN
     triples temporal CV of Q (d=+1.5).

  2. `snake_arm_inflates_action_std` — DDQN nearly doubles
     `q_action_std_late` (cross-action SD), d=+0.65 z=+2.5 sig.
     Substantive form of "DDQN INFLATES Q distribution" — the
     three Q-exploding DDQN seeds drag σ_Q up while the
     remaining 27 match vanilla.

  3. `snake_arm_outcome_marginal_independent` — PC finds
     `arm ⫫ outcome` marginally. The bimodal seed distribution
     (3/30 DDQN seeds Q-explode, 1 outcome outlier at 7.67, 26
     match vanilla) renders Cohen's d=+0.22 NS uninformative.

All 3 HELD → cluster SUPPORTED. Snake adds the 5th env to the
cross-env outcome classification (1 Asterix + 1 Breakout + 1 SI
+ 1 Freeway + 1 Snake) with a 4th regime category.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.snake_clip_ratchet_regime import (
    snake_arm_drives_temporal_cv,
    snake_arm_inflates_action_std,
    snake_arm_outcome_marginal_independent,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    snake_arm_drives_temporal_cv,
    snake_arm_inflates_action_std,
    snake_arm_outcome_marginal_independent,
)
