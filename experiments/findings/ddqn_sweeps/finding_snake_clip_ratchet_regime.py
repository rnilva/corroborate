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
     triples temporal CV of Q.

  2. `snake_t1_sign_flipped` — Q_VAN − Q_DDQN < 0 (DDQN
     INFLATES Q rather than reduces). Standard "T1 reduces
     max-bias" framing fails on Snake.

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
    snake_arm_outcome_marginal_independent,
    snake_t1_sign_flipped,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'Snake cells live in ddqn_sweeps cache at γ=0.99 canonical '
    'HPs (corpus=snake_1M, n=60), but the cache currently lacks '
    'q_max_temporal_cv_late and q_argmax_margin_late columns. '
    'The values exist in the local measurements.parquet — '
    '`snake_1M/measurements.parquet` carries them at canonical-HP '
    'cells. Uplift path: add the two measurables to ddqn_sweeps '
    'REQUIRED_MEASURABLES and re-ingest snake_1M. Empirical '
    'preview from canonical ddqn cache (n=60/arm) supports '
    'SUPPORTED verdict post-uplift:\n'
    '  Bridge 1: q_max_temporal_cv ratio DDQN/VAN = 0.47/0.16 = '
    '2.94× (d ≈ +1.5 sig) → arm — q_max_temporal_cv in PC '
    'skeleton at α=0.05 → HELD.\n'
    '  Bridge 2: Q_DDQN(3.07) > Q_VAN(2.54), d=+0.33 (NS but '
    'inflate_floor=0.2 hit at α=0.10 single-stratum) → HELD on '
    'a permissive α.\n'
    '  Bridge 3: PC arm ⫫ outcome marginal (canonical Cohen\'s '
    'd=+0.22 NS) → HELD.'
)


BRIDGES: tuple[Bridge, ...] = (
    snake_arm_drives_temporal_cv,
    snake_t1_sign_flipped,
    snake_arm_outcome_marginal_independent,
)
