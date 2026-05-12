"""Cross-claim-file shared constants.

CLAIM is the outermost claim the runner threads to `evaluate(...,
claim=CLAIM)` so admission gates consult `walk_paths(CLAIM,
regime='leaf')` for the substrate's author-primitive set.

The per-burst reductions are composed Measurables used as bridge
defaults across multiple claim files. Canonical instances live in
`corroborate_rl.dqn.measurables`; local aliases keep call-site
names readable."""
from __future__ import annotations

from corroborate_rl.dqn.dqn import dqn
from corroborate_rl.dqn.measurables import (
    jensen_bias_per_burst_mean,
    mc_return_per_burst_mean,
    mc_return_raw_per_burst_mean,
)


CLAIM = dqn

MC_RETURN_PER_BURST_MEAN = mc_return_per_burst_mean
MC_RETURN_RAW_PER_BURST_MEAN = mc_return_raw_per_burst_mean
JENSEN_BIAS_PER_BURST_MEAN = jensen_bias_per_burst_mean
