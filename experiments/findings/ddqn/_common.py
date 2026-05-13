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
from collections.abc import Mapping

import numpy as np
import numpy.typing as npt

from corroborate.measurables import Measurable
from corroborate_rl.dqn.measurables import (
    argmax_entropy_per_burst,  # pyright: ignore[reportUnknownVariableType]
    bootstrap_gap_magnitude_per_burst,  # pyright: ignore[reportUnknownVariableType]
    jensen_bias_per_burst_mean,
    mc_return_per_burst_mean,
    mc_return_raw_per_burst_mean,
)


CLAIM = dqn

# Explicit per-burst Measurable types — local aliases for
# bridge call-site readability AND to give pyright a fully-
# resolved Measurable[Mapping[str, object], NDArray] type.
type _PerBurstMeasurable = Measurable[
    Mapping[str, object], npt.NDArray[np.floating],
]

from typing import cast as _cast

MC_RETURN_PER_BURST_MEAN: _PerBurstMeasurable = mc_return_per_burst_mean
MC_RETURN_RAW_PER_BURST_MEAN: _PerBurstMeasurable = mc_return_raw_per_burst_mean
JENSEN_BIAS_PER_BURST_MEAN: _PerBurstMeasurable = jensen_bias_per_burst_mean
BOOTSTRAP_GAP_MAGNITUDE_PER_BURST: _PerBurstMeasurable = _cast(
    _PerBurstMeasurable, bootstrap_gap_magnitude_per_burst,
)
ARGMAX_ENTROPY_PER_BURST: _PerBurstMeasurable = _cast(
    _PerBurstMeasurable, argmax_entropy_per_burst,
)
