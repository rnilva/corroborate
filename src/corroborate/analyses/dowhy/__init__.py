"""DoWhy-backed analyses — backdoor ATE + refutations + the
two stratum-level link families.

The top-level `dowhy` module here registers the three primitive
analyses (`backdoor_ate`, `placebo_refutation`,
`random_common_cause_refutation`) plus the typed Result
dataclasses. Sibling modules in this subpackage build on those
primitives for higher-level questions.
"""
from corroborate.analyses.dowhy.dowhy import (  # noqa: F401
    BackdoorResult, RefutationResult,
    backdoor_ate, placebo_refutation,
    random_common_cause_refutation,
)
from corroborate.analyses.dowhy.mediation_dowhy import (  # noqa: F401
    mediation_dowhy as _mediation_dowhy,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.dowhy.paired_continuous_do_dowhy import (  # noqa: F401
    paired_continuous_do_dowhy as _paired_continuous_do_dowhy,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.dowhy.stratum_baseline_predictor_link_dowhy import (  # noqa: F401
    stratum_baseline_predictor_link_dowhy as _stratum_vanilla_pred_link,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.dowhy.stratum_delta_link_dowhy import (  # noqa: F401
    stratum_delta_link_dowhy as _stratum_delta_link_dowhy,  # pyright: ignore[reportUnusedImport]
)

__all__ = [
    'BackdoorResult',
    'RefutationResult',
    'backdoor_ate',
    'placebo_refutation',
    'random_common_cause_refutation',
]
