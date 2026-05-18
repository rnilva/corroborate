"""Link-shape analyses — predictor→target relationship within /
across strata.

Includes `stratum_link_moderation_dowhy` (UNCONSUMED — held as
the methodologically-sound moderation-on-link primitive pending
a consumer bridge; see its docstring).
"""
from corroborate.analyses.link.cross_stratum_arm_diff_partial_spearman import (  # noqa: F401
    cross_stratum_arm_diff_partial_spearman as _csadps,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.link.cross_stratum_arm_diff_slope import (  # noqa: F401
    cross_stratum_arm_diff_slope as _csads,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.link.cross_stratum_property_slope import (  # noqa: F401
    cross_stratum_property_slope as _csps,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.link.paired_link_per_burst import (  # noqa: F401
    paired_link_per_burst as _link_per_burst,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.link.stratum_link_moderation_dowhy import (  # noqa: F401
    stratum_link_moderation_dowhy as _stratum_link_moderation,  # pyright: ignore[reportUnusedImport]
)
