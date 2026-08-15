"""Paired-shape analyses — `paired_g` and family.

Side-effect submodule imports populate the `@analysis` registry
on `import corroborate.analyses.paired` (and on
`import corroborate.analyses`, which cascades).
"""
from corroborate.analyses.paired.arm_mean_diff import (  # noqa: F401
    arm_mean_diff as _arm_mean_diff,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.paired.bootstrap_paired_g import (  # noqa: F401
    bootstrap_paired_g as _bootstrap_pg,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.paired.cliff_delta_paired import (  # noqa: F401
    cliff_delta_paired as _cliff_delta,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.paired.factorial_2x2 import (  # noqa: F401
    factorial_2x2_interaction as _factorial,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.paired.mundlak_decomposition import (  # noqa: F401
    mundlak_decomposition as _mundlak,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.paired.mundlak_paired_g_per_burst import (  # noqa: F401
    mundlak_paired_g_per_burst as _mundlak_pgpb,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.paired.paired_comparison import (  # noqa: F401
    paired_comparison as _paired_comparison,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.paired.paired_directional import (  # noqa: F401
    paired_directional as _paired_directional,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.paired.paired_g import (  # noqa: F401
    paired_g as _paired_g,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.paired.paired_g_per_burst import (  # noqa: F401
    paired_g_per_burst as _per_burst,  # pyright: ignore[reportUnusedImport]
)
