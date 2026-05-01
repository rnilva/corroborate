"""Framework analyses — registered by import.

Each submodule registers one or more analyses via `@analysis`.
Importing `corroborate.analyses` (or any submodule) populates
the registry; bridges consume by parameter name."""
from corroborate.analyses import (  # noqa: F401
    dowhy as _dowhy,  # pyright: ignore[reportUnusedImport]
    factorial_2x2 as _factorial,  # pyright: ignore[reportUnusedImport]
    meta_regression_paired_g as _mr,  # pyright: ignore[reportUnusedImport]
    meta_regression_per_burst as _mr_per_burst,  # pyright: ignore[reportUnusedImport]
    mundlak_decomposition as _mundlak,  # pyright: ignore[reportUnusedImport]
    mundlak_paired_g_per_burst as _mundlak_pgpb,  # pyright: ignore[reportUnusedImport]
    paired_g as _paired_g,  # pyright: ignore[reportUnusedImport]
    paired_g_among_solvers as _solvers,  # pyright: ignore[reportUnusedImport]
    paired_g_per_burst as _per_burst,  # pyright: ignore[reportUnusedImport]
    paired_g_pooled as _pooled,  # pyright: ignore[reportUnusedImport]
    universe_scope as _universe,  # pyright: ignore[reportUnusedImport]
    tautology_audit as _audit,  # pyright: ignore[reportUnusedImport]
    verdict_distribution as _verdict_dist,  # pyright: ignore[reportUnusedImport]
)

__all__: list[str] = []
