"""Framework analyses — registered by import.

Each submodule registers one or more analyses via `@analysis`.
Importing `corroborate.analyses` (or any submodule) populates
the registry; bridges consume by parameter name."""
from corroborate.analyses import (  # noqa: F401
    meta_regression_paired_g as _mr,  # pyright: ignore[reportUnusedImport]
    paired_g as _paired_g,  # pyright: ignore[reportUnusedImport]
)

__all__: list[str] = []
