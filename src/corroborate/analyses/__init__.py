"""Framework analyses — registered by import.

Each submodule registers one or more analyses via `@analysis`.
Importing `corroborate.analyses` (or any submodule) populates
the registry; bridges consume by parameter name."""
from corroborate.analyses import paired_g as _paired_g  # noqa: F401  # pyright: ignore[reportUnusedImport]

__all__: list[str] = []
