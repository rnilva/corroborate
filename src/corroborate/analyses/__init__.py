"""Framework analyses — typed `@analysis`-decorated primitives
consumed by Bridges via fixture-injection (see
`corroborate.bridge.analysis`).

**This module exists for its import side effect only.** Each
submodule below registers one or more analyses with the global
analysis registry; importing `corroborate.analyses` (or any
specific analysis submodule) populates the registry.
`__all__ = []` because consumers do NOT `from corroborate.analyses
import X` — they reference analyses by name via Bridge
parameter declarations:

    @claim_bridge
    def my_bridge(paired_g: PairedGResult, ...) -> Verdict:
        ...

The framework's fixture-injection in `bridge.analysis` looks up
the parameter name (`paired_g`) against the registry that this
import populates. The `_*` aliases and `pyright: ignore` comments
suppress unused-import warnings on the side-effect imports.

Available analyses are discoverable via
`corroborate.bridge.analysis.registered_names()`."""
from corroborate.analyses import (  # noqa: F401
    arm_mean_diff as _arm_mean_diff,  # pyright: ignore[reportUnusedImport]
    bootstrap_paired_g as _bootstrap_pg,  # pyright: ignore[reportUnusedImport]
    cliff_delta_paired as _cliff_delta,  # pyright: ignore[reportUnusedImport]
    dowhy as _dowhy,  # pyright: ignore[reportUnusedImport]
    factorial_2x2 as _factorial,  # pyright: ignore[reportUnusedImport]
    meta_regression_paired_g as _mr,  # pyright: ignore[reportUnusedImport]
    meta_regression_per_burst as _mr_per_burst,  # pyright: ignore[reportUnusedImport]
    mundlak_decomposition as _mundlak,  # pyright: ignore[reportUnusedImport]
    mundlak_paired_g_per_burst as _mundlak_pgpb,  # pyright: ignore[reportUnusedImport]
    paired_comparison as _paired_comparison,  # pyright: ignore[reportUnusedImport]
    paired_g as _paired_g,  # pyright: ignore[reportUnusedImport]
    paired_g_among_solvers as _solvers,  # pyright: ignore[reportUnusedImport]
    paired_g_per_burst as _per_burst,  # pyright: ignore[reportUnusedImport]
    paired_g_pooled as _pooled,  # pyright: ignore[reportUnusedImport]
    paired_link_per_burst as _link_per_burst,  # pyright: ignore[reportUnusedImport]
    paired_link_per_env as _link_per_env,  # pyright: ignore[reportUnusedImport]
    paired_continuous_do_dowhy as _paired_continuous_do_dowhy,  # pyright: ignore[reportUnusedImport]
    link_attenuation_dowhy as _link_attenuation_dowhy,  # pyright: ignore[reportUnusedImport]
    paired_arm_spearman as _paired_arm_spearman,  # pyright: ignore[reportUnusedImport]
    paired_delta_link_dowhy as _paired_delta_link_dowhy,  # pyright: ignore[reportUnusedImport]
    proportion_mediated as _proportion_mediated,  # pyright: ignore[reportUnusedImport]
    partial_spearman_paired as _psp,  # pyright: ignore[reportUnusedImport]
    cross_config_paired_slope as _ccps,  # pyright: ignore[reportUnusedImport]
    within_arm_link as _within_arm_link,  # pyright: ignore[reportUnusedImport]
    tautology_audit as _audit,  # pyright: ignore[reportUnusedImport]
    verdict_distribution as _verdict_dist,  # pyright: ignore[reportUnusedImport]
)

__all__: list[str] = []
