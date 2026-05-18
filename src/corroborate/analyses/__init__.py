"""Framework analyses — typed `@analysis`-decorated primitives
consumed by Bridges via fixture-injection (see
`corroborate.bridge.analysis`).

**This module exists for its import side effect only.** Each
subpackage below registers one or more analyses with the global
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
in each subpackage suppress unused-import warnings on the
side-effect imports.

Subpackage layout (post-D consolidation, 2026-05-17):

- `paired/`       — paired-shape analyses (paired_g family)
- `link/`         — predictor→target link analyses
- `spearman/`     — within / JCI / partial Spearman family
- `panel/`        — `per_stratum_panel` helper + stratum-panel
                    analyses (`stratum_panel`, `stratum_effect_panel`,
                    `stratified_arm_diff_pooled`, `meta_regression_*`)
- `dowhy/`        — DoWhy backdoor + refutations + stratum-link
                    DoWhy primitives
- `diagnostic/`   — `tautology_audit`, `verdict_distribution`

Top-level analyses (don't fit a single bucket yet):
- `pc_discovery`  — constraint-based PC algorithm (different ID
                    family from DoWhy estimators)

Shared infrastructure (top-level, leading underscore):
- `_dedup_diagnostics`, `_dowhy_internal`, `_result_protocols`

Available analyses are discoverable via
`corroborate.bridge.analysis.registered_names()`.
"""
from corroborate.analyses import (  # noqa: F401
    diagnostic as _diagnostic,  # pyright: ignore[reportUnusedImport]
    dowhy as _dowhy,  # pyright: ignore[reportUnusedImport]
    link as _link,  # pyright: ignore[reportUnusedImport]
    paired as _paired,  # pyright: ignore[reportUnusedImport]
    panel as _panel,  # pyright: ignore[reportUnusedImport]
    pc_discovery as _pc_discovery,  # pyright: ignore[reportUnusedImport]
    spearman as _spearman,  # pyright: ignore[reportUnusedImport]
)

__all__: list[str] = []
