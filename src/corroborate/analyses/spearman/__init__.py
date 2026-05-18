"""Spearman-based analyses — JCI (Fisher-z-pooled) (partial-)Spearman.

The canonical primitive is `partial_spearman` (unified, subsumes
the 5 legacy `stratified_*` / `per_burst_*_jci_*` variants —
see CLAUDE.md canonical-analyses table). `stratum_panel_jci_spearman`
is kept separate: it's a stratum-panel falsification primitive
(per-stratum marginal-vs-stratified ρ comparison) with a different
shape than the per-cell-or-per-burst observation-iterating
`partial_spearman`.
"""
from corroborate.analyses.spearman.partial_spearman import (  # noqa: F401
    partial_spearman as _partial_spearman,  # pyright: ignore[reportUnusedImport]
)
from corroborate.analyses.spearman.stratum_panel_jci_spearman import (  # noqa: F401
    stratum_panel_jci_spearman as _stratum_panel_jci_spearman,  # pyright: ignore[reportUnusedImport]
)
