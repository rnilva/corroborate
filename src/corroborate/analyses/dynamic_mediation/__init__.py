"""Trajectory-resolved mediation primitives — package surface.

Two `@analysis`-decorated primitives sit in this package:

  - `dynamic_partial_spearman` — per-burst partial-Spearman
    magnitude (`rho_partial[b]`). Iterates one ρ per BURST INDEX
    over per-burst `List(Float64)` columns. Sibling to the static
    `partial_spearman`.
  - `dynamic_pc_adjacency` — per-burst PC-style Fisher-z CI test
    (depth-0 marginal + depth-1 partial). Reports per-burst edge
    presence and counts across the trajectory. Sibling to
    `corroborate.graph.discovery.discover_adjacency`.

Both surface the shared `TimeAggregationStatus` enum
(`SIGN_FLIP_DETECTED`, `WEAK_TIME_VARYING`, `CONSISTENT_DIRECTION`,
`UNDERPOWERED_BURSTS`) — the trajectory analogue of
`mediation_dowhy`'s `LinearityStatus`, surfacing burst-pool
pathologies as a typed value rather than a runtime gotcha.

Inputs are PER-BURST `List(Float64)` columns on `cells` (a
`pl.DataFrame`), produced by the substrate's `_per_burst`
measurables (see e.g. `bootstrap_gap_magnitude_per_burst` in
`corroborate_rl.dqn.measurables`). Each cell carries an array of
length `n_bursts` at the named columns. The primitives align by
burst index across cells within a stratum; cells with shorter
trajectories still contribute their prefix (the per-burst valid
count `n_per_burst[b]` grows as the longer cells continue
contributing past shorter cells' tails — "ragged tail"
semantics, the less-information-losing form vs truncating all
cells to the shortest stratum length).

Granularity contract (shared between primitives):

  - `mediator_per_burst`, `outcome_per_burst`: str column names on
    `cells` carrying `List[Float64]` of length `n_bursts`, OR
    `Measurable[..., NDArray]` instances (mirroring
    `partial_spearman`'s lazy-evaluation pattern — the framework
    reads the Measurable's cached column if present, else
    evaluates against the cell record). Scalar columns trigger a
    structural raise via the `_as_float_list` shape contract.
  - `arm_field`: str column name on `cells` carrying a per-cell
    string arm tag. The arm is numerically encoded (sorted-unique
    → integer code) for the rank-based Spearman; only the
    *partition* between arms matters for ρ.

Per-burst alignment is "ragged tail": `n_bursts = max trajectory
length`, `n_per_burst[b]` shrinks as shorter cells drop off.

Shared infrastructure lives in `_common.py` (classifier, encoding,
ragged-tail extraction). The two primitives sit in their own
modules (`partial_spearman.py`, `pc_adjacency.py`) and re-export
their public typed result + the `@analysis`-decorated function
through this package `__init__.py`.

The static `partial_spearman` should NOT be used on per-burst data
when burst dynamics are non-monotonic — see
`findings_per_burst_mediation_trajectory` and
`DYNAMIC_MEDIATION_DESIGN.md` for the empirical motivation.
"""
from __future__ import annotations

# Re-export the shared types so consumers can `from
# corroborate.analyses.dynamic_mediation import
# TimeAggregationStatus`. The module file `_common.py` is internal
# (leading-underscore signals that), but the typed enum + Stratum
# alias are part of the package's public typed contract.
from corroborate.analyses.dynamic_mediation._common import (
    ClusterBootstrapEdgeCounts,
    ClusterBootstrapInterval,
    FisherZDLPool,
    Stratum,
    TimeAggregationStatus,
    classify_status as classify_status,
    cluster_bootstrap_edge_counts as cluster_bootstrap_edge_counts,
    cluster_bootstrap_pool as cluster_bootstrap_pool,
    fisher_z_dl_pool as fisher_z_dl_pool,
)
from corroborate.analyses.dynamic_mediation.partial_spearman import (
    DynamicMediationResult,
    dynamic_partial_spearman,
)
from corroborate.analyses.dynamic_mediation.pc_adjacency import (
    DynamicPCResult,
    dynamic_pc_adjacency,
)


__all__ = [
    'ClusterBootstrapEdgeCounts',
    'ClusterBootstrapInterval',
    'DynamicMediationResult',
    'DynamicPCResult',
    'FisherZDLPool',
    'Stratum',
    'TimeAggregationStatus',
    'dynamic_partial_spearman',
    'dynamic_pc_adjacency',
]
