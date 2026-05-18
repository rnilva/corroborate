"""Action_duplicate FA-gradient-density mechanism cluster.

Three bridges encode the substantive empirical claims about how
`ActionDuplicate(k=K)` amplifies vanilla bias:

  Edge 1 — jens scales with K: vanilla bias grows monotonically
    with K within each env. Empirical: FR ρ ≈ +1, MC ρ ≈ +1.

  Edge 2 — Q drifts with K (one of two downstream channels):
    on unsolvable envs, Q grows upward while MC stays at the
    failure floor; on solvable envs, Q stays put and MC drops.
    This bridge tests the Q-drift channel.

  Edge 3 — per-action σ_Q scales with K: the proximate
    mechanism. Per-action output weights split gradient signal
    by K, raising estimation noise. Cache currently lacks
    q_action_std_late on k-sweep corpora → POWER_INSUFFICIENT
    until uplifted.

EXPECTED: SUPPORTED on bridges 1+2 (data is in cache);
POWER_INSUFFICIENT on bridge 3 (column missing). The composed
verdict at cluster level is UNDERPOWERED. BLOCKED_ON names the
re-ingest path that would land bridge 3.

PRE-REGISTERED DRIFT (2026-05-18): Bridge 3 prediction committed
at this commit hash. The per-corpus measurements.parquet ALREADY
shows the σ_Q scaling (FR k=1→4: 0.0050→0.0070; MC: 0.115→0.167;
ρ ≈ +1 in each). Once a re-ingest uplifts `q_action_std_late`
into the ddqn_sweeps cache, Bridge 3 should DRIFT to HELD (the
ρ values above are well past the bridge's predicted_direction
threshold). If Bridge 3 lands with ρ < +0.5 OR sign-flips, the
σ_Q-scaling mechanism walks back — Finding 7 of the case study
report becomes "amplification real, mechanism not isolated".
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.action_duplicate_fa_mechanism import (
    vanilla_jens_amplifies_with_k,
    vanilla_q_late_drifts_with_k,
    vanilla_sigma_q_scales_with_k,
)


EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'Bridge 3 (vanilla_sigma_q_scales_with_k) needs '
    'q_action_std_late in the ddqn_sweeps cache on k-sweep corpora '
    '(action_dim_inflated_fourrooms_postfix, k_sweep_mountaincar). '
    'The measurement.parquet files locally have the values '
    '(measured at original ingest), but the ddqn_sweeps cache '
    'parquet only retains columns referenced by bridges or '
    'REQUIRED_MEASURABLES. After adding q_action_std_late to a '
    'REQUIRED_MEASURABLES surface (or via this very bridge\'s '
    'target field), a re-ingest will uplift the column. Empirical '
    'preview from per-corpus measurements: FR σ_Q 0.0050→0.0070 '
    '(k=1→4, ρ ≈ +1), MC σ_Q 0.115→0.167 (ρ ≈ +1). Bridge 3 '
    'should fire HELD once the cache picks up the column.'
)


BRIDGES: tuple[Bridge, ...] = (
    vanilla_jens_amplifies_with_k,
    vanilla_q_late_drifts_with_k,
    vanilla_sigma_q_scales_with_k,
)
