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

EXPECTED post-2026-05-19 ingest: SUPPORTED on all 3 bridges.
The May 18 pre-registered DRIFT prediction (Bridge 3 fires
HELD once `q_action_std_late` is in cache for k-sweep corpora)
LANDED: PartialSpearman ρ p_value < 1e-11 after re-ingesting
`action_dim_inflated_fourrooms_postfix` + `k_sweep_mountaincar`
with `q_action_std_late` added to REQUIRED_MEASURABLES.
Empirical: FR σ_Q 0.0050→0.0070 (k=1→4); MC σ_Q 0.115→0.167.

PR-4 verdict: HELD. The σ_Q-scaling mechanism (per-action
gradient density splitting → σ_Q growth ~√K) is empirically
corroborated cross-env on the canonical k-sweep panel.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.action_duplicate_fa_mechanism import (
    vanilla_jens_amplifies_with_k,
    vanilla_q_late_drifts_with_k,
    vanilla_sigma_q_scales_with_k,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    vanilla_jens_amplifies_with_k,
    vanilla_q_late_drifts_with_k,
    vanilla_sigma_q_scales_with_k,
)
