"""Expectile reproduces DDQN's mechanism-link disconnect on FR.

The disconnect (FINDINGS rev 9): DDQN cuts the Jensen-bias mech
AND outcome rises in aggregate AND the per-burst link between
Δ_jens and Δ_outcome is null. The mech→outcome story is
non-compositional at the stratum level.

This Finding asks: does the same disconnect reproduce when the
DDQN swap is replaced by an expectile-greedify swap (a different
bias-correction family)? Cluster shape (3 bridges):

- mech arm: `expectile_reduces_jensen_gap_more_than_ddqn__fourrooms`
  HELD-INVERSE (expectile cuts mech, even more aggressively than
  DDQN — Strategy 2's premise).
- outcome arm: `expectile_outcome_effect__fourrooms` HELD-DIRECT
  (expectile improves outcome vs vanilla in aggregate).
- link arm: `expectile_link_null__fourrooms` HELD-null
  (per-burst Δ_jens → Δ_outcome link is null — the disconnect
  signature).

All three HELD → SUPPORTED (disconnect reproduces); any REFUTED
→ REFUTED; mix → UNDERPOWERED. The cluster-shaped framing is
what gives the disconnect claim its scientific content; no
single bridge can test "marginal effect AND null link" because
those are two different sample shapes (paired-d vs per-burst-Δ
DoWhy backdoor).

**Why a Finding and not a bridge with `depends_on`**: the
framework currently composes verdicts across bridges only at
the Finding level (CLAUDE.md §"Findings"; see also
`CHAINED_BRIDGES_DESIGN.md` for the planned bridge-level
`depends_on`). A bridge that fixture-injects another bridge's
verdict isn't supported today, so the cluster lives here.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.dqn_bridges import (
    expectile_link_null__fourrooms,
    expectile_outcome_effect__fourrooms,
    expectile_reduces_jensen_gap_more_than_ddqn__fourrooms,
)


# EMPIRICAL state: the expectile arm is not yet in the canonical
# cache (the bridges resolve to POWER_INSUFFICIENT today), so the
# cluster verdict is UNDERPOWERED. Pin to that + name the data
# gap in BLOCKED_ON. When the expectile_3way corpus is ingested
# this clears and the cluster's actual verdict surfaces.
EXPECTED: ClusterVerdict = ClusterVerdict.UNDERPOWERED


BLOCKED_ON: str | None = (
    'expectile_3way corpus not ingested into the canonical '
    'ddqn cache — all three constituent bridges resolve to '
    'POWER_INSUFFICIENT. Ingest the corpus (or move expectile '
    'cells into a co-ingested sibling) to surface the actual '
    'cluster verdict.'
)


BRIDGES: tuple[Bridge, ...] = (
    expectile_reduces_jensen_gap_more_than_ddqn__fourrooms,
    expectile_outcome_effect__fourrooms,
    expectile_link_null__fourrooms,
)
