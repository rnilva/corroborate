"""Finding hand-roll #3 — asymmetric rescue envelope at rs=0.1.

The CONTRAST case: bridges share `(do(DDQN), outcome_native)` but
have DIFFERENT scopes (per-env) — three separate sub-claims, no
shared extent, not a cluster. After the simplification (drop
`require_shared_extent` knob), the envelope and cluster Findings
have **identical walk shape** — `composed_verdict` doesn't care
whether the bridges happen to share extent or not. The author
just lists which bridges support the claim.

Substantive claim: at reward_scale=0.1, DDQN rescues under-
learning vanilla on FourRooms but does NOT rescue on Acrobot or
CartPole. The asymmetric envelope is the conjunction of:
- positive HELD: `ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1`
- null HELD × 2: `ddqn_does_not_rescue__{acrobot,cartpole}_rs_0p1`

Expected SUPPORTED. Currently the two null sides return
POWER_INSUFFICIENT (CIs span the ceiling), so the envelope walks
UNDERPOWERED — honest drift signal: the FR side held but the
nulls haven't decisively landed on the postfix cache.
"""
from __future__ import annotations

import json
from pathlib import Path

from corroborate.bridge.bridge import Bridge
from corroborate.bridge.verdict import Verdict
from corroborate.graph.causal import (
    ClusterVerdict, PostEvalEntry, composed_verdict, evaluated_graph,
)

from experiments.findings import ddqn as HYPOTHESIS
from experiments.findings.ddqn.rs_rescue import (
    ddqn_does_not_rescue__acrobot_rs_0p1,
    ddqn_does_not_rescue__cartpole_rs_0p1,
    ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1,
)


HEADLINE: str = (
    'Rescue at rs=0.1 is FourRooms-specific — positive HELD on FR + '
    'null HELDs on Acrobot/CartPole — DDQN rescue does NOT generalize '
    'beyond the underlearning regime of FourRooms.'
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BRIDGES: tuple[Bridge, ...] = (
    ddqn_rescues_underlearning_vanilla__fourrooms_rs_0p1,
    ddqn_does_not_rescue__acrobot_rs_0p1,
    ddqn_does_not_rescue__cartpole_rs_0p1,
)


def _main() -> None:
    short = HYPOTHESIS.__name__.split('.')[-1]
    repo_root = Path(__file__).resolve().parents[3]
    run_json = repo_root / 'experiments/findings' / f'{short}.run.json'
    snapshot = json.loads(run_json.read_text())
    post_eval = {
        b['bridge_name']: PostEvalEntry(
            verdict=Verdict(b['verdict']),
            extent_hash=int(b['extent_hash']),
        )
        for b in snapshot['bridges']
    }
    g = evaluated_graph(HYPOTHESIS.BRIDGES, post_eval)
    verdict = composed_verdict(g, bridges=BRIDGES)
    drift = verdict != EXPECTED
    print(f'HEADLINE: {HEADLINE}')
    print()
    print(f'  walk verdict:  {verdict.value}')
    print(f'  expected:      {EXPECTED.value}')
    print(f'  drift:         {drift}')
    if drift:
        print()
        print('  → claim flipped relative to authoring snapshot.')


if __name__ == '__main__':
    _main()
