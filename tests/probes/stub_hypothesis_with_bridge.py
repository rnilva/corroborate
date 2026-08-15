"""On-disk Hypothesis with one bridge, for `--render` CLI tests.

`tests.probes.stub_hypothesis` deliberately declares zero bridges
(the smallest Protocol-conforming shape); the `--render` success
path needs at least one authored edge to draw, so this sibling
carries a single interventional bridge. The CLI tests patch
`corroborate.cli.hypothesis.run`, so the bridge body never runs
against data — only its authored topology matters.
"""
from __future__ import annotations

from corroborate.bridge import Bridge
from corroborate.bridge.bridge import claim_bridge
from corroborate.bridge.verdict import Verdict
from corroborate.core.claim import claim
from corroborate.core.intervention import DoEffect, Intervention
from corroborate.graph.causal import Direction


@claim
def _treatment_op(x: int) -> int:
    return x


INTERVENTION = DoEffect(
    arms=(
        (),
        (Intervention(slot_path='op', replacement=_treatment_op),),
    ),
)


@claim_bridge(
    source=INTERVENTION,
    target='checkpoint_return',
    direction=Direction.DIRECT,
    pair_by=('seed',),
    predicted_direction='a_gt_b',
)
def _return_edge() -> Verdict:
    return Verdict.HELD


BRIDGES: tuple[Bridge, ...] = (_return_edge,)
FINDINGS: tuple[()] = ()
