"""Finding hand-roll #1 — REACH bias-correction link is causally
corroborated.

Module-level attributes encode the claim's structural anatomy;
the `walk(g)` callable IS the verdict. The shape this file's
attributes settle into is the input we'll use to author a typed
`Finding` Protocol mirroring
`corroborate.core.hypothesis.Hypothesis`.

Substantive claim: on the REACH-polarity cohort (FourRooms,
Acrobot, MountainCar, MetaMaze) under scope `DDQN_RELEVANT_SCOPE`
(G1 premise active ∧ G2 argmax-vulnerable ∧ standard config), the
edge `jensen_gap → eval_best_burst_mean` is corroborated by the
DoWhy refutation triple (backdoor ATE + placebo + RCC).
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
from experiments.findings.ddqn.bias_correction import (
    reach_link_backdoor_ate_negative,
    reach_link_placebo_refuted,
    reach_link_rcc_robust,
)


# ============ Finding Protocol attributes ============


HEADLINE: str = (
    'REACH bias-correction link is causally corroborated — '
    'DoWhy refutation triple admits on jensen_gap → '
    'eval_best_burst_mean across FourRooms / Acrobot / '
    'MountainCar / MetaMaze under DDQN_RELEVANT_SCOPE.'
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


# Typed bridge references — pyright catches rename / delete at
# import; no string magic. Mirrors `Hypothesis.BRIDGES` shape.
BRIDGES: tuple[Bridge, ...] = (
    reach_link_backdoor_ate_negative,
    reach_link_placebo_refuted,
    reach_link_rcc_robust,
)


# ============ Smoke runner ============


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
        print('  → claim flipped relative to authoring snapshot. '
              'Investigate: new data carved the cluster, bridge '
              'renamed/deleted, or scope refactored.')


if __name__ == '__main__':
    _main()
