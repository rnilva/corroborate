"""Finding hand-roll #2 — MetaMaze γ-amplification REFUTED.

Stress-tests the Finding shape under `EXPECTED=REFUTED`. Same
Protocol surface as `finding_reach_bias_link.py` (the SUPPORTED
hand-roll), opposite expected verdict. Validates that the surface
is symmetric under negation.

Substantive claim (REFUTED): the predicted "high-γ amplifies
DDQN benefit on chain-depth-sensitive MetaMaze" fails on the
postfix corpus. Both mean- and median-aggregated sibling walks
(do(DDQN) → eval_best_burst_mean under `_METAMAZE_GAMMA_SCOPE`)
return REFUTED — DDQN HURTS at γ=0.999 on MetaMaze, not amplifies.
See `findings_metamaze_gamma_link.md` for the empirical
decomposition.
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
from experiments.findings.ddqn.within_env import (
    metamaze_link_steeper_at_high_gamma,
    metamaze_link_steeper_at_high_gamma__median,
)


HEADLINE: str = (
    'MetaMaze γ-amplification REFUTED — predicted high-γ ↑ DDQN '
    'benefit; both mean + median sibling walks return refuted on '
    'the (do(DDQN), eval_best_burst_mean) cluster under '
    '_METAMAZE_GAMMA_SCOPE on the postfix corpus.'
)


EXPECTED: ClusterVerdict = ClusterVerdict.REFUTED


BRIDGES: tuple[Bridge, ...] = (
    metamaze_link_steeper_at_high_gamma,
    metamaze_link_steeper_at_high_gamma__median,
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
