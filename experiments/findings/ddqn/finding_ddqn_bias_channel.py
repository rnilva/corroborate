"""Bias-channel: DDQN's outcome benefit on {Acrobot, Freeway} is
mediated by jensen_gap reduction (classical Hasselt-2016).

DoWhy backdoor + refutation per-env at canonical 1M:

  Acrobot-v1:
    ATE_marg = +1.83, ATE | jens = +0.90 → 51% absorbed
    placebo ATE = 0.000 ✓, RCC drift = 0.003 ✓
  Freeway-MinAtar:
    ATE_marg = +5.21, ATE | jens = +0.24 → **95% absorbed**
    placebo ATE = 0.000 ✓, RCC drift = 0.014 ✓

Both ATE estimates pass placebo refutation (zero) and
random-common-cause robustness (drift < 5% of ATE). Conditioning
on jensen_gap absorbs majority of the treatment effect — jens
is the load-bearing mediator on these 2 envs.

Memory: `findings_ddqn_mediator_heterogeneity` documents the
broader per-env causal picture (Breakout, PacMan have a
DIFFERENT channel via argmax_entropy_late; Asterix/SI/MetaMaze
are causally ambiguous).

EXPECTED: SUPPORTED. Both envs show clean bias mediation with
refutation passes. The Hasselt-2016 bias-reduction-mediates-
outcome theory holds at canonical scope on these 2 envs (and
ONLY these 2 envs — see sibling finding_ddqn_policy_structure_
channel for the entropy-mediated envs).

Reproducer: `scripts/per_env_dowhy_mediation.py`."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED
BLOCKED_ON: str | None = 'bridge implementations deferred — DoWhy per-env mediation primitive needs scope wrapping'
BRIDGES: tuple[Bridge, ...] = ()
