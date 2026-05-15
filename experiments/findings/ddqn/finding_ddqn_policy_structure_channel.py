"""Policy-statistics-channel (revised framing): DDQN's outcome
benefit on {Breakout, PacMan} is mediated by argmax-distribution
statistics, NOT by jensen_gap. The MECHANISM (state-conditional
policy modification vs Q-flat noise) is not identified from
current measurables.

DoWhy backdoor + refutation per-env at canonical 1M:

  Breakout-MinAtar:
    ATE_marg = +17.67
    ATE | entropy = +8.42 → 52% absorbed (entropy mediates)
    ATE | persistence = ... → 52% absorbed (persistence also mediates)
    ATE | jens = +45.29 (sign-FLIPPED collider behavior — jens
        is NOT a clean mediator)
    placebo ATE = 0.000 ✓, RCC drift = 0.022 ✓
  PacMan-jumanji:
    ATE_marg = +165.67
    ATE | entropy = +56.54 → 66% absorbed
    ATE | persistence → 94% absorbed (alternative summary, same
        underlying mediator)
    ATE | jens = +156.98 (only 5% absorbed)
    placebo ATE = 0.000 ✓, RCC drift = 2.80 ✓ (1.7% of ATE)

**Caveat (important)**: `argmax_entropy_late` is the Shannon
entropy of the MARGINAL action distribution over late training,
not state-conditional. Same entropy can reflect EITHER (a)
state-differentiated argmax (different argmax per state region
— true policy structure modification) OR (b) Q-flat indecisive
policy (DDQN flattens Q across actions). The substrate's
current measurables cannot directly distinguish (a) from (b)
without a state-conditional argmax measurable, which requires
state-hashing for image-obs envs (MinAtar, PacMan) — not
currently available.

**Partial cross-check** via `argmax_persistence_late`
(temporal-local argmax consistency): if DDQN's effect were
pure Q-flat noise (b), we'd expect LOW persistence + HIGH
entropy on DDQN. But persistence MEDIATES similarly to entropy
on Breakout (52%) and PacMan (94%), and DDQN's persistence is
not catastrophically lower than vanilla's. Both proxies aligned
→ not pure Q-flat noise; some state-conditional structure
likely present. But this doesn't directly identify it.

EXPECTED: SUPPORTED at the observational mediation level.
The mechanistic interpretation between (a) and (b) is open;
verification requires substrate extension (image bucket-hash
+ state-conditional argmax measurable). See related deferred
project `project_image_state_hash_for_substrate.md`.

The collider behavior of jens on Breakout + PacMan (sign-
flipped slope when conditioning on jens; near-zero absorption)
robustly distinguishes these envs from the bias-channel envs
(Acrobot, Freeway) where jens absorbs 51-95% cleanly.

Reproducer: `scripts/per_env_dowhy_mediation.py`."""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED
BLOCKED_ON: str | None = 'bridge implementations deferred — DoWhy per-env mediation primitive needs scope wrapping'
BRIDGES: tuple[Bridge, ...] = ()
