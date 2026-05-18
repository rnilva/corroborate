"""Cross-env regime classification of DDQN's outcome effect at γ=0.999.

Four bridges, one per MinAtar env, jointly assert that DDQN's
outcome sign at γ=0.999 is REGIME-DEPENDENT — at least three
distinct signs jointly observed within a single env family:

  Asterix-MinAtar  → HARM    (d=-0.80 z=-3.1)
  Breakout-MinAtar → HELP    (d=+0.66 z=+2.6)
  SpaceInvaders    → HELP    (d=+2.16 z=+8.4)
  Freeway-MinAtar  → NEUTRAL (d=+0.10 z=+0.4)

If all 4 bridges admit (each env's predicted sign HELDs), the
cluster is SUPPORTED. This is the substantive cross-env claim
that the "DDQN universally helps" framing from Hasselt 2016 is
FALSIFIED on MinAtar at the long-horizon edge.

The regime classifier that explains these signs is `vanilla's
jens scaling exponent under γ→1`:
  - super-linear (Asterix 622×) → Q-EXPLODED → HARM
  - moderate (Breakout 113×, SI 133×) → Q-STRUCTURED → HELP
  - flat (Freeway 1.5×) → Q-COLLAPSED → NEUTRAL

Captured in the regime-classification memory; this Finding is
the formal substantive claim.

PRE-REGISTERED DRIFT (2026-05-18): k=2 sweep is running; k=4 to
follow. Prediction committed at this commit hash: at k=2 and k=4
the cross-env classifier STAYS SUPPORTED with intensified d
magnitudes — Asterix harm intensifies (d ≤ -1.0 by k=4 OR
saturates), Breakout help strengthens (d_out ≥ +0.85 by k=4 per
the √log amplification), SI help strengthens (already huge at
k=1, may plateau). Freeway stays NEUTRAL. If ANY env's sign flips
at k=2/k=4, the regime classification walks back — likely the
4-bin classifier needs k-conditioning. The T3a panel extension
(separately running) adds Snake γ=0.999 (CLIP-RATCHET cross-γ
test), PacMan γ=0.999 (predicted Q-STRUCTURED), LunarLander
γ=0.999 (novel slot — could expose a 5th regime).
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.outcome_regime_g999_cross_env import (
    ddqn_harms_asterix_g999__cross_env_regime,
    ddqn_helps_breakout_g999__cross_env_regime,
    ddqn_helps_si_g999__cross_env_regime,
    ddqn_neutral_freeway_g999__cross_env_regime,
)


EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED


BLOCKED_ON: str | None = None


BRIDGES: tuple[Bridge, ...] = (
    ddqn_harms_asterix_g999__cross_env_regime,
    ddqn_helps_breakout_g999__cross_env_regime,
    ddqn_helps_si_g999__cross_env_regime,
    ddqn_neutral_freeway_g999__cross_env_regime,
)
