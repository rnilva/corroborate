"""Synthetic Type-A/B controlled-substrate pre-registered Finding (v3).

v1 → v2 → v3 evolution (both prior versions scrapped pre-sweep):

- **v1**: a bandit in a tuxedo (`s' = (s+1) mod L`); rvs knob
  confounded |Q|, Δ_v, AND Var_a[Q*] in lockstep; no FA-capacity
  axis. Scrapped per `/tmp/synthetic_env_roast.md`.
- **v2**: action-dependent transitions fixed v1's bandit structure,
  but the α knob modulated per-step REWARD-SAMPLING NOISE rather
  than the Q-target-side Var_a[V*(s')] that Cor 3.2's σ_clip
  actually concerns. n_seeds=12 (under-powered); L=64 with 32-unit
  hidden was 12.8× over-parameterized (no FA-binding); μ_best=0.05
  put |Q*| ≈ 50 (50× under natural-env scale). Walk-back paths
  pre-laundered every observed-data shape as publishable.
  Scrapped per `/tmp/synthetic_v2_roast.md`.
- **v3**: anisotropy primitive on the Q-TARGET side. State-baked
  payoffs `mu_state(s) = peak_value · β^(s mod K)` set
  Var_a[V*(s')] directly; `peak_value=1.0` matches natural-env
  Q scale; `noise_sigma=0.02` matches Asterix knife-edge σ/Δ ≈ 2%;
  L=1024 with hidden=[16] aliases 4096 Q-values through 16-dim
  bottleneck (genuine FA-binding); n_seeds=27 ≥ 27 (within
  budget; v2 critic's recommendation was ≥30, see §Honest gaps).

v3 panel has TWO structural axes + γ substrate:

- L = n_states ∈ {32, 1024} (FA-capacity)
- β = beta ∈ {0.0, 0.5, 0.9} (Type-A/B on Q-target side)
- γ ∈ {0.95, 0.99, 0.999} (substrate axis)

6 envs × 3 γ × 2 arms × 27 seeds = 972 cells (≤ 1000 budget).

The v3 bridges:

- **P1 (PRIMARY)**: ρ(β, d_out) ≤ −0.5 pooled across γ. The
  load-bearing prediction; n_strata=18 ≥ min_strata=10.
- **D1 (DIAGNOSTIC)**: ρ(argmax_margin, d_out) ≥ +0.5. Inverse
  parameterization of P1; corroborates the mechanism interpretation.
- **D2 (DIAGNOSTIC)**: ρ(β, d_out) ≤ −0.6 at γ=0.999 sub-scope.
  Structurally fires POWER_INSUFFICIENT at n_strata=6 < 10;
  diagnostic value via observed ρ magnitude compared to pooled.
- **N1 (PROPERLY-POWERED NULL)**: |ρ(L, d_out)| < 0.30 AND
  p > 0.30 at β=0. Capacity-alone shouldn't drive d_out
  direction. v2's N1 noise-permissive |ρ|≤0.3 alone was ~70%
  type-I; v3 requires the dual rho+p criterion.

## Expected verdict pre-sweep

EXPECTED = EMPTY_EXTENT until the sweep ingests. The Finding fires
UNDERPOWERED [blocked] until then; framework's DRIFT detection
fires when the post-ingest verdict deviates from this pin.

## REFUTATION criterion (load-bearing pre-registered retraction)

The v2 critic noted: "the walk-back paths accept every observed-
data shape as publishable — that's not commitment, it's a flowchart
of post-hoc framings." v3 commits to a SPECIFIC data shape that,
if observed, forces the substrate-author to retract the claim:

**If P1 fires NO_EFFECT-NULL (|ρ| < 0.20) with n_strata ≥ 15
admitted** (an adequately-powered null at the primary covariate),
the v3 synthetic substrate FAILS to reproduce the natural-env
Asterix Type-B mechanism. This is a RETRACTION, not a walk-back:
the substrate-author cannot claim "synthetic substrate enables
causal env-feature identification of the Asterix harm regime."
The Finding's EXPECTED would be repinned to NO_EFFECT-NULL with
BLOCKED_ON=None, and the substrate paper would carry the
retraction as a methodology-demonstration finding ("this attempt
fails because Q-target-side anisotropy alone, in a chain MDP
with calibrated knife-edge σ/Δ, does not reproduce Asterix's
mechanism").

## Diagnostic predictions (disambiguation surface)

- **P1 HELD + D1 HELD**: β-driven mechanism corroborated;
  argmax-margin is the proximate mediator.
- **P1 HELD + D1 NULL**: β matters but argmax-margin isn't the
  mediator. Substantive open question (which channel?).
- **P1 HELD + D2 ρ stronger than pooled**: γ-amplification
  confirmed; the chain-amplification story carries.
- **P1 HELD + D2 ρ NULL**: γ doesn't modulate the β effect in
  synthetic; chain amplification is natural-env-specific.
- **N1 HELD + P1 HELD**: clean separation — β drives, L doesn't.
- **N1 SIGN_FLIP + P1 HELD**: capacity has an independent channel
  beyond β at Type-A; walks back the "single causal axis" claim.
- **N1 HELD + P1 NULL**: substrate is QUIET — neither β nor L
  drives DDQN's sign. Different from REFUTATION because N1 is
  consistent with HELD or NULL; needs the REFUTATION clause's
  adequately-powered check on P1.

## Critic recommendations addressed

1. **μ_best ≈ 1**: peak_value=1.0 (was μ_best=0.05). |Q*| at
   γ=0.999 is now 1000, matching natural-env Asterix Q≈436.
2. **σ at 1-3% of Δ_v**: noise_sigma=0.02·peak_value. At β=0
   (Δ_v=peak), σ/Δ=2%; at β=0.5, σ/Δ=4%; at β=0.9, σ/Δ=20% (FA-
   residual dominates argmax — the knife-edge end of the regime).
3. **L ≥ 1024 with hidden ≤ 16**: L ∈ {32, 1024}; hidden=[16].
   At L=1024 the FA must alias 4096 Q-values through 16-dim
   hidden → genuine capacity-binding.
4. **n_seeds ≥ 30**: PARTIAL — n_seeds=27 (cell budget binding
   at 972 ≤ 1000). 27 is closer to 30 than v2's 12; the per-
   stratum d SE ≈ sqrt(4/27) ≈ 0.385 vs v2's 0.577.
5. **Anisotropy primitive on Q-target side**: YES, state-baked
   `mu_state(s) = peak_value · β^(s mod K)`. Var_a[V*(s')] is
   hand-set via the cross-action shape, NOT Var_a[reward noise].

## Companion docs

- `docs/PRE_REGISTRATION_synthetic_bias_typeb.md` — predictions
  + v1/v2/v3 evolution + REFUTATION clause.
- `/tmp/synthetic_v2_roast.md` — the hostile v2 review v3 addresses.
- `/tmp/synthetic_env_roast.md` — the v1 review.
- `src/corroborate_rl/corroborate_rl/synthetic_bias_typeb.py` —
  the v3 env module.
- `experiments/configs/synthetic_bias_typeb_v3_sweep.yaml` — the
  v3 sweep config.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.synthetic_bias_typeb import (
    ddqn_harm_amplified_at_g999__synthetic_typeb_v3,
    ddqn_harms_under_high_beta__synthetic_typeb_v3,
    ddqn_helps_when_argmax_margin_wide__synthetic_typeb_v3,
    n_states_alone_does_not_drive_dout__synthetic_typeb_v3,
)


# Pre-registration: EXPECTED = EMPTY_EXTENT until sweep ingests.
# All four bridges fire POWER_INSUFFICIENT (n_strata=0) without
# data. The framework's DRIFT detection fires when the post-ingest
# composed verdict deviates from this pin.
EXPECTED: ClusterVerdict = ClusterVerdict.EMPTY_EXTENT


BLOCKED_ON: str | None = (
    'Pre-registered 2026-05-19 BEFORE v3 sweep ran (v1 + v2 both '
    'scrapped pre-sweep — see docs/PRE_REGISTRATION_synthetic_bias_typeb.md '
    '+ /tmp/synthetic_v2_roast.md). Awaiting ingest of '
    '`experiments/data/synthetic_bias_typeb_v3_pilot` (6 envs × 3 γ × '
    '2 arms × 27 seeds = 972 cells). Predicted post-ingest '
    'EXPECTED=SUPPORTED iff P1 HELDs (ρ(β, d_out) ≤ −0.5, p ≤ 0.05) '
    'AND D1 HELDs (ρ(argmax_margin, d_out) ≥ +0.5) AND N1 HELDs-as-'
    'null (|ρ(L, d_out)| < 0.30 at β=0). REFUTATION fires if P1 '
    'NO_EFFECT-NULL at adequately-powered n_strata ≥ 15 — retracts '
    'the synthetic-substrate identification claim (see Finding '
    'docstring §REFUTATION criterion). Drift on any bridge fires '
    'DRIFT at hypothesis run-time.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_harms_under_high_beta__synthetic_typeb_v3,
    ddqn_helps_when_argmax_margin_wide__synthetic_typeb_v3,
    ddqn_harm_amplified_at_g999__synthetic_typeb_v3,
    n_states_alone_does_not_drive_dout__synthetic_typeb_v3,
)
