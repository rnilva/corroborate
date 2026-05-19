"""Synthetic Type-A/B controlled-substrate pre-registered Finding (v3.1).

v1 → v2 → v3 → v3.1 evolution (all prior versions scrapped pre-sweep):

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
- **v3**: state-baked deterministic payoff `mu_state(s) =
  peak · β^(s mod K)`. Value iteration confirmed two STRUCTURAL
  flaws (`/tmp/synthetic_v3_review.md`): (i) `Var_a[V*(s'_a)] = 0`
  identically at every β (modular periodicity made every reachable
  successor sit on the same V* orbit); (ii) Q* had only K=4
  distinct values across L=1024 states (no FA-capacity binding).
  v3 was the SAME conceptual error as v2 (per-step reward variance
  vs Q-target-side variance), just relocated.
- **v3.1**: RANDOM per-state payoffs
  `mu_state[s] = peak · (1 - spread + spread · U_s)` with
  `U_s ~ U(0, 1)` seeded by `payoff_seed`. Value iteration
  confirms `Var_a[V*(s'_a)] > 0` at every spread > 0 (scales
  monotonically) and Q* has ~L distinct entries (no modular
  collapse). The L axis now GENUINELY binds FA capacity.

v3.1 panel has THREE structural axes + γ substrate:

- L = n_states ∈ {32, 1024} (FA-capacity)
- payoff_spread ∈ {0.0, 0.25, 0.5, 0.75, 1.0} (anisotropy on
  Q-target side)
- payoff_seed ∈ {0, 1, 2} (cross-realisation averaging)
- γ ∈ {0.95, 0.99, 0.999} (substrate axis)

30 envs × 3 γ × 2 arms × 8 seeds = 1440 cells (≤ 1500 budget).

The v3.1 bridges:

- **P1 (PRIMARY)**: ρ(payoff_spread, d_out) ≤ −0.5 pooled across γ.
  The load-bearing prediction; n_strata=90 ≥ min_strata=10.
- **D2 (DIAGNOSTIC, PRE-REGISTERED POWER_INSUFFICIENT)**:
  ρ(payoff_spread, d_out) ≤ −0.6 at γ=0.999 sub-scope. v3.1
  pre-registers the PREDICTED verdict as POWER_INSUFFICIENT
  (matching the structural diagnostic), addressing the v3
  reviewer's discipline issue #1.
- **N1 (PROPERLY-POWERED NULL)**: |ρ(L, d_out)| < 0.30 AND
  p > 0.30 at spread=0. Capacity-alone shouldn't drive d_out
  direction when the env is structurally isotropic.

**v3 → v3.1 bridge surgery**: D1 (`ddqn_helps_when_argmax_margin_wide`)
was DROPPED. v3's argmax_margin = peak·(1-β) was a closed-form
function of β → D1's ρ was rank-equivalent to P1's ρ exactly
(the v3 reviewer's discipline issue #2: "two HELDs here look like
corroboration but carry one bit of information"). v3.1's
random per-state payoffs make argmax_margin a per-realisation
RANDOM variable not derivable from the env name; no closed-form
covariate. Dropping D1 honestly reduces the bridge count from
4 → 3.

## Expected verdict pre-sweep

EXPECTED = EMPTY_EXTENT until the sweep ingests. The Finding fires
UNDERPOWERED [blocked] until then; framework's DRIFT detection
fires when the post-ingest verdict deviates from this pin.

## REFUTATION criterion (load-bearing pre-registered retraction)

The v2 critic noted: "the walk-back paths accept every observed-
data shape as publishable — that's not commitment, it's a flowchart
of post-hoc framings." v3.1 inherits v3's binding commitment to a
SPECIFIC observed-data shape that retracts the substrate-
identification claim:

**If P1 fires NO_EFFECT-NULL (|ρ| < 0.20) with n_strata ≥ 15
admitted** (an adequately-powered null at the primary covariate),
the v3.1 synthetic substrate FAILS to reproduce the natural-env
Asterix Type-B mechanism. This is a RETRACTION, not a walk-back:
the substrate-author cannot claim "synthetic substrate enables
causal env-feature identification of the Asterix harm regime."
The Finding's EXPECTED would be repinned to NO_EFFECT-NULL with
BLOCKED_ON=None, and the substrate paper would carry the
retraction as a methodology-demonstration finding ("this attempt
fails because Q-target-side anisotropy alone, in a chain MDP
with calibrated knife-edge σ/Δ AND non-modular Q*, does not
reproduce Asterix's mechanism").

The REFUTATION criterion's threshold n_strata=15 is binding under
v3.1's 90-stratum primary panel: cell-budget-induced dropout
would have to exceed 83% before the criterion becomes vacuous.

## Diagnostic predictions (disambiguation surface)

- **P1 HELD + D2 ρ stronger than pooled**: γ-amplification
  confirmed; the chain-amplification story carries.
- **P1 HELD + D2 ρ NULL**: γ doesn't modulate the spread effect in
  synthetic; chain amplification is natural-env-specific.
- **N1 HELD + P1 HELD**: clean separation — spread drives, L
  doesn't (at the spread=0 baseline).
- **N1 SIGN_FLIP + P1 HELD**: capacity has an independent channel
  beyond spread at the isotropic baseline; walks back the
  "single causal axis" claim.
- **N1 HELD + P1 NULL**: substrate is QUIET — neither spread nor
  L drives DDQN's sign. Different from REFUTATION because N1 is
  consistent with HELD or NULL; needs the REFUTATION clause's
  adequately-powered check on P1.

## Critic recommendations addressed (v3.1)

1. **μ_best ≈ 1**: peak_value=1.0 (was μ_best=0.05). |Q*| at
   γ=0.999 is ≤ 1000, matching natural-env Asterix Q≈436.
2. **σ at 1-3% of Δ_v**: noise_sigma=0.02·peak_value. At
   spread=1.0 the per-state argmax-margin distribution has
   median 0.12; σ/Δ at the median is 17% (FA-residual regime).
   The heterogeneous spread of argmax-margins across states
   is closer to natural-env Asterix than v3's uniform per-env
   margin.
3. **L ≥ 1024 with hidden ≤ 16**: L ∈ {32, 1024}; hidden=[16].
   v3.1 fix: Q* has ~1024 distinct entries at L=1024 (verified
   by VI), so 16-dim hidden GENUINELY aliases 1024 distinct
   V*-values → real FA-binding (v3 had only 16 distinct Q*
   entries → 16-dim hidden trivially represented them).
4. **n_seeds ≥ 30**: PARTIAL — n_seeds=8 per env. The v3.1
   panel inflates the env-axis count (30 envs across 3
   payoff_seeds × 5 spreads × 2 L) to substitute env-level
   replication for seed-level replication. Cross-env averaging
   over payoff_seed smooths the per-realisation topology while
   the spread axis still has 30 distinct (L, seed) pairs at each
   spread level. Effective n at each spread level = 6 envs ×
   3 γ × 8 seeds = 144 cells.
5. **Anisotropy primitive on Q-target side, NON-PERIODIC**:
   FULLY ADDRESSED. `mu_state[s] = peak · (1 - spread + spread ·
   U_s)` with `U_s ~ U(0, 1)`. Verified by value iteration:
   `Var_a[V*(s'_a)] > 0` at every spread > 0 (v3 had this = 0);
   Q* has ~L distinct entries at L=1024 (v3 had only 16).

## Companion docs

- `docs/PRE_REGISTRATION_synthetic_bias_typeb.md` — predictions
  + v1/v2/v3/v3.1 evolution + REFUTATION clause.
- `/tmp/synthetic_v3_review.md` — the v3 review (structural flaws).
- `/tmp/synthetic_v2_roast.md` — the v2 review.
- `/tmp/synthetic_env_roast.md` — the v1 review.
- `src/corroborate_rl/corroborate_rl/synthetic_bias_typeb.py` —
  the v3.1 env module.
- `experiments/configs/synthetic_bias_typeb_v3_1_sweep.yaml` — the
  v3.1 sweep config.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.synthetic_bias_typeb import (
    ddqn_harm_amplified_at_g999__synthetic_typeb_v31,
    ddqn_harms_under_high_spread__synthetic_typeb_v31,
    n_states_alone_does_not_drive_dout__synthetic_typeb_v31,
)


# Pre-registration: EXPECTED = EMPTY_EXTENT until sweep ingests.
# All three bridges fire POWER_INSUFFICIENT (n_strata=0) without
# data. The framework's DRIFT detection fires when the post-ingest
# composed verdict deviates from this pin.
EXPECTED: ClusterVerdict = ClusterVerdict.EMPTY_EXTENT


BLOCKED_ON: str | None = (
    'Pre-registered 2026-05-19 BEFORE v3.1 sweep ran (v1 + v2 + v3 '
    'all scrapped pre-sweep — see docs/PRE_REGISTRATION_synthetic_bias_typeb.md '
    '+ /tmp/synthetic_v3_review.md). Awaiting ingest of '
    '`experiments/data/synthetic_bias_typeb_v3_1_pilot` (30 envs × 3 γ × '
    '2 arms × 8 seeds = 1440 cells). Predicted post-ingest '
    'EXPECTED=SUPPORTED iff P1 HELDs (ρ(payoff_spread, d_out) ≤ −0.5, '
    'p ≤ 0.05) AND N1 HELDs-as-null (|ρ(L, d_out)| < 0.30 at '
    'spread=0). D2 pre-registered as POWER_INSUFFICIENT (structural). '
    'REFUTATION fires if P1 NO_EFFECT-NULL at adequately-powered '
    'n_strata ≥ 15 — retracts the synthetic-substrate identification '
    'claim (see Finding docstring §REFUTATION criterion). Drift on '
    'any bridge fires DRIFT at hypothesis run-time.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_harms_under_high_spread__synthetic_typeb_v31,
    ddqn_harm_amplified_at_g999__synthetic_typeb_v31,
    n_states_alone_does_not_drive_dout__synthetic_typeb_v31,
)
