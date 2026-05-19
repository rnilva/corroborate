"""Synthetic Type-A/B controlled-substrate pre-registered Finding (v3.2).

v1 → v2 → v3 → v3.1 → v3.2 evolution (all prior versions scrapped pre-sweep):

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
  Pre-launch review (`/tmp/synthetic_v3_1_review.md`) surfaced an
  attenuation problem at n_seeds=8 (per-stratum d-SE ≈ 0.51,
  reliability r ≈ 0.20, observed ρ for true −0.5 → −0.22 lands
  inside the |ρ|<0.20 REFUTATION band with ~30-43% probability).
- **v3.2** (current): three cheap fixes from the v3.1 review.
  (a) **n_seeds 8 → 16** — per-stratum d-SE → 0.36; reliability
  r → 0.32; false-REFUTATION rate at true ρ=−0.5 drops ~30-43%
  → ~15%. (b) **Drop L=32 envs** (recovers cell budget; L=1024 is
  where the FA-binding substantive claim lives; L=32 was a
  near-tabular baseline). (c) **REFUTATION null_threshold 0.20
  → 0.15** — matches the attenuation arithmetic at n_seeds=16.
  The N1 (L-axis null) bridge becomes structurally degenerate at
  single-L and is REMOVED from the Finding's BRIDGES tuple. The
  v3.2 sweep YAML carries the pre-registered set.

v3.2 panel has TWO structural axes + γ substrate:

- payoff_spread ∈ {0.0, 0.25, 0.5, 0.75, 1.0} (anisotropy on
  Q-target side)
- payoff_seed ∈ {0, 1, 2} (cross-realisation averaging)
- γ ∈ {0.95, 0.99, 0.999} (substrate axis)

15 envs × 3 γ × 2 arms × 16 seeds = 1440 cells (≤ 1500 budget).

The v3.2 bridges:

- **P1 (PRIMARY)**: ρ(payoff_spread, d_out) ≤ −0.5 pooled across γ.
  The load-bearing prediction; n_strata=45 (15 envs × 3 γ) ≥
  min_strata=10. REFUTATION null_threshold tightened to 0.15.
- **D2 (DIAGNOSTIC, PRE-REGISTERED POWER_INSUFFICIENT)**:
  ρ(payoff_spread, d_out) ≤ −0.6 at γ=0.999 sub-scope.
  n_strata=15 sits AT the helper's min_strata=15 boundary →
  STRUCTURAL POWER_INSUFFICIENT.

**v3.1 → v3.2 bridge surgery**: N1
(`n_states_alone_does_not_drive_dout__synthetic_typeb_v31`) was
DROPPED in v3.2. With L=1024 the only registered L value, the
L-as-Spearman-covariate has a single value at spread=0 → ρ is
undefined. The L-as-modulator question is structurally
unaddressable at the v3.2 panel; a future v4 with a richer L
axis would restore it.

## Substantive scope (v3.2 narrowing)

v3.2 tests Q-target-side anisotropy → DDQN harm in the
**FA-residual-heavy regime (σ/Δ ≈ 47%)**, NOT the Asterix 1-3%
knife-edge regime. The v3.1 reviewer verified via VI that σ_clip
and Δ_v both scale linearly with `payoff_spread`, so σ/Δ stays
ratio-invariant at ~0.466 across all spread > 0. A HELD verdict
supports the mechanism shape at FA-heavy σ/Δ; it does NOT
establish that Asterix's knife-edge mechanism is the same. The
σ/Δ-knife-edge regime test remains open and would require a v4
with decoupled σ_clip / Δ_v knobs.

## Expected verdict pre-sweep

EXPECTED = EMPTY_EXTENT until the sweep ingests. The Finding fires
UNDERPOWERED [blocked] until then; framework's DRIFT detection
fires when the post-ingest verdict deviates from this pin.

## REFUTATION criterion (load-bearing pre-registered retraction)

The v2 critic noted: "the walk-back paths accept every observed-
data shape as publishable — that's not commitment, it's a flowchart
of post-hoc framings." v3.2 inherits v3.1's binding commitment to
a SPECIFIC observed-data shape that retracts the substrate-
identification claim, with the threshold tightened to match the
v3.2 attenuation arithmetic:

**If P1 fires NO_EFFECT-NULL (|ρ| < 0.15) with n_strata ≥ 15
admitted** (an adequately-powered null at the primary covariate),
the v3.2 synthetic substrate FAILS to reproduce the natural-env
Asterix Type-B mechanism AT THE FA-RESIDUAL-HEAVY σ/Δ ≈ 47%
REGIME v3.2 tests. This is a RETRACTION, not a walk-back:
the substrate-author cannot claim "synthetic substrate enables
causal env-feature identification of the Asterix harm regime at
this σ/Δ regime." The Finding's EXPECTED would be repinned to
NO_EFFECT-NULL with BLOCKED_ON=None, and the substrate paper would
carry the retraction as a methodology-demonstration finding ("this
attempt fails because Q-target-side anisotropy alone, in a chain
MDP at σ/Δ ≈ 47%, does not reproduce Asterix's mechanism").

The REFUTATION criterion's threshold n_strata=15 is binding under
v3.2's 45-stratum primary panel: cell-budget-induced dropout would
have to exceed 67% before the criterion becomes vacuous.

## Diagnostic predictions (disambiguation surface)

- **P1 HELD + D2 ρ stronger than pooled**: γ-amplification
  confirmed; the chain-amplification story carries.
- **P1 HELD + D2 ρ NULL**: γ doesn't modulate the spread effect in
  synthetic; chain amplification is natural-env-specific.
- **P1 SIGN_FLIP**: spread has opposite mechanism in synthetic at
  the FA-residual-heavy σ/Δ regime — substantive walk-back of the
  σ_clip → argmax-corruption mechanism.
- **P1 NULL (n≥15, |ρ|<0.15)**: REFUTATION — the v3.2 substrate
  fails to reproduce the natural-env mechanism at σ/Δ ≈ 47%.

## Critic recommendations addressed (v3.2)

Inherits v3.1's structural fixes + adds three power / threshold
fixes from the v3.1 review:

1. **μ_best ≈ 1**: peak_value=1.0 (v3.1+). |Q*| at γ=0.999 is
   ≤ 1000, matching natural-env Asterix Q≈436.
2. **σ/Δ regime**: noise_sigma=0.02·peak_value. VI confirms σ_clip
   and Δ_v both scale linearly with payoff_spread; σ/Δ ≈ 0.466
   ratio-invariant across all spread > 0. v3.2 explicitly scopes
   the substantive claim to this FA-residual-heavy regime (NOT
   Asterix 1-3% knife-edge).
3. **L = 1024 with hidden = 16**: v3.2 narrowed (L=32 dropped).
   Q* has ~1024 distinct entries at L=1024 (verified by VI), so
   16-dim hidden GENUINELY aliases 1024 distinct V*-values → real
   FA-binding (v3 had only 16 distinct Q* entries → 16-dim hidden
   trivially represented them).
4. **n_seeds**: 8 → 16 in v3.2. Per-stratum d-SE 0.51 → 0.36;
   reliability 0.20 → 0.32; attenuation factor 0.44 → 0.57; false-
   REFUTATION rate at true ρ=−0.5 drops 30-43% → ~15%. Still
   below the critic's n=30 target, but the cell budget binds and
   v3.2 substitutes env-axis ensembling (15 envs across 5 spreads
   × 3 payoff_seeds at L=1024) for additional seed replication.
5. **REFUTATION threshold**: 0.20 → 0.15 in v3.2. Matches the
   attenuation arithmetic; keeps the REFUTATION clause's binding
   nature while dropping the false-REFUTATION rate to ~15%.
6. **Anisotropy primitive on Q-target side, NON-PERIODIC**:
   FULLY ADDRESSED in v3.1 (preserved in v3.2).
   `mu_state[s] = peak · (1 - spread + spread · U_s)` with
   `U_s ~ U(0, 1)`. Verified by value iteration:
   `Var_a[V*(s'_a)] > 0` at every spread > 0 (v3 had this = 0);
   Q* has ~L distinct entries at L=1024 (v3 had only 16).

## Acknowledged limit (NOT fixed in v3.2)

- **σ/Δ ≈ 47% is far from Asterix 1-3% knife-edge**. The v3.2
  env's payoff_spread knob moves σ_clip and Δ_v in lockstep
  (both ∝ spread), so σ/Δ is ratio-invariant at ~0.466. The
  substantive claim is narrowed to "FA-residual-heavy regime"
  accordingly. A future v4 with decoupled σ_clip / Δ_v knobs
  would test the knife-edge regime.
- **lr=3e-4 × γ=0.999 confound**. At γ=0.999, the Bellman
  residual scale is ~1000× peak; lr=3e-4 fixed means gradients
  are ~1000× weaker relative to Q-scale than at γ=0.95. The
  L=1024 × γ=0.999 cells will likely be partially-trained
  rather than purely FA-bound. This is the v2 critic's
  lr-confound carried forward; v3.2 does not address it. The
  D2 γ-amplification diagnostic should be read with this in
  mind (it tests γ-effect at fixed lr, not γ-effect at
  optimization-equilibrated lr).
- **payoff_spread confounds problem difficulty**. At spread=1,
  the optimal policy must navigate a heterogeneous reward
  landscape; at spread=0, every state is identical and reward
  is policy-independent. d_out at spread=0 is expected ≈ 0 by
  construction. The PRIMARY P1 bridge cannot fully isolate
  "Var_a[V*] effect" from "optimization-difficulty effect" —
  the v3.1 review flagged this as a real interpretation gap.

## Companion docs

- `docs/PRE_REGISTRATION_synthetic_bias_typeb.md` — predictions
  + v1/v2/v3/v3.1/v3.2 evolution + REFUTATION clause.
- `/tmp/synthetic_v3_1_review.md` — the v3.1 pre-launch review
  that motivated the v3.2 fixes.
- `/tmp/synthetic_v3_review.md` — the v3 review (structural flaws
  v3.1 addressed).
- `/tmp/synthetic_v2_roast.md` — the v2 review.
- `/tmp/synthetic_env_roast.md` — the v1 review.
- `src/corroborate_rl/corroborate_rl/synthetic_bias_typeb.py` —
  the v3.1+ env module (env definition unchanged in v3.2; only
  the registered subset narrows).
- `experiments/configs/synthetic_bias_typeb_v3_2_sweep.yaml` — the
  v3.2 sweep config.
"""
from __future__ import annotations

from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict

from experiments.findings.ddqn_sweeps.synthetic_bias_typeb import (
    ddqn_harm_amplified_at_g999__synthetic_typeb_v32,
    ddqn_harms_under_high_spread__synthetic_typeb_v32,
)


# Pre-registration: EXPECTED = EMPTY_EXTENT until sweep ingests.
# Both bridges fire POWER_INSUFFICIENT (n_strata=0) without data.
# The framework's DRIFT detection fires when the post-ingest
# composed verdict deviates from this pin.
EXPECTED: ClusterVerdict = ClusterVerdict.EMPTY_EXTENT


BLOCKED_ON: str | None = (
    'Pre-registered 2026-05-19 BEFORE v3.2 sweep ran (v1 + v2 + v3 '
    'all scrapped pre-sweep; v3.1 superseded pre-launch per '
    '/tmp/synthetic_v3_1_review.md — see '
    'docs/PRE_REGISTRATION_synthetic_bias_typeb.md). Awaiting '
    'ingest of `experiments/data/synthetic_bias_typeb_v3_2_pilot` '
    '(15 envs × 3 γ × 2 arms × 16 seeds = 1440 cells). Predicted '
    'post-ingest EXPECTED=SUPPORTED iff P1 HELDs '
    '(ρ(payoff_spread, d_out) ≤ −0.5, p ≤ 0.05) at the '
    'FA-residual-heavy σ/Δ ≈ 47% scope v3.2 tests. D2 pre-registered '
    'as POWER_INSUFFICIENT (structural at n_strata=15). REFUTATION '
    'fires if P1 NO_EFFECT-NULL (|ρ| < 0.15, tightened from v3.1''s '
    '0.20) at adequately-powered n_strata ≥ 15 — retracts the '
    'synthetic-substrate identification claim at the FA-residual-heavy '
    'σ/Δ regime (see Finding docstring §REFUTATION criterion). '
    'Drift on any bridge fires DRIFT at hypothesis run-time.'
)


BRIDGES: tuple[Bridge, ...] = (
    ddqn_harms_under_high_spread__synthetic_typeb_v32,
    ddqn_harm_amplified_at_g999__synthetic_typeb_v32,
)
