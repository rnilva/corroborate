# DDQN Universe Bridges — Findings Summary

The framework's three-verdict architecture (mechanism / link / outcome) applied
to a 17-env corpus + ~10 designed intervention sweeps. Bridges authored in
`experiments/findings/ddqn/` are the typed, corroborable form of each finding below.

## Mechanism: bias correction (CLAIMs 1 & 2)

`do(arm=DDQN) ↓ jensen_gap` is broadly corroborated. The dormancy bridge
(`ddqn_refuted_when_dormancy_fires`) cleanly identifies the necessary-condition
scope: when `jensen_dormancy_gap > 0`, the Hasselt premise is inactive and
DDQN's mechanism cannot fire. Cross-validated on Asterix sync=10k (100% cells
dormant), Catch (saturated), CartPole sync=10k.

## Link: bias-correction → outcome (CLAIMs 4–10)

Genuinely env-conditional. The link is prominent in a band on
**(chain-depth × Q-stability)**:

| env / regime | link verdict | evidence |
|---|---|---|
| FourRooms-misc | HELD clean | per-burst r=-0.94, β=-1.0, dormancy never fires |
| Acrobot-v1 γ=0.999 | HELD per-burst (CLAIM 10) | 4 bridges all HELD: ATE=-0.63, placebo=0, RCC drift=0, plc=1.0 |
| MetaMaze-misc γ=0.999 | HELD (CLAIM 5b) | chain-depth amplifier engages at high γ |
| DiscountingChain-bsuite γ-cond | HELD (CLAIM 5c) | bootstrap-heavy regime |
| SpaceInvaders 1M | PHASE 1 only (CLAIM 8) | early bursts active; late Q-explosion crossover |
| MinAtar 1M sync=100 (Asterix/Breakout/Freeway) | dies after Phase 1 | β collapses as Q diverges |
| CartPole-v1 | dormant or saturated | no clean regime in our sweeps |

**Bootstrap fraction is the load-bearing link-side scope feature** —
`bootstrap_fraction_drives_g_link__net_of_dormancy` (CLAIM 4) shows it predicts
g_link strength after partialling out dormancy.

**CLAIM 11: extreme Q-divergence attenuates the link** (companion to dormancy):
when `q_divergence_score = vanilla_jens / (r_max / (1-γ)) > 1000`, g_link
attenuates significantly. DoWhy backdoor ATE = -0.21 (binary above-1000 vs
band, n=13), placebo refutation passes (ratio 0%), RCC drift = 0.012. Pearl-
rung-2 corroboration via Asterix sync × training-length: sync=1000/100k gives
q_div=0.02 with g_link=+0.21; sync=100/1M gives q_div=17000 with g_link=-0.23.

Together with CLAIM 2 (dormancy), CLAIM 11 bounds the link-active band:
**0.02 < q_divergence_score < 1000** is where DDQN's link operates. Below:
mech inactive (dormancy). Above: Q-explosion overwhelms link translation.

**n-step falsification (CLAIM 9)** corroborates necessity: Δ → 0 monotonically
as bootstrap dependence shrinks (n=1: Δ=+0.087, p=3e-4; n=10: ~0). Negative-
prediction test that confounds rarely produce; theorem's necessary condition
holds; rules out alternative mechanisms.

## Under-learning rescue (CLAIM 7 + 7b)

Pearl-rung-2 dominance at low reward scale on FourRooms:

- rs=0.1: DDQN's response curve dominates vanilla's (+0.49 native outcome)
- rs=0.3: same pattern (CLAIM 7b)
- JCI confirms: `vanilla_native ⊥ ddqn_native | log_rs` — two independent
  reward-scale-response curves, not a causal arrow between arm outputs.

**Continuous-α intervention** (`dampened_alpha_envs`, 5α × 4 envs × 30 seeds × rs=0.1):
- FourRooms reproduces the full chain — outcome slope +0.49, jensen slope -0.43
  across α ∈ {0, 0.25, 0.5, 0.75, 1.0}, monotonic 8× change
- DeepSea / DiscountingChain / MNISTBandit don't fire at rs=0.1 — they're not
  in the rescue regime
- **Cross-env scope is "env in under-learning band", not `|A|`**

The α sweep validates the rescue mechanism interventionally: as α grows from 0
(vanilla) to 1 (DDQN), outcome rises and jensen falls in lockstep.

## What we've REFUTED

- **`|A| ∈ {3, 4}` upper bound** (cross-env artifact): `action_duplicate`
  intervention on FourRooms shows DDQN benefit grows monotonically 6× from
  |A|=4 to |A|=16. Hasselt floor is monotonically positive in |A|, no upper
  boundary intrinsic to the algorithm.
- **mc_variance as link attenuator** (CLAIM 6 SHADOW): refuted via CV
  decomposition (β=+0.02, p=0.46 with `log_mc_cv`) — the variance-scaled
  signal was a reward-magnitude confound. CLAIM 7 (under-learning rescue) is
  the surviving causal story.
- **L2 regularization recovers Acrobot Goldilocks**: weight_decay ∈ {1e-4, 1e-3}
  on Adam doesn't break the chain-depth-amplifier ceiling at γ=0.999. The
  ceiling is intrinsic, not a γ-as-regularizer artifact.

## Methodological framework (canonical)

- **Per-burst link analysis** (`paired_link_per_burst` + `phase_link_consistency`)
  is canonical for Q-non-monotonic envs. Scalar mech-link slopes silently
  combine causally opposite phases (e.g., SpaceInvaders Phase 1 link active vs
  Phase 2 reversed → scalar averages to ~0).
- **Hidden mediator audit**: when adjusting for candidate mediators changes the
  inferred causal channel, the original observational correlation is suspect.
  CartPole's link signal at sync=1000 has multiple correlated channels
  (q_std, q_mag, q_cv); q_cv is 97% predicted by q_std × q_mag at that regime,
  so they're not independent channels.
- **Conditioning rule**: link analyses must condition on `mech HELD`. Mech-
  dormant cells (Q underestimating) get UNTESTABLE, not NULL — the framework
  refuses to collapse.
- **`max(0, …)` clamp on `jensen_gap`** hides per-burst link signal in
  underestimation regimes; pair with `jensen_dormancy_gap` invariant to
  distinguish true zero-bias from negative-bias dormancy.

## What's genuinely OPEN

1. **MinAtar at sync=10k × 1M** (running). If Q-stabilization rescues Phase 2/3
   link, the FourRooms pattern generalizes. If not, dense-MinAtar is a separate
   failure mode requiring its own scope claim.
2. **Single-variable scope predicate for link prominence**: `bootstrap_fraction
   × effective_horizon` is on the right axis (chain-amp regime) but doesn't
   predict link strength quantitatively across all envs. Multi-feature
   conjunction may be irreducible (see `findings_ddqn_scope_synthesis.md`).
3. **CartPole's small-effect regime**: at sync=1000, 57% cells are dormant, 43%
   mech-active. Even mech-active cells show q_mag-channel hurt instead of
   bias-correction help. Needs new measurables (gradient stability, network
   curvature, exploration coverage) to identify the missing channel.

## Bottom line

DDQN's **mechanism** (bias correction) is broadly active and theoretically
supported (Hasselt 2010); the framework's dormancy invariant cleanly bounds
the necessary scope.

DDQN's **link** (mechanism → outcome) operates in a regime band: enough chain
depth for bias to compound × enough Q-stability for the correction to translate.
FourRooms is the cleanest case; Acrobot at γ=0.999 (per-burst), MetaMaze at
high γ, DiscountingChain in chain-amp regime, and SpaceInvaders Phase 1 give
multi-env corroboration. CartPole and dense-MinAtar are open — their link
verdicts depend on whether the chain-depth × Q-stability band engages, which
in turn depends on HP regime (sync_period, γ, reward_scale).

The framework's contribution is keeping the mech / link / outcome verdicts
separate. Treating an underpowered or regime-dependent link as either NULL or
HELD smuggles the regime question past the reader. The bridges enumerated above
are the typed, regime-aware claims that survive interventional corroboration.
