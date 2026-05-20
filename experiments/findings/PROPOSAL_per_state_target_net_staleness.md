# Proposal: Per-state target-net staleness as a loop-targeted bias-chain interruption

**Status**: Draft proposal, pre-registration target. 2026-05-20.

## 1. Empirical setup

The framework's analysis of DDQN's mechanism at γ→1 sparse-reward
identified two robust facts at SI γ=0.999 (n=60, both arms, n=30
seeds per arm):

1. **Bias reduction does not mediate outcome.** The standard story —
   "DDQN reduces overestimation bias → outcome rescue" — is refuted
   by PC discovery + partial-Spearman mediation analysis. `jensen_gap`
   absorbs only 8% of arm→outcome (shrink from |marginal ρ|=0.80
   to 0.733 with p=2.6e-12). PC removes the arm→outcome edge under
   separating sets that DON'T include the bias measure. The
   per-state-conditional version of this analysis corroborates the
   pattern at FR γ=0.999 in a separate Finding
   (`finding_jens_does_not_mediate_outcome_at_fr_g999_mlp`).

2. **Within-episode policy looping IS the mediator.** A measurable
   that counts the fraction of late-window training steps whose
   state-hash also appears in the trailing 64-step window
   (`state_repeat_rate_within_episode_window64_late`, episode-boundary
   artifact controlled) absorbs **84% of arm→outcome** as a single
   mediator (shrink to 0.16 of marginal ρ, p=0.347 NS — conditional
   independence achieved). The arm difference is large: vanilla 0.845
   → DDQN 0.789, Cohen's d = -3.46, p = 2.1e-15.

## 2. Mechanism characterization

The proposed underlying mechanism (consistent with Lemma 2 in
`THEORY_bootstrap_dominance.md`):

At γ→1 sparse-reward, vanilla's bootstrap target
`r + γ · max_a Q(s', a)` is dominated by the chain term `γ max Q`
because `r ≈ 0`. When state `s` is revisited K times within an
episode, its `Q(s, ·)` accumulates K rounds of max-of-K positive
bias amplification. The positive amplification at `s` makes the
greedy action there increasingly attractive, biasing the policy
to return to `s` again — closing a positive-feedback trajectory loop.

The lit-review found NO published work that operationalizes this
specific mechanism (within-episode bias-chain amplification on
revisited states in online value-based RL). The closest
adjacents:
- **Lipton 2016 "Sisyphean Curse"**: across-episode revisits via
  replay forgetting (different mechanism).
- **Yang 2023 "Q-Value Divergence in Offline RL"**: names
  "self-excitation" but offline scope.
- **Wang 2022 "Does DQN Learn?"**: proves attractors exist
  theoretically but doesn't measure trajectory loops.
- **Schaul 2022 "Policy Churn"**: sibling axis (between-snapshot
  temporal, not within-rollout spatial).

DDQN's clip (decouple argmax-selection from value-evaluation) is
known to reduce overestimation bias in aggregate. The framework's
mediation analysis shows this aggregate reduction does NOT carry
the outcome rescue at γ→1 sparse-reward; per-state cycle
interruption does.

## 3. Proposed mechanism

**Per-state target-net staleness (PS-stale)**: at loop-revisited
states, compute the bootstrap target using a more-stale snapshot
of the target net. Formally:

```
target = r + γ · max_a Q(s', a; θ_lookup(s'))

  θ_lookup(s') = θ_stale_old   if  recency(s') > threshold
               = θ_target      otherwise

  recency(s') = count of times s' was visited in the last W
                training steps (e.g., W = 64)
```

where:
- `θ_target` is the standard target net synced every K steps
  (canonical: K=100).
- `θ_stale_old` is an EVEN MORE stale snapshot — a copy of
  `θ_target` from M target-sync events ago (e.g., M=10 → 10K
  training steps stale).
- `threshold` is the loop-detection threshold (e.g., 3 visits in
  64 steps).

**Conceptually**: at loop-states, the bootstrap pulls the Q
estimate toward an OLDER snapshot that hasn't accumulated as much
bias from the recent loop iterations. The per-state cycle
amplification is broken without disturbing non-loop state-action
pairs.

**Mathematical intuition**: Lemma 2's chain `Q_{t+1} ←
γ · max Q_t` compounds bias when `max Q_t` is itself bias-inflated.
At loop states where bias has already accumulated `K` rounds via
recent revisits, using a `θ_stale_old` snapshot from BEFORE those
K rounds breaks the bias propagation chain. The local fixed-point
at the loop attractor changes from `γb/(1−γ)` (Lemma 2 asymptote)
to `γb · (1 − γ^K)/(1 − γ)` (truncated to K rounds), which is
substantially lower at γ→1.

## 4. Distinctness from prior work

| | What it modifies | Per-state? | Targets the loop mechanism? |
|---|---|---|---|
| Count-based exploration (Strehl 2008, Bellemare 2016) | reward / policy | partially (bonus depends on N(s)) | indirectly (modifies behavior, not Q update) |
| RND / curiosity bonuses | reward | partially | indirectly |
| Polyak / soft target updates | target net (all states) | no | uniformly, not loop-specific |
| DDQN (Hasselt 2010/2016) | argmax computation (all states) | no | aggregate bias reduction, not per-state |
| Maxmin / TQC | aggregation over multiple nets | no | uniform variance reduction |
| **PS-stale (proposed)** | **target net (selectively at loop states)** | **yes** | **directly — per-state cycle interruption** |

The distinguishing axis: PS-stale modifies the Q UPDATE RULE
(specifically the bootstrap target computation) in a STATE-CONDITIONAL
way, targeted at the loop signature. No prior method has this
combination of (a) modifying Q updates not behavior, AND
(b) state-conditional targeting of loop trajectories.

## 5. Implementation sketch

Substrate change in `corroborate_rl.dqn.claims.bootstrap`:

```python
@claim
def bootstrap_per_state_stale(
    online_params: Params,
    target_params: Params,
    stale_old_params: Params,
    obs: jax.Array,
    next_obs: jax.Array,
    action: jax.Array,
    reward: jax.Array,
    done: jax.Array,
    gamma: float,
    state_hash_window: jax.Array,  # recent (window, ) state hashes
    state_hash_now: jax.Array,     # current hash
    *,
    greedification: GreedificationFn = greedify,
    loop_threshold: int = 3,
    q_function: QFunction = mlp_q,
) -> jax.Array:
    """Per-state target-net staleness: at high-recency states,
    use stale_old_params for bootstrap target evaluation; else
    use target_params (canonical DDQN-style target net).

    state_hash_window: rolling buffer of last W state hashes in
    current episode (zero-padded at episode start).
    """
    recency_count = jnp.sum(state_hash_window == state_hash_now)
    is_loop = recency_count > loop_threshold
    # Compute both target-net Q evaluations
    q_target = q_function(target_params, next_obs)       # (n_actions,)
    q_stale_old = q_function(stale_old_params, next_obs) # (n_actions,)
    # Select based on loop status
    q_for_target = jnp.where(is_loop, q_stale_old, q_target)
    a_star = greedification(online_params, next_obs, q_function)
    bootstrap_value = q_for_target[a_star]
    return reward + gamma * (1.0 - done) * bootstrap_value
```

State management:
- Maintain `state_hash_window` as a (W,) circular buffer per cell
- Maintain `stale_old_params` as a SECOND target net, synced
  with period M × K (M=10, K=100 → stale_old syncs every 1000
  training steps)

Plug-and-play with existing `DoEffect` intervention infrastructure:
```python
Intervention(
    slot_path='bootstrap',
    replacement=partial(bootstrap_per_state_stale,
                        loop_threshold=3,
                        stale_period_mult=10),
)
```

## 6. Pre-registered test design

**Four-arm `DoEffect`** at SI γ=0.999 × MLP × B=32 × 1M, n=30
seeds per arm:

| arm | name | base | replacement(s) |
|---|---|---|---|
| A | baseline (vanilla) | `bootstrap` | (none) |
| B | DDQN canonical | `bootstrap` | `greedification=double_greedify` |
| C | PS-stale alone | `bootstrap` | `bootstrap_per_state_stale` |
| D | PS-stale + DDQN | `bootstrap` | `bootstrap_per_state_stale` + `greedification=double_greedify` |

**Pre-registered predictions** (committed in bridge source-hash
BEFORE materialization):

1. **PS-stale REDUCES within-episode loops more than DDQN**:
   `Δ_C = D_arm_C(repeat_rate_within_episode) − D_arm_A(...)`
   should be more negative than `Δ_B = D_arm_B(...) − D_arm_A(...)`.
   Predicted direction: `Δ_C < Δ_B`. d_floor on the contrast: -0.5.

2. **PS-stale OUTCOME at SI γ=0.999 is comparable to DDQN**:
   `Δ_C(outcome) ≥ 0.7 · Δ_B(outcome)`. We don't predict PS-stale
   beats DDQN — they target different facets of the bias-chain.

3. **PS-stale × DDQN is approximately ADDITIVE on outcome**:
   `Δ_D(outcome) ≈ Δ_B(outcome) + Δ_C(outcome) − overlap_term`.
   If interventions act on the SAME mechanism (cycle interruption),
   the combination should be near-redundant (`Δ_D ≈ max(Δ_B, Δ_C)`).
   If they act on DIFFERENT facets, the combination should be near-
   additive. Predicted: PARTIAL additivity (one of them already
   addresses the per-state loop; small additional gain expected).

**Cross-env replication tests**:

- **FR γ=0.999** (loop-allowing env): replicate the four-arm test
  on FR γ=0.999 (currently sweeping for the diagnostic baseline).
  Predicted: same direction of all three predictions.
- **Acrobot γ=0.999** (loop-free env, physics-momentum):
  Predicted: PS-stale gives NO BENEFIT over vanilla (no loops to
  break). This is the negative-control.

If PS-stale gives benefit at Acrobot, the mechanism story is
WRONG (the intervention is doing something we don't understand).

## 7. Falsifiability conditions

PS-stale's mechanism story is REFUTED if any of the following
materialize:

1. PS-stale does NOT reduce within-episode revisit rate at SI
   γ=0.999 (intervention doesn't bite at the named mechanism).
2. PS-stale improves outcome at Acrobot γ=0.999 (it's doing
   something other than what we said).
3. PS-stale's benefit at SI γ=0.999 is FULLY redundant with
   DDQN's (combo = max, no additive gain) — implies PS-stale
   isn't distinctly addressing per-state cycle interruption.
4. The repeat-rate mediation strength at SI γ=0.999 with the
   PS-stale arm DOESN'T shift (84% → comparable). Should drop
   if the mechanism is being interrupted.

## 8. Limitations & open questions

1. **State_hash dependence**: PS-stale depends on a non-degenerate
   state hash. Envs without a registered hash (currently MetaMaze,
   Catch-bsuite, etc.) cannot use this mechanism. The
   `_FOURROOMS_HASH` was registered specifically for this purpose
   (commit `42ead69`).

2. **`stale_period_mult` and `loop_threshold` hyperparameters**:
   not theoretically derived. Initial values from intuition
   (M=10, threshold=3). A sweep over these would be needed to
   characterize sensitivity.

3. **Memory/compute cost**: maintains an additional target-net
   snapshot. Roughly 2× the target-net memory of DDQN. Negligible
   relative to total training.

4. **Asterix exception**: PS-stale's prediction at Asterix γ=0.999
   is UNCLEAR. Asterix is loop-allowing but vanilla benefits from
   Q-magnitude anisotropy (per
   `findings_asterix_breakout_channel_asymmetry_g999`). PS-stale
   should reduce the anisotropy → outcome could go either way.
   This is a discriminator test: if PS-stale HARMS Asterix similar
   to DDQN, that corroborates "the harm is via per-state Q
   interruption." If PS-stale doesn't harm Asterix while DDQN
   does, the Asterix harm has a different mechanism than the SI
   rescue.

5. **What if the empirical FR sweep doesn't replicate the SI
   pattern**? Then PS-stale's grounding weakens (SI-only
   evidence). Should wait for the FR sweep (currently running,
   PID 2370678) to confirm cross-env transfer before committing
   to the implementation.

## 9. Order of operations

1. **Wait for FR γ=0.999 sweep to finish** (PID 2370678) — confirms
   loop hypothesis transfers cross-env. If yes, proceed.
2. **Implement `bootstrap_per_state_stale`** in
   `corroborate_rl.dqn.claims.bootstrap`. Adds state-hash-window
   tracking + second target-net snapshot.
3. **Author bridges** in `experiments/findings/ddqn_three_conditions/`:
   pre-registered direction + verdict matrix.
4. **Run sweep**: 4-arm × n=30 × 1M at SI + FR γ=0.999
   (n=240 cells total, several GPU hours).
5. **Materialize verdicts**. Drift detector catches deviations
   from pre-registration.
6. **Cross-env negative-control**: Acrobot γ=0.999 with PS-stale,
   predicted NO_EFFECT.

If the FR sweep currently running shows NO loop signal (loop
hypothesis fails to transfer), this proposal is shelved pending
re-evaluation of the SI-only mechanism story.
