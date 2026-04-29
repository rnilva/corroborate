# corroborate

Find the *scope* of a mechanism claim, then verify the *causal
chain* that explains it.

## What this is for

A mechanism claim is an authored algorithmic intervention plus
a theorem justifying its effect. Hasselt 2010's DDQN claim, for
instance: *swap argmax-Q-target-net for double-action-selection,
because action-selection / value-evaluation decoupling reduces
the Jensen-gap-induced overestimation bias.* The intervention is
the swap; the theorem names a *premise* (the gap is what makes
single-DQN biased).

Real performance of such claims is heterogeneous — DDQN helps in
some envs, hurts in others. The literature usually reports a
single unconditional verdict that obscures this. `corroborate`
makes the heterogeneity legible in two phases:

1. **Find scope.** Find the cleavage axis along which the
   mechanism's effect splits the corpus. The framework's
   preferred axis is the *invariance gap* — the residual of the
   theorem's premise, measured per env. Where the gap is large,
   the mechanism's target is salient; where the gap is small,
   the mechanism has nothing to grip on. Scope is empirical, not
   authored.

2. **Verify the causal chain.** Test each edge of the chain
   `env feature → invariance gap → mechanism activation →
   outcome` as a typed `ClaimedEdge` with a Pearl tier
   (associational / interventional) and a power-aware verdict
   (`HELD` / `NO_EFFECT` / `POWER_INSUFFICIENT` /
   `HELD_WITH_SCOPE_FLAG` / `INVARIANT_VIOLATION`).

The invariance gap is the load-bearing node — both the
scope-defining feature (Phase 1) and the causal mediator
(Phase 2). **Authoring an invariant per mechanism claim is the
substrate-author's primary commitment.**

## Three composable capabilities

The framework provides:

- **(a) Intervention study** — `apply_interventions(theory,
  arms)` literally re-runs the system with a swapped `@claim`.
  Active intervention, not observational reconstruction.
- **(b) Falsification** — power-aware verdict trichotomy with
  explicit MDE tracking. "Below MDE" is a first-class verdict,
  distinct from both confirmation and refutation.
- **(c) Causal discovery** — typed `CausalGraph` of
  `BridgeEdge`s with Pearl tier and direction.

These are substrate, not application. Phase 1 + Phase 2 compose
them; one-shot reproducibility audits and dialectic loops compose
them differently. All three reuse the same primitives.

## Status

Pre-v0. The acceptance test is a DDQN study reproducing the
§3 verdict pattern (`mechanism HELD ↛ outcome HELD ↛ link HELD`)
across 17 envs from `PAPER_NOTES.md`. v1 promotes that to
scope-finding on `vanilla_jensen_gap_mean` as the operational
invariance gap.
