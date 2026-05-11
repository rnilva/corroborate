# PC adjacency discovery — SURVIVE cohort

**Cohort:** SURVIVE polarity (`env_reward_polarity > 0.3`) ∩
`q_div < 1.0` (Q-bounded) ∩ `jensen_dormancy_gap < 0.05`
(mech-active) ∩ standard config.
**n_cells:** 304 (Asterix-MinAtar 165 + Breakout-MinAtar 47 +
CartPole-v1 47 + SpaceInvaders-MinAtar 34 + PacMan-jumanji 11).
Freeway-MinAtar drops out (q_div > 1.0 in most in-scope cells).
Acrobot is REACH polarity. CartPole's |A|=2 gate isn't applied
on SURVIVE since G2's argmax-vulnerable test is for the REACH
mech HELD path, not SURVIVE's bounded-Q regime.
**Algorithm:** conservative-PC at depth ≤ 1, JCI by `env_name`, α=0.05.

## CPDAG

**Edges retained (adjacency layer — 10 of 21):**

```
  argmax_H — eff_h     argmax_H — is_ddqn
  outcome  — jens      outcome  — q_div
  is_ddqn  — jens      is_ddqn  — q_div     is_ddqn — stale
  jens     — q_div     jens     — stale     q_div   — stale
```

**Edges removed (notable separating sets):**

| edge | separating set Z |
|---|---|
| argmax_H ⊥ outcome  | {is_ddqn} |
| argmax_H ⊥ jens     | {stale} |
| argmax_H ⊥ q_div    | {stale} |
| argmax_H ⊥ stale    | {eff_h}, {jens}, {q_div} |
| eff_h ⊥ outcome     | {argmax_H}, {is_ddqn} |
| eff_h ⊥ is_ddqn     | ∅ |
| eff_h ⊥ jens        | {stale} |
| eff_h ⊥ q_div       | {stale} |
| eff_h ⊥ stale       | {jens}, {q_div} |
| outcome ⊥ is_ddqn   | ∅ (marginal indep at depth 0) |
| outcome ⊥ stale     | {jens}, {q_div} |

## Headline contrast — REACH vs SURVIVE

**REACH** (`{FourRooms, Acrobot, MountainCar, MetaMaze}`, n=346):
`is_ddqn` is **graph-disconnected** — every is_ddqn→· edge
removed at depth ≤ 1.

**SURVIVE** (this analysis, n=304):
`is_ddqn` is **connected to 4 mediators** (jens, q_div, argmax_H,
stale) but **NOT to outcome marginally** (`is_ddqn ⊥ outcome | ∅`).
This is the classic "mech HELD ↛ link HELD" pattern, but with
the mediator surface visible in the adjacency.

The structural difference:
- On REACH, treatment doesn't even move the mediators detectably
  (within JCI by env) — staleness invariant, jens shielded by
  q_div's algebraic identity. PC has no edge to orient.
- On SURVIVE Q-bounded scope, treatment moves jens AND q_div AND
  argmax_H AND staleness — all four mediator candidates wire up
  to is_ddqn. But the chain to outcome is blocked: `argmax_H ⊥
  outcome | {is_ddqn}` (argmaxH's correlation with outcome is
  fully mediated by treatment, not direct), and `is_ddqn ⊥
  outcome | ∅` (treatment's marginal effect on outcome is null).

This independently validates the three-gate framework
(`project_three_gate_proposal.md`): on SURVIVE Q-bounded scope
G1 is active (mediators move), but G2/G3 fail (outcome doesn't
respond at this cohort's scope).

## PC orientation caveat

The CPDAG's directed edges include several **physically incorrect
orientations** that PC's v-structure detection produces because
treatment / outcome aren't tagged as exogenous / terminal:

```
  jens     → is_ddqn       ← nonsense: treatment is exogenous
  q_div    → is_ddqn       ← nonsense
  stale    → is_ddqn       ← nonsense
  is_ddqn  → argmax_H      ← physically reasonable
  outcome  → jens          ← nonsense: outcome is downstream
  outcome  → q_div         ← nonsense
```

PC algorithms infer orientation from conditional-independence
asymmetries (v-structures). When `is_ddqn` appears as the
unshielded triple center between (jens, stale, q_div) — i.e.
`jens — is_ddqn — stale` with is_ddqn NOT in the separating set
of jens⫫stale — PC labels is_ddqn as a collider and orients
both edges INTO is_ddqn. This is the algorithm working correctly
on the data; the orientation is wrong only because PC doesn't
know is_ddqn is an intervention.

**Domain-knowledge override:** every is_ddqn edge should be
oriented as `is_ddqn → ·` (treatment is exogenous; nothing
causes it). Every outcome edge should be oriented as
`· → outcome` (outcome is the terminal target). With these
overrides:

```
  is_ddqn → jens
  is_ddqn → q_div
  is_ddqn → stale
  is_ddqn → argmax_H
  jens    → outcome
  q_div   → outcome
  eff_h   → argmax_H
  outcome ← stale   (already domain-correct in CPDAG)
  jens — q_div, jens — stale, q_div — stale  (mediator-cluster, ambiguous)
```

This is the structural-DAG read of the cohort: treatment moves
all four mediators; jens and q_div are both inputs to outcome;
staleness reaches outcome via {jens, q_div}; eff_h reaches
outcome via argmax_H.

## Methodology lesson

Conservative-PC at depth-1 with JCI by env is the right
**descriptive** tool but requires manual orientation for
intervention variables. Future iterations should either:
1. Tag `is_ddqn` as a context variable for the JCI framework
   (Mooij et al. 2020 §6 — `discover_adjacency` doesn't yet
   support this directly), OR
2. Run separately on (vanilla cells, DDQN cells) and report
   per-arm adjacency differences, OR
3. Manual orient-by-domain post-processing as done above.

## Sibling questions

- **Q-explosion-included sibling.** Same cohort minus the
  `q_div < 1.0` filter — recovers Asterix sync=100,
  SpaceInvaders 1M, where DDQN actively AMPLIFIES bias
  (memory `findings_q_amplification_cartpole.md`). Expect
  is_ddqn → jens edge with positive sign, mediator-cluster
  structure shifted.
- **Per-env PC.** Run on each SURVIVE env independently to
  resolve the mediator-cluster ambiguity (jens — q_div — stale
  triangle).

## Files

- `run_pc_survive.py` — discovery script.
- `pc_adjacency.json` — full output.
