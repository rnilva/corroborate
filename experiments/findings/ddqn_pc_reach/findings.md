# PC adjacency discovery — REACH cohort

**Cohort:** REACH envs `{FourRooms-misc, Acrobot-v1, MountainCar-v0,
MetaMaze-misc}` ∩ `_DDQN_RELEVANT_SCOPE` (G1 ∧ G2, standard config).
**n_cells:** 346 (Acrobot 107 + FourRooms 68 + MetaMaze 111 + MountainCar 60).
**Algorithm:** conservative-PC (Ramsey-Zhang-Spirtes 2006) at depth ≤ 1.
**CI test:** stratified partial Spearman ρ, JCI by `env_name`, Fisher-z pool.
**α:** 0.05.

**Variables:** `is_ddqn` (binary treatment indicator), `jens` (jensen_gap),
`q_div` (q_divergence_score), `argmax_H` (argmax_entropy_late), `stale`
(target_staleness_late), `eff_h` (effective_horizon), `outcome`
(eval_best_burst_mean).

## CPDAG

**Directed edges (9):**

```
  jens     → outcome
  jens     → q_div
  q_div    → outcome
  eff_h    → q_div
  eff_h    → argmax_H
  argmax_H → outcome
  stale    → outcome
  stale    → argmax_H
  stale    → q_div
```

**Undirected (1):** `eff_h — jens` (direction not determinable from
conditional independence alone).

**Ambiguous triples (5):** all around `eff_h` — its role as collider
vs. non-collider in (argmax_H, jens, q_div, outcome) chains is
undetermined at depth-1 conservative PC.

## 2026-05-11 reviewer correction

The original headline "is_ddqn graph-disconnected on REACH" was
OVERSOLD. Two of the six removed `is_ddqn`-edges follow
mechanically from the algebraic identity `q_div = jens/(R/(1−γ))`
(within-γ ρ(qdiv, jens) = +0.974 per `tautology_audit.json`).
Re-running PC with qdiv excluded from the variable set
(`run_pc_reach_no_qdiv.py`, `pc_adjacency_no_qdiv.json`)
recovers a clean treatment edge:

**Edges retained (6 of 15 when qdiv excluded):**
- **`is_ddqn — jens`** (the key empirical treatment edge,
  hidden by the qdiv algebraic shadow in the original run)
- `eff_h — jens`
- `outcome — jens`, `outcome — stale`
- `argmax_H — eff_h`, `argmax_H — stale`

**Edges still removed:**
- `is_ddqn ⊥ outcome | ∅` — link null on REACH preserved
- `is_ddqn ⊥ stale | ∅` — staleness still treatment-invariant
- `is_ddqn ⊥ {eff_h, argmax_H} | {jens}` — these channels are
  downstream of jens, not direct treatment effects

**CPDAG (qdiv excluded, manual orientation: is_ddqn → · always,
· → outcome always):**

```
is_ddqn ──→ jens ──→ outcome
              │
              ↓
            eff_h ──→ argmax_H
                       ↑
            stale ─────┘
            stale ──→ outcome
```

**Revised headline:** treatment acts cleanly on jensen_gap; jens
mediates to outcome and to eff_h/argmax_H; staleness is a
separate non-treatment-modulated channel that also reaches
outcome. The "link null on REACH" finding (`is_ddqn ⊥ outcome | ∅`
at depth 0) is preserved — it's the marginal independence
itself, not a graph-disconnection finding.

## Headline finding (legacy, qdiv-included run) — `is_ddqn` is graph-disconnected

PC removes every `is_ddqn → ·` edge:

| edge | separating set Z |
|---|---|
| is_ddqn ⊥ outcome  | ∅ (marginal indep) |
| is_ddqn ⊥ stale    | ∅ |
| is_ddqn ⊥ argmax_H | {outcome}, {q_div}, {eff_h}, {jens} (4 sepsets) |
| is_ddqn ⊥ eff_h    | {q_div}, {argmax_H}, {jens} |
| is_ddqn ⊥ jens     | {q_div} |
| is_ddqn ⊥ q_div    | {jens} |

`is_ddqn ⊥ outcome | ∅` at depth 0 — **DDQN's marginal effect on
outcome is null on REACH within env-stratification.** Cross-env
paired-Δ analyses (memory `findings_ddqn_convergence.md`,
`findings_action_dim_sweep.md`) reach the same conclusion: link
null on most REACH envs; the bias-reduction-to-outcome arrow
breaks. PC's depth-1 conservative test recovers this without
the paired-Δ machinery.

`is_ddqn ⊥ jens | {q_div}` is the algebraic shadow at work:
`q_div = jens / (R / (1−γ))` is structurally tied to jens, so
conditioning on one removes the marginal information of the
other. PC can't disentangle which of jens/qdiv is "the cause"
of treatment's effect — they're collinear within (env, γ).

**Quantified via `tautology_audit`** (companion script
`run_tautology_audit.py`, output `tautology_audit.json`):
within-γ Spearman ρ on the same REACH cohort (n=343):

| candidate           | ρ vs jens (within γ) | HP-R² on γ | ρ vs outcome (within γ) |
|---|---|---|---|
| q_divergence_score  | **+0.974** (p≈0)     | 0.054      | −0.825                  |
| effective_horizon   | +0.778 (p≈0)         | **0.918**  | −0.137                  |
| argmax_entropy_late | +0.275 (p=2e-7)      | 0.323      | +0.289                  |
| target_staleness    | −0.246 (p=4e-6)      | 0.162      | +0.763                  |

The audit's reads-jaccard structural check returned 0 for every
candidate (qdiv's registered reads name `'jensen_gap'`
symbolically but jens's own reads are `{predicted_q_at_start,
mc_return}` — the audit doesn't traverse transitive dependencies).
**The shadow diagnostic is the stratified ρ**: qdiv is
near-collinear with jens within γ (ρ=0.97), consistent with
the algebraic identity; eff_h is 92% determined by γ alone, so
its coupling with jens via γ-shared structure is what the PC
removal saw. argmax_H and stale carry only modest within-γ
coupling — they're genuinely separate mediators.

Caveat: the audit's `flagged_no_residual_signal` flag triggers
on LOW within-stratum ρ (HP-shadow semantics), not HIGH (which
would be the algebraic-shadow signature). So the audit reports
CLEAN for qdiv even though ρ=0.97 — read the raw ρ rather than
the flag for shadow-of-jens detection.

`is_ddqn ⊥ stale | ∅` confirms cross-env that DDQN does not
shift target staleness on average — consistent with
`findings_target_staleness_collinear.md`: staleness is the
algorithmic mediator only when sync varies (treatment via sync,
not via DDQN). At fixed sync within REACH, staleness is
treatment-invariant.

## What PC retains — the descriptive structure

The retained edges form an **outcome-target hub**: every variable
except `is_ddqn` connects to `outcome` directly. The implicit
DAG (if we orient `eff_h — jens` and resolve ambiguities):

```
  stale ──→ outcome ←── jens ←── eff_h?
    │         ↑           │
    │         q_div  ──── │
    │         ↑           │
    ↓         │           │
  argmax_H ←──┘           │
    ↑                     │
    └─── eff_h ───────────┘
```

`jens ↔ q_div` is the algebraic identity (algebraically directed
both ways within fixed (env, γ); cross-γ leakage breaks the
shadow as established in `q_divergence_shadowed_by_jens`).
`stale → argmax_H → outcome` is a plausible causal chain (target
drifts → online network's argmax distribution narrows → outcome
changes). PC can't distinguish chain from common-cause structure.

## Methodological note — descriptive, not causal-of-treatment

This discovery surfaces **observational dependence structure**,
not the causal effect of DDQN. The treatment `is_ddqn` falls out
because:
1. its marginal effect on outcome is null on REACH (matches
   established findings);
2. its mediating channels (jens, q_div) are mutually shielded by
   algebra (qdiv = jens × const within stratum), so PC can't tag
   either as the necessary mediator.

For the "DDQN→outcome via jens" arrow, the right tool is
DoWhy `backdoor_ate` on Δ_jens conditioning (see
`reach_link_backdoor_ate_negative`), not PC adjacency. PC's
value here is descriptive: it shows the dependency cluster
around `outcome` is independent of treatment, validating the
"link null on REACH" narrative without per-pair-Δ machinery.

## Sibling questions

- **Per-env PC (no JCI pooling).** Run conservative-PC on each
  REACH env separately (n=60-111). Per-env adjacencies may
  resolve the eff_h orientation ambiguity that pooled-JCI can't
  disentangle.
- **Wider variable set.** Add `gamma`, `sync_period`, `n_actions`
  as graph nodes. If `is_ddqn ⊥ outcome | {gamma}`, that's
  Simpson's paradox cross-γ.
- **SURVIVE cohort PC.** Repeat on `{CartPole, Asterix, Breakout,
  Freeway, SpaceInvaders}` ∩ scope. Different separating-set
  structure expected (mech HELD on |A|=2 reverses for CartPole;
  Q-amplification on MinAtar).

## Files

- `run_pc_reach.py` — discovery script.
- `pc_adjacency.json` — full output (variables, edges, separating
  sets, ambiguous triples).
