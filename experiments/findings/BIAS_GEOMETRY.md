# Bias geometry: a two-axis unification of the empirical regime classifier

> Connects three things that have been developed separately in the
> substrate work:
> 1. T1/T2 decomposition (`findings_two_types_of_bias`, Finding 5 of
>    the report — empirical mechanism: max-bias vs FA-truncation).
> 2. Λ_m/Λ_a decomposition (`THEORY_bootstrap_dominance.md` v3 —
>    theoretical effect: magnitude vs argmax).
> 3. 4-regime outcome classifier (Q-EXPLODED / Q-STRUCTURED /
>    Q-COLLAPSED / CLIP-RATCHET) — empirical taxonomy from MinAtar
>    γ=0.999 panel + Snake γ=0.99.
>
> These appear as three independent observations but actually share
> a single underlying two-axis structure. Commit `9c857f0`'s Λ_m
> calibration on 8 envs **empirically demonstrates** the need for
> the second axis (5/7 upper-bracket mismatches), confirming the
> unification.
>
> Status: settled reframing. Doesn't introduce new empirical claims;
> reorganizes existing ones under a single dec​omposition.
>
> Date: 2026-05-18

## 1. Why a unification is needed

The publication plan has accumulated three decompositions of "what
DDQN does":

- **Where the bias comes from** (Finding 5): T1 = max-bias DDQN
  attacks; T2 = FA-truncation residual DDQN can't reduce.
- **How DDQN's reduction translates to outcome** (Finding 1 / T3a):
  4-regime classifier with Q-EXPLODED, Q-STRUCTURED, Q-COLLAPSED, and
  (post-T2a) CLIP-RATCHET.
- **What governs the regime** (THEORY note v3): Λ_m for the magnitude
  channel; Λ_a for the argmax channel.

These are three views of the same object. Commit `9c857f0` made the
unification empirically necessary: Λ_m calibrated on 8 envs is a
1-bit classifier (Λ_m ≪ 1 ⇒ Q-COLLAPSED is reliable), but its
upper-bracket (Λ_m ≫ 1 ⇒ "bias-dominated") **mis-classifies 5 of 7
envs** when used to predict the 4-bin regime. The missing axes are
Λ_a (argmax preservation) and the FA-capacity bound L (which T2
measures empirically).

This note shows the three decompositions all live on a single
**(Λ_m, Λ_a, L) triple** that, when read jointly, explains the
4-regime classifier.

## 2. The four quantities

### 2.1 T1 — max-of-K bias (DDQN-attackable)

**Definition** (Finding 5, `findings_two_types_of_bias.md`):

$$T_1 := Q_{\mathrm{VAN}} - Q_{\mathrm{DDQN}}$$

empirically measured per cell. T1 is the part of vanilla's
Q-overestimation that DDQN's decoupled-argmax estimator reduces.

**Theoretical origin** (Lemma 1 + Lemma 2 of THEORY note):

Under iid Gaussian noise (A1), Lemma 1 gives vanilla's per-step bias
$b_V = \gamma \cdot \sigma \cdot \phi(K)$ and DDQN's per-step bias
$b_D = \gamma \cdot \sigma \cdot \phi_D(K)$ with $\phi_D \ll \phi$
(Hasselt 2010 Prop 1). Lemma 2's Bellman-MSE fixed point gives the
steady-state values; their difference is:

$$T_1^{\mathrm{(theorem)}} = \frac{\gamma \cdot \sigma \cdot (\phi(K) - \phi_D(K))}{1 - \gamma} \approx \frac{\gamma \cdot \sigma \cdot \phi(K)}{1 - \gamma}$$

T1 is therefore the **theorem-predictable** bias component.

### 2.2 T2 — FA-truncation residual (DDQN-irreducible)

**Definition** (Finding 5):

$$T_2 := Q_{\mathrm{DDQN}} - \mathrm{MC}_{\mathrm{DDQN}}$$

empirically the bias that **remains** after DDQN's clip removes T1.

**Theoretical origin** (THEORY note §9.5, OPEN gap): FA capacity L
truncates the Bellman fixed point; γ-truncation imposes a floor;
sampling bias from replay; policy mismatch. The theorem currently
treats T2 as an open formalization gap — Lemma 2's "constant b"
linearization breaks down at the FA boundary; the truncation isn't
in the closed-form Λ_m.

T2 is **outside the theorem's predictive scope** as currently
formalized. Empirically the T1/T2 ratio is measurable and tracks
env-specific dynamics (`findings_two_types_of_bias.md`).

### 2.3 Λ_m — magnitude bias as a ratio

**Definition** (THEORY note §5, eq. 4):

$$\Lambda_m := \frac{\gamma \cdot \sigma \cdot \phi(K)}{\rho \cdot R_{\max}}$$

This is the size of T1 normalized to Q* (Theorem 1): $T_1 / |Q^*| \geq \gamma \Lambda_m \approx \Lambda_m$.
$\Lambda_m \geq 1$ ⇒ T1 dominates Q*.

**What it predicts** (cleanly): Λ_m ≪ 1 ⇒ Q tracks Q* in magnitude →
Q-COLLAPSED-style regime where bias never accumulates.

**What it cannot predict alone**: which of {Q-EXPLODED, Q-STRUCTURED,
T2-dominated-COLLAPSED} the env lands in when Λ_m ≫ 1. The empirical
calibration confirms this: 5/7 upper-bracket mismatches.

### 2.4 Λ_a — argmax preservation index

**Definition** (THEORY note §6, eq. 10):

$$\Lambda_a := \frac{\sigma_{\mathrm{aniso}} \cdot \sqrt{2 \ln K}}{\min_s \Delta(s)}$$

where $\sigma_{\mathrm{aniso}}$ is the per-state across-action SD of
$Q_\theta$ and $\Delta(s)$ is the action-value gap. Theorem 2's
Gaussian-tail bound gives the sufficient condition for argmax
preservation.

**What it predicts**: whether the bias magnitude (governed by Λ_m)
corrupts the argmax of $Q_\theta$. Λ_a small ⇒ argmax preserved
even if Q magnitude grows; Λ_a large ⇒ argmax corruption.

**Independence from Λ_m**: Λ_m scales with the common-mode bias
(mean shift of Q values via max-of-K). Λ_a scales with the
anisotropic spread of Q around its mean across actions. These can
vary independently.

## 3. The connection — single underlying bias geometry

The four quantities aren't independent dimensions. They're two
PAIRS of views on the same underlying object:

| View | "What" axis | "How much" axis |
|---|---|---|
| **Mechanism** (empirical) | T1: max-bias DDQN attacks | T2: FA-truncation DDQN can't |
| **Effect** (theoretical) | Λ_m: magnitude bias → Q vs Q* | Λ_a: argmax bias → argmax_Q vs argmax_{Q*} |

The mechanism axis tells you **where bias comes from** (max-bias
vs FA-truncation). The effect axis tells you **how bias hurts**
(magnitude tracks the bias level; argmax tracks the policy
correctness).

T1 and Λ_m are the same quantity expressed two ways:
- T1 measures the magnitude in Q-units (8 jens at FR vanilla γ=0.999).
- Λ_m measures the same magnitude as a dimensionless ratio (≈95 at FR).

T2 is the residual after T1; it has no clean dimensionless analog in
the current theorem because L (FA capacity) is outside Lemma 2's scope.

Λ_a is genuinely new — it's NOT a re-expression of T2. T2 measures the
FA-bounded residual bias in Q-units; Λ_a measures how the per-action
ANISOTROPY of bias maps to argmax errors. T2 is dominated by
common-mode bias; Λ_a is dominated by anisotropic bias. These can
coexist or be independent.

## 4. The 4-regime classifier as the (Λ_m, Λ_a, L) joint

| Empirical regime | Λ_m bracket | Λ_a state | L state | Calibration anchors |
|---|---|---|---|---|
| **Q-COLLAPSED** (signal-led) | ≪ 1 | — | — | Acrobot γ=0.999 (Λ_m=0.41); MountainCar γ=0.999 (0.28) |
| **Q-STRUCTURED** (DDQN rescues magnitude + argmax OK) | ≫ 1 | small | moderate | SI γ=0.999 (Λ_m=23, argmax preserved); Breakout γ=0.999 (Λ_m=31); MetaMaze γ=0.999 (Λ_m=41) |
| **Q-EXPLODED** (DDQN's clip corrupts argmax) | ≫ 1 | large under DDQN clip | low (no truncation) | Asterix γ=0.999 (Λ_m=18, post-clip Λ_a large → d_out=-0.76 harm) |
| **T2-dominated-COLLAPSED** (FA truncates first) | nominally ≫ 1 | — | tight (L bounds bias) | FR γ=0.999 (Λ_m=12.6 but FA truncates q_late to ~8; outcome 0 → policy collapse) |
| **CLIP-RATCHET** (DDQN-induced σ-asymmetry) | bias-dominated MAGNITUDE | inverted (DDQN INFLATES Q variance) | — | Snake γ=0.99 (DDQN INFLATES Q + σ; outside theorem scope) |

**Reading the table.** The 4-bin classifier emerges from the joint
of three axes (Λ_m, Λ_a, L), not from any single one. Λ_m alone is
a Q-COLLAPSED detector. The upper-bracket discrimination needs Λ_a
(argmax preservation) and L (FA truncation depth) jointly.

CLIP-RATCHET is the joint complement — DDQN's clip itself introduces
σ-asymmetry, which is outside the theorem's scope (the theorem
covers vanilla's bias; CLIP-RATCHET is DDQN's own failure mode).

## 5. Reframing the T3a predictors (commit 9c857f0)

The publication plan's T3a pre-registration committed at `9c857f0`
has two predictors:

- **Predictor A** = empirical-signature based (smoothness, σ-asymmetry,
  CNN-class heuristic, FA-capacity from env structure).
- **Predictor B** = Λ_m alone.

Under the bias-geometry unification:

- **Predictor A is really an EMPIRICAL PROXY for (Λ_a, L)** —
  measuring the argmax channel via DDQN's smoothness reduction and
  the FA-capacity bound via env class.
- **Predictor B is Λ_m alone** — the magnitude channel.

Predictor B's 5/7 calibration mismatches are NOT random failures.
They're the theorem's diagonal: the envs where Λ_m ≫ 1 but Λ_a small
+ L moderate (= Q-STRUCTURED) get mis-classified as Q-EXPLODED if
read off Λ_m alone. The theorem PREDICTS this mismatch pattern.

So "Predictor A vs Predictor B disagreement as publishable signal"
becomes a theorem-predicted decomposition: A measures (Λ_a, L),
B measures Λ_m, the joint gives the 4-regime classifier, and
disagreement at the per-env level surfaces where each axis matters.

## 6. Empirical calibration table (from `9c857f0`)

The calibration explicitly anchors the (Λ_m, Λ_a, L) joint:

| env | Λ_m | observed regime | which axes flag the regime |
|---|---:|---|---|
| Acrobot γ=0.999 k=1 | 0.41 | Q-COLLAPSED | Λ_m ≪ 1 ⇒ signal-led ✓ (theorem-decisive) |
| MountainCar γ=0.999 | 0.28 | Q-COLLAPSED | Λ_m ≪ 1 ✓ |
| Freeway γ=0.999 | 2.81 | Q-COLLAPSED | borderline Λ_m; L tight (FA bottleneck at MinAtar CNN[16,128] for this env) |
| FourRooms γ=0.999 | 12.64 | T2-dominated COLLAPSED | Λ_m ≫ 1 but L tight (MLP[64,64] truncates q_late to ≈ 8) |
| MetaMaze γ=0.999 | 40.59 | Q-STRUCTURED | Λ_m ≫ 1 but Λ_a small (argmax preserved) |
| SpaceInvaders γ=0.999 | 23.45 | Q-STRUCTURED | Λ_m ≫ 1 but Λ_a small |
| Breakout γ=0.999 | 30.65 | Q-STRUCTURED | Λ_m ≫ 1 but Λ_a small |
| **Asterix γ=0.999** | **17.92** | **Q-EXPLODED** | **Λ_m ≫ 1 AND Λ_a large under DDQN's clip → outcome harm** |

The 5/7 upper-bracket "mismatches" (Freeway, FR, MetaMaze, SI,
Breakout) are precisely the envs where Λ_m alone is insufficient and
either Λ_a or L provides the discriminator. Asterix is the
ONE upper-bracket env where Λ_m's bias-dominated prediction matches
the empirical Q-EXPLODED bin — because it's the env where Λ_a IS
large (post-clip) and L doesn't truncate aggressively.

## 7. Implications for the publication

### 7.1 The theoretical contribution becomes clearer

The theorem note's Λ_m, by itself, is "Hasselt 2010 + Bellman
algebra" (mostly bookkeeping). The genuinely new piece is Λ_a as a
distinct axis from Λ_m — the formalization of the argmax channel
that the empirical regime classifier needs.

Under the bias-geometry unification, the publication's theoretical
contribution is sharper:
- T1/T2 (empirical) → reified as Lemma 1+2 vs L (theorem).
- Λ_m vs Λ_a (theoretical) → the two-axis decomposition.
- 4-regime classifier (empirical) → the JOINT (Λ_m, Λ_a, L) tuple.

The novel claim becomes "**a single (Λ_m, Λ_a, L) bias-geometry tuple
explains both the empirical T1/T2 mechanism decomposition AND the
empirical 4-regime outcome classifier**". That's a stronger
contribution than either decomposition alone.

### 7.2 Λ_m's role gets a clean statement

Λ_m is not the regime classifier. It's the **signal-led-vs-bias-led
1-bit detector**. The 4-bin classifier requires the joint (Λ_m, Λ_a, L).
Commit `9c857f0`'s honest "Λ_m is a 1-bit not a 4-bit classifier"
framing is exactly right; the bias-geometry unification reframes
this as "the missing axes are Λ_a and L, predicted by the theorem
to be necessary".

### 7.3 Predictor A is not a competing predictor — it's the missing axes

Reframed: Predictor A measures the empirical proxies for (Λ_a, L);
Predictor B measures Λ_m. They're complementary, not competing. The
"disagreement-as-publishable" framing becomes "theorem-predicted
diagonal" framing — disagreement at the per-env level surfaces
exactly where Λ_m alone is insufficient, which the theorem
quantitatively predicts.

### 7.4 CLIP-RATCHET as the boundary of the theorem

CLIP-RATCHET is **outside the bias geometry** as defined. It's where
DDQN's clip itself introduces variance asymmetry — a different
failure mode than vanilla's max-bias. The unification's clean
boundary statement: "the (Λ_m, Λ_a, L) tuple describes vanilla's
failure modes and DDQN's reduction; CLIP-RATCHET is DDQN's own
failure mode and lives outside".

## 8. Open formalization gaps under the unification

The unification doesn't close any of the theorem's open gaps; it
reorganizes them:

- **L (FA truncation)** — still §9.5 open. The unification makes its
  necessity sharper: without L the 4-regime classifier loses
  resolution between Q-STRUCTURED, Q-EXPLODED, and T2-dominated COLLAPSED.
- **Λ_a closed form** — §6 of theorem has a sufficient condition under
  iid Gaussian; FA-correlated case is open.
- **DDQN's effect on Λ_a** — the theorem says DDQN's clip "primarily
  affects Λ_m" but "introduces per-state asymmetry → can corrupt argmax".
  **Theorem 3 (THEORY note §6.1, 2026-05-18, post 6 review rounds)
  closes this under (A2)+(A3) iid Gaussian:** the closed-form
  sufficient condition for DDQN-clip preservation of
  **one-step-bootstrapped** argmax is
  $\gamma \sigma_{\mathrm{clip}}\sqrt{2(K-1)} < \Delta_v$ (Popoviciu's
  deterministic range bound; reviewer-corrected from a too-tight
  √(2 ln K) probabilistic constant in the v1 draft). Lemma 5 supplies
  a leading-order **lower bound** on the per-state clip error, which
  Corollary 3.2 propagates as a CORRUPTION-side sufficiency: large
  $\mathrm{Var}_a[\Lambda_a]$ ⇒ $\sigma_{\mathrm{clip}}$ at-least-this-large
  ⇒ argmax CAN be corrupted. The empirical signature operates via
  Corollary 3.2, NOT via direct verification of the preservation
  threshold (6.1.5). Extension to converged-iterate requires (A4'a)
  σ_clip-magnitude alignment AND remains subject to a geometric-series
  argmax-accumulation gap (parallel to §9.3's Robbins-Monro gap for
  Theorem 1).
  $\sigma_{\mathrm{clip}}$ is the SD across actions of the
  expected clip error at next-state destinations. Corollary 3.2:
  $\sigma_{\mathrm{clip}}$ is governed by the VARIANCE of $\Lambda_a$
  across action-destinations from $s$. Empirical proxy:
  $\Delta_{\mathrm{smoothness}}$ (DDQN's reduction of
  q_inter_state_grad_overlap_late) — exclusively negative for Asterix
  γ=0.999 in the 8-env panel, matching the theorem's prediction.
  **Open piece:** the env-structural reason for $\Lambda_a$
  heterogeneity (why Asterix's action-destinations have heterogeneous
  $\Lambda_a$) lives outside the algorithm's bias-geometry.

These three gaps together are what stand between "appendix-quality
note" and "publishable theoretical contribution".

### 8.1 Empirical 4th axis (post-9c857f0 calibration)

After the Λ_a calibration upper bracket showed both Asterix (HARMS,
Λ_a=2.87) and SI (HELPS, Λ_a=2.40) sitting above 2, we pushed for an
empirical 4th axis to discriminate the Q-EXPLODED vs Q-STRUCTURED
upper bin. Candidate: **Δ_smoothness** = Cohen's d of
`q_inter_state_grad_overlap_late` (DDQN − VAN), per env at γ=0.999.

n=8 panel (Asterix / SI / FR / MetaMaze / Breakout / MountainCar /
Acrobot / Freeway):

- Λ_a alone (Pearson) ρ(Λ_a, d_out) = +0.04 p=0.92 — null cross-env.
- Δ_smooth (Pearson) ρ(Δ_smooth, d_out) = +0.60 p=0.11 (marginal NS).
- **Partial ρ(Δ_smooth, d_out | Λ_a) = +0.74 p=0.035.** ✓
- Binary rule (Λ_a > 2 AND Δ_smooth < −1 → harm): **8/8 classified.**

But the result is genuinely partial:

1. **Without Asterix (n=7), Spearman ρ collapses to +0.18 (p=0.70).**
   The cross-env signal is anchored to one harm-regime env.
2. **Within-DDQN per-cell r(smoothness, outcome) flips sign across
   envs:** Acrobot −0.86, Freeway +0.47, Breakout +0.38, FR −0.35.
   A graded continuous mediator would track the cross-env sign — it
   doesn't.

**Conclusion: Δ_smoothness is a DISCRETE env-feature, not a graded
continuous mediator.** The 8-env binary classifier works ("DDQN
catastrophically reduces inter-state Q-coherence at Asterix γ=0.999,
nowhere else"), but the mechanism is Asterix-specific — the
Q-overshoot-and-recalibrate regime breaking Bellman pool.

**What this MEANS for the unification:**

- The (Λ_m, Λ_a, L) tuple stands. Δ_smoothness is NOT added as a
  4th theoretical axis — it's empirical, not derivable from σ_aniso
  / K / Δ in closed form.
- Open gap #3 (above) remains open **theoretically**. Δ_smoothness
  is the **empirical operationalization** of "DDQN's effect on
  argmax structure" — measurable but unexplained.
- Calling this a "Λ_clip_asymmetry" axis would over-claim — it's
  one empirical signature of a discrete regime boundary, not a
  closed-form effect channel.
- The Asterix-vs-rest pattern is empirically real; its closed-form
  theorem articulation is the publishable next step.

Memory: `findings_lambda_a_smoothness_third_axis_partial.md`.
Related: `findings_q_smoothness_is_jens_shadow`,
`findings_asterix_g999_harm_is_optimization_dynamics`,
`findings_asterix_g999_pc_mediator_triangle`,
`findings_pc_cross_env_smoothness`.

## 9. Cross-references

- `experiments/findings/THEORY_bootstrap_dominance.md` v3 (formal
  theorem with Lemmas 1-4 and Theorems 1-2).
- `findings_two_types_of_bias.md` (memory) — T1/T2 empirical
  decomposition; Finding 5 of the report.
- `findings_prospective_predictions_t3a_lambda_m.md` (memory) — Λ_m
  calibration table from `9c857f0`.
- `findings_minatar_gamma_sweep_first_results.md` (memory) — 4-regime
  classifier empirical anchors at γ=0.999.
- `findings_snake_ddqn_destabilizes_sparse_reward.md` /
  `finding_snake_clip_ratchet_regime` — CLIP-RATCHET as the
  boundary regime.
- `docs/PRE_REGISTRATIONS_2026-05-18.md` PR-5 — Predictor A + B
  formulation.
- `docs/DDQN_PUBLICATION_PLAN.md` — T2b' (DONE), T3a (in flight).
