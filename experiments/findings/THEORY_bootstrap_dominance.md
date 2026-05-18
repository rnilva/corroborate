# Bootstrap-Dominance: a formal regime classification for vanilla Q-learning at γ → 1

> Formal version v3 (post-review). Distinguishes proved-from-literature
> results from new claims; states assumptions explicitly; flags
> open-formalization gaps.
>
> Status: draft theory note. Appendix-quality rigor for the lemmas
> that are derivable from Hasselt 2010 + standard SGD/Bellman theory.
> Open formalization gaps explicitly listed in §6.
>
> Author: corroborate-rl team
> Date: 2026-05-18 (v3-formal)
> Prior versions: v1 (single-Λ overreach, walked back); v2 (sufficiency
> heuristic, informal lemmas).

## 1. Setup + Notation

Consider an MDP $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ with discrete
action set $|\mathcal{A}| = K$. Let $Q^*(s, a)$ be the true optimal
action-value function. We use a parametric Q-network $Q_\theta$ trained
by Bellman regression on mini-batches sampled from a replay buffer.

**Parameters and observables:**
- $\gamma \in (0, 1)$ — discount factor.
- $\sigma(s) := \mathrm{SD}_a [Q_\theta(s, a)]$ — within-state across-action
  standard deviation of the Q-network's output, also called
  `q_action_std_late` in the substrate measurables. We treat this as a
  slowly-varying parameter; reported values are late-window means.
- $\rho \in (0, 1]$ — fraction of replay-buffer transitions with non-zero
  reward.
- $R_{\max}$ — upper bound on per-transition reward magnitude.
- $B$ — mini-batch size.
- $\alpha \in (0, 1)$ — Q-update step size (scalar SGD coefficient).
- $T_{\mathrm{sync}}$ — target-network sync period (number of updates
  between target $\bar\theta \leftarrow \theta$).
- $L$ — FA capacity bound on representable Q magnitude.

**Vanilla DQN target:**
$$y_i^{(\mathrm{V})} = r_i + \gamma \cdot \max_{a'} Q_{\bar\theta}(s'_i, a')$$

**DDQN target:**
$$y_i^{(\mathrm{D})} = r_i + \gamma \cdot Q_{\bar\theta}\!\bigl(s'_i, \arg\max_{a'} Q_\theta(s'_i, a')\bigr)$$

**Standing assumptions** (used throughout this note, declared up front):
- (A1) iid Gaussian noise on Q estimates: $Q_\theta(s, a) = Q^*(s, a) + \varepsilon_{s, a}$
  with $\varepsilon_{s, a} \sim \mathcal{N}(0, \sigma^2)$ iid across $a$
  at a fixed $s$.
- (A2) Robbins-Monro step sizes: $\sum_t \alpha_t = \infty$,
  $\sum_t \alpha_t^2 < \infty$. (For analytic results we use a small
  constant $\alpha$ as the approximation.)
- (A3) Bounded FA capacity: $|Q_\theta(s, a)| \leq L$ representable.
- (A4) Bounded replay: $\rho \in (0, 1]$, $|R| \leq R_{\max}$.

(A1) is unrealistic in deep RL (FA features correlate noise across
actions); see §6 for the conservative interpretation in that case.

## 2. Lemma 1 — Max-of-K positive bias (Hasselt 2010)

**Lemma 1.** Under (A1), with $\phi(K) := \mathbb{E}[\max_{a=1..K} \varepsilon_a]/\sigma$
the *expected maximum of K iid standard Gaussians*:

$$\mathbb{E}\!\left[\max_a Q_\theta(s, a)\right] - \max_a Q^*(s, a) \;\leq\; \sigma \cdot \phi(K). \tag{1}$$

**Tightness.** Equality in (1) holds when all $Q^*(s, a)$ are equal
(the noise determines the argmax). When the action-value gap
$\Delta(s) := \max_a Q^*(s, a) - \max_{a \neq a^*} Q^*(s, a)$ is large
compared to $\sigma$, the bias is much smaller than $\sigma \phi(K)$
(the noise rarely changes which action's estimate is largest).
Specifically (Thrun-Schwartz 1993): for large $\Delta/\sigma$, the bias
is $O(\sigma e^{-\Delta^2 / 2\sigma^2})$.

**Proof reference.** Hasselt 2010, §3.1, Equation (4). $\phi(K)$ is
the standard order-statistic constant; see David & Nagaraja 2003,
Theorem 10.5.4 for closed-form moments.

**Exact $\phi(K)$ values** (Monte Carlo $N = 2 \times 10^6$):

| $K$ | $\phi(K)$ exact | $\sqrt{2 \ln K}$ asymptote |
|---:|---:|---:|
| 2 | 0.564 | 1.177 |
| 3 | 0.845 | 1.482 |
| 4 | 1.029 | 1.665 |
| 5 | 1.163 | 1.794 |
| 6 | 1.267 | 1.893 |
| 10 | 1.539 | 2.146 |
| 18 | 1.821 | 2.404 |

The asymptotic $\sqrt{2 \ln K}$ bound over-estimates by 30–60% at $K \leq 10$.
Use exact $\phi(K)$ for empirical Λ_m calculations.

## 3. Lemma 2 — Bootstrap-chain fixed point

**Lemma 2.** Under (A4) and a constant per-step bias $b > 0$ in the
bootstrap target, the iterated Q-update
$$Q_{t+1} = (1 - \alpha) Q_t + \alpha \bigl(r + \gamma (Q_t + b)\bigr)$$
has expected fixed point
$$Q_\infty = Q^* + \frac{\gamma b}{1 - \gamma}, \quad \text{where } Q^* = \frac{r}{1 - \gamma}. \tag{2}$$

**Proof.** Setting $Q_{t+1} = Q_t = Q_\infty$ at the fixed point:
$$
Q_\infty = (1 - \alpha) Q_\infty + \alpha r + \alpha \gamma Q_\infty + \alpha \gamma b
$$
$$
\Rightarrow 0 = \alpha r + (\alpha \gamma - \alpha) Q_\infty + \alpha \gamma b
$$
$$
\Rightarrow (1 - \gamma) Q_\infty = r + \gamma b
$$
$$
\Rightarrow Q_\infty = \frac{r}{1 - \gamma} + \frac{\gamma b}{1 - \gamma} = Q^* + \frac{\gamma b}{1 - \gamma}. \square
$$

**Caveat (non-linearity).** In practice $b = \gamma \sigma \phi(K)$
depends on $\sigma$, which depends on the current Q magnitude. So the
"constant $b$" assumption is local linearization. Iteratively, $\sigma$
may grow with $Q$ until FA capacity $L$ truncates. The Lemma 2 form is
an **upper bound** on the asymptote in the linearized regime;
empirically the FA-truncated asymptote is smaller (e.g., 8 vs the
Lemma-2 bound of 18 at FR × MLP × γ=0.999).

## 4. Lemma 3 — DDQN's bias direction

**Lemma 3** (Hasselt 2010 Proposition 1, paraphrase). Under (A1) with
two independent estimators $Q^{(1)}, Q^{(2)}$ having iid Gaussian noise
with std $\sigma$:

$$\mathbb{E}\!\left[Q^{(2)}\!\bigl(\arg\max_a Q^{(1)}(s, a)\bigr)\right] \;\leq\; \mathbb{E}\!\left[\max_a Q^{(1)}(s, a)\right]. \tag{3}$$

That is: the decoupled-argmax estimator (used by DDQN with online/target
networks) has **strictly less bias** than the single-Q max estimator
under iid noise. The DDQN estimator's bias is in fact often *negative*
(under-estimation), which is the structural reason for DDQN's lower Q
magnitudes in practice (Hasselt 2010 §4, "Theorem 1 remark").

**Quantitative bound.** Hasselt 2010 §4.1 gives an explicit formula for
the DDQN bias under iid Gaussian noise. We denote DDQN's bias factor
$\phi_D(K)$ with the property $|\phi_D| \leq c \cdot \phi(K)$ for some
$c \in (0, 0.2)$ for the K-range we consider ($K \leq 18$). Exact $\phi_D$
depends on the joint distribution of $Q^{(1)}, Q^{(2)}$ (their
temporal correlation through target syncing). We treat the conservative
bracket as sufficient for regime classification.

**Caveat.** Online and target networks are NOT independent in practice
(target is a delayed copy of online, syncing every $T_{\mathrm{sync}}$
steps). The bias-direction result still applies — Hasselt 2010 §4 holds
under positive correlation — but the magnitude bound tightens.

## 5. Theorem 1 — Magnitude regime sufficiency

Define the **magnitude bias-dominance ratio**:

$$\Lambda_m \;:=\; \frac{\gamma \cdot \sigma \cdot \phi(K)}{\rho \cdot R_{\max}}. \tag{4}$$

This is the ratio of per-step Lemma-1 bias contribution to per-step
expected reward signal contribution to the Bellman target.

**Theorem 1** (Magnitude regime sufficiency). Under (A1)–(A4) and the
Lemma-2 linearization, the relative bias at the Q-update fixed point
satisfies the lower bound:

$$\frac{Q_\infty^{(\mathrm{V})} - Q^*}{Q^*} \;\geq\; \gamma \cdot \Lambda_m. \tag{5}$$

For $\gamma \approx 1$ (the limit of interest for this study),
$\gamma \cdot \Lambda_m \approx \Lambda_m$. Equivalently: when
$\Lambda_m \geq 1/\gamma$ (i.e. $\Lambda_m \gtrsim 1$ at high $\gamma$),
the expected fixed-point Q value's bias exceeds the magnitude of $Q^*$
itself; Q is dominated by bias rather than tracking the true value
function.

**Proof.** Combining Lemma 1 and Lemma 2 with $b = \gamma \sigma \phi(K)$:
- $Q_\infty^{(\mathrm{V})} - Q^* = \gamma b / (1 - \gamma) = \gamma^2 \sigma \phi(K) / (1 - \gamma)$ (Lemma 2 with vanilla $b$).
- $Q^* = E[r] / (1 - \gamma)$. For sparse rewards with density $\rho$ and magnitude $R_{\max}$, $E[r] \leq \rho R_{\max}$ in steady state, so $Q^* \leq \rho R_{\max} / (1 - \gamma)$.
- Therefore $|Q_\infty^{(\mathrm{V})} - Q^*| / |Q^*| \geq \gamma^2 \sigma \phi(K) / (\rho R_{\max}) = \gamma \cdot \Lambda_m$. At $\gamma = 0.999$ this is within 0.1% of $\Lambda_m$; at $\gamma = 0.99$ within 1%. $\square$

**Corollary 1.1** (DDQN's role on the magnitude channel). Substituting
Lemma 3's bias factor $\phi_D(K)$ for $\phi(K)$:

$$\Lambda_m^{(D)} \;=\; \frac{\gamma \cdot \sigma \cdot \phi_D(K)}{\rho \cdot R_{\max}} \;\leq\; 0.2 \cdot \Lambda_m^{(V)}. \tag{6}$$

DDQN reduces $\Lambda_m$ by at least $5\times$ at fixed $\sigma$ and $\rho$.
At γ=0.999 with $\Lambda_m^{(V)} \approx 95$ (FR × MLP), DDQN's
$\Lambda_m^{(D)} \leq 19$. Still nominally $\geq 1$, so DDQN is still in
the bias-dominated regime by Theorem 1, but the absolute bias is much
smaller in magnitude.

**Corollary 1.2** (γ-amplification). Since
$Q_\infty - Q^* = \gamma b / (1 - \gamma)$, the bias compounds with
$1/(1-\gamma)$. At γ = 0.99 the amplifier is 99; at γ = 0.999 it is 999.
Even tiny per-step biases compound catastrophically in the high-γ
limit, holding $\Lambda_m$ constant.

**Theorem 1's domain.** This covers Q-magnitude bias divergence. It says
nothing about argmax preservation. See §6 for the orthogonal axis.

## 6. Theorem 2 — Argmax preservation sufficient condition

Define the **anisotropic component of the cross-action Q noise**:
$$\sigma_{\mathrm{aniso}}(s) := \sqrt{\mathrm{Var}_a\!\bigl[Q_\theta(s, a) - \mathbb{E}_a Q_\theta(s, \cdot)\bigr]}. \tag{7}$$

This is $\sigma$ projected onto the "across-actions" component; it
governs which action $\arg\max_a Q_\theta(s, a)$ picks vs which
$\arg\max_a Q^*(s, a)$ would pick.

Define the **action-value gap**:
$$\Delta(s) := Q^*(s, a^*(s)) - \max_{a \neq a^*(s)} Q^*(s, a), \tag{8}$$
the difference between the optimal and second-best action's true values
at state $s$.

**Theorem 2** (Argmax preservation sufficient condition). Under (A1)
with iid Gaussian noise, the probability that $\arg\max_a Q_\theta(s, a) = a^*(s)$
satisfies:
$$P\bigl[\hat{a}(s) = a^*(s)\bigr] \;\geq\; 1 - (K - 1) \cdot \exp\!\bigl(-\Delta(s)^2 / (4 \sigma_{\mathrm{aniso}}(s)^2)\bigr). \tag{9}$$

**Proof.** For each non-optimal action $a$, the event
$Q_\theta(s, a) > Q_\theta(s, a^*)$ requires the noise to overcome the
gap $\Delta(s)$. Under iid Gaussian, this is a normal-tail probability:
$P[Q_\theta(s, a) > Q_\theta(s, a^*)] = P[\varepsilon_a - \varepsilon_{a^*} > \Delta(s)]$,
where $\varepsilon_a - \varepsilon_{a^*}$ is $\mathcal{N}(0, 2\sigma_{\mathrm{aniso}}^2)$.
Applying the standard Gaussian sub-Gaussian tail bound
$P(Z > t) \leq \exp(-t^2/2)$ for $Z \sim \mathcal{N}(0, 1)$ with
$t = \Delta(s) / \sqrt{2 \sigma_{\mathrm{aniso}}^2}$ gives the per-action
upper bound $\exp(-\Delta(s)^2 / (4 \sigma_{\mathrm{aniso}}^2))$. Union
bound over the $K - 1$ non-optimal actions yields (9). $\square$

Define the **argmax-preservation ratio** (a scale-matching heuristic
index, not a tight constant extracted from (9)):
$$\Lambda_a := \frac{\sigma_{\mathrm{aniso}} \cdot \sqrt{2 \ln K}}{\min_s \Delta(s)}. \tag{10}$$

**Corollary 2.1**. When $\Lambda_a \ll 1$ uniformly across states,
argmax preservation holds with high probability per (9). When
$\Lambda_a \gtrsim 1$, argmax may be corrupted. More precisely,
inverting (9): preservation at confidence $\delta$ requires
$\Delta(s) \geq 2\sigma_{\mathrm{aniso}}(s) \cdot \sqrt{\ln((K-1)/\delta)}$,
which differs from (10)'s definition by a multiplicative constant of
order unity. (10) is the right asymptotic-order index; the multiplicative
constant depends on the confidence band.

**Independence from $\Lambda_m$.** $\Lambda_m$ scales with the
common-mode bias (the mean shift of all Q values upward via the max
operator + γ amplification). $\Lambda_a$ scales with the anisotropic
spread of $Q_\theta$ around its mean across actions at fixed $s$, vs
the true gap $\Delta(s)$. These quantities can vary independently:
- High $\Lambda_m$, low $\Lambda_a$: vanilla Q magnitudes balloon
  uniformly, but the argmax stays correct (Asterix γ=0.999 case).
- Low $\Lambda_m$, high $\Lambda_a$: Q magnitudes track $Q^*$, but the
  anisotropic noise corrupts argmax (e.g., near-equal-value bandit
  states).
- High $\Lambda_m$ + high $\Lambda_a$: catastrophic divergence
  (FR γ=0.999 case).

DDQN's clip primarily affects $\Lambda_m$ (reduces the common-mode bias
via Lemma 3); its effect on $\Lambda_a$ depends on whether the clip's
per-state adjustment is uniform or differential across actions —
typically the latter (clip = `min(target_max, target_at_online_argmax)`
introduces per-state asymmetry). This is the structural reason DDQN
sometimes corrupts argmax even while reducing Q magnitude (Asterix γ=0.999).

## 6.1 Theorem 3 — DDQN's clip and agent-argmax preservation

Theorem 2 gives the argmax-preservation condition for *vanilla*
Q-learning: when $\Lambda_a \ll 1$ the noise on $Q_\theta$ doesn't
overcome $\Delta(s)$. Theorem 3 takes the next step — characterizing
when *DDQN's clip*, applied on top of vanilla, **introduces a
NEW source of argmax fragility** beyond what Theorem 2 covers.

### Setup

DDQN's bootstrap target at next-state $s'$:
$$V_d(s') = Q_{\mathrm{target}}\bigl(s',\,\hat{a}_{\mathrm{online}}(s')\bigr),$$
where $\hat{a}_{\mathrm{online}}(s') = \arg\max_{a'} Q_{\mathrm{online}}(s', a')$.

Vanilla's bootstrap: $V_v(s') = \max_{a'} Q_{\mathrm{target}}(s', a')$.

Define the **clip error** at $s'$:
$$\Delta_{\mathrm{clip}}(s') := V_v(s') - V_d(s') \geq 0, \tag{6.1.1}$$
non-negative because vanilla takes the max. When $\hat{a}_{\mathrm{online}} = \arg\max_{a'} Q_{\mathrm{target}}$,
$\Delta_{\mathrm{clip}} = 0$; otherwise it equals the gap from
$Q_{\mathrm{target}}$'s max to its value at online's argmax.

**Notation bridge (three distinct gaps).** Theorem 3 navigates
three different action-value gaps; the bridge:

| Symbol | Definition | Where used |
|---|---|---|
| $\Delta(s)$ | True gap $Q^*(s, a^*) - \max_{a \neq a^*} Q^*(s, a)$ (eq. 8). | §6's $\Lambda_a$. |
| $\Delta_T(s')$ | Same true gap of $Q^*$ at the bootstrap next-state $s'$. | Lemma 5's $p_{\neq}$ bound. |
| $\Delta_v(s)$ | Bias-shifted gap of the **vanilla agent's** estimate $Q_v$ at the controlled state $s$. | Theorem 3's conclusion (6.1.5). |

Under (A1)+(A2)+(A3), $\sigma_T(s') = \sigma_{\mathrm{aniso}}(s') = $
§2's $\sigma$ at $s'$. Under Λ_m ≫ 1, vanilla's positive bias shifts
every $Q_v(s, a)$ upward by approximately
$\gamma/(1-\gamma) \cdot \sigma_{s'_a} \cdot \phi(K)$. To first order
(under $\sigma_{s'_a}$ uniform across $a$), this is common-mode →
$\Delta_v(s) \approx \Delta(s)$.

**Consistency caveat.** Heterogeneity of $\sigma_{s'_a}$ across
action-destinations — exactly the condition that drives
$\sigma_{\mathrm{clip}} \neq 0$ in (6.1.4) — is also the condition
that breaks the common-mode argument: $\Delta_v - \Delta = O(\gamma\phi(K) \cdot \sigma_T\text{-variation})$.
So the regime Theorem 3 actually targets (σ_clip > 0) is the regime
where $\Delta_v \neq \Delta$ at the same order as $\sigma_{\mathrm{clip}}$
itself. The three-gap distinction is bookkeeping at the same
asymptotic order as Lemma 5's lower-bound residual: keep $\Delta_v$,
$\Delta_T$, $\Delta$ as separate symbols because at the regime
boundary they pick up sign-dependent second-order corrections that
are part of $\sigma_{\mathrm{clip}}$'s closed form.

### Assumptions

(A2) **Online–target decoupling.** Conditional on $Q^*$,
$Q_{\mathrm{online}}(s', \cdot)$ and $Q_{\mathrm{target}}(s', \cdot)$ are
independent and have a common per-action noise std $\sigma_T(s')$.
This holds in the hard-sync limit with sync period $T \to \infty$
(target frozen relative to current online; the frozen state under
stationarity of $Q^*$ is independent of current online); progressively
violated as Polyak $\tau \to 1$ (target $\to$ online, fully
coupled). Standard DDQN's $\tau \approx 0.005$ violates A2
substantially — A2 is a clean-room assumption for the
closed form, not a description of typical deployments.

(A3) **iid Gaussian cross-action noise at $s'$.** Q-estimation noise
across actions at $s'$ is iid $\mathcal{N}(0, \sigma_T^2(s'))$, with
true action-value gap $\Delta_T(s')$. Common std across online and
target (under A2). FA-correlated noise violates (A3); §9.4 flags
this as open.

(A2) + (A3) together specialize §1's (A1) to the bootstrap target's
cross-action noise structure at each next-state.

### Per-state clip error

**Lemma 5.** Under (A2)+(A3), the expected clip error at $s'$
satisfies the leading-order lower bound:
$$\mathbb{E}[\Delta_{\mathrm{clip}}(s')] \;\geq\; p_{\neq}(s') \cdot \sigma_T(s') \cdot \eta(K), \tag{6.1.2}$$
with equality only in the noise-dominated limit
$\sigma_T \gg \max_a \Delta_a$; the residual on the right side
contributes a non-negative correction.
where:
- $p_{\neq}(s')$ = probability that
  $\hat{a}_{\mathrm{online}} \neq \arg\max_{a'} Q_{\mathrm{target}}$.
  Under (A2)+(A3) (common per-action std $\sigma_T$),
  $p_{\neq}(s') \leq (K-1) \exp\!\bigl(-\Delta_T(s')^2/(4\sigma_T(s')^2)\bigr)$
  — same exponential tail as eq. (9) of Theorem 2; the factor of 4
  in the exponent absorbs the paired-Gaussian variance doubling
  $(\sigma_{\mathrm{online}}^2 + \sigma_{\mathrm{target}}^2 = 2\sigma_T^2$ under common std).
- $\eta(K) = \mathbb{E}\bigl[\max_a Z_a - Z_{\hat{a}^{\mathrm{unif}}} \,\big|\, \hat{a}^{\mathrm{unif}} \neq \arg\max\bigr]$,
  the expected gap from the max to a uniformly-chosen non-max
  action of iid $\mathcal{N}(0, 1)$. Under (A2)+(A3) target's noise
  is independent of online's selection mechanism, so the
  conditioning on "disagreement" is **structurally vacuous** —
  $\eta(K)$ coincides with the unconditional max-vs-iid-non-max gap
  of target's noise. Asymptotically $\eta(K) \approx \phi(K) = \Theta(\sqrt{2 \ln K})$
  (same leading constant as §2's exact-$\phi$ table; numeric values
  carry over).

**Proof sketch.** The exact identity is
$\mathbb{E}[\Delta_{\mathrm{clip}}] = p_{\neq} \cdot \mathbb{E}[\text{gap} \mid \text{disagree}]$.
Decompose the gap into noise + true-Q components:
$$\text{gap} = (Q^*(a^*_T) - Q^*(\hat{a}_{\mathrm{online}})) + (\varepsilon^{\mathrm{tgt}}_{a^*_T} - \varepsilon^{\mathrm{tgt}}_{\hat{a}_{\mathrm{online}}}).$$
Under (A2)+(A3), target's noise is iid Gaussian across actions and
independent of online's selection mechanism. Conditional on
disagreement (a specific online choice $a \neq a^*_T$):
- The **noise term** has expected magnitude
  $\sigma_T \cdot \eta(K)$ regardless of which non-max action $a$
  online picked (target's iid structure doesn't depend on online's
  choice; this is the max-vs-iid-non-max gap of target's noise).
- The **true-Q term** is selection-biased: online preferentially
  picks small-$\Delta_a$ actions (its noise has an easier time
  flipping rank at small-gap destinations), so
  $\mathbb{E}[\Delta_a \mid \text{disagree}]$ is small relative to
  $\max_a \Delta_a$. The selection bias makes this residual small
  but non-negative.

So $\mathbb{E}[\text{gap} \mid \text{disagree}] = \sigma_T \cdot \eta(K) + \mathbb{E}[\Delta_a \mid \text{disagree}]$ —
**both terms non-negative**. The bound (6.1.2) captures only the
noise term; the selection-biased true-Q residual is a non-negative
correction that's small in the noise-dominated limit
($\sigma_T \gg \max_a \Delta_a$) and larger in the sub-noise regime
($\sigma_T \ll \max_a \Delta_a$, where $p_{\neq}$ itself is also
small). This is the structural reason for the **lower-bound
direction**: the residual is one-sided, so the noise term alone is
a floor on $\mathbb{E}[\Delta_{\mathrm{clip}}]$, not an unbiased
approximation. Absolute-magnitude expressions are treated in the
Status section.

For $p_{\neq}$, the union bound gives
$p_{\neq} \leq P(\hat{a}_{\mathrm{online}} \neq a^*_T) + P(\hat{a}_{\mathrm{target}} \neq a^*_T) \leq 2(K-1) \exp(-\Delta_T^2/(4\sigma_T^2))$
from the symmetric union over both estimators disagreeing with
$a^*$. For Lemma 5's displayed form we cite the single-estimator
bound $(K-1) \exp(\cdot)$; the factor-of-2 looseness is absorbed
into the slack at large-$\sigma$ saturation (Remark below). $\square$

**Remark on saturation.** Inside Lemma 5's product, the
$p_{\neq}$ factor is upper-bounded by $(K-1) \exp(\cdot)$ (eq.
9-style union); this bound saturates at $K-1$ when $\sigma_T \to \infty$,
while the TRUE $p_{\neq}$ saturates at $(K-1)/K$ (uniform-random
argmax in extreme noise). So the $p_{\neq}$ FACTOR is loose by
$\sim K$ at the saturation limit. (This is a one-sided looseness
inside the p_≠ term, distinct from Lemma 5's lower-bound direction
on the overall product.) The factor's saturation drives the
monotonicity caveat in (6.1.7) below.

### Agent argmax preservation

The DDQN agent's value at state $s$, action $a$, at a **one-step
bootstrap with shared next-state value estimator**:
$$Q_d(s, a) = \mathbb{E}[r(s, a)] + \gamma \cdot \mathbb{E}\bigl[V_d(s'_a)\bigr] = Q_v(s, a) - \gamma \cdot \mathbb{E}\bigl[\Delta_{\mathrm{clip}}(s'_a)\bigr], \tag{6.1.3}$$
where $s'_a$ is the next-state under action $a$. DDQN compresses each
$Q$-value DOWNWARD by an amount that depends on the destination $s'_a$.

**Caveat.** (6.1.3) holds at a one-step bootstrap; the joint fixed-point
difference $Q_d^* - Q_v^*$ between the two algorithms' steady-state
estimates involves a geometric series of clip errors propagated
through the bootstrap recursion. For Theorem 3's argmax-preservation
conclusion the one-step form is the right object — argmax is
preserved at each update iff its accumulated form is — but the
empirical $\sigma_{\mathrm{clip}}$ from late-window traces conflates
one-step clip errors with their geometric-series fixed-point
expansion. Order-of-magnitude alignment between the two forms is
the implicit assumption.

If $\mathbb{E}[\Delta_{\mathrm{clip}}(s'_a)]$ is **uniform across $a$**,
DDQN shifts $Q_d(s, \cdot)$ by a constant → agent's argmax preserved.
If $\mathbb{E}[\Delta_{\mathrm{clip}}(s'_a)]$ **varies across $a$**, the
relative ranking can flip.

Define the **clip-error anisotropy at $s$**:
$$\sigma_{\mathrm{clip}}(s) := \sqrt{\mathrm{Var}_a\!\bigl[\mathbb{E}[\Delta_{\mathrm{clip}}(s'_a)]\bigr]}. \tag{6.1.4}$$

**Theorem 3** (DDQN-clip argmax preservation, sufficient condition).
Under (A2)+(A3) + one-step bootstrap form (eq. 6.1.3), the agent's
**one-step-bootstrapped** argmax at $s$ under DDQN satisfies:
$$\arg\max_a Q_d(s, a) = \arg\max_a Q_v(s, a) \quad \text{whenever} \quad \gamma \cdot \sigma_{\mathrm{clip}}(s) \cdot \sqrt{2(K-1)} \;<\; \Delta_v(s), \tag{6.1.5}$$
where $\Delta_v(s)$ is the vanilla agent's action-value gap at $s$.

The extension to **converged-iterate** argmax (the empirical proxy
operates on late-window Q estimates, near DDQN's converged-Q
attractor $Q_d^*$, not at a one-step bootstrap) introduces a
load-bearing **alignment assumption** plus an **open limitation**:

(A4'a) **σ_clip magnitude alignment** (load-bearing assumption).
$\sigma_{\mathrm{clip}}(s)$ evaluated on DDQN's converged-iterate
$Q_d^*$ is order-of-magnitude aligned with $\sigma_{\mathrm{clip}}(s)$
evaluated on a one-step bootstrap from the same iterate. This bounds
the cross-action clip-error variance the empirical signature picks up.

**Geometric-series argmax-accumulation gap** (open limitation,
parallel to §9.3's Robbins-Monro convergence gap for Theorem 1).
Converged-iterate $Q_d^*$'s argmax fragility at $s$ relates to one-step
clip anisotropy via a geometric-series accumulation that the present
derivation does NOT cover. (A4'a) gives the noise MAGNITUDE; the
bridge from one-step argmax preservation (proved in eq. 6.1.5) to
converged argmax preservation (what the empirical signature observes)
is structurally similar in standing to Theorem 1's missing Robbins-Monro
piece — present, open, doesn't invalidate the calibration use of the
regime classification but limits the algebraic claim to one-step.

Theorem 3's empirical signature relies on (A4'a) being approximately
true PLUS the geometric-series gap being benign on the canonical
calibration scope. The algebraic derivation covers one-step only.

**Empirical update (2026-05-18, `experiments.findings.theorem3`).** The
geometric-series gap was tested via a typed hypothesis panel on
14-corpus / 840-cell MinAtar γ-sweep (commits `9f4cc0d` +
`35cdfde`):

1. *Horizon normalisation*: dividing per-burst $\Lambda_a$ by the
   Bellman effective-horizon factor $(1 - \gamma^t)/(1 - \gamma)$
   at the burst's training step $t$ is mathematically vacuous at
   the burst granularity — $\gamma^{20000} \approx 0$ for any
   $\gamma < 1$, so the factor saturates to a per-cell constant
   that doesn't affect the growth ratio. **Bellman bias accumulation
   saturates within the first burst.**

2. *Sub-burst direct test*: $\Lambda_a$ measured in the steps
   100-1000 transient and paired against Bellman's predicted
   growth $1/(1-\gamma^{550})$ via Spearman ρ on the 840-cell
   panel returns POWER_INSUFFICIENT (p=0.31). The empirical
   sub-burst-to-tail growth ratio is **env- and K-dependent
   (1.05 to 15.4×) while Bellman's predictor is γ-only (1.00 at
   γ=0.95, 2.36 at γ=0.999)**. The Bellman predictor explains
   essentially none of the cross-env variance.

Both probes converge on the same conclusion: **the formal
geometric-series gap is empirically moot for the σ_Λa signature.**
NN training dynamics + FA fitting + replay-buffer effects drive
σ_Λa's training-trajectory non-stationarity — all outside Theorem
3's algebraic scope. For converged-tail measurements (where the
empirical signature operates), Bellman accumulation has long
saturated; Theorem 3's empirical signature stands in better
standing than the original open-limitation framing implied.

The §6.1 caveat is retained as honest theoretical bookkeeping;
the empirical probe shows it doesn't bite at the measurement
scale the empirical signature uses.

**Proof.** From (6.1.3),
$Q_d(s, a) = Q_v(s, a) - \gamma \mathbb{E}[\Delta_{\mathrm{clip}}(s'_a)]$.
For argmax to be preserved, the maximum pairwise deviation among the
$K$ values $\{\mathbb{E}[\Delta_{\mathrm{clip}}(s'_a)]\}_{a \in \mathcal{A}}$
must be smaller than $\Delta_v(s) / \gamma$. These $K$ values are
deterministic (expectations marginalising over Q-noise), not iid
samples; the relevant bound is therefore the deterministic
**Popoviciu / sample-range** bound: for $K$ real numbers with sample
SD $\sigma$, the range $(\max - \min)$ satisfies
$\max - \min \leq \sigma \cdot \sqrt{2(K-1)}$ (tight for the
$\pm\sigma\sqrt{(K-1)/K}$ extremal configuration; cf. Popoviciu 1935).
Setting $\sigma = \sigma_{\mathrm{clip}}(s)$ and requiring
$\gamma \cdot (\max - \min) < \Delta_v(s)$ yields (6.1.5). $\square$

**Remark.** A tighter probabilistic bound of
$\sigma_{\mathrm{clip}}(s) \cdot 2 \sqrt{2 \ln K}$ holds under the
additional assumption that the destination map $a \mapsto s'_a$
induces sub-Gaussian fluctuations across $a$ — but that's an extra
modelling commitment beyond (A2)+(A3). Popoviciu is the assumption-free
deterministic bound; we adopt it.

### Λ_clip — DDQN-induced argmax-corruption index

Mirroring (10) for the agent's argmax under DDQN:
$$\Lambda_{\mathrm{clip}} := \frac{\gamma \cdot \sigma_{\mathrm{clip}}(s) \cdot \sqrt{2(K-1)}}{\min_s \Delta_v(s)}. \tag{6.1.6}$$

**Corollary 3.1** (clip-effect regime decomposition).
- $\Lambda_{\mathrm{clip}} \ll 1$: DDQN's clip preserves vanilla's
  argmax → outcome reflects only the magnitude-channel improvement
  ($\Lambda_m$ reduction). Q-STRUCTURED regime.
- $\Lambda_{\mathrm{clip}} \gtrsim 1$: DDQN's clip CAN corrupt agent's
  argmax → outcome harm despite magnitude improvement. Q-EXPLODED
  regime.

This is the second axis the empirical regime classifier required.

### Connecting $\sigma_{\mathrm{clip}}$ to $\Lambda_a$ heterogeneity

Substituting Lemma 5's leading-order term into (6.1.4) gives:
$$\sigma_{\mathrm{clip}}(s)^2 \;\gtrsim\; \mathrm{Var}_a\!\left[ p_{\neq}(s'_a) \cdot \sigma_T(s'_a) \cdot \eta(K) \right]. \tag{6.1.7}$$
The $\gtrsim$ inherits Lemma 5's one-sided lower bound: the
disagreement-driven Var$_a$ component is a lower bound on
$\sigma_{\mathrm{clip}}^2$; selection-biased residuals and σ_T·φ(K)
anisotropy contribute additively.

Since $p_{\neq}(s')$ depends on $\Lambda_a(s') = \sigma_T(s')\sqrt{2\ln K}/\Delta_T(s')$
through the non-linear $\exp(-1/\Lambda_a^2)$-shape of (9) applied to
the bootstrap target's argmax, the right side is **monotonically
increasing in $\mathrm{Var}_a[\Lambda_a(s'_a)]$ within the transition
band of $p_{\neq}$** (where $\Lambda_a$ is order-unity, $\sigma_T$
and $\eta(K)$ approximately constant across destinations).
$p_{\neq}$ is sigmoidal: $\to 0$ at $\Lambda_a \ll 1$ (no argmax
fragility), $\to (K-1)/K$ at $\Lambda_a \gg 1$ (uniform-random
argmax in extreme noise). In both extremes
$\mathrm{Var}_a[p_{\neq}] \to 0$ even as $\mathrm{Var}_a[\Lambda_a]$
may grow → the dependency is non-monotone globally, monotone only
within the transition band.

**Corollary 3.2** (qualitative). DDQN's clip corrupts agent's argmax
when the action-destinations $\{s'_a : a \in \mathcal{A}\}$ are
distributed across the transition band with heterogeneous $\Lambda_a$.
When all destinations are saturated to the same regime ($\Lambda_a \ll 1$
or $\gg 1$), $p_{\neq}$ varies little across $a$,
$\sigma_{\mathrm{clip}} \approx 0$, and DDQN's clip preserves argmax.

This identifies the discriminator between Q-STRUCTURED and Q-EXPLODED
at the upper Λ_m bracket: **heterogeneity of $\Lambda_a$ across
action-destinations**, not $\Lambda_a$'s magnitude alone.

**Logical-role note (load-bearing).** Theorem 3 and Corollary 3.2
support OPPOSITE-direction sufficient conditions, each using Lemma 5
in the right direction:

- **Theorem 3 (eq. 6.1.5) proves: small $\sigma_{\mathrm{clip}}$ ⇒
  argmax preserved.** The proof uses Popoviciu deterministically on
  $\sigma_{\mathrm{clip}}$ taken as a given quantity from definition
  (6.1.4); it does NOT invoke Lemma 5.
- **Corollary 3.2 motivates: large $\mathrm{Var}_a[\Lambda_a]$ ⇒
  $\sigma_{\mathrm{clip}}$ large ⇒ argmax CAN be corrupted.** Lemma 5's
  lower bound (6.1.2) is the correct direction here:
  $\sigma_{\mathrm{clip}}^2 \gtrsim \mathrm{Var}_a[p_{\neq} \sigma_T \eta(K)]$
  (eq. 6.1.7) says "$\sigma_{\mathrm{clip}}$ is at-least-this-much
  large when $\Lambda_a$ heterogeneity is large."

The empirical signature ($\sigma_{\Lambda_a}^{\mathrm{env}} \to d_{\mathrm{out}}$
cross-env correlation) runs through Corollary 3.2's
**corruption-side** chain, not through (6.1.5)'s preservation
threshold. Lemma 5 supports the corruption side; it does NOT support
verifying (6.1.5) directly (that would need an UPPER bound on
$\sigma_{\mathrm{clip}}$, which the present derivation doesn't
produce).

### Empirical operationalization

$\sigma_{\mathrm{clip}}(s)$ is not directly measurable, but its
qualitative signature IS:

- When $\sigma_{\mathrm{clip}}$ is large, DDQN's clip introduces
  state-dependent compression of $Q_d$ → the inter-state coherence
  of $Q_d$'s gradient drops vs vanilla's.

**Bridging assumptions (load-bearing).** The empirical signature
operates at scopes the algebraic derivation doesn't directly cover.
Two distinct bridging assumptions:

1. **Stationarity of bias-geometry across states.** The
   $\texttt{q\_inter\_state\_grad\_overlap\_late}$ proxy averages
   across the visited state distribution;
   $\sigma_{\mathrm{clip}}(s)$ is defined at one state $s$ across
   action-destinations. We assume envs with high cross-action
   $\sigma_{\mathrm{clip}}$ at typical states also exhibit
   cross-state heterogeneity (the env's bias-geometry structure is
   roughly stationary).
2. **Cross-seed-as-cross-action proxy ergodicity.**
   $\sigma_{\Lambda_a}^{\mathrm{env}}$ measures cross-seed SD of
   seed-mean $\Lambda_a$ (a per-cell scalar). For this to track
   $\mathrm{Var}_a[\Lambda_a(s'_a)]$ (cross-action heterogeneity at
   one state), seeds must sample state-distributions ergodically
   such that seed-mean Λ_a varies if and only if cross-action
   Λ_a varies. Two seeds with similar per-cell scalars at the
   same env can have very different cross-action heterogeneity;
   the proxy is structurally measuring inter-seed convergence
   variance, which under ergodicity correlates with cross-action
   variance but isn't the same object.

Both assumptions are plausible but neither is in the algebraic
derivation. The empirical operationalization is contingent on
both holding to within order-of-magnitude.

**Two empirical proxies, capturing orthogonal aspects of $\sigma_{\mathrm{clip}}$**:

- $\Delta_{\mathrm{smoothness}} := d\!\bigl[\texttt{q\_inter\_state\_grad\_overlap\_late}\bigr]_{\mathrm{DDQN} - \mathrm{VAN}}$
  (Cohen's $d$, per env). Captures DDQN's reduction of inter-state
  Q-coherence — the DOWNSTREAM signature of high $\sigma_{\mathrm{clip}}$.
- $\sigma_{\Lambda_a}^{\mathrm{env}}$ = cross-seed SD of per-cell
  $\Lambda_a^{\mathrm{cell}} = \texttt{q\_action\_std\_late} \cdot \sqrt{2\ln K} / (\texttt{q\_argmax\_margin\_late} + \varepsilon)$
  on the vanilla arm. Captures the UPSTREAM input variance
  $\mathrm{Var}_a[\Lambda_a(s'_a)]$ via cross-seed state-distribution variation.

**Prediction.** Envs where Theorem 3's $\sigma_{\mathrm{clip}}$ is
large should show $\Delta_{\mathrm{smoothness}} < 0$ AND high
$\sigma_{\Lambda_a}^{\mathrm{env}}$.

**Non-refuting calibration at a power-feasible scale** (8-env panel
at γ=0.999, see `findings_lambda_a_smoothness_third_axis_partial.md`
and `findings_theorem3_sigma_clip_validation.md`). The n=8 panel
could have refuted the theorem with a ρ near zero or wrong-signed;
it didn't. Causal grounding still requires within-env intervention:

- $\Delta_{\mathrm{smoothness}}$: Asterix has $d = -2.13$, the only
  env in the panel with $d \lesssim -1$. All other envs are in
  $[-0.46, +0.67]$. Discrete env-feature.
- $\sigma_{\Lambda_a}^{\mathrm{env}}$ vs $d_{\mathrm{out}}$: Spearman
  $\rho = -0.778$ ($p = 0.023$); without Asterix Spearman
  $\rho = -0.937$ ($p = 0.002$) — graded predictor.
- Partial $\rho(\sigma_{\Lambda_a}, d_{\mathrm{out}} \mid \Delta_{\mathrm{smoothness}}) = -0.669$ ($p = 0.069$) — orthogonal axes.

The 8-env panel is **calibration**, not a falsifying test. A clean
falsifying test of (6.1.5) requires either: (i) a within-env
intervention that manipulates $\Lambda_a$ heterogeneity directly
(e.g., action-duplicate $k > 1$ creating known cross-action symmetry,
predicting $\Delta_{\mathrm{smoothness}}$ shift); or (ii) an out-of-sample
$\Lambda_{\mathrm{clip}} > 1$ prediction on a held-out env not used
for calibration. Both deferred.

### Status

**What Theorem 3 proves**: a closed-form *sufficient* condition for
DDQN's clip preserving the agent's argmax (eq. 6.1.5), with an
$\Lambda_{\mathrm{clip}}$ ratio that scales with the heterogeneity of
$\Lambda_a$ across action-destinations (Corollary 3.2). Under
(A2)+(A3) the derivation is fully algebraic.

**What Theorem 3 does NOT prove**:
- That $\Lambda_{\mathrm{clip}} > 1$ ⇒ argmax corrupted (necessity).
  Theorem 3 is sufficiency only, like Theorems 1 + 2.
- That outcome $d < 0$ ⟺ $\Lambda_{\mathrm{clip}} > 1$. The mapping
  from argmax corruption to outcome is downstream of policy
  rollouts + reward distribution; out of scope per §10.
- A closed-form for $\sigma_{\mathrm{clip}}$ under FA-correlated noise.
  (A3) iid Gaussian is the same scope-limit as Theorem 2 (cf. §9.4).
  Under FA correlation, $\sigma_{\mathrm{clip}}$ takes a more complex
  form that the present derivation doesn't cover.
- A theoretical derivation of why specifically Asterix γ=0.999 has
  large $\sigma_{\mathrm{clip}}$ (the env-structural reason for
  heterogeneous action-destinations). This is an env-property,
  outside the algorithm's bias-geometry.
- A tight constant in (6.1.5). We adopt Popoviciu's $\sqrt{2(K-1)}$
  as the assumption-free deterministic bound; the proof currently
  treats the $K$ destination expectations as deterministic numbers.
  A sub-Gaussian probabilistic re-grounding would give the tighter
  $2\sqrt{2 \ln K}$ asymptote at large $K$ (e.g., $K=18$: Popoviciu
  $\sqrt{34} \approx 5.83$ vs $2\sqrt{2 \ln 18} \approx 4.81$), but
  requires an extra modelling commitment we don't make.
- A tight expression for $\mathbb{E}[\Delta_{\mathrm{clip}}(s')]$.
  The exact identity is
  $\mathbb{E}[\Delta_{\mathrm{clip}}] = \sigma_T \cdot \phi(K) + p_{\neq} \cdot \mathbb{E}[\Delta_a \mid \text{disagree}]$,
  where $\sigma_T \cdot \phi(K)$ is target's intrinsic max-of-K bias
  (always present, regardless of disagreement). Lemma 5 bounds only
  the disagreement-driven residual. If $\sigma_T(s'_a)$ is uniform
  across action-destinations, $\sigma_T \cdot \phi(K)$ is
  common-mode → contributes $0$ to $\mathrm{Var}_a$ → doesn't affect
  Theorem 3's argmax conclusion. But under
  $\sigma_T$-heterogeneity (the very condition that drives
  $\sigma_{\mathrm{clip}} \neq 0$ via (6.1.7)),
  $\sigma_T(s'_a) \cdot \phi(K)$ is ALSO anisotropic across $a$ and
  contributes to $\sigma_{\mathrm{clip}}^2$. Lemma 5's bound is
  therefore loose in BOTH absolute clip-error magnitude AND in the
  argmax-relevant $\mathrm{Var}_a$ quantity at the
  $\sigma_T$-heterogeneous boundary. The Theorem 3 conclusion still
  holds qualitatively (any anisotropy drives $\sigma_{\mathrm{clip}}$
  up) but the closed-form ratio between $\sigma_{\mathrm{clip}}$ and
  $\mathrm{Var}_a[\Lambda_a]$ requires both terms.

**Closes open gap §9.4 partially.** Theorem 3 provides the
"argmax channel under DDQN's clip" closed form; the FA-correlation
case remains open. The Asterix-vs-rest empirical pattern is now
theorem-articulated under (A2)+(A3); the unexplained piece is the
env-structural source of $\Lambda_a$ heterogeneity, which lives
outside the algorithm and is appropriately classed as a
substrate-empirical question.

## 7. Lemma 4 — Per-batch SGD gradient is B-invariant in expectation

**Lemma 4**. Under iid sampling from the replay buffer and standard
Bellman-MSE loss $L = \frac{1}{B} \sum_i (Q_\theta(s_i, a_i) - y_i)^2$:

$$\mathbb{E}\!\left[\nabla_\theta L\right] \;=\; \mathbb{E}_{(s,a,y) \sim D}\!\left[\nabla_\theta (Q_\theta(s, a) - y)^2\right] \tag{11}$$

is independent of batch size $B$. The variance of $\nabla_\theta L$ scales
as $\mathrm{Var}[\nabla_\theta L] = O(1/B)$.

**Proof reference.** Standard SGD result; Bertsekas-Tsitsiklis 1996,
§4. The expected gradient direction is the population gradient
direction; minibatch averaging reduces variance only.

**Corollary 4.1** (Batch-size invariance of the regime). Theorem 1's
$\Lambda_m$ does not depend on $B$. The expected regime classification
(bias-dominated vs reward-led) is therefore B-invariant.

**Caveat.** In non-convex stochastic optimization, finite-sample SGD
trajectories can escape unfavorable attractors via lucky high-variance
steps, especially at small $B$. So while expected divergence direction
is B-invariant, the *probability of escape* from bias-attraction during
a finite-T training run may depend on B. Theorem 1 covers the
expected-fixed-point regime; per-trajectory escape probability is
outside its scope.

## 8. Empirical anchors

**FR × MLP[64,64] × unshaped × γ=0.999** (canonical 1M corpus + warmup probe):

| Observable | Value | Source |
|---|---:|---|
| $\gamma$ | 0.999 | config |
| $K$ | 4 | env |
| $\phi(K) = \phi(4)$ exact | 1.029 | §2 |
| $\sigma_{\mathrm{V}}$ (q_action_std_late) | 0.018 | measured ($n=30$) |
| $\sigma_{\mathrm{D}}$ | 0.001 | measured |
| $\rho$ (training-phase) | $1.95 \times 10^{-4}$ | measured (195 / 1M) |
| $R_{\max}$ | 1.0 | env |
| $b_{\mathrm{V}} = \gamma \sigma \phi(K)$ | 0.0185 | Lemma 1 |
| Per-step reward signal $= \rho R_{\max}$ | $1.95 \times 10^{-4}$ | Lemma 3 |
| **$\Lambda_m^{(\mathrm{V})}$** | **94.9** | Theorem 1 (eq. 4) |
| Analytic asymptote $\gamma b/(1-\gamma)$ | 18.4 | Lemma 2 (eq. 2) |
| Observed $q_{\mathrm{late, V}}$ (FA-truncated) | $\approx 8$ | measured |
| **$\Lambda_m^{(\mathrm{D})}$** ($\phi_D \approx 0.1 \phi$) | **9.5** | Corollary 1.1 |
| Observed $q_{\mathrm{late, D}}$ | $\approx 1$ | measured |

$\Lambda_m^{(\mathrm{V})} \approx 95 \gg 1$ → Theorem 1 classifies the
vanilla regime as bias-dominated. Observed FA-truncated $q_{\mathrm{late}}$
sits below the Lemma-2 asymptote because the bias chain saturates the
FA capacity before reaching the analytic fixed point. The regime
classification (bias-dominated) is consistent with both the analytic
and empirical state.

DDQN's $\Lambda_m^{(\mathrm{D})} \approx 10$ is nominally still
bias-dominated, but Corollary 1.1's $\geq 5\times$ reduction maps to
$30\times$ reduction in absolute $q_{\mathrm{late}}$ (8 → 1) —
consistent with FA truncation of a smaller asymptote.

## 9. Open formalization gaps

The following items are **not** rigorously proven in this note:

### 9.1 Hasselt's iid Gaussian assumption (A1) violated by FA

In deep RL with shared-feature networks, $Q_\theta(s, a_1)$ and
$Q_\theta(s, a_2)$ share early-layer features and are NOT iid. The
correlation structure can both reduce $\sigma_{\mathrm{aniso}}$ (shared
features → similar per-action values) AND tighten Lemma 1's bound.

**Status.** The lemmas still hold as upper bounds (positive bias persists
under correlation), but the constant $\phi(K)$ is replaced by a smaller,
FA-dependent function. A rigorous treatment would specify a FA family
(e.g., bounded-Lipschitz 2-layer MLPs) and derive the correlated-noise
bound. Empirically the bound is conservative: observed FR vanilla bias
is $\approx 8$, theoretical FA-free Lemma-2 bound is $18$.

### 9.2 DDQN's exact $\phi_D(K)$

Hasselt 2010 §4.1 gives the bias formula under independent estimators.
Online and target networks are NOT independent; they share the
trajectory. The exact $\phi_D$ depends on the joint distribution
$(Q^{(1)}, Q^{(2)})$ which evolves with training. We treat
$|\phi_D| \leq 0.2 \phi(K)$ as a conservative empirical bracket.

**Status.** Closed form unavailable without specifying the
target-network update dynamics. Empirical estimation (per-corpus
measurement of $\phi_D$) is feasible but not done here.

### 9.3 Robbins-Monro convergence under bias

Theorem 1 invokes the Lemma-2 fixed point under the iterated update.
Robbins-Monro / Bertsekas-Tsitsiklis convergence theorems assume
unbiased gradient signals; the biased Bellman target violates this.
Convergence-to-fixed-point under biased updates is covered by a
different family of results (Tsitsiklis-Van Roy 1997 "deadly triad"
analyses; finite-time bounds in Anschel et al. 2017).

**Status.** Theorem 1's "expected fixed point" interpretation is
heuristic — under biased updates the iterates may not converge; they
may oscillate or diverge to the FA bound $L$. The Lemma-2 fixed point
is an upper bound on what the iterates could reach in the linearized
regime.

### 9.4 The argmax channel under FA correlation

Theorem 2's iid-Gaussian assumption gives a sufficient condition under
that noise model. Under FA-correlated noise, $\sigma_{\mathrm{aniso}}$
is smaller AND systematically biased toward specific actions (e.g., the
initialization's argmax for an under-trained Q). The argmax-preservation
condition becomes a more complex function of the feature representation.

**Status.** Partially closed by Theorem 3 (§6.1) for the iid case
extended to DDQN's clip. Theorem 2 covers vanilla's argmax-preservation
sufficiency; Theorem 3 covers DDQN's clip-induced argmax-corruption
sufficiency. Both rest on (A1)/(A2)+(A3) iid Gaussian assumptions.
FA-correlated noise remains open; the empirical substrate-level work
(`finding_asterix_g999_pc_mediator_triangle`,
`findings_pc_cross_env_smoothness`,
`findings_lambda_a_smoothness_third_axis_partial`) characterizes the
argmax channel qualitatively under FA correlation. Theorem 3 gives the
under-iid closed form that the empirical $\Delta_{\mathrm{smoothness}}$
proxies in the FA-correlated regime.

### 9.5 Missing parameters: $\alpha$, $T_{\mathrm{sync}}$, $L$

Theorem 1 does not include $\alpha$ (step size), $T_{\mathrm{sync}}$
(target update period), or $L$ (FA capacity). The effects of these:
- $\alpha$ controls convergence rate to the fixed point; too small slows
  the iteration toward the bias asymptote; too large amplifies noise.
- $T_{\mathrm{sync}}$ damps the bootstrap chain: at high
  $T_{\mathrm{sync}}$, the target is stale, slowing bias compounding.
- $L$ truncates the fixed point: $|Q_\infty| \leq L$ caps the bias
  regardless of the Lemma-2 algebraic bound.

**Status.** These can be folded into Theorem 1 as a refined statement:
$\Lambda_m \geq C(\alpha, T_{\mathrm{sync}}, L) \Rightarrow$ bias-dominated.
$C$ would absorb the missing parameters. This refinement is left to
future work.

## 10. What this note CAN'T prove and won't claim

- **Necessary conditions.** $\Lambda_m < 1$ doesn't guarantee that
  Q-learning converges to $Q^*$. Other failure modes (exploration
  starvation, FA mis-specification, non-stationary policy dynamics)
  exist. Theorem 1 is sufficiency for *bias-dominated divergence*, not
  necessity.

- **Outcome (policy performance) prediction.** $\Lambda_m$ governs the
  Q-magnitude channel; $\Lambda_a$ governs the argmax channel;
  $\Lambda_{\mathrm{clip}}$ (Theorem 3) governs DDQN's clip-induced
  argmax channel; outcome is downstream of argmax via ε-greedy rollouts
  + reward distribution. No clean closed-form mapping from
  $(\Lambda_m, \Lambda_a, \Lambda_{\mathrm{clip}})$ to outcome Δ —
  that's why the empirical regime classifier (Q-EXPLODED /
  Q-STRUCTURED / Q-COLLAPSED / CLIP-RATCHET) carries the publication
  work, not this theorem.

- **Source of $\Lambda_a$ heterogeneity across action-destinations**
  (Corollary 3.2). The env-structural reason why some envs (Asterix
  γ=0.999) have heterogeneous next-state $\Lambda_a$ while others
  (SI γ=0.999) have homogeneous next-state $\Lambda_a$ lives outside
  the bias-geometry — it's a property of the env's reward landscape
  and dynamics, not the algorithm. Theorem 3 takes this heterogeneity
  as input, doesn't predict it.

- **CLIP-RATCHET regime** (Snake γ=0.99). DDQN's clip introduces
  σ-asymmetry across seeds — a different mechanism not captured by
  Lemma 3's bias-reduction framing. Theorem 1 + 2 don't apply here.

## 11. References

- Hasselt, H. (2010). *Double Q-learning*. NeurIPS. — Proposition 1 +
  §4.1 give Lemma 3 + the exact bias formula under iid.
- Hasselt, H., Guez, A., & Silver, D. (2016). *Deep reinforcement
  learning with double Q-learning*. AAAI. — Empirical Atari claim,
  online/target version of Lemma 3.
- Thrun, S., & Schwartz, A. (1993). *Issues in using function
  approximation for reinforcement learning*. — Upper bound on
  Q-overestimation; Lemma 1 reference.
- David, H. A., & Nagaraja, H. N. (2003). *Order Statistics* (3rd ed.).
  Wiley. — Theorem 10.5.4 gives closed-form moments of max-of-K
  Gaussians; underpins Lemma 1's exact $\phi(K)$ values.
- Bertsekas, D. P., & Tsitsiklis, J. N. (1996). *Neuro-Dynamic
  Programming*. — Robbins-Monro convergence; Lemma 4's
  B-invariance of expected gradient.
- Tsitsiklis, J. N., & Van Roy, B. (1997). *An analysis of temporal-
  difference learning with function approximation*. IEEE TAC. — Deadly
  triad; finite-FA convergence analyses.
- Anschel, O., Baram, N., & Shimkin, N. (2017). *Averaged-DQN: variance
  reduction and stabilization for deep reinforcement learning*. ICML.
  — Bias-vs-variance tradeoff under biased Q-updates.
- Watkins, C. J. C. H. (1992). *Q-learning*. Machine Learning. —
  Original convergence proof for tabular Q-learning.

## 12. Empirical-corroboration anchors (this study)

- `findings_fr_gamma_why_transfers_to_minatar.md` — joint partial
  ρ(γ, jens | self_ref + σ_action) shrinks 76-87% at SI/Asterix/Breakout
  γ=0.999 (Lemma-2 mediator structure transfers).
- `findings_warmup_anchor_intervention.md` — v2 vanilla outcome flat
  across warmup ∈ {100, 10k, 100k} (warmup raises ρ slightly; Λ_m stays
  ≫ 1; bias regime persists). Theorem 1 prediction held empirically.
- `finding_shaping_decouples` — under reward shaping, ρ → ρ' ≫ ρ
  densifies signal; Λ_m drops; vanilla outcome recovers. Theorem 1
  corroborated.
- `finding_acrobot_chain_does_not_replicate` — Acrobot γ=0.999 has
  dense -1 reward → ρ ≈ 1, Λ_m ≈ 0.03 ≪ 1; vanilla outcome doesn't
  collapse. Theorem 1 corroborated.
- `findings_si_corroborates_regime_classification.md` — SI γ=0.999
  fits Q-STRUCTURED (Λ_m moderate); DDQN helps via Corollary 1.1.

**Falsifiable prediction (in flight):** batch-size sweep at FR
γ=0.999 × MLP × 1M, B ∈ {128, 512, 2048} (B=32 anchor from existing
warmup probes). Lemma 4 → Corollary 4.1 predicts vanilla `jensen_gap`
≈ invariant in B. Refutation criterion: sig negative trend
ρ(B, jens) ≤ -0.5. See `experiments/configs/fr_batch_size_sweep.yaml`.
