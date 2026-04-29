# Findings — DDQN case study

Empirical findings from the DDQN acceptance corpus. Dated entries
to track when claims were authored vs. observed.

---

## 2026-04-29 (revised same day) — Substrate units bug + HP-conditioning of solve verdicts

### Bug

The framework's eval (`rl/dqn/eval.py:60`) records the *discounted*
Monte-Carlo return: `mc_return = Σ_t γ^t * r_t`. Literature solve
thresholds (gymnasium docs, Y&T 2019, Osband 2019) are *raw*
episodic return. Initial `SOLVE_THRESHOLDS` table used raw values
against the framework's discounted output — units mismatch.

Concrete symptom: CartPole-v1 baseline `outcome.eval_best_burst_mean
= 99.343` across all 200k corpus runs and across 178/180 cells in
a 36-config × 5-seed HP sweep. The exact value is `100 *
(1 - 0.99^500)` — the discounted return for a 500-step max-length
CartPole episode at γ=0.99. The agent IS reaching the optimal
policy at peak; it's just reported in discounted units. The raw
threshold of 475 (gymnasium docs) was unreachable in discounted
units (max ≈ 99.34).

### Fix

`rl/env_solve_thresholds.py` updated: each Tier-1 / Tier-2 entry
now carries the discounted-equivalent threshold at γ=0.99 with
the conversion documented in `source`:
- CartPole-v1: 475 → 99.0
- Acrobot-v1: -100 → -63.4
- MountainCar-v0: -110 → -67.3
- MinAtar (4): Y&T 50%-baseline × discount factor (≈ 0.2 for L=500)
- bsuite envs: slight downward adjustment (γ^L attenuation)

### Convergence-class shifts (200k corpus, baseline arm)

| Env | Old class | New class |
|---|---|---|
| Breakout-MinAtar | unsolved | **solved** (best=1.0, final=1.0) |
| CartPole-v1 | unsolved | **partial** (best=0.80, final=0.20) |
| MountainCar-v0 | solved | **unsolved** (-76.7 < tighter -67.3) |
| Acrobot-v1 | solved | solved |
| Catch / DeepSea / DiscountingChain / UmbrellaChain | solved | solved |
| Asterix / Freeway / MNISTBandit | unsolved | unsolved |

Solved subset is now 6 envs (was 6 — different composition).

### CartPole HP sweep — vanilla DQN solves CartPole with right HPs

Pre-registered hypothesis: a different HP regime should solve
CartPole at 200k *without* any new mechanism (DDQN, PER, n-step,
etc.). Tested on the 36-config × 5-seed sweep:

- **2 of 36 configs achieve final_solve = 1.0** — every seed
  stably solves at end of training. Both: `lr=1e-3, capacity=50k,
  sync=100`, varying batch_size ∈ {32, 64}.
- **11 of 36 configs have final_solve ≥ 0.5** — stable solve.
- **19 of 36 configs have best_solve ≥ 0.8** — peak is solved
  but degrades by final eval.
- The original corpus's HP config (`capacity=10000`,
  `batch_size=32`, `lr=1e-3`, `sync=100`) gets best_solve=1.0,
  final_solve=0.0 — **the agent solves CartPole at peak but
  catastrophically forgets**. Increasing replay capacity to 50k
  fixes the stability problem. Mechanism-level unchanged.

### Reading

1. **Vanilla DQN at appropriate HPs solves CartPole at 200k.**
   No new mechanism (DDQN, PER, n-step, dueling) is needed.
   Capacity=50k is the load-bearing HP for stability; the
   original corpus used 10k.

2. **The earlier convergence audit was contaminated by two bugs**:
   raw-vs-discounted units mismatch + HP-undertuning. The "5 of 13
   thresholdable envs are unsolved" finding from 2026-04-29 morning
   was over-stated. Corrected: 4 of 13 are unsolved (Asterix,
   Freeway, MNISTBandit, MountainCar), and at least CartPole is
   HP-fixable without mechanism changes.

3. **The §3 verdict pattern survives the correction.** On the
   new solved subset (6 envs), DDQN's mechanism g = -0.925, outcome
   g = -0.032 — directionally identical to the morning's finding.
   The headline conclusion (DDQN's bias-reduction activates but
   doesn't propagate to outcome) is robust to the threshold
   correction.

4. **Implication for any future DDQN claim**: the existing 200k
   corpus's results are conditional on `capacity=10k` HPs that
   don't reach stable solve on multiple envs. A re-run of DDQN at
   the better HPs (capacity=50k, etc.) might shift the outcome
   verdict. Future intervention studies should sweep HPs first to
   establish a converging baseline before testing mechanism-level
   interventions.

### Reproduction

```
# Convergence audit with corrected discounted thresholds:
uv run python experiments/convergence_audit.py --total-steps 200000

# CartPole HP sweep (15 min on CPU):
uv run python experiments/cartpole_hp_sweep.py
```

---

## 2026-04-29 — DDQN's bias-reduction mechanism activates on
## converged envs, but does not propagate to outcome.

### Setup

- Corpus: `runs_with_mediators.parquet`, 1080 cells at total_steps
  = 200000, 18 envs × 30 seeds × 2 arms (vanilla / DDQN).
- Outcome path: `outcome.eval_best_burst_mean` (Hasselt-convention
  reporting).
- Mechanism path: `mechanism.jensen_gap` (Hasselt 2010 §3 —
  empirical positive Jensen bias `max(0, mean(predicted_q_at_start
  − mc_return))` over eval-burst arrays).

### Convergence audit

Per-env solve thresholds from `rl/env_solve_thresholds.py`
(literature: gymnasium docs + Osband 2019 / bsuite scoring;
derived: 50% of Young & Tian 2019 MinAtar-DQN baseline). Applied
via `rl/convergence.classify_envs` to the baseline arm at 200k.

| Class | n | Envs |
|---|---|---|
| solved (final-mean ≥ thresh in ≥50% of cells) | 6 | Acrobot-v1, Catch-bsuite, DeepSea-bsuite, DiscountingChain-bsuite, MountainCar-v0, UmbrellaChain-bsuite |
| partial (0 < solve rate < 0.5) | 2 | MemoryChain-bsuite, SpaceInvaders-MinAtar |
| unsolved (final-mean ≥ thresh in 0 cells) | 5 | **CartPole-v1**, all 4 MinAtar-Atari (Asterix, Breakout, Freeway, MNISTBandit-bsuite) |
| absent (no defensible threshold) | 5 | BernoulliBandit, FourRooms, GaussianBandit, MetaMaze, Pong |

Critically, **5 of 13 thresholdable envs are not converged at
200k**, including CartPole-v1 (vanilla DQN reaches mean 98 / 500,
threshold 475). Catastrophic forgetting is visible in many envs
(Freeway 0.93 → 0.14, GaussianBandit 9.1 → -15.5).

### §3 verdict pattern shifts under convergence-conditioning

| Edge | All 18 envs | Solved 6 envs |
|---|---|---|
| Mechanism g (Δ jensen_gap) | -0.349 (I²=0.86) | **-0.985 (I²=0.93)** |
| Outcome g (Δ best_burst_mean) | +0.086 (I²=0.68) | **-0.025 (I²=0.41)** |

### Reading

1. **Hasselt's bias-reduction mechanism is partly vindicated.**
   On the converged subset, DDQN reduces the Jensen overestimation
   gap by approximately one standard deviation (g = -0.985) — the
   mechanism activates strongly. The framework's gap-grounded
   Scope was right that the gap is the load-bearing scope-defining
   feature; it just had to be conditioned on convergence to surface.

2. **The link to outcome is empirically broken.** Even on envs
   where the bias-reduction activates, the outcome effect is
   essentially zero (g = -0.025, slightly negative). Bias
   reduction does not propagate to better return on this corpus.

3. **The "DDQN modestly helps" finding from the unrestricted
   analysis is a convergence artifact.** The +0.086 outcome g
   comes from underconverged envs where DDQN's "help" reflects
   noisier mid-trajectory dynamics rather than improved final
   policy. Once we restrict to envs where vanilla DQN has reached
   a learned policy, DDQN's outcome contribution disappears.

### Honest scope of this claim

- **Corpus-conditional.** The finding is for total_steps=200k, on
  this 18-env grid, with this set of HPs. We do not claim DDQN's
  bias-reduction-without-outcome-propagation generalizes to
  longer training horizons or different env distributions.
- **Pearl-rung 2 unconditional for arm-effect verdicts** (paired
  Hedges' g across seeds), Pearl-rung 1 for the link (Pearson r
  across envs of per-env effect sizes). The link is observational;
  the dowhy backdoor estimates of mediator → outcome (in the
  `dowhy_ddqn` smoke) confirm at rung-2-conditional that
  jensen_gap → outcome is below threshold even with backdoor
  adjustment.
- **No held-out validation**, since the existing corpus has been
  fully explored. Any sharper claim would need a fresh sweep on
  the held-out env subset (12 of 18 envs reserved per
  `rl/env_splits.py` once that ships).

### Implications for next steps

- The framework's gist ("find scope, find causal chain") survives
  this case study. The framework correctly finds that the gap is
  the scope-defining feature *when scope is properly conditioned*.
  Where the framework's verdict diverges from the literature
  reading is on the chain: the link `gap → outcome` is empirically
  broken even when the upstream `arm → gap` mechanism activates.
- "What's DDQN's actual gap?" — closed (or shelved) by accepting
  this terminal verdict: the bias-reduction mechanism IS what DDQN
  does, but on this corpus that mechanism doesn't translate to
  outcome.
- The next study should set DDQN as the default arm and intervene
  on other components (sync period, n-step return, PER) to
  characterize the effects that DO propagate to outcome at this
  training horizon.

### Reproduction

```
uv run python experiments/convergence_audit.py --total-steps 200000
```
