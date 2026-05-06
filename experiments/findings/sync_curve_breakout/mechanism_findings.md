# DDQN's mechanism: bootstrap conservatism at high sync_period

**Date:** 2026-05-05
**Companion to:** `findings.md` (link panel) and `outcome_findings.md`
(per-sync paired_g).

## The question

Why does DDQN HURT outcome at sync=10000 on Breakout-MinAtar (g=−0.71,
p=5e-4) where Q-explosion is suppressed? The Hasselt theorem assumes
DDQN reduces overestimation; here it INCREASES bias on average. What's
the algorithmic mechanism?

## The data

Per-arm per-burst diagnostics from `traces.parquet` step-level columns
binned into 20 bursts of 50k steps each, on Breakout-MinAtar, 30 paired
seeds × 4 sync values.

### Headline: argmax disagreement and Q-magnitude grow with sync

| sync | argmax_disagree @ b0 | DDQN Q / vanilla Q @ b0 | DDQN Q / vanilla Q @ b19 |
|---|---|---|---|
| 100 | 17.5% | 0.84 | 0.60 |
| 1000 | 23.6% | 0.86 | 1.11 |
| 3000 | 36.0% | 0.83 | 0.57 |
| **10000** | **61.4%** | **0.64** | 1.13 |

**Interpretation.** argmax disagreement is the rate at which the online
network's argmax differs from the target network's argmax — i.e. how
often DDQN's bootstrap rule actually picks a *different* action than
vanilla's. With slow sync, online drifts far from target between syncs,
and disagreement skyrockets in the early bursts (62% at sync=10000 b0
vs 17% at sync=100 b0).

When they disagree, DDQN's bootstrap target is **strictly lower** than
vanilla's: vanilla picks `max_a Q_target(s',a)`, DDQN picks
`Q_target(s', argmax_online Q_online(s',a))` ≤ max. So at sync=10000 b0,
DDQN's targets are systematically smaller for **62% of training steps**.
Result: DDQN's Q grows to only 64% of vanilla's by b0 (the largest early
suppression in the panel).

## Cross-seed σ_Q does NOT drive sync=10000 failure

The original Q-amplification hypothesis from
`findings_q_amplification_cartpole.md` predicted DDQN > vanilla in cross-
seed σ_Q. Empirically on Breakout:

| sync | σ_Q ratio (DDQN/vanilla) @ b8 | @ b19 |
|---|---|---|
| 100 | 0.42× | 0.52× |
| 1000 | 0.69× | 1.44× |
| 3000 | 0.75× | 0.69× |
| 10000 | **1.33×** | **1.07×** |

sync=10000 shows only mild cross-seed amplification (1.07-1.33×) —
*much smaller* than sync=1000 late-burst amplification (1.44×). Yet
sync=10000 is where DDQN HURTS most. **Q-amplification per the CartPole
finding is real but is not the dominant story on Breakout-at-sync=10000.**

## The bootstrap-conservatism mechanism

1. **Slow sync → high online-target divergence early.** sync=10000 means
   only 5 target syncs in the first 50k training steps. Online evolves
   rapidly during the burn-in but target stays stale — argmax disagreement
   at burst 0 is **62%** at sync=10000 vs **17%** at sync=100.

2. **DDQN's bootstrap rule punishes disagreement.** Every time
   `argmax_online ≠ argmax_target`, DDQN's bootstrap value ≤ vanilla's
   (strictly less when the action distribution favors a different action).
   At 62% disagreement, this is the dominant case.

3. **Smaller bootstrap targets → slower Q growth → more conservative
   greedy policy.** DDQN's online_max_q at sync=10000 burst 0 is 0.11 vs
   vanilla's 0.18 — **DDQN is at 64% of vanilla's Q magnitude after just
   50k steps**, the largest early-Q gap in the entire sync panel.

4. **Conservative early policy → worse exploration → permanent disadvantage.**
   DDQN's eval-time policy at burst 0 returns MC=0.64 vs vanilla's 1.55
   (60% worse). The trajectory deficit never closes — DDQN MC@b19=3.45
   vs vanilla 3.93. By burst 16-19, DDQN's Q has caught up and even
   overshot vanilla's (1.13× ratio), but the policy quality already lost
   the race. **Q catches up; policy doesn't.**

5. **At sync=100, this same mechanism (DDQN suppressed early) DOES NOT
   hurt** — because at sync=100, vanilla's Q-explosion is so dominant
   that its early policy is also degraded by Q-noise. DDQN's conservatism
   isn't a relative liability there. Hence the null at sync=100.

6. **At sync ∈ {1000, 3000} (Goldilocks):** disagreement is moderate
   (24-36% at b0), DDQN's early Q-suppression is mild (0.83× of vanilla),
   and the resulting stability bonus exceeds the policy-conservatism
   penalty. Outcome g≈+0.2 ns.

## What "DDQN helps Hasselt premise active" actually requires

The bridge author should require all four:

1. `argmax_disagree_rate @ early bursts` is moderate (not too high)
2. DDQN's `online_max_q` early ratio is not too suppressed (>0.75 of vanilla)
3. Late-burst Q does not amplify past vanilla (DDQN/vanilla < 1.0)
4. Mean Δ_outcome > 0 (the actual evaluation criterion)

The single hyperparameter `sync_period` controls all four jointly:
- Too small → vanilla Q-explosion overwhelms (1 fails on σ-front)
- Too large → bootstrap conservatism cripples early policy (2-3 fail)
- Just right → moderate disagreement, modest Q-suppression, stable late
  trajectory (1-3 satisfied)

The Goldilocks band is **sync ∈ [1000, 3000] on Breakout** — outside it,
two distinct failure modes engage.

## Implication for cross-env scope

The `findings_q_amplification_cartpole.md` failure mode (σ_Q amplification)
applies to CartPole. The Breakout-at-sync=10000 failure mode is bootstrap
conservatism (Q under-suppression early). They're distinct.

When Freeway+SI complete:
- If Freeway also shows large early argmax disagreement at sync=10000
  with DDQN Q-ratio < 0.7, expect bootstrap-conservatism harm there too.
- If SI shows smaller disagreement (different action structure, fewer
  actions to disagree on?), DDQN at sync=10000 might not hurt SI.

## Reproduction

```bash
PYTHONPATH=. uv run python experiments/findings/sync_curve_breakout/run_mechanism_analysis.py
```

Output: `mechanism_panel.json`. Requires the Breakout sync=100 trace
shards restored from cloud (~3.6 GB):

```bash
PYTHONPATH=. uv run python -c "
from pathlib import Path
from corroborate.corpus.cloud import restore
restore(Path('experiments/data/minatar_1M'), files=[
    'tmp/arm002__Breakout-MinAtar__vanilla_dqn__traces.parquet',
    'tmp/arm003__Breakout-MinAtar__ddqn__traces.parquet',
])
"
```
