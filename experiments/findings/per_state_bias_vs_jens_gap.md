# Per-state cumulative bias vs bias-at-start (jens_gap)

## Hypothesis tested

User's principled reasoning: bias-at-start (`predicted_q_at_start − mc_return`,
the basis of `jensen_gap`) is confounded — long trajectories have s_0
sitting at maximal chain depth, so bias-at-start mixes per-step
bias-rate with chain-length. The de-confounded measure should be the
mean-per-state cumulative bias along the visited trajectory:
`mean_t [Q(s_t) − G(s_t)]` where `G(s_t)` is the realized return
from t onward.

If cumulative bias is the operative variable, the chain-traced mean
should be a tighter predictor of Δ_outcome than bias-at-start.

## Substrate change

Added per-state probes to `eval_episode`/`eval_burst` (corroborate_rl
substrate, `dqn/eval.py`):
- `predicted_q_per_step[b, k, t]`: max-Q at each visited state
- `mc_return_from_step[b, k, t]`: realized discounted return from t
- `active_per_step[b, k, t]`: 1.0 while episode running

Added measurables:
- `mean_per_state_cumulative_bias_late` (scalar, late-window mean)
- `mean_per_state_cumulative_bias_per_burst` (per-burst array)

Wired into `dqn_default_measurables()`.

## Sweeps

Three small sweeps, GPU (RTX 5080), 30 seeds × 200k steps each:
- FR γ=0.99 (5 min)
- Acrobot γ=0.999 (9 min)
- MetaMaze γ=0.999 (5 min)

Trace data persists at `experiments/data/per_state_bias_probe_{fr,acrobot,metamaze}/`.

## Result: hypothesis NOT supported

| env | within-cell (n≈30) | per-burst pooled (n≈170-270) |
|---|---|---|
| | r(Δy, Δjens) / r(Δy, Δpstate) | r(Δy, Δjens) / r(Δy, Δpstate) |
| FR γ=0.99 | -0.83 / -0.47 | -0.79 / -0.80 |
| Acrobot γ=0.999 | -0.53 / -0.39 | -0.89 / -0.63 |
| MetaMaze γ=0.999 | -0.43 / **-0.67** | -0.58 / -0.42 |

Per-burst pooled (higher n) consistently shows jens at least as good
or better. Partial Spearman ρ(Δy, Δjens | Δpstate) survives (-0.21
to -0.45, all p<0.001) on all three envs; the per-state partial is
small (-0.18 to -0.29) or even positive (+0.17 on MetaMaze).

**Within-cell on MetaMaze did show per-state dominate** (joint OLS
β_pstate=-2.65 p<0.001), but the joint regression coefficients
suggest multicollinearity ping-pong (β_jens flips POSITIVE in same
regression). At n=30 with two tightly-coupled regressors, this is
not a robust finding.

## Why bias-at-start wins

Outcome (`eval_best_burst_mean`) is measured at s_0 — it's the
realized discounted return from episode start. `jensen_gap` is the
agent's "delusion" at s_0. Both quantities live at s_0 by
construction, so they should correlate tightly — DDQN reducing the
s_0 delusion is essentially aligning predicted Q with realized
return at the same state.

The mean-per-state measure averages over OTHER visited states whose
bias is at smaller (shallower) remaining-chain depths. Useful for
mechanistic accounts; noisier as a one-number summary when the
target you're correlating with is itself a start-state quantity.

## Stopping point

The bias-at-start vs per-state question is closed. `jensen_gap` is
the proximal predictor of `Δ_outcome` on the textbook regimes
(FR γ=0.99, Acrobot γ=0.999, MetaMaze γ=0.999).

Reproduction:
- Substrate: `src/corroborate_rl/corroborate_rl/dqn/eval.py`,
  `src/corroborate_rl/corroborate_rl/dqn/measurables.py`
  (mean_per_state_cumulative_bias_{late,per_burst}).
- Configs: `experiments/configs/per_state_bias_probe_{fr,acrobot,metamaze}.yaml`.
- Analysis: `scripts/per_state_bias_analysis.py`.

The deeper question — what's UPSTREAM of bias-at-start reduction
(the algorithmic mechanism)? — is distinct and remains open. DDQN's
algorithmic correction magnitude (`target_max −
target_q_at_online_argmax` per step) is the natural next probe.
