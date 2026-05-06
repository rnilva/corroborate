# Continuous env_reward_polarity measurable — endogenous polarity proxy

**Date:** 2026-05-05
**Followup to:** `polarity_proof_findings.md` (formal proof of polarity-
flips-eff_h-mediator-sign with categorical hand-coding).

## Headline

> **`env_reward_polarity` measurable** — per-cell scalar Pearson r between
> `episode_length` and `mc_return` over (burst × eval-episode) — recovers
> the hand-coded categorical polarity at Spearman ρ = +0.88, p = 0.021,
> with 6/6 sign agreement on testable envs.

## Per-env continuous polarity (vanilla baseline cells, 1410 total)

| env | n | mean polarity | sd | range | categorical | match |
|---|---|---|---|---|---|---|
| MountainCar-v0 | 10 | **−0.996** | 0.003 | [−1.000, −0.993] | −1 | ✓ |
| Acrobot-v1 | 30 | **−0.948** | 0.047 | [−0.996, −0.846] | −1 | ✓ |
| FourRooms-misc | 570 | **−0.903** | 0.032 | [−1.000, −0.826] | −1 | ✓ |
| SpaceInvaders-MinAtar | 30 | +0.266 | 0.317 | [−0.438, +0.814] | +1 | ✓ |
| Asterix-MinAtar | 30 | **+0.472** | 0.091 | [+0.314, +0.643] | +1 | ✓ |
| Breakout-MinAtar | 30 | **+0.991** | 0.004 | [+0.982, +0.997] | +1 | ✓ |

Spearman ρ(continuous, categorical) = +0.878, p = 0.021 (n_envs=6).
Sign agreement: 6/6, binomial p = 0.016.

## Why this matters

The categorical polarity proof needed a hand-coded env catalogue
(`POLARITY = {'Acrobot-v1': 'goal', ...}`). That's substrate-specific
human knowledge bolted into the analysis script.

The continuous version is **endogenously computable** from any cell's
trace data — no env catalogue needed. It generalizes to any new env
the substrate produces. Just compute `env_reward_polarity` per cell;
its sign and magnitude tell you the env's reward structure.

This is exactly the pattern the framework's docs recommend: prefer
endogenous measurables over HP-derived or hand-coded scope predicates.
Compare:
- HP-derived (`gamma`): not predictive cross-env per earlier analyses
- Hand-coded categorical polarity: required env catalogue, only
  applies to known envs
- **Endogenous continuous polarity: works on any env, derived from
  trace data, registered as `@measurable`**

## Per-cell variation

Per-cell σ within env is small (0.003-0.317), confirming polarity is
mostly env-determined. But there IS some cell-level variation — most
visible on SpaceInvaders (σ=0.317, range [−0.44, +0.81]). This may
carry signal: cells where DDQN's policy "mode" dominates may push the
within-env polarity slightly. Worth investigating as a per-cell
moderator if the pooled-env mean version proves underpowered.

## Bridge implication

The two paired bridges sketched in `polarity_proof_findings.md`:
```python
@claim_bridge(scope=POLARITY_GOAL & NOT_Q_EXPLODED, ...)
def eff_h_mediates_g_link__goal_envs(...): ...

@claim_bridge(scope=POLARITY_SURVIVAL & NOT_Q_EXPLODED, ...)
def eff_h_mediates_g_link__survival_envs(...): ...
```

Can now use `pl.col('env_reward_polarity') < -0.3` and
`pl.col('env_reward_polarity') > +0.3` as endogenous scope predicates,
not hand-coded env name `is_in(...)` lists. This is the framework-
proper form.

A more refined alternative: a SINGLE bridge with the polarity as a
continuous moderator in a meta-regression, predicting `slope_y_on_m`
on the polarity covariate. The expected slope is positive (more
positive polarity → more positive coupling slope). This expresses
the polarity finding as ONE typed claim instead of two paired ones.

## What's missing for a complete proof

The continuous polarity here was computed on cells from local traces
only (5 envs from expectile_3way + 4 MinAtar from sync_intervention =
6 distinct envs). To validate against the full polarity panel (8
envs in the formal proof), need:

- CartPole, DCC, DeepSea, MemoryChain, UmbrellaChain, MetaMaze, Pong
  cells with `episode_length` traces — would require cloud restore
  (CartPole's traces are in ddqn corpus; MetaMaze in
  gamma_sweep_metamaze*; etc.)

For now the continuous measurable is authored; the full validation
across all 8 polarity-panel envs is deferred until traces restore.

## Reproduction

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. uv run python \
    experiments/findings/sync_curve_breakout/run_polarity_continuous.py
```

Output: `polarity_continuous_panel.json`.
