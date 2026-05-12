# Cross-env link power: REACH-polarity has structural predictor, SURVIVE doesn't

## Question

Within the CLAIM 17 link-active scope (bounded Q + bf > 0.5 +
mech active + standard config), what predicts the *magnitude* of
DDQN's outcome benefit (mean_dY) across envs?

## Method

- Filter `ddqn` corpus to MODULE_SCOPE (no bsuite) +
  CLAIM 17 scope.
- Per (env, hp_config), aggregate over seeds: paired Δ_jens,
  Δ_outcome, plus env-feature averages (effh, bf, q_div).
- **Strict mech-HELD filter**: paired-t p<0.05 AND frac<0 ≥ 0.65.
  This rules out configs where mean Δ_jens < 0 is driven by a few
  outlier seeds (~50% of seeds split, mech precariously held).
  CartPole's "negative mean_dY" reading from looser filtering was
  largely an artifact of unstrict mech-HELD configs.
- Stratify envs by polarity class:
  - **REACH** (negative env_reward_polarity, shorter-is-better):
    FourRooms, Acrobot, MetaMaze, MountainCar.
  - **SURVIVE** (positive polarity, longer-is-better): CartPole,
    Asterix-MinAtar, Breakout-MinAtar, Freeway-MinAtar,
    SpaceInvaders-MinAtar.

## Result

**REACH (n_envs=4)** — chain-amplifier theory's textbook prediction
fires cleanly at the cross-env level:

| env | mean_dY | effh | mean_dJ |
|---|---|---|---|
| MetaMaze γ=0.999 | +2.13 | 110 | -7.7 |
| Acrobot γ=0.99/0.95 | +0.31 | 32 | -1.1 |
| FourRooms γ=0.99 | +0.11 | 38 | -0.3 |
| MountainCar γ var | -0.004 | 40 | -1.2 |

- r(mean_dY, **effh**) = **+0.975** (p=0.025)
- r(mean_dY, **mean_dJ**) = **-0.986** (p=0.014)

Both are env-structural (effh determined by γ × bf) and
cumulative-bias proxies, tightly correlated. The chain-depth
ordering predicts cross-env link power.

**SURVIVE (n_envs=4 strict-held)** — no clean structural predictor:

| env | mean_dY | effh | mean_dJ |
|---|---|---|---|
| SpaceInvaders | +2.56 | 70 | -33.8 |
| Breakout | +0.21 | 37 | -21.9 |
| Asterix | +0.06 | 53 | -11.4 |
| CartPole | 0.00 (saturated) | 73 | -18.0 |

- r(mean_dY, effh) = +0.408 (ns)
- r(mean_dY, mean_dJ) = -0.903 (Pearson, p=0.097, borderline)

Effh doesn't structurally order SURVIVE envs. Observed |Δ_jens|
(empirical bias-reduction magnitude) tracks mean_dY moderately.

## Interpretation

**Polarity-class is a moderator of which env-feature drives
cross-env link power.** On REACH envs, longer chain → more
bias compounding → bigger DDQN outcome benefit at the env-mean
level (the textbook DDQN paper story). On SURVIVE envs, the
chain-amplifier integrates bias differently: every-step positive
reward + Q overestimation produces uniform Q inflation that
doesn't always degrade policy → DDQN's correction doesn't
reliably translate to outcome at structural levels. Empirical
|Δ_jens| still has signal but env-features don't.

**CartPole's negative-dY apparition was an artifact** of
permissive mech-HELD filtering. Under strict filter
(paired-t p<0.05, frac<0 ≥ 0.65), the only CartPole config that
qualifies is at the env's outcome ceiling (saturated, mean_dY=0).
The earlier `cartpole_sync_1k` mean_dY=-0.12 reading came from a
config where ~50% of seed-pairs have Δ_jens > 0 — mech
precariously held, not a real "mech-HELD ∧ link-failed" signal.

## Bridge

`effh_predicts_link_power__reach_envs` in `ddqn/`.
Scope: CLAIM 17 + REACH (env_reward_polarity < -0.3).
Analysis: meta_regression_per_burst with effective_horizon as
covariate. HELD when β(effective_horizon) ≥ 0.005 AND
significant. Predicted direction `a_gt_b`.

## Reproduction

- Cross-env analysis script (inline in conversation log).
- Bridge: `experiments/findings/ddqn/:effh_predicts_link_power__reach_envs`.
- Cache: `experiments/data/cache/dqn_bridges.parquet` (15110
  cells, used while ddqn cache rebuilds).
