# Dormancy case studies — DDQN helps despite Jensen-premise inactive

## What this asks

The dormancy bridge (`ddqn_refuted_when_dormancy_fires`) claims:
**when the Jensen-bias premise is dormant (`v_jens ≤ 0` for some non-trivial
floor `σ_Q × √(2 log |A|)`), DDQN's bias-correction has nothing to operate
on, so Δ_outcome ≈ 0 on dormant cells.**

This file lists concrete configs that contradict the necessary-condition
reading.

## Why config-level, not seed-pair-level

The framework's seed-pairing critique applies here too: DDQN seed-N's
trajectory diverges from vanilla seed-N's from step 1, so pair-Δ doesn't
extract within-seed information. The natural unit of comparison for a case
study is the **config** `(env, sync_period, total_steps, eval_every)` —
aggregate the 30 seeds in each arm, compare mean outcomes via Welch t.

(Earlier seed-pair-level inspection on the same data turned up 54 "rock-
solid" pair cells and 16 with Δ_o ≥ 1.0 outcome point — those were
between-seed noise reading like signal. At config level, mean Δ_o on
Breakout flips negative; only one config survives. Documented here to flag
that seed-pair case-study counts inflate when seed Δ_o has σ comparable to
mean Δ_o, which is exactly the regime where the seed-pairing critique
warns of phantom signal.)

## Scope construction

Aggregate per `(env, sync_period, total_steps, eval_every)` × arm:
- `o_mean = mean(eval_best_burst_mean over seeds)`, `o_sd`, `n_seeds`
- `jens_max = max(jensen_gap over seeds)` (strict: 0 ⟺ no seed ever
  overshot MC)
- `dorm_mean = mean(jensen_dormancy_gap_at_best_burst over seeds)`

Pair the two arms by config key. `Δ_o = o_mean_d − o_mean_v`; Welch SE
`Δ_o_se = sqrt(o_sd_v²/n_v + o_sd_d²/n_d)`. Filter:

1. **`jens_max_v == 0`** — no seed in vanilla arm ever overshot MC.
2. **`dorm_mean_v ≥ 0.1`** — non-trivial dormancy floor at the
   best-outcome burst (mean over seeds).
3. **`Δ_o > 0`** — DDQN's mean outcome exceeds vanilla's.

## Result — 4 strict configs

| env | sync | T | n_v | dorm_v | o_v | o_d | Δ_o | Δ_o_se | z | Δ_q_late |
|---|---|---|---|---|---|---|---|---|---|---|
| **SpaceInvaders-MinAtar** | 3000 | 200k | 30 | 0.149 | 15.81 | 16.31 | **+0.502** | 0.154 | **+3.26** | −0.48 |
| SpaceInvaders-MinAtar | 1500 | 200k | 30 | 0.203 | 15.50 | 15.68 | +0.186 | 0.169 | +1.10 | −0.77 |
| Breakout-MinAtar | 1500 | 200k | 30 | 0.220 | 7.30 | 7.19 | **−0.107** | 0.228 | −0.47 | −0.89 |
| Breakout-MinAtar | 3000 | 200k | 30 | 0.246 | 7.41 | 7.07 | **−0.338** | 0.230 | −1.47 | −0.53 |

**Only 1 config** survives strict rock-solid dormancy AND Welch-
significant DDQN benefit:

- **SpaceInvaders-MinAtar, sync=3000, T=200k**: 30 vanilla seeds, 30 DDQN
  seeds. No vanilla seed ever showed Jensen overestimation. Dormancy floor
  at best burst averages 0.149 across seeds. Mean outcome: 15.81 →
  16.31 (Δ=+0.502, z=3.26, p≈0.001). DDQN's `q_late_mean` is 0.48
  LOWER than vanilla's.

The Breakout configs that earlier appeared as "case studies" at the
seed-pair level FAIL at config level: mean Δ_o is negative. The
seed-pair "win cells" were balanced by "lose cells" the previous count
filtered out.

## Adjacent: Acrobot configs (not strictly rock-solid)

| env | sync | T | n_v | jens_max_v | jens_median_v | o_v | o_d | Δ_o | z |
|---|---|---|---|---|---|---|---|---|---|
| Acrobot-v1 | 100 | 200k | 30 | (>0) | (≥0) | −5.27 | −5.25 | +0.023 | +0.55 |
| Acrobot-v1 | 100 | 1M | 60 | (>0) | (≥0) | −53.15 | −52.51 | **+0.644** | **+2.14** |

Acrobot at sync=100 T=1M: Δ_o=+0.64 with z=+2.14 (sig at α=0.05). But
Acrobot doesn't qualify for the strict scope — `jens_max_v > 0` (some
vanilla seeds in this config DO show overestimation). The per-env d=+0.43
the dormancy bridge reports was a SEED-LEVEL filter on (jensen_dormancy
_gap_at_best_burst >= 0.05); after aggregating to config level, Acrobot is
not a "dormant-config" case study — it's a mixed config where dormant
seeds and active seeds coexist, and DDQN helps in aggregate.

This is consistent with the bridge's POW_INSUF verdict: the per-env CI
straddles +0.2 because the env's cells span both dormant and active modes.

## What channel is DDQN operating on?

For the single surviving rock-solid config (SpaceInvaders sync=3000
T=200k):

- **`Δ_q_late = −0.48`**: DDQN's mean Q late in training is 0.48 lower
  than vanilla's. Vanilla already underestimates (Q ≤ MC); DDQN under-
  estimates further.
- The Hasselt-2010 `min(Q_online[a*], Q_target[a*])` upper-bound on
  bootstrap targets keeps Q magnitudes smaller even when there's no
  overestimation to correct. On a dense-reward MinAtar environment with
  fixed-length episodes, this Q-magnitude regularization can translate
  into a small (~0.5 reward point) outcome benefit.

## Implication for the bridge claim

After honest aggregation, the empirical challenge to the dormancy claim
narrows substantially:

- **Universal-null reading** ("DDQN ≈ 0 whenever Jensen dormant") is
  contradicted by SpaceInvaders sync=3000 T=200k — but by only ~+0.5
  outcome points (n=30 each arm, z=3.26). This is a small substantive
  effect, not a refutation of the underlying theory.
- **Cross-config consistency**: Breakout matched-scope configs go the
  other way (Δ_o = −0.1 to −0.3). DDQN's Q-magnitude regularization
  helps on SI but doesn't on Breakout under identical (sync, T) ranges.
- **The bridge's POW_INSUF verdict remains the right call**: per-env
  CIs straddle the +0.2 substantive ceiling; pooled-d aggregate is
  small. The case-study evidence is "a single config showing a small
  but real positive effect" — not enough to flip to
  INVARIANT_VIOLATION.

The seed-pair-level case studies hugely overstated the number and
magnitude of dormant cells where DDQN helps. The methodologically
honest count is **1 config out of the corpus**, with effect size
~0.5 outcome points.

## Reproducer

```python
import polars as pl
df = pl.read_parquet('experiments/data/cache/ddqn.parquet')
ddqn = 'bootstrap=partial(Claim:bootstrap;greedification=Claim:double_greedify)'
keys = ['env_name', 'sync_period', 'total_steps', 'eval_every']
base = df.filter(pl.col('arm_key').is_in([ddqn, 'baseline']))
agg = base.group_by(keys + ['arm_key']).agg([
    pl.col('eval_best_burst_mean').mean().alias('o_mean'),
    pl.col('eval_best_burst_mean').std().alias('o_sd'),
    pl.col('eval_best_burst_mean').len().alias('n'),
    pl.col('jensen_gap').max().alias('jens_max'),
    pl.col('jensen_dormancy_gap_at_best_burst').mean().alias('dorm_mean'),
])
van = agg.filter(pl.col('arm_key')=='baseline').drop('arm_key').rename(
    {c: c+'_v' for c in agg.columns if c not in keys+['arm_key']})
ddq = agg.filter(pl.col('arm_key')==ddqn).drop('arm_key').rename(
    {c: c+'_d' for c in agg.columns if c not in keys+['arm_key']})
cp = van.join(ddq, on=keys, how='inner').with_columns([
    (pl.col('o_mean_d') - pl.col('o_mean_v')).alias('Δo'),
    (pl.col('o_sd_v').pow(2)/pl.col('n_v') + pl.col('o_sd_d').pow(2)/pl.col('n_d')).sqrt().alias('Δo_se'),
])
case_studies = cp.filter(
    (pl.col('jens_max_v') == 0.0)
    & (pl.col('dorm_mean_v') >= 0.1)
)
```
