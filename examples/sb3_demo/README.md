# SB3 walkthrough: external runs to a verdict

This example measures the integration cost of using corroborate
on training runs produced by an implementation it has never
seen — here, ordinary
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
DQN. The training script is exactly what an SB3 tutorial writes —
no corroborate imports, no recording code, no extra files — and
the analysis reads SB3's own artifacts.

The question: whether gamma = 0.99 outperforms gamma = 0.80 on
CartPole-v1 at 25k steps. `sb3_claim.py` codifies it as an
executable, data-independent claim; `analyze.py` loads the runs,
explores them with plain polars, then evaluates the claim.

(Runs logged your own way — a directory of plain JSON records —
load via `corroborate.data.load_runs` instead, and any DataFrame
with the named columns evaluates directly.)

## 1. Train (`train.py`, pure SB3)

```bash
uv run examples/sb3_demo/train.py --seeds 3 --steps 25000   # ~10 min on CPU
```

Two gamma values, three paired seeds each: construct `DQN`, train
with an `EvalCallback` (5 evaluation episodes every 5k steps),
`model.save()`. What lands on disk is SB3's output, nothing else:

```text
runs/
  gamma080-s0/model.zip           model.save()
  gamma080-s0/evaluations.npz     EvalCallback's evaluation log
  ...
```

Artifacts from a real run of this script are committed, so the
analysis can be run without training.

## 2. Load (`analyze.py`, corroborate's side)

```bash
uv run --with 'stable-baselines3>=2.3' examples/sb3_demo/analyze.py
```

`corroborate_rl.sb3.load_sb3_runs` reads the folder into one row
per run. Configuration comes from each checkpoint's own `data`
record — `model.save()` dumps the algorithm's resolved state, and
intersecting it with the DQN constructor's signature separates
what was *configured* (`gamma`, `buffer_size`, `seed`, …) from
runtime state (`num_timesteps`, the decayed `exploration_rate`).
Evaluations come from `evaluations.npz`, aggregated per
checkpoint into `return_mean`, `return_auc`, and one
`return_mean_at_<step>` column each. The checkpoint doesn't
record which environment it trained on, so the analyst states
that known context in plain polars (`with_columns`).

```text
loaded: 6 runs × 26 columns
┌─────────────┬──────┬───────┬─────────────┐
│ id          ┆ seed ┆ gamma ┆ return_mean │
╞═════════════╪══════╪═══════╪═════════════╡
│ gamma080-s0 ┆ 0    ┆ 0.8   ┆ 157.0       │
│ gamma080-s1 ┆ 1    ┆ 0.8   ┆ 205.8       │
│ gamma080-s2 ┆ 2    ┆ 0.8   ┆ 173.6       │
│ gamma099-s0 ┆ 0    ┆ 0.99  ┆ 98.8        │
│ gamma099-s1 ┆ 1    ┆ 0.99  ┆ 238.0       │
│ gamma099-s2 ┆ 2    ┆ 0.99  ┆ 86.2        │
└─────────────┴──────┴───────┴─────────────┘
```

## 3. Explore

The result is an ordinary polars DataFrame; exploration is
ordinary polars. The trajectory columns make training dynamics —
not just the final mean — visible:

```text
mean return per checkpoint (seeds pooled per condition):
┌───────┬───────┬───────┬───────┬───────┬───────┐
│ gamma ┆ 5000  ┆ 10000 ┆ 15000 ┆ 20000 ┆ 25000 │
╞═══════╪═══════╪═══════╪═══════╪═══════╪═══════╡
│ 0.8   ┆ 189.5 ┆ 173.1 ┆ 184.8 ┆ 248.5 ┆ 178.8 │
│ 0.99  ┆ 196.5 ┆ 175.5 ┆ 184.9 ┆ 145.0 ┆ 141.0 │
└───────┴───────┴───────┴───────┴───────┴───────┘

Δ(return_mean) per seed (gamma 0.99 − 0.80):
┌──────┬───────┬───────┬───────┐
│ seed ┆ 0.8   ┆ 0.99  ┆ delta │
╞══════╪═══════╪═══════╪═══════╡
│ 0    ┆ 157.0 ┆ 98.8  ┆ -58.2 │
│ 1    ┆ 205.8 ┆ 238.0 ┆ 32.2  │
│ 2    ┆ 173.6 ┆ 86.2  ┆ -87.4 │
└──────┴───────┴───────┴───────┘
```

The two conditions track each other through 15k steps — then
gamma 0.99 falls away over the last two checkpoints while gamma
0.80 holds. In these three paired seeds, that trajectory is
consistent with a late decline at the higher gamma. It is
descriptive evidence, not enough by itself to distinguish
degradation from noisy or slower learning.

## 4. Author and evaluate the claim

The claim module contains the scientific test, not the data:

```python
@claim_bridge(
    source='gamma',
    target='return_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    pair_by=('seed',),
    scope=(
        (pl.col('env_id') == 'CartPole-v1')
        & pl.col('gamma').is_in([0.80, 0.99])
    ),
    predicted_direction='a_gt_b',
)
def higher_gamma_improves_return(
    paired_directional: PairedDirectionalResult,
    *,
    alpha: float = 0.05,
    sesoi_dz: float = 0.5,
    minimum_pairs: int = 3,
) -> tuple[Verdict, RefutationClass | None]:
    del alpha, sesoi_dz, minimum_pairs
    return paired_directional_verdict(paired_directional)
```

The bridge is exactly what a native one looks like — no external
special case. `tier=INTERVENTIONAL` on the assigned parameter
says the contrast was executed; the scope pins which two values
it compares. Evaluation adds one fact from the data side: which
columns were *configuration* — recovered from the checkpoints
themselves, never hand-listed:

```python
evaluation = evaluate(
    higher_gamma_improves_return, df,
    leaves=sb3_config_columns(RUNS, DQN),
)
```

Conditions then derive from the `gamma` column's scoped values
(labelled `gamma=0.8` / `gamma=0.99`, ascending — the claim's
sign lives in `predicted_direction`), and the admission gates
check the contrast over exactly the cells this claim admits:
`contrast_present` blocks if the record doesn't actually vary
gamma in scope, `contrast_isolation` blocks if another
*configuration* column moved together with gamma (a confound
riding the contrast — an unregistered column that co-varies only
warns, since a label is harmless and only you can say which it
is), and `pair_completeness` warns when a seed is missing one
condition. A blocked claim gets the verdict `INADMISSIBLE` —
quality problems land on the verdict record, not in a separate
report.

```text
claim: gamma 0.99 > gamma 0.80 on return_mean
  n_pairs=3  mean_diff=-37.8 (CI -142.9..+67.3)
  dz=-0.61  p=0.7981
  verdict: POWER_INSUFFICIENT (UNDERPOWERED)
```

At this training length the point estimate runs against the claim,
and three pairs are not enough evidence to settle it either way.
The verdict is therefore `POWER_INSUFFICIENT` rather than
`NO_EFFECT`; with more seeds (`--seeds 8`) the same test module
follows whatever the compatible data supports.

## 5. Grow the record

The record is live. Run more seeds, re-load, and the same claim
recomputes on the larger run set — batches concatenate with plain
`pl.concat`, because a growing study is one run set that happens
to arrive in parts. A verdict that moves as evidence accretes is
the system working; the hypothesis layer's drift discipline
exists precisely to notice it. The gates re-check contrast
presence, isolation, and pairing on every evaluation, over
whatever the record has become.

## Producer-side cost

Zero. Not zero files — zero *anything*: `train.py` is the
tutorial-shaped SB3 script, and the analysis reads the artifacts
SB3 already writes. If you have a folder of checkpoint zips and
`EvalCallback` logs from last month, this works on it today.
