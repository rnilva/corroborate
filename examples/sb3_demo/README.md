# SB3 walkthrough: external runs to a verdict

This example measures the integration cost of using corroborate
on training runs produced by an implementation it has never
seen — here, ordinary
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
DQN. The training script contains no corroborate imports and
writes no corroborate-specific files.

The question: whether gamma = 0.99 outperforms gamma = 0.80 on
CartPole-v1 at 25k steps. `sb3_claim.py` codifies it as an
executable, data-independent claim; `analyze.py` loads the runs,
explores them with plain polars, then evaluates the claim.

(Runs logged your own way — monitor CSVs, a tensorboard scrape —
skip the loader entirely: read them into a DataFrame and evaluate
the claim against it directly.)

## 1. Train (`train.py`, pure SB3)

```bash
uv run examples/sb3_demo/train.py --seeds 3 --steps 25000   # ~10 min on CPU
```

Two gamma values, three paired seeds each. At five checkpoints,
each run is evaluated once per fixed evaluation seed. The script
writes what any careful experimenter records anyway: `runs.jsonl`
(one line per run: id + which config ran it), one resolved-config
JSON per run, `evaluations.jsonl`, and an optional
`provenance.json`.

Files from a real run of this script are committed, so the
analysis can be run without training.

## 2. Load (`analyze.py`, corroborate only)

```bash
uv run python examples/sb3_demo/analyze.py
```

`load_runs` reads the directory into one row per run: config
flattened to dotted-path columns, evaluations aggregated per
checkpoint, `return_mean` / `return_auc` / one
`return_mean_at_<step>` column per checkpoint derived. It is a
reader, not a gatekeeper — no verdicts here.

```text
loaded: 6 runs × 21 columns
┌─────────────┬──────┬───────┬─────────────┐
│ id          ┆ seed ┆ gamma ┆ return_mean │
╞═════════════╪══════╪═══════╪═════════════╡
│ gamma080-s0 ┆ 0    ┆ 0.8   ┆ 156.2       │
│ gamma080-s1 ┆ 1    ┆ 0.8   ┆ 162.6       │
│ gamma080-s2 ┆ 2    ┆ 0.8   ┆ 240.2       │
│ gamma099-s0 ┆ 0    ┆ 0.99  ┆ 100.2       │
│ gamma099-s1 ┆ 1    ┆ 0.99  ┆ 166.4       │
│ gamma099-s2 ┆ 2    ┆ 0.99  ┆ 105.8       │
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
│ 0.8   ┆ 210.9 ┆ 172.7 ┆ 182.9 ┆ 208.0 ┆ 186.3 │
│ 0.99  ┆ 164.3 ┆ 172.9 ┆ 199.7 ┆ 135.8 ┆ 124.1 │
└───────┴───────┴───────┴───────┴───────┴───────┘

Δ(return_mean) per seed (gamma 0.99 − 0.80):
┌──────┬───────┬───────┬────────┐
│ seed ┆ 0.99  ┆ 0.8   ┆ delta  │
╞══════╪═══════╪═══════╪════════╡
│ 0    ┆ 100.2 ┆ 156.2 ┆ -56.0  │
│ 1    ┆ 166.4 ┆ 162.6 ┆ 3.8    │
│ 2    ┆ 105.8 ┆ 240.2 ┆ -134.4 │
└──────┴───────┴───────┴────────┘
```

Gamma 0.99 is behind at 5k, level by 10k, ahead at 15k — then
falls away over the last two checkpoints while gamma 0.80 holds.
In these three paired seeds, that trajectory is consistent with a
late decline at the higher gamma. It is descriptive evidence, not
enough by itself to distinguish degradation from noisy or slower
learning.

## 4. Author and evaluate the claim

The claim module contains the scientific test, not the data:

```python
@claim_bridge(
    source='gamma',
    contrast=(0.80, 0.99),
    target='return_mean',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    pair_by=('seed',),
    scope=pl.col('env_id') == 'CartPole-v1',
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

`contrast=(0.80, 0.99)` is the claim's own statement of which two
parameter values it compares — conditions are derived from the
`gamma` column, so there are no producer arm labels anywhere.
Evaluation is one call:

```python
evaluation = evaluate(higher_gamma_improves_return, df)
```

Before the statistics run, admission gates check the contrast
over exactly the cells this claim admits: `contrast_isolation`
blocks if any other column moved together with gamma (a confound
riding the contrast), and `pair_completeness` warns when a seed
is missing one condition. A blocked claim gets the verdict
`INADMISSIBLE` — quality problems land on the verdict record,
not in a separate report.

```text
claim: gamma 0.99 > gamma 0.80 on return_mean
  n_pairs=3  mean_diff=-62.2 (CI -179.0..+54.6)
  dz=-0.90  p=0.8698
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
exists precisely to notice it.

The same claim module also runs against anyone else's runs with
the same measurable schema — the contrast values live in the
claim, so nothing producer-specific needs to be supplied at
evaluation time. The gates re-check isolation and pairing on
every evaluation, over whatever the record has become.

## Producer-side cost

Zero corroborate-specific files. `train.py` records run ids,
resolved configs, evaluations, and provenance — records a
careful experiment keeps regardless of what analyses them later.
