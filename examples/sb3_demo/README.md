# SB3 walkthrough: external runs to a verdict

This example measures the integration cost of using corroborate
on training runs produced by an implementation it has never
seen — here, ordinary
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
DQN. The training script contains no corroborate imports.

The study: whether gamma = 0.99 outperforms gamma = 0.80 on
CartPole-v1 at 25k steps. `sb3_claim.py` codifies that question as
an executable, data-independent `@claim_bridge`; `analyze.py`
first explores the adapted run set, then evaluates that authored
test against the verified record.

(Runs logged your own way — monitor CSVs, a tensorboard scrape —
need none of the bundle machinery: read them into a DataFrame,
`Panel.from_dataframe(df)`, and skip straight to step 2. The
bundle path below is for a producer handing over a structured
record the adapter can verify.)

## 1. Train (`train.py`, pure SB3)

```bash
uv run examples/sb3_demo/train.py --seeds 3 --steps 25000   # ~10 min on CPU
```

Two conditions, three paired seeds each. At five checkpoints,
each run is evaluated once per fixed evaluation seed. The script
writes the bundle: `contract.json` (the study description, about
25 lines — the only corroborate-specific file the producer
authors), `runs.jsonl`, `evaluations.jsonl`, `provenance.json`,
and one resolved-config JSON per run.

A bundle from a real run of this script is committed, so step 2
can be run without training.

## 2. Verify and adapt (`analyze.py`, corroborate only)

```bash
uv run python examples/sb3_demo/analyze.py
```

The adapter verifies the bundle before any analysis. Output from
the committed bundle:

```text
admissible: True
  [VERIFIED    ] provenance_recorded: execution provenance recorded for producer 'stable-baselines3 DQN demo producer'
  [ATTESTED    ] provenance_attested: producer identity and invocation are attested by the record, not mechanically verified
  [VERIFIED    ] pairs_complete: verified 3 complete pairs
  [VERIFIED    ] config_isolation: all resolved configs share one template after removing only gamma and seed
  [VERIFIED    ] evaluation_complete: verified 6 × 5 × 5 evaluation extent
  [UNVERIFIABLE] assignment: assignment process was not mechanically recorded
  [VERIFIED    ] rows_derived: derived 6 seeded-run rows
```

Statements the files can prove are `VERIFIED` (pair
completeness, that the configurations differ only in gamma).
Statements only the producer can make remain `ATTESTED` or
`UNVERIFIABLE`. A malformed bundle raises with the same
vocabulary instead of parsing permissively.

The receipt makes no claim about when the test module was
authored, and the record carries no seal: evidence is a live,
growing thing here, and its integrity over time belongs to the
producer's version control.

## 3. Explore

The adapted run set is a `Panel`; its cells are a polars
DataFrame that registered analyses accept directly. Nothing in
this step requires a declared design.

```text
panel: 6 seeded runs × 17 columns
┌─────────────┬──────────┬──────┬───────┬─────────────┐
│ id          ┆ arm_key  ┆ seed ┆ gamma ┆ return_mean │
╞═════════════╪══════════╪══════╪═══════╪═════════════╡
│ gamma080-s0 ┆ gamma080 ┆ 0    ┆ 0.8   ┆ 156.2       │
│ gamma080-s1 ┆ gamma080 ┆ 1    ┆ 0.8   ┆ 162.6       │
│ gamma080-s2 ┆ gamma080 ┆ 2    ┆ 0.8   ┆ 240.2       │
│ gamma099-s0 ┆ gamma099 ┆ 0    ┆ 0.99  ┆ 100.2       │
│ gamma099-s1 ┆ gamma099 ┆ 1    ┆ 0.99  ┆ 166.4       │
│ gamma099-s2 ┆ gamma099 ┆ 2    ┆ 0.99  ┆ 105.8       │
└─────────────┴──────────┴──────┴───────┴─────────────┘
```

The adapter derives one `return_mean_at_<step>` column per
evaluation checkpoint, so the trajectory — not just the final
mean — is explorable:

```text
mean return per checkpoint (seeds pooled per condition):
┌──────────┬───────┬───────┬───────┬───────┬───────┐
│ arm_key  ┆ 5000  ┆ 10000 ┆ 15000 ┆ 20000 ┆ 25000 │
╞══════════╪═══════╪═══════╪═══════╪═══════╪═══════╡
│ gamma080 ┆ 210.9 ┆ 172.7 ┆ 182.9 ┆ 208.0 ┆ 186.3 │
│ gamma099 ┆ 164.3 ┆ 172.9 ┆ 199.7 ┆ 135.8 ┆ 124.1 │
└──────────┴───────┴───────┴───────┴───────┴───────┘
```

Gamma 0.99 is behind at 5k, level by 10k, ahead at 15k — then
falls away over the last two checkpoints while gamma 0.80 holds.
In these three paired seeds, that trajectory is consistent with a
late decline at the higher gamma. It is descriptive evidence, not
enough by itself to distinguish degradation from noisy or slower
learning.

A descriptive paired probe quantifies the same contrast:

```text
probe: Δ(return_mean) = -62.2 ± 40.0  g=-0.51  pairs helped: 33% of 3
```

## 4. Author and evaluate the claim test

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

There are no bundle paths, observed values, or arm labels in this
module. `analyze.py` binds the external record only at the call
site:

```python
evaluation = evaluate(
    higher_gamma_improves_return,
    panel.cells,
    recorded_contrast=study.contrast,
)
```

The recorded contrast must match the bridge's `gamma` source.
Its verified baseline/treatment keys are injected into the named
`paired_directional` analysis, whose typed result is retained in
`evaluation.analysis_results` before the bridge body maps it to a
verdict:

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

The record is live. Run more seeds (`--seeds 8`), re-adapt, and
the same claim module recomputes on the larger run set — batches
of the same study carry the same recorded contrast, so their
panels pool. A verdict that moves as evidence accretes is the
system working; the hypothesis layer's drift discipline exists
precisely to notice it.

The same bridge also runs against any other adapted study with
the same measurable schema and contrast path. Producer condition
names may differ: `recorded_contrast` supplies them at runtime.
The adapter continues to verify configuration isolation, pair
completeness, and evaluation extent; the claim module continues
to own the estimand and decision rule.

## Producer-side cost

The corroborate-specific work in `train.py` is `contract.json`:
study id, the contrast parameter (`gamma`) with its two condition
names and values, the pairing field (`seed`), the scope
(`env_id`), and the evaluation extent. Verification, Panel
construction, and analysis happen on corroborate's side.
