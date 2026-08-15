# SB3 walkthrough: external runs to a verdict

This example measures the integration cost of using corroborate
on training runs produced by an implementation it has never
seen — here, ordinary
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
DQN. The training script contains no corroborate imports.

The study: whether gamma = 0.99 outperforms gamma = 0.80 on
CartPole-v1 at 25k steps.

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

## 2. Verify, adapt, analyse (`analyze.py`, corroborate only)

```bash
uv run python examples/sb3_demo/analyze.py
```

The adapter verifies the bundle before any analysis. Output from
the committed bundle:

```text
admissible: True
  [VERIFIED    ] manifest_files: verified 10 sealed files
  [VERIFIED    ] bundle_digest: bundle digest 46b2721f04d3…
  [VERIFIED    ] provenance_recorded: execution provenance recorded for producer 'stable-baselines3 DQN demo producer'
  [ATTESTED    ] provenance_attested: producer identity and invocation are attested by the record, not mechanically verified
  [VERIFIED    ] pairs_complete: verified 3 complete pairs
  [VERIFIED    ] config_isolation: all resolved configs share one template after removing only gamma and seed
  [VERIFIED    ] evaluation_complete: verified 6 × 5 × 5 evaluation extent
  [UNVERIFIABLE] protocol: no prospective protocol committed; the design is admitted retrospectively
  [UNVERIFIABLE] assignment: assignment process was not mechanically recorded
  [VERIFIED    ] rows_derived: derived 6 seeded-run rows
```

Statements the files can prove are `VERIFIED` (the seal, pair
completeness, that the configurations differ only in gamma).
Statements only the producer can make remain `ATTESTED` or
`UNVERIFIABLE`. A malformed bundle raises with the same
vocabulary instead of parsing permissively.

The adapted run set is a `Panel`:

```text
panel: 6 seeded runs × 12 columns
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

## 3. The pre-registered test

The claim is declared before outcomes are read
(`DirectionalDesign`: one-sided, alpha 0.05, SESOI dz = 0.5,
3 planned pairs):

```text
claim: gamma 0.99 > gamma 0.80 on return_mean
  n_pairs=3  mean_diff=-62.2 (CI -179.0..+54.6)
  dz=-0.90  p=0.8698
  verdict: POWER_INSUFFICIENT (UNDERPOWERED)
```

At this training length the point estimate runs against the
claim, and three pairs are not enough evidence to settle it
either way. The verdict is therefore `POWER_INSUFFICIENT` rather
than `NO_EFFECT`; with more seeds (`--seeds 8`) the verdict
follows whatever the data supports.

## Producer-side cost

The corroborate-specific work in `train.py` is `contract.json`:
study id, the contrast parameter (`gamma`) with its two condition
names and values, the pairing field (`seed`), the scope
(`env_id`), and the evaluation extent. Sealing, verification,
Panel construction, and analysis happen on corroborate's side.
