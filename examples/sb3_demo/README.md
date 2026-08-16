# SB3 walkthrough: external runs to a verdict

This example measures the integration cost of using corroborate
on training runs produced by an implementation it has never
seen — here, ordinary
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
DQN. The training script contains no corroborate imports.

The study: whether gamma = 0.99 outperforms gamma = 0.80 on
CartPole-v1 at 25k steps. Like most real studies it starts by
looking: the adapted run set is explored first, and the
directional claim is declared afterwards — with the receipt
recording exactly that register.

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

The `protocol` line is the study's epistemic register, typed: no
design was sealed before the runs existed, so whatever is claimed
later is admitted retrospectively. That is the ordinary register
of exploratory work — recorded, not penalised.

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
At this training length the intuitive claim runs the other way,
and the trajectory says why a final-checkpoint reading alone
would mislead: the deficit is late-training degradation, not
slower convergence still in progress.

A design-free paired probe quantifies the same contrast:

```text
probe: Δ(return_mean) = -62.2 ± 40.0  g=-0.51  pairs helped: 33% of 3
```

## 4. A directional claim under a declared design

Exploration sharpened the question; now the claim is stated as a
frozen design (`DirectionalDesign`: one-sided, alpha 0.05, SESOI
dz = 0.5, 3 planned pairs) and tested:

```text
claim: gamma 0.99 > gamma 0.80 on return_mean
  n_pairs=3  mean_diff=-62.2 (CI -179.0..+54.6)
  dz=-0.90  p=0.8698
  verdict: POWER_INSUFFICIENT (UNDERPOWERED)
```

The design is declared at analysis time, and the receipt already
recorded what that means: retrospective. At this training length
the point estimate runs against the claim, and three pairs are
not enough evidence to settle it either way. The verdict is
therefore `POWER_INSUFFICIENT` rather than `NO_EFFECT`; with more
seeds (`--seeds 8`) the verdict follows whatever the data
supports.

## 5. From exploration to confirmation

The exploratory pass produced a directional, scoped claim — and a
mechanism suspicion (late-training degradation at high gamma)
worth its own study. To test the claim in the confirmatory
register, seal the design *before* training the fresh seeds: the
contract's optional `prospective_protocol` field commits a
document (by path and SHA-256) naming the confirmatory pair keys,
the per-condition configurations, and the evaluation extent. The
adapter then verifies the digest and that the executed study
matches the sealed design (`protocol_committed` and
`protocol_design_match`, both `VERIFIED`) — "pre-registered"
becomes a machine-checked property of the record rather than a
sentence in prose.

## Producer-side cost

The corroborate-specific work in `train.py` is `contract.json`:
study id, the contrast parameter (`gamma`) with its two condition
names and values, the pairing field (`seed`), the scope
(`env_id`), and the evaluation extent. Sealing, verification,
Panel construction, and analysis happen on corroborate's side.
