# Corroborate on your own runs: an SB3 walkthrough

This demo answers one question: **how much work is it to point
corroborate at training runs produced by an implementation it has
never seen?** Here that implementation is ordinary
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
DQN — no corroborate imports anywhere in the training code.

The study: does gamma = 0.99 beat gamma = 0.80 on CartPole-v1 at
25k steps? (Spoiler: the honest answer is better than a yes.)

## 1. Train — pure SB3 (`train.py`)

```bash
uv run examples/sb3_demo/train.py --seeds 3 --steps 25000   # ~10 min on CPU
```

Two conditions × 3 paired seeds; at five checkpoints each run is
evaluated with five fixed evaluation seeds. Alongside the runs it
writes the **bundle**: `contract.json` (the ~25-line study
description — the only corroborate-specific thing the producer
authors), `runs.jsonl`, `evaluations.jsonl`, `provenance.json`,
and one resolved-config JSON per run.

A committed bundle from a real run of this script is included, so
you can skip training and go straight to step 2.

## 2. Verify, adapt, analyse — corroborate (`analyze.py`)

```bash
uv run python examples/sb3_demo/analyze.py
```

The adapter is a *verifier*, not a file reader. Real output:

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

Note what the receipt refuses to do: statements the files can
prove are `VERIFIED` (the seal, pair completeness, that the
configs differ in *only* gamma); statements only the producer can
make stay `ATTESTED`/`UNVERIFIABLE` rather than being silently
upgraded. A broken bundle fails closed with the same vocabulary.

Then the run set is a `Panel`:

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

## 3. The verdict — and the point of the framework

The claim was declared *before* the outcomes were read
(`DirectionalDesign`: one-sided, alpha 0.05, SESOI dz = 0.5,
3 planned pairs):

```text
claim: gamma 0.99 > gamma 0.80 on return_mean
  n_pairs=3  mean_diff=-62.2 (CI -179.0..+54.6)
  dz=-0.90  p=0.8698
  verdict: POWER_INSUFFICIENT (UNDERPOWERED)
```

The intuitive claim did not survive contact with the data — the
point estimate runs the *other way* at this training length — and
with three pairs the evidence settles nothing. A benchmark table
would have printed two means and let the reader over-conclude;
the framework's verdict is "we cannot tell yet", which is
exactly what three seeds of evidence supports. Train more seeds
(`--seeds 8`) and the verdict machinery will move to whatever the
data actually earns — in the predicted direction or against it.

## What the producer had to author

The entire corroborate-specific burden was `contract.json`:
study id, which config key is the contrast (`gamma`), the two
condition names and values, the pairing field (`seed`), the scope
(`env_id`), and the evaluation extent. Everything else —
sealing, verification, admissibility, Panel construction,
analysis — is corroborate's side of the boundary.
