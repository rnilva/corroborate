# corroborate

A framework for testing the mechanism claims behind reinforcement
learning algorithms.

The frozen study accompanying the RLC 2026 workshop paper
(*Corroborate: A Framework for Testing Mechanism Claims in
Reinforcement Learning*), including the paper PDF, its data, and
the figure pipelines, is on the
[`submission` branch](../../tree/submission). `main` is the
current framework.

## Overview

A paper that proposes a mechanism (for example, Double-DQN's
claim that decoupled action selection reduces overestimation
bias) proves a theorem under conditions the implementation then
relaxes. Whether the prediction holds in practice is an empirical
question, and testing it is closer to causal inference than to
comparing learning curves: the mechanism is entangled with the
rest of the implementation, and the relevant quantities are
confounded.

corroborate provides the apparatus for this kind of test:

1. Component boundaries are drawn at theorems. A mechanism is
   isolated as a functional unit that can be swapped without
   changing the rest of the algorithm, so a comparison between
   conditions is a controlled contrast.
2. Hypotheses are executable. Each edge of a hypothesis is a
   small program called a bridge. Its decorator declares the
   claim's commitments (scope, evidential tier, predicted
   direction); its body runs as a test and returns a verdict.
3. Inconclusive results are a distinct verdict.
   `POWER_INSUFFICIENT` is separate from both `HELD` and
   `NO_EFFECT`, and it propagates through chains of composed
   claims rather than being collapsed into "no effect".

The demonstration study applies this to Double-DQN across twelve
environments: the bias-reduction claim holds in most of them,
while the further claim that bias reduction improves outcomes
does not survive testing.

## Terminology

The paper uses plain terms; some code identifiers and logged
column names predate them and are kept for compatibility:

| paper | code |
|---|---|
| condition (one algorithm variant under study) | `arm`, `arm_key` |
| evaluation window (block of greedy eval episodes) | `burst` (e.g. `eval_best_burst_raw_mean`) |
| run set (the logged collection of runs) | corpus (`runs.parquet` + sidecars) |
| seeded run at an environment–condition pair | cell |
| metric | `@measurable` |
| the implementation under study | "substrate" (e.g. the `--substrate` CLI flag) |

## Example bridge

```python
from functools import partial
from corroborate import claim_bridge
from corroborate.bridge.bridge import Direction, Tier
from corroborate.bridge.verdict import Verdict
from corroborate.core.intervention import DoEffect, Intervention

# The mechanism as a unit of intervention: swap greedification,
# leave everything else unchanged.
DDQN_SWAP = Intervention(
    slot_path='bootstrap',
    replacement=partial(bootstrap, greedification=double_greedify),
)
INTERVENTION = DoEffect(arms=((), (DDQN_SWAP,)))  # baseline, treatment

@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    predicted_direction='a_lt_b',
)
def ddqn_reduces_jensen_gap(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
    *,
    stratify_by: tuple[str, ...] = ('env_name',),
) -> Verdict:
    del stratify_by
    return stratified_arm_diff_pooled.verdict
```

The decorator holds the claim's declared commitments; the body
maps statistics to a verdict. The named parameter is a registered
analysis that the framework runs against the scoped run set and
injects by name. A hypothesis is a Python package exposing a
tuple of bridges plus cluster-level findings; running it prints
the verdict table:

```bash
uv run corroborate hypothesis experiments.findings.<name>
# Optionally render the evidence graph:
uv run corroborate hypothesis experiments.findings.<name> --render evidence.svg
```

## Using external runs

Training runs produced by other codebases evaluate without
modifying the training code, and without any corroborate-specific
files on the producer's side. For stable-baselines3,
`corroborate_rl.sb3.load_sb3_runs` reads the artifacts SB3
already writes — `model.save()` checkpoint zips plus
`EvalCallback` `evaluations.npz` — into a DataFrame, one row per
run, configuration flattened to dotted-path columns,
per-checkpoint outcome aggregates derived. For runs logged in
your own format, `corroborate.data.load_runs` reads a directory
of plain JSON records into the same shape. Both are readers, not
gatekeepers.

The claim is an ordinary, data-independent `@claim_bridge`
module — the same shape as a native bridge, with no external
special case. `tier=INTERVENTIONAL` on the assigned parameter
says the contrast was executed (outside the framework); the
scope pins which values it compares:

```python
@claim_bridge(
    source='gamma',
    target='return_mean',
    tier=Tier.INTERVENTIONAL,
    pair_by=('seed',),
    scope=pl.col('gamma').is_in([0.80, 0.99]),
    predicted_direction='a_gt_b',
    ...
)
def higher_gamma_improves_return(...) -> ...: ...

df = load_runs('path/to/runs')
result = evaluate(
    higher_gamma_improves_return, df,
    leaves=config_columns('path/to/runs'),
)
```

`leaves` registers which columns were *configuration* — derived
from the record's own config files, the external counterpart of
walking a native claim composition; nobody types a list of
names. Conditions then derive from the source column's scoped
values — no producer arm labels exist anywhere — and
study-design quality is checked where the hypothesis layer owns
it, by admission gates over exactly the cells the claim admits:
`contrast_present` blocks when the record doesn't vary the
parameter in scope, `contrast_isolation` blocks (verdict
`INADMISSIBLE`) when another configuration column moved together
with it, and `pair_completeness` warns on the verdict record
when a pairing unit is missing one condition.

The record is live: run more seeds, re-load, and the same bridge
recomputes — batches concatenate with plain `pl.concat`, and a
verdict that moves with the evidence is the system working. (Runs
logged your own way need no loader at all: any DataFrame with the
named columns evaluates directly, and `Panel.from_dataframe`
offers the exploration surface.)

[`examples/sb3_demo/`](examples/sb3_demo/) walks through this
with stable-baselines3 DQN: a tutorial-shaped training script
with zero recording code, loading of SB3's own artifacts,
descriptive exploration in plain polars, and evaluation of an
executable claim module through the same bridge path used by
native studies. It runs on CPU in a few minutes, and artifacts
from a real run are committed so the analysis half can be run
without training.

## Repository layout

| path | contents |
|---|---|
| `src/corroborate/` | the framework: claims, bridges, verdicts, causal-graph evaluation, registered analyses, run-set storage |
| `src/corroborate_rl/` | the DQN/JAX implementation used by the frozen study |
| `examples/sb3_demo/` | external-data walkthrough (stable-baselines3) |
| `docs/HYPOTHESIS_AS_GRAPH.md` | authoring model: a hypothesis as a causal graph with bridges as edges |
| `REPRODUCIBILITY.md` | bitwise vs scientific reproducibility under XLA configuration |
| `CLAUDE.md` | contributor documentation: typing discipline, vocabulary, analysis catalogue |
| `tests/` | framework tests, including a linear-Gaussian SCM control with closed-form expectations |

Checks:

```bash
uv run pyright          # strict mode, 0 errors
uv run pytest tests/    # fast cohort; `-m ''` includes the slow end-to-end tests
```

## Running studies

An implementation plugs into the sweep CLI through a typed
entry-point module and a YAML configuration:

```bash
# Train the conditions across environments and seeds:
uv run corroborate sweep run --substrate corroborate_rl.dqn_sweep \
    --device gpu experiments/configs/<sweep>.yaml

# Ingest the logged run set and evaluate a hypothesis against it:
uv run corroborate hypothesis experiments.findings.<name> \
    --ingest <run-set-dir>

# Inventory local and cloud-archived run sets:
uv run corroborate catalogue experiments/data --remote-prefix s3://<your-bucket>/
```

Run sets archive to any S3-compatible store using botocore's
standard credential chain. Trace columns are evicted locally once
cloud-recoverable and restored on demand.

## License

MIT — see [`LICENSE`](LICENSE).
