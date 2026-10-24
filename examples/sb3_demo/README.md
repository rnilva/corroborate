# SB3 walkthrough: external runs to a verdict

This example measures the integration cost of using corroborate
on training runs produced by an implementation it has never
seen — here, ordinary
[stable-baselines3](https://github.com/DLR-RM/stable-baselines3)
DQN. The training script is what an SB3 tutorial writes, logging
through SB3's own `Logger`; the analysis reads the Logger's
files directly.

The question: whether gamma = 0.99 outperforms gamma = 0.80 on
CartPole-v1 at 25k steps. `sb3_claim.py` codifies it as an
executable, data-independent claim; `analyze.py` loads the runs,
explores them with plain polars, then evaluates the claim.

## 1. Train (`train.py`, pure SB3)

```bash
uv run examples/sb3_demo/train.py --seeds 3 --steps 25000   # ~10 min on CPU
```

Two gamma values, three paired seeds each: construct `DQN`, point
its Logger at a folder (`configure(str(run_dir), ['json'])`),
train with an `EvalCallback` (5 evaluation episodes every 5k
steps). The one line of contact with the analysis side is the
loop recording the constructor arguments next to SB3's own
records:

```python
model.set_logger(configure(str(run_dir), ['json']))
for key, value in config.items():
    model.logger.record(f'config/{key}', value)
```

What lands on disk is the Logger's own output, nothing else:

```text
runs/
  gamma080-s0/progress.json     one JSON record per Logger dump()
  ...
```

Each dump carries whatever was recorded since the last one:
`rollout/ep_rew_mean` and `time/total_timesteps` on every dump,
`train/loss` once learning starts, `eval/mean_reward` on
evaluation dumps, `config/*` on the first. Files from a real run
of this script are committed, so the analysis can be run without
training.

## 2. Load (`analyze.py`, corroborate's side)

```bash
uv run examples/sb3_demo/analyze.py
```

`corroborate_rl.sb3_logger.load_sb3_logger_runs` reads the folder
into a `Panel` — one row per progress file, plus the two facts a
bare frame cannot carry: the configuration registry and
provenance. The reader is mechanical. Keys are kept verbatim;
`config/*` keys — the producer's own statement of what was
configured — become scalar columns and the registry; every other
key stays a record: an array column with one slot per dump —
NaN for numeric keys, null otherwise, where the key wasn't
recorded — aligned by `time/total_timesteps`
(itself just a record column). Nothing is renamed, reduced, or
classified at load time — no terminal means, no AUC, no "outcome"
column role. What one run reached is that run's own record;
which point matters is the claim's declaration.

```text
loaded: 6 runs × 23 columns
registry: config/buffer_size, config/env_id, config/exploration_fraction,
          config/gamma, config/learning_rate, config/learning_starts,
          config/seed, config/target_update_interval, config/train_freq
```

## 3. Explore

`panel.df` is an ordinary polars DataFrame; the record arrays
explode into an ordinary long frame, so the evaluation trajectory
— not just any single point — is explorable directly:

```text
mean eval reward per evaluation point (seeds pooled per condition):
┌──────────────┬───────┬───────┬───────┬───────┬───────┐
│ config/gamma ┆ 5000  ┆ 10000 ┆ 15000 ┆ 20000 ┆ 25000 │
╞══════════════╪═══════╪═══════╪═══════╪═══════╪═══════╡
│ 0.8          ┆ 189.5 ┆ 173.1 ┆ 192.6 ┆ 233.7 ┆ 188.0 │
│ 0.99         ┆ 196.5 ┆ 175.5 ┆ 184.1 ┆ 145.4 ┆ 131.7 │
└──────────────┴───────┴───────┴───────┴───────┴───────┘

Δ(eval reward at 25k) per seed (gamma 0.99 − 0.80):
┌─────────────┬───────┬───────┬────────┐
│ config/seed ┆ 0.8   ┆ 0.99  ┆ delta  │
╞═════════════╪═══════╪═══════╪════════╡
│ 0           ┆ 165.4 ┆ 102.4 ┆ -63.0  │
│ 1           ┆ 168.6 ┆ 184.6 ┆ 16.0   │
│ 2           ┆ 230.0 ┆ 108.2 ┆ -121.8 │
└─────────────┴───────┴───────┴────────┘
```

The two conditions track each other through 15k steps — then
gamma 0.99 falls away over the last two evaluation points while
gamma 0.80 holds. In these three paired seeds, that trajectory is
consistent with a late decline at the higher gamma. It is
descriptive evidence, not enough by itself to distinguish
degradation from noisy or slower learning.

## 4. Author and evaluate the claim

The claim module contains the scientific test, not the data —
including its horizon. `eval_reward_at_25k` is a registered
measurable reducing the producer's own record columns; a run
never evaluated at step 25 000 reads NaN — missing evidence,
never silently rebased to an earlier point:

```python
@measurable(
    name='eval_reward_at_25k',
    reads=('eval/mean_reward', 'time/total_timesteps'),
)
def eval_reward_at_25k(record: Mapping[str, object]) -> float: ...

GAMMA_EFFECT = DoEffect.from_values(
    source='config/gamma',
    reference=0.80,
    treatment=0.99,
)

@claim_bridge(
    source=GAMMA_EFFECT,
    target='eval_reward_at_25k',
    direction=Direction.DIRECT,
    tier=Tier.INTERVENTIONAL,
    pair_by=('config/seed',),
    scope=pl.col('config/env_id') == 'CartPole-v1',
    predicted_direction='a_gt_b',
)
def higher_gamma_improves_return(
    paired_directional: PairedDirectionalResult,
    *,
    alpha: float = 0.05,
    sesoi_dz: float = 0.5,
    minimum_pairs: int = 3,
) -> tuple[Verdict, RefutationClass | None]:
    return paired_directional_verdict(paired_directional)
```

The bridge is exactly what a native one looks like — no external
special case: a `DoEffect` in source position declaring the exact
reference and treatment values of an externally-executed
contrast. The Panel already carries the one fact the gates need
from the data side — which columns were *configuration*, stated
by the producer's own `config/*` records, never hand-listed — so
evaluation is the claim and the record:

```python
evaluation = evaluate(higher_gamma_improves_return, panel)
```

(No materialisation step: the analysis resolves the registered
measurable per row through the registry.)

Exact matches to 0.80 and 0.99 receive the stable symbolic arm
identities `baseline` and `treatment`; neither membership nor
orientation is inferred from observed support or display labels.
Any additional gamma levels remain in the accumulated record
but sit outside this declared contrast, available to other
claims over the same Panel.

The admission gates then check the contrast over exactly the
rows this claim admits. The registry is authoritative about what
was configured: the declared source must be a registered
configuration column (a measurement cannot be the assigned
parameter of an intervention — BLOCK), a registered leaf that
moves with gamma inside a seed pair is a co-varied knob (a
confound — BLOCK, unless it was assigned jointly, in which case
the declaration widens to
`from_values(reference={...}, treatment={...})`), and an
unregistered column moving with the contrast warns — a record
array counts by its distinct values, so a knob logged per dump
instead of under `config/` is still seen. A label is harmless,
an unlogged knob is not, and only the author can say which. `contrast_present` blocks when a declared arm is
absent from the record, and `pair_completeness` warns when a
seed is missing one condition. A blocked claim gets the verdict
`INADMISSIBLE` — quality problems land on the verdict record,
not in a separate report. What no registry can attest —
assignment, randomisation, hidden confounding — stays outside
the framework's claims for external runs.

```text
claim: gamma 0.99 > gamma 0.80 on eval_reward_at_25k
  n_pairs=3  mean_diff=-56.3 (CI -172.8..+60.3)
  dz=-0.81  p=0.8530
  verdict: POWER_INSUFFICIENT (UNDERPOWERED)
```

At this training length the point estimate runs against the claim,
and three pairs are not enough evidence to settle it either way.
The verdict is therefore `POWER_INSUFFICIENT` rather than
`NO_EFFECT`; with more seeds the same test module follows
whatever the compatible data supports.

## 5. Grow the record

The record is live, and growth is plain concatenation — there is
no record-wide derivation to redo. More seeds land as more
progress files (train into the same folder, or a new one and
point the reader at it); a longer-budget batch is just rows whose
arrays extend further. The claim's horizon is pinned in the claim
(`eval_reward_at_25k`), so old and new rows stay comparable by
construction: a row without the declared horizon reads NaN and
the `target_finiteness` warning reports what that costs, per
evaluation. Batches loaded
separately pool with `concat_panels` (the pooled Panel keeps
carrying the union registry). A verdict that moves as evidence
accretes is the system working; the declared estimand stays fixed
while the gates re-check arm presence, configuration isolation,
and pairing over whatever the record has become.

Whether two rows are the same experimental unit is never the
reader's call: `pair_by=('config/seed',)` declares it, and the
paired analysis raises on duplicate `(condition, seed)` buckets
rather than silently averaging a re-run into a fresh seed.

## 6. Reuse a graph of claims as evidence grows

```bash
uv run python -m examples.sb3_demo.growing_analysis
```

`early_claim.py` adds a question at 15k steps, independently of the
original 25k claim in `sb3_claim.py`. `hypothesis.py` composes their
bridges. It has no data paths and no global intervention: the contrast
belongs to each bridge. These are two budget-specific tests on the same
seeds, not independent replications or a causal explanation of learning.

The short `growing_analysis.py` example loads the committed records,
evaluates the graph on the first two seeds, then includes the third:

```python
first = records.narrow(pl.col('config/seed') < 2)
second = records.narrow(pl.col('config/seed') >= 2)

initial = run(hypothesis, data=first, use_cache=False)
updated = run(
    hypothesis,
    data=concat_panels((first, second)),
    use_cache=False,
)
```

This split demonstrates growth using data already available; it does not
claim the two batches were collected at different times. In normal use,
load separately arriving batches through the same reader. The claim
definitions stay fixed, and the typed results expose the sample counts,
estimates, admission checks and verdicts over the current observations.
No cache, manifest or report file is required to evaluate the hypothesis.

The responsibilities stay separate:

| Part | Owns |
|---|---|
| `train.py` | Experiment execution and producer logging |
| `load_sb3_logger_runs` | Mechanical reading, configuration and provenance |
| `sb3_claim.py`, `early_claim.py`, `hypothesis.py` | Measurements, questions, decision rules and graph composition |
| `analyze.py`, `growing_analysis.py` | Interactive exploration and presentation |

Copying or restoring files belongs in auxiliary tooling. It must not
redefine the measurements, select successful runs or supply prewritten
verdicts to the analysis modules.

## Producer-side cost

One loop. `train.py` is the tutorial-shaped SB3 script plus the
`config/*` record loop — the producer stating what was
configured, in the producer's own logging vocabulary. If you have
folders of `progress.json` from last month *with* config keys
recorded, this works on them today; without config keys, the
record columns still load and explore, and `evaluate` reports the
missing registry on the verdict record instead of guessing.
