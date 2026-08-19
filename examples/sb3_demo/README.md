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
load via `corroborate.data.load_runs` instead into the same
shape, and any DataFrame with the named columns evaluates
directly.)

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

`corroborate_rl.sb3.load_sb3_runs` reads the folder into a
`Panel` — one row per run in `panel.cells`, plus the two facts a
bare frame cannot carry: the configuration registry and
provenance. Configuration comes from each checkpoint's own `data`
record — `model.save()` dumps the algorithm's resolved state, and
intersecting it with the DQN constructor's signature separates
what was *configured* (`gamma`, `buffer_size`, `seed`, …; entries
SB3 could not JSON-encode, like `train_freq`, are kept as opaque
but equality-comparable strings) from runtime state
(`num_timesteps`, the decayed `exploration_rate`). Evaluations
come from `evaluations.npz`, aggregated into `return_mean` at the
record-wide terminal evaluation point (null for a run not
evaluated there — never silently rebased to an earlier horizon,
with `return_terminal_n`/`_attempted` recording what it stands
on), `return_auc` for runs covering the full grid, and one
`return_mean_at_<step>` column per point. The checkpoint doesn't
record which environment it trained on, so the analyst stamps
that known context with `with_columns` — which stays a Panel;
analyst context does not join the configuration registry.

```text
loaded: 6 runs × 30 columns
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

`panel.cells` is an ordinary polars DataFrame; exploration is
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
GAMMA_EFFECT = DoEffect.from_values(
    source='gamma',
    reference=0.80,
    treatment=0.99,
)

@claim_bridge(
    source=GAMMA_EFFECT,
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

The bridge is exactly what a native one looks like — no external
special case: a `DoEffect` in source position, here declaring the
exact reference and treatment values of an externally-executed
contrast rather than slot-swap arms. The population scope only
selects CartPole-v1. The Panel already carries the one fact the
gates need from the data side — which columns were
*configuration*, recovered from the checkpoints themselves,
never hand-listed — so evaluation is the claim and the record:

```python
evaluation = evaluate(higher_gamma_improves_return, panel)
```

Exact matches to 0.80 and 0.99 receive the stable symbolic arm
identities `baseline` and `treatment`; neither membership nor
orientation is inferred from observed support or display labels.
Any additional gamma levels remain in the accumulated record
but sit outside this declared contrast, available to other
claims over the same Panel.

The admission gates then check the contrast over exactly the
cells this claim admits. The Panel's registry is authoritative
about what was configured: the declared source must be a registered
configuration column (a measurement cannot be the assigned
parameter of an intervention — BLOCK), a registered leaf that
moves with gamma inside a seed pair is a co-varied knob (a
confound — BLOCK, unless it was assigned jointly, in which case
the declaration widens to
`from_values(reference={...}, treatment={...})`), and an
unregistered column moving with the contrast warns — a label is
harmless, an unlogged knob is not, and only the author can say
which. `contrast_present` blocks when a declared arm is absent
from the record, and `pair_completeness` warns when a seed is
missing one condition. A blocked claim gets the verdict
`INADMISSIBLE` — quality problems land on the verdict record,
not in a separate report. What no registry can attest —
assignment, randomisation, hidden confounding — stays outside
the framework's claims for external runs.

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
recomputes on the larger run set — batches pool with
`concat_panels`, because a growing study is one run set that
happens to arrive in parts (the pooled Panel keeps carrying the
union registry). A verdict that moves as evidence accretes is
the system working. The declared estimand stays fixed while the
gates re-check arm presence, configuration isolation, and pairing
over whatever the record has become.

## Producer-side cost

Zero. Not zero files — zero *anything*: `train.py` is the
tutorial-shaped SB3 script, and the analysis reads the artifacts
SB3 already writes. If you have a folder of checkpoint zips and
`EvalCallback` logs from last month, this works on it today.
