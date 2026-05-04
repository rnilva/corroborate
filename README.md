# corroborate

Find the *scope* of a mechanism claim, then verify the *causal
chain* that explains it.

## What this is for

A mechanism claim is an authored algorithmic intervention plus a
theorem justifying its effect. Hasselt 2010's DDQN claim, for
instance: *swap argmax-Q-target-net for double-action-selection,
because action-selection / value-evaluation decoupling reduces
the Jensen-gap-induced overestimation bias.* The intervention is
the swap; the theorem names a *premise* (the gap is what makes
single-DQN biased).

Real performance of such claims is heterogeneous — DDQN helps in
some envs, hurts in others. The literature usually reports a
single unconditional verdict that obscures this. `corroborate`
makes the heterogeneity legible in two phases:

1. **Find scope.** Find the cleavage axis along which the
   mechanism's effect splits the corpus. The framework's
   preferred axis is the *invariance gap* — the residual of the
   theorem's premise, measured per env. Where the gap is large,
   the mechanism's target is salient; where the gap is small,
   the mechanism has nothing to grip on. Scope is empirical,
   not authored.

2. **Verify the causal chain.** Test each edge of the chain
   `env feature → invariance gap → mechanism activation →
   outcome` as a typed `Bridge` with a Pearl tier
   (associational / interventional) and a power-aware Verdict.

The invariance gap is the load-bearing node — both the
scope-defining feature (Phase 1) and the causal mediator
(Phase 2). **Authoring an invariant per mechanism claim is the
substrate-author's primary commitment.**

## Three composable capabilities

The framework provides:

- **(a) Intervention study** — `apply_interventions(base,
  interventions)` re-runs the system with typed structural
  swaps on the claim graph. Active intervention, not
  observational reconstruction.
- **(b) Falsification** — power-aware verdict trichotomy with
  explicit MDE tracking and an `xfail`-style
  `predicted_direction='null'` analog. "Below MDE" is a
  first-class verdict, distinct from both confirmation and
  refutation.
- **(c) Causal discovery** — typed `CausalGraph` of
  `BridgeEdge`s with Pearl tier and direction; conservative-PC
  adjacency + DoWhy backdoor + refutations as registered
  analyses.

These are substrate, not application. Phase 1 + Phase 2 compose
them; one-shot reproducibility audits and dialectic loops compose
them differently. All three reuse the same primitives.

## Authoring shape

A bridge file is a Python module exporting `INTERVENTION:
DoEffect` + `BRIDGES: tuple[Bridge, ...]` — anything structurally
satisfying `corroborate.core.hypothesis.Hypothesis` works
(modules, classes-with-`ClassVar`s, frozen dataclasses).

```python
from functools import partial
from corroborate import claim_bridge
from corroborate.bridge import Direction, Tier, Verdict
from corroborate.core.intervention import DoEffect, Intervention

# Typed structural delta on the claim graph.
DDQN_SWAP = Intervention(
    slot_path='bootstrap',
    replacement=partial(bootstrap, greedification=double_greedify),
)
INTERVENTION = DoEffect(treatment=(DDQN_SWAP,), baseline=())

@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    predicted_direction='a_lt_b',
    pair_by=('seed',),
)
def ddqn_reduces_jensen_gap(paired_g: PairedGResult) -> Verdict:
    if paired_g.n_pairs < 30:
        return Verdict.POWER_INSUFFICIENT
    if paired_g.g < -0.3 and paired_g.p_value < 0.05:
        return Verdict.HELD
    return Verdict.NO_EFFECT
```

The `paired_g` parameter (no default) is a fixture — the
framework's `@analysis`-registered `paired_g` runs against the
bridge-filtered cells and the result is injected by name. The
function body is the threshold logic; the decorator args are the
edge metadata. See `experiments/findings/dqn_bridges.py` for
the canonical zoo (32 bridges).

## Verdict + predicted_direction

`Verdict ∈ {HELD, NO_EFFECT, POWER_INSUFFICIENT,
HELD_WITH_SCOPE_FLAG, INVARIANT_VIOLATION}` paired with
`PredictedDirection ∈ {'a_gt_b', 'a_lt_b', 'two_sided', 'null',
None}`. The convention is uniform: **HELD = prediction
confirmed**, regardless of which direction was predicted.

| `predicted_direction` | HELD encodes |
|---|---|
| `'a_gt_b'` | positive-direction effect detected |
| `'a_lt_b'` | negative-direction effect detected |
| `'two_sided'` | non-zero effect detected |
| `'null'` | **no effect detected (xfail-style: prediction-of-null confirmed)** |

The `(verdict, predicted_direction)` tuple disambiguates "DDQN
reduces bias" (HELD, `'a_lt_b'`) from "DDQN's outcome benefit
is null" (HELD, `'null'`) — both are HELD because both predictions
were confirmed; the predicted direction names which prediction.

## Running it

```bash
# Author bridges in `experiments/findings/<X>.py`; run them
# against a corpus via the canonical CLI:
uv run python scripts/run_hypothesis.py experiments.findings.dqn_bridges \
    --data experiments/data/<corpus>/runs.parquet
```

`runner.run(h: Hypothesis | str, *, data, cache_path, ...)` is
the library entry; the CLI is a thin argparse wrapper. The
runner caches measurables per-hypothesis under
`experiments/data/cache/<short>.parquet` so re-evaluating bridges
on the same corpus skips recomputation.

For YAML-authored sweeps:
```bash
uv run python experiments/run_yaml_sweep.py \
    experiments/configs/<sweep_name>.yaml
```

## Status

Pre-v0. The acceptance test is a DDQN-vs-vanilla study
reproducing the `mechanism HELD ↛ outcome HELD ↛ link HELD`
verdict pattern across the canonical 17-env corpus. Current
state: 32 bridges across `experiments/findings/dqn_bridges.py`
+ `ddqn_universe.py`, exercising the typed Phase-6 contract
(`Hypothesis` Protocol + typed `DoEffect` Interventions) end-to-
end.

## Documentation

- `CLAUDE.md` — typing discipline, vocabulary, canonical
  analyses, contributor instructions.
- `ANALYSIS_RECIPE.md` — post-sweep analysis sequence (classify
  cells → bridges → meta-regression → PC → robustness →
  per-burst → tautology audit → data-driven intervention
  selection).
- `SCOPE_SEARCH.md` — the scope-finding procedure (Phase 1).
- `LIFECYCLE.md` — corpus + verdict lifecycle from cell-runner
  to bridge evaluation.
- `FUTURE_WORKS.md` — explicit deferrals and open questions.
- `FINDINGS.md` — historical narrative log of empirical findings.
