# corroborate

Test the *mechanism* behind an RL algorithm, not just its score.

> **Paper artifact.** The frozen DDQN study that accompanies
> *Corroborate: A Framework for Testing Mechanism Claims in
> Reinforcement Learning* (Finding the Frame @ RLC 2026) —
> including the paper PDF, its data, and its figure pipelines —
> lives on the [**`submission` branch**](../../tree/submission).
> `main` is the living framework.

## The problem

A paper that proposes a mechanism — say Double-DQN's *decoupled
action selection reduces overestimation bias* — proves a theorem
under conditions the implementation then relaxes. Whether the
analytical prediction survives in the practical regime is an
empirical question, and a hard one: the mechanism is entangled
with the rest of the implementation, the relevant quantities are
confounded, and the test is closer to causal inference than to
plotting curves. The apparatus built for that test in one paper
is rarely reusable by another.

`corroborate` is a framework for exactly that part. It combines
three principles:

1. **Component boundaries drawn at theorems.** The mechanism is
   isolated as a functional unit that can be swapped without
   disturbing the rest of the algorithm, so a comparison is a
   controlled causal contrast — an intervention, not an
   observational reconstruction.
2. **Hypotheses as executable claims.** Each edge of a
   hypothesis is a small program called a **bridge**: its
   decorator declares the claim's commitments — scope (which
   environments and conditions it applies to), evidential tier
   (associational vs interventional), predicted direction —
   *before any data is seen*, and its body runs as a test that
   returns a verdict.
3. **"We cannot tell" is a verdict.** `POWER_INSUFFICIENT` is
   first-class, distinct from both `HELD` and `NO_EFFECT`, and it
   propagates through chains of composed claims instead of
   collapsing into "no effect."

The demonstration study renders the field's understanding of DDQN
as explicit per-environment claims: bias reduction holds in most
environments; the analytical prediction that it *improves
outcomes* is where the chain breaks — and the framework says so
without overclaiming.

## Terminology: paper ↔ code

The paper uses plain words; some code identifiers and logged
column names predate them and are kept for compatibility:

| paper | code |
|---|---|
| condition (one algorithm variant under study) | `arm`, `arm_key` |
| evaluation window (block of greedy eval episodes) | `burst` (e.g. `eval_best_burst_raw_mean`) |
| run set (the logged collection of runs) | corpus (`runs.parquet` + sidecars) |
| seeded run at an environment–condition pair | cell |
| metric | `@measurable` |
| the implementation under study | "substrate" (e.g. the `--substrate` CLI flag) |

## A bridge, concretely

```python
from functools import partial
from corroborate import claim_bridge
from corroborate.bridge import Direction, Tier, Verdict
from corroborate.core.intervention import DoEffect, Intervention

# The mechanism as a unit of intervention: swap greedification,
# leave everything else untouched.
DDQN_SWAP = Intervention(
    slot_path='bootstrap',
    replacement=partial(bootstrap, greedification=double_greedify),
)
INTERVENTION = DoEffect(arms=((), (DDQN_SWAP,)))  # baseline, treatment

@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',              # the bias the theorem talks about
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    predicted_direction='a_lt_b',     # committed before data is seen
    stratify_by=('env_name',),        # per-environment claims
)
def ddqn_reduces_jensen_gap(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
) -> Verdict:
    return stratified_arm_diff_pooled.verdict
```

The decorator is the claim's declared commitments; the body maps
statistics to a verdict. The named parameter is a registered
analysis the framework runs against the scoped run set and
injects by name. A **hypothesis** is a Python package exposing a
tuple of bridges plus cluster-level **findings**; running it
prints the verdict table.

```bash
uv run corroborate hypothesis experiments.findings.<name>
```

## What's in this repo

| path | contents |
|---|---|
| `src/corroborate/` | the framework: claims, bridges, verdicts, causal-graph evaluation, registered analyses, run-set storage |
| `src/corroborate_rl/` | the DQN/JAX implementation under study (plugs into the framework's sweep CLI) |
| `docs/HYPOTHESIS_AS_GRAPH.md` | the organizing principle: a hypothesis is a causal graph, bridges are its edges |
| `REPRODUCIBILITY.md` | what same-seed actually buys you: bitwise vs scientific reproducibility under XLA configuration |
| `CLAUDE.md` | contributor doc: typing discipline, vocabulary, which analysis to reach for |
| `tests/` | framework tests + a synthetic linear-Gaussian SCM positive control that recovers a known chain in closed form |

Quality gates (both green on a clean checkout):

```bash
uv run pyright          # strict mode, 0 errors
uv run pytest tests/    # fast cohort; `-m ''` adds the slow DQN end-to-end tests
```

## Running studies

The framework is study-agnostic; an implementation plugs in
through a typed entry-point module and a sweep YAML:

```bash
# Train the conditions across environments and seeds (GPU optional):
uv run corroborate sweep run --substrate corroborate_rl.dqn_sweep \
    --device gpu experiments/configs/<sweep>.yaml

# Ingest the logged run set and evaluate a hypothesis against it:
uv run corroborate hypothesis experiments.findings.<name> \
    --ingest <run-set-dir>

# Inventory local + cloud-archived run sets:
uv run corroborate catalogue experiments/data --remote-prefix s3://<your-bucket>/
```

Run sets archive to any S3-compatible store (credentials via
botocore's standard chain); per-window trace columns are evicted
locally once cloud-recoverable and restored on demand. The
`submission` branch is the fully-worked example of all of this,
with the shipped data to reproduce the paper offline.

## License

MIT — see [`LICENSE`](LICENSE).
