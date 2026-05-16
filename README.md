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
# Author bridges in `experiments/findings/<X>.py`. Three CLI modes
# (CACHE_ADDITIVITY.md):
PYTHONPATH=. uv run python scripts/run_hypothesis.py \
    experiments.findings.dqn_bridges                  # read-only (default)
PYTHONPATH=. uv run python scripts/run_hypothesis.py \
    experiments.findings.dqn_bridges --check          # drift report, no work
PYTHONPATH=. uv run python scripts/run_hypothesis.py \
    experiments.findings.dqn_bridges \
    --ingest <corpus>[,<corpus>...]                   # named ingest
PYTHONPATH=. uv run python scripts/run_hypothesis.py \
    experiments.findings.dqn_bridges \
    --ingest-all experiments/data/                    # walk full root
```

`runner.run(h: Hypothesis | str, *, data, cache_path, ...)` is
the library entry; the CLI is a thin argparse wrapper. Per-corpus
`measurements.parquet` stores are the source of truth; the
per-hypothesis cache (`experiments/data/cache/<short>.parquet`)
is a projection.

### Sweeps + traces

```bash
# YAML-authored sweep — writes per-arm subdirs, merges to top-
# level `<out_dir>/{runs,traces}.parquet`, archives to cloud.
# Drops a `.in_progress` sentinel for the duration so concurrent
# `--ingest-all` walks skip the half-built corpus.
set -a && . .env && set +a   # AWS creds for archive
PYTHONPATH=. uv run python scripts/run_sweep.py \
    experiments/configs/<sweep>.yaml

# Inspect a trace's schema + which measurables it can satisfy
# without materialising data:
PYTHONPATH=. uv run python scripts/trace_schema.py \
    experiments/data/<corpus>/traces.parquet \
    experiments.findings.<bridges_module>
```

Per-step trace columns (`online_max_q_per_step`, etc.) are the
heavyweight part of a corpus (~MB per cell). The runner reads
them only when computing measurables, then evicts them
(CORPUS_INTEGRITY.md CI7). For OOM-prone cases, the opt-in
`corpus.measurements.compute_trace_measurables_streaming`
iterates row-groups instead of materialising the full join.

## Cache + cloud

Two storage layers + two corpus roots:

```
PER-CORPUS STORES                     PER-HYPOTHESIS CACHE
  source of truth                       derived projection
─────────────────────────             ────────────────────
experiments/data/<corpus>/            experiments/data/cache/<hyp>.parquet
experiments/probes/<corpus>/          experiments/findings/<hyp>.run.json
  ├── runs.parquet                      ├── (verdict snapshot)
  ├── measurements.parquet              └── (rebuilt each ingest)
  ├── measurements.hashes.json
  ├── traces.parquet
  └── _remote.json (cloud manifest)
                              ↕
                 s3://corroborate-archive/<corpus>/
                 (MANIFEST.json mirror)
```

`experiments/data/` is canonical sweep output; `experiments/probes/`
is ad-hoc pilots. **Both roots carry corpora** — any inventory or
catalogue walk must include both, and `--ingest <name>` resolves
relative names against `experiments/data/` only, so probes need
absolute paths.

### Discovery — `catalogue`

```bash
# Inventory all corpora across local + cloud, with status per row
# (CLOUD_AND_LOCAL / CLOUD_EVICTED / LOCAL_ONLY / IN_PROGRESS_SCAFFOLD).
uv run python -m corroborate catalogue \
    experiments/data experiments/probes \
    --remote-prefix s3://corroborate-archive/

# Per-(corpus, arm) leaf-signature view — the configurational
# fingerprint each sweep arm holds constant vs sweeps over.
uv run python -m corroborate catalogue \
    experiments/data experiments/probes \
    --leaves --leaves-wide
```

### Archive lifecycle

```bash
set -a && . .env && set +a   # AWS creds live in .env

# After a sweep completes, archive its parquets to cloud:
uv run python -m corroborate archive experiments/data/<corpus> \
    --remote s3://corroborate-archive/<corpus>

# Inspect what a corpus has archived:
uv run python -m corroborate ls experiments/data/<corpus>

# Restore (e.g. to re-derive trace-dependent measurables):
uv run python -m corroborate restore experiments/data/<corpus>

# Purge LOCAL copies of cloud-archived files (manifest preserved,
# `restore` stays available). NEVER `rm` cloud-backed files
# directly — `purge` validates the manifest first.
uv run python -m corroborate purge experiments/data/<corpus>
```

### Cache mechanics

- **Per-corpus measurements** (`<corpus>/measurements.parquet`):
  column-additive. Each `@measurable` is hashed (closure +
  reads + name); the `measurements.hashes.json` sidecar tracks
  which hashes are current. Drifted / missing measurables are
  recomputed on `--ingest` (CACHE_ADDITIVITY.md C2/C3).
- **Per-hypothesis cache** (`cache/<hyp>.parquet`): a
  `diagonal_relaxed` concat over the per-corpus stores, filtered
  by the hypothesis's `MODULE_SCOPE`. Rebuilt atomically each
  directory-walk ingest. Bridges scope on its columns.
- **Corpus stamp**: cells carry a `corpus` column derived from
  the leaf directory name — `<name>` for top-level corpora,
  `<parent>/<name>` for nested sub-corpora (parent dir has its
  own `runs.parquet`). Distinguishes sub-corpora that share leaf
  names across different parents
  (`findings_corpus_name_leaf_collision.md`).
- **Trace eviction (CI7)**: trace columns are heavy (~MB per
  cell) and cloud-recoverable. Local `traces.parquet` is evicted
  post-measurable-compute when the cloud manifest confirms a
  sha256-matching archived copy.
- **`.in_progress` sentinel**: a sweep mid-flight drops this
  file; `--ingest-all` walks skip the corpus until it lands
  (CORPUS_INTEGRITY.md CI1).

### When to reach for which command

| situation | command |
|---|---|
| "what corpora do I have, locally and in cloud?" | `corroborate catalogue` (above) |
| "I added a new @measurable, refresh the cache" | `run_hypothesis --ingest <corpus>[,…]` or `--ingest-all <root>` |
| "I deleted a sweep dir, the cache still has its cells" | `run_hypothesis --evict <corpus>` |
| "drift check, no work" | `run_hypothesis --check` |
| "trace cols missing on a new measurable" | runner auto-restores from cloud; ensure `.env` is sourced |
| "free disk on a cloud-backed corpus" | `corroborate purge` (NEVER `rm`) |

See `CACHE_ARCHITECTURE.md` for the full picture, `CACHE_ADDITIVITY.md`
for the named-ingest contract, and `CORPUS_INTEGRITY.md` for the
CI1–CI8 invariants the runner enforces.

## Status

Pre-v0. The acceptance test is a DDQN-vs-vanilla study
reproducing the `mechanism HELD ↛ outcome HELD ↛ link HELD`
verdict pattern across the canonical 17-env corpus. Current
state: 32 bridges across `experiments/findings/dqn_bridges.py`
+ `ddqn/`, exercising the typed Phase-6 contract
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
- `CACHE_ARCHITECTURE.md` / `CACHE_ADDITIVITY.md` /
  `CACHE_BUILD.md` — two-layer cache: per-corpus stores
  (column-additive, closure-hash drift) vs per-hypothesis cache
  (atomically-rebuilt projection). `CORPUS_INTEGRITY.md` —
  the CI1–CI8 invariants the runner enforces on ingest.
- `FUTURE_WORKS.md` — explicit deferrals and open questions.
- `FINDINGS.md` — historical narrative log of empirical findings.
