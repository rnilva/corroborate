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

Two layers stack: individual bridges (one falsifiable edge each)
and hypothesis modules (the package surface that composes them).
The canonical primary usage is hypothesis modules — modular
packages organized per claim and per finding, conventionally
under `experiments/findings/<name>/` in the consuming project.
(The frozen DDQN study on the `submission` branch is the
fully-worked example throughout this README.) **`Panel`** is the
substrate-author's pre-authoring exploration surface that
precedes both.

### Day-1/2 exploration via `Panel`

Before authoring bridges, a substrate author probes the data to
decide the scope predicates + cluster shape the bridges will
encode. `Panel` is the typed entry point — no `@claim_bridge`
harness, no ingest dance, no need to declare a hypothesis
module first:

```python
from corroborate.data import Panel, DerivedSpec
import polars as pl

# Three load entry points: from one corpus, multiple corpora, or
# an existing per-hypothesis cache.
panel = Panel.from_cache('experiments.findings.<hyp>')
# panel = Panel.from_corpora(['experiments/data/<corp_a>', ...])
# panel = Panel.from_corpus('experiments/data/<one_corpus>')

# Narrow scope — extends the panel's scope_chain so later
# inspectors know what filtered down to this cohort.
canonical = panel.narrow(pl.col('gamma') == 0.999).narrow(
    pl.col('action_duplicate_k').is_null()
    | (pl.col('action_duplicate_k') == 1),
)

# Probe per-stratum diagnostics — surfaces HP-mixing across
# corpora, cohort heterogeneity, finite-fraction per measurable.
diag = canonical.diagnostics
for stratum, n in diag.n_cells_per_stratum.items():
    if diag.nonunique_configs_per_stratum[stratum] > 1:
        print(f'⚠ {stratum} carries {n} cells across multiple configs')

# Analyze with framework primitives directly — `panel.cells` is
# a pl.DataFrame, which is the canonical @analysis input.
from corroborate.analyses.spearman.partial_spearman import partial_spearman
res = partial_spearman.fn(
    canonical.cells,
    x='jensen_dormancy_gap', y='jensen_gap',
    conditioning=(), stratify_by='env_name', min_stratum_size=30,
)

# Per-stratum aggregate via DerivedSpec — same kernel the
# framework's per-stratum analyses use.
sigmas = canonical.derive(DerivedSpec(
    column='lambda_a_late',
    aggregator='std',
    cell_filter=pl.col('arm_key') == 'baseline',
))

# Promote: when exploration finds the right scope, write the
# narrowed cohort as a per-hypothesis cache (+ sources sidecar).
canonical.to_cache('experiments.findings.<new_hyp>')
```

See `experiments/findings/hasselt_clean/_exploration.py` on the
`submission` branch for a fully-worked Day-1/2 substrate-author
walk-through that decides the scope predicates ultimately encoded
in `hasselt_clean/_scope.py`.

### A bridge

```python
from functools import partial
from corroborate import claim_bridge
from corroborate.bridge import Direction, Tier, Verdict
from corroborate.core.intervention import DoEffect, Intervention

DDQN_SWAP = Intervention(
    slot_path='bootstrap',
    replacement=partial(bootstrap, greedification=double_greedify),
)
# Two-arm contrast: arms[0] is the empty-tuple baseline (vanilla);
# arms[1] is the treatment (DDQN swap).
INTERVENTION = DoEffect(arms=((), (DDQN_SWAP,)))

@claim_bridge(
    source=INTERVENTION,
    target='jensen_gap',
    direction=Direction.INVERSE,
    tier=Tier.INTERVENTIONAL,
    predicted_direction='a_lt_b',
    stratify_by=('env_name',),
)
def ddqn_reduces_jensen_gap(
    stratified_arm_diff_pooled: StratifiedArmDiffPooledResult,
) -> Verdict:
    return stratified_arm_diff_pooled.verdict
```

`stratified_arm_diff_pooled` is a **fixture** — the framework's
`@analysis`-registered `stratified_arm_diff_pooled` runs against
the bridge-filtered cells and the result is injected by name. The
function body is the threshold logic (the canonical primitive
returns its own `verdict` per the verdict trichotomy + scope-flag
discipline); decorator args are edge metadata. See CLAUDE.md
§"Canonical analyses" for which primitive to reach for in which
shape — `paired_g` is **off-limits for RL substrate bridges**
because seeds pseudo-replicate across strata; `stratified_arm_diff_pooled`
is the canonical cross-env / cross-config pool.

### A hypothesis module

A hypothesis module / package satisfies the
`corroborate.core.hypothesis.Hypothesis` Protocol by exposing
six module-level names. Canonical layout (the frozen study's
`experiments/findings/ddqn/__init__.py`, `submission` branch):

```python
import corroborate.analyses                  # populate analysis registry
import corroborate_rl.dqn.measurables        # populate measurable registry

from experiments.findings.ddqn._arms import INTERVENTION
from experiments.findings.ddqn._common import CLAIM
from experiments.findings.ddqn._scope import MODULE_SCOPE

# Per-claim sub-modules export their own BRIDGES tuples; the
# package re-aggregates.
from experiments.findings.ddqn.bias_correction import BRIDGES as _BIAS
from experiments.findings.ddqn.mediation import BRIDGES as _MEDIATION
from experiments.findings.ddqn.q_shape_mediation import BRIDGES as _Q_SHAPE
# ... (one sub-module per theoretical claim)

# Per-finding sub-modules each define EXPECTED + BLOCKED_ON +
# BRIDGES (cluster-shaped claims; see "Findings" below).
from experiments.findings.ddqn import (
    finding_hasselt_chain,
    finding_polarity_conditional_chain,
    # ...
)

BRIDGES = (*_BIAS, *_MEDIATION, *_Q_SHAPE, ...)
FINDINGS = (finding_hasselt_chain, finding_polarity_conditional_chain, ...)

REQUIRED_MEASURABLES: tuple[str, ...] = (
    'q_per_burst', 'state_conditional_argmax_entropy_late', ...,
)
```

Six load-bearing module-level names:

| name | role |
|---|---|
| `INTERVENTION: DoEffect` | typed structural delta on the claim graph (the "do" the bridges arm against) |
| `CLAIM` | outermost claim for endogeneity gating |
| `MODULE_SCOPE: pl.Expr` | scope universe applied at ingest time — every cell that enters the cache satisfies it. Bridges scope further on its columns. |
| `BRIDGES: tuple[Bridge, ...]` | flat tuple of all `@claim_bridge`-decorated functions in the package |
| `FINDINGS: tuple[Finding, ...]` | cluster-shaped claims composed from subsets of `BRIDGES` (see below) |
| `REQUIRED_MEASURABLES: tuple[str, ...]` | opt-in measurables to pre-populate at ingest, even when no current bridge consumes them. Validated against the registry at `_validate_hypothesis` |

**Sub-module shape** — the study's `ddqn/` exemplifies one
decomposition that the framework doesn't enforce; siblings adapt
as needed:

- **`ddqn/`** is the most decomposed: package-private constants in
  `_arms.py` / `_common.py` / `_scope.py` / `_verdicts.py`; one
  `<claim>.py` per theoretical unit (`bias_correction.py`,
  `mediation.py`, `q_shape_mediation.py`, ...) each exporting its
  own `BRIDGES` tuple; one `finding_<name>.py` per cluster-shaped
  finding (each exports `EXPECTED`, `BLOCKED_ON`, `BRIDGES`).
- **`ddqn_sweeps/`** is flatter — no underscore-privates; reuses
  `INTERVENTION` and `CLAIM` from `ddqn._arms` / `ddqn._common`;
  inlines `MODULE_SCOPE`. Per-claim and per-finding files only.
- **`ddqn_three_conditions/`** has `_arms.py` / `_common.py` /
  `_measurables.py` (hypothesis-local `@measurable` registration
  for derived columns: `shaping_kind`, `fa_kind`, `k_eff`) but
  collapses all four bridges into one `conditions.py`. The
  per-claim split is a bridge-count heuristic, not a rule.

Each `<claim>.py` is a flat module with `@claim_bridge`-decorated
functions and a closing `BRIDGES = (bridge_a, bridge_b, ...)`
tuple — the per-claim sub-modules are NOT packages, just regular
Python modules.

The frozen study (`submission` branch) carries three hypothesis
packages:

- `experiments/findings/ddqn/` — the DDQN canonical study.
  ~58 bridges across 9 per-claim sub-modules; 9 findings spanning
  Hasselt's chain, mediation, channel decomposition, polarity-
  conditional moderation. (Note: two extra `finding_*.py` files
  on disk — `finding_ddqn_bias_channel.py`,
  `finding_ddqn_policy_structure_channel.py` — exist but aren't
  registered in `FINDINGS`; treat as orphan / dead code.)
- `experiments/findings/ddqn_sweeps/` — companion HP-sweep bridges
  (n-step, reward-scale, Polyak-τ, γ-sweep, ...). ~15 bridges,
  3 findings. Loose `MODULE_SCOPE` (just `~bsuite`) so each bridge
  opts INTO its own HP regime via per-bridge `scope=`.
- `experiments/findings/ddqn_three_conditions/` — multi-stratum
  panel claims (linear-FA caps Type 1; shaping decouples; k-sweep
  panel). 4 bridges, 1 finding (the panel cluster).

## Findings — cluster-shaped claims on the post-evaluated graph

A `Finding` is a typed subgraph of a Hypothesis's evaluated
causal graph asserting an aggregate verdict. Module-level
attributes only (mirrors `Hypothesis`):

```python
# finding_hasselt_chain.py (frozen study, `submission` branch)
from corroborate.bridge.bridge import Bridge
from corroborate.graph.causal import ClusterVerdict
from experiments.findings.ddqn.bias_correction import (
    algorithm_reduces_bootstrap_gap_magnitude,
    bootstrap_gap_predicts_jens__theorem,
    intervention_outcome_link_null__mech_conditioned,
    mc_disc_raw_coupled__per_env_jci,
)

EXPECTED: ClusterVerdict = ClusterVerdict.SUPPORTED
BLOCKED_ON: str | None = None
BRIDGES: tuple[Bridge, ...] = (
    mc_disc_raw_coupled__per_env_jci,
    algorithm_reduces_bootstrap_gap_magnitude,
    bootstrap_gap_predicts_jens__theorem,
    intervention_outcome_link_null__mech_conditioned,
)
```

The framework derives the cluster's verdict via `composed_verdict(g,
bridges=BRIDGES)`: every named bridge admits → `SUPPORTED`; any
refutes → `REFUTED`; mix admit / unevaluated → `UNDERPOWERED`;
all members admit zero cells → `EMPTY_EXTENT` (corpus can't
distinguish them).

`EXPECTED` pins the **EMPIRICAL** state (not the theoretical
claim). When data hasn't caught up, pin to the current verdict +
set `BLOCKED_ON` to a non-`None` string naming the gap; the
renderer surfaces `[blocked]` for the pending state and `← DRIFT`
only when the actual verdict diverges from `EXPECTED`. See
CLAUDE.md §"Findings" for the full authoring discipline.

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
# Run a hypothesis module / package. Four CLI modes:
uv run corroborate hypothesis \
    experiments.findings.<hyp>                        # read-only (default)
uv run corroborate hypothesis \
    experiments.findings.<hyp> --check                # drift report, no work
uv run corroborate hypothesis \
    experiments.findings.<hyp> \
    --ingest <corpus>[,<corpus>...]                   # named ingest
uv run corroborate hypothesis \
    experiments.findings.<hyp> \
    --ingest-all experiments/data/                    # walk full root
```

The legacy `python scripts/run_hypothesis.py ...` invocation
continues to work as a back-compat shim that forwards to the same
`corroborate.cli.hypothesis.main`.

The runner imports the hypothesis module, validates the
`Hypothesis` Protocol shape, populates / extends
`experiments/data/cache/<hyp>.parquet` with the cells `BRIDGES`
need, evaluates each bridge, then surfaces:

- per-bridge verdicts (HELD / NO_EFFECT / POWER_INSUFFICIENT /
  ...);
- per-finding cluster verdicts via `composed_verdict(graph,
  bridges=f.BRIDGES)` — each Finding's runtime verdict gets
  compared to its author-pinned `EXPECTED`, with `← DRIFT`
  flagged when they diverge;
- a snapshot at `experiments/findings/<short>.run.json` (where
  `<short> = h.__name__.split('.')[-1]`) — the audit baseline
  alongside the bridges file.

`--check` itself does NOT diff against the snapshot. It runs
two drift checks without computing anything: (a) measurable-
closure drift via per-corpus `<corpus>/measurements.hashes.json`,
and (b) cache-source drift via `<cache>.sources.json`. Finding
verdict drift surfaces only on a full run (above).

`runner.run(h: Hypothesis | str, *, data, cache_path, ...)` is
the library entry; the CLI is a thin argparse wrapper. Per-corpus
`measurements.parquet` stores are the source of truth; the
per-hypothesis cache is a projection.

### Sweeps + traces

```bash
# YAML-authored sweep via the framework CLI. Substrate (e.g. the
# in-tree DQN substrate at `corroborate_rl.dqn_sweep`) plugs in
# via a typed `SWEEP_ENTRY_POINTS` module-level export; framework
# knows nothing about JAX. Writes per-arm subdirs, merges to
# top-level `<out_dir>/{runs,traces}.parquet`, archives to cloud.
# Drops a `.in_progress` sentinel for the duration so concurrent
# `--ingest-all` walks skip the half-built corpus.
set -a && . .env && set +a   # AWS creds for archive
uv run --package corroborate_rl corroborate sweep run \
    --substrate corroborate_rl.dqn_sweep --device gpu \
    experiments/configs/<sweep>.yaml

# Optional: pre-register bridge commitments at sweep launch
# (writes <out_dir>/pre_registration.json immutably) by adding
# a `pre_registered_bridges:` list to the YAML. Audit post-sweep:
corroborate audit pre-registration <out_dir>

# Inspect a trace's schema + which measurables it can satisfy
# without materialising data:
PYTHONPATH=. uv run python scripts/trace_schema.py \
    experiments/data/<corpus>/traces.parquet \
    experiments.findings.<bridges_module>
```

Per-step trace columns (`online_max_q_per_step`, etc.) are the
heavyweight part of a corpus (~MB per cell). The runner reads
them only when computing measurables, then evicts them
(corpus-integrity invariant CI7; the CI1–CI8 catalogue lives in
`CORPUS_INTEGRITY.md` on the `submission` branch). For OOM-prone
cases, the opt-in
`corpus.measurements.compute_trace_measurables_streaming`
iterates row-groups instead of materialising the full join.

## Cache + cloud

Two storage layers + two corpus roots:

```
PER-CORPUS STORES                     PER-HYPOTHESIS CACHE
  source of truth                       derived projection
─────────────────────────             ────────────────────
experiments/data/<corpus>/            experiments/data/cache/
experiments/probes/<corpus>/            ├── <hyp>.parquet      (cells × measurables)
  ├── runs.parquet                      └── <hyp>.sources.json (per-corpus input provenance)
  ├── measurements.parquet
  ├── measurements.hashes.json        experiments/findings/<hyp>.run.json
  ├── traces.parquet                    └── (verdict snapshot)
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
uv run corroborate catalogue \
    experiments/data experiments/probes \
    --remote-prefix s3://corroborate-archive/

# Per-(corpus, arm) leaf-signature view — the configurational
# fingerprint each sweep arm holds constant vs sweeps over.
uv run corroborate catalogue \
    experiments/data experiments/probes \
    --leaves --leaves-wide
```

### Cloud credentials + preflight

The framework uses **botocore's standard credential resolution
chain** — no `.env` auto-loading, no custom providers. Configure
via any of:

1. **`~/.aws/credentials` (recommended)**: profile-based, picked
   up automatically. Pair with `~/.aws/config` for the
   per-service `endpoint_url` (R2 / non-AWS S3 backends need it):
   ```ini
   # ~/.aws/credentials
   [r2]
   aws_access_key_id = ...
   aws_secret_access_key = ...

   # ~/.aws/config
   [profile r2]
   region = auto
   services = r2-endpoint

   [services r2-endpoint]
   s3 =
     endpoint_url = https://<account>.r2.cloudflarestorage.com
   ```
   Then: `corroborate archive <dir> --remote s3://... --profile r2`
2. **Environment variables** (`AWS_ACCESS_KEY_ID`,
   `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`): work without
   profile config. If you keep them in `.env`, source it
   explicitly (`set -a && . .env && set +a`) — there's no
   auto-load.
3. **IAM role** (EC2 / ECS): zero config; the chain picks it up.

Every cloud-touching CLI entry runs a **preflight check** that
fails fast with a typed stage + actionable hint. Covered:

- `corroborate {archive, restore, ls, catalogue --remote-prefix}`
  — preflights against the per-command remote (always for
  `archive`, derived from the local manifest for `restore` / `ls`,
  from `--remote-prefix` for `catalogue`).
- `corroborate sweep run` — preflights when the sweep YAML has
  `archive_remote` set; **fails before the sweep loop kicks off**
  (which can be hours of compute). Skip with `--skip-preflight`.
- `corroborate hypothesis` (and the `scripts/run_hypothesis.py`
  shim) — preflights when `--ingest` / `--ingest-all` is used AND
  `--no-restore` is NOT set AND any corpus under the ingest scope
  (2 levels deep, mirroring the catalogue's `_MAX_DEPTH`) carries
  `_remote.json`. Skip with `--skip-preflight`.

Stages:

| stage              | example hint                                                                                  |
|--------------------|------------------------------------------------------------------------------------------------|
| `no_credentials`   | "Set AWS_ACCESS_KEY_ID or pass --profile <name>."                                              |
| `auth_failed`      | "Credentials accepted by boto3 but rejected by the bucket (HTTP 403). Verify key/secret."     |
| `bucket_missing`   | "No bucket at that name. Check the URI AND AWS_ENDPOINT_URL (R2 ≠ s3.amazonaws.com)."         |
| `network`          | "Could not reach endpoint. Check AWS_ENDPOINT_URL."                                            |

Python API: `from corroborate._internals.cloud_auth import preflight`
— call explicitly for library use.

### Archive lifecycle

```bash
# Per-command --profile flag (or rely on AWS_PROFILE env var):
uv run corroborate archive experiments/data/<corpus> \
    --remote s3://corroborate-archive/<corpus> --profile r2

# Inspect what a corpus has archived:
uv run corroborate ls experiments/data/<corpus> --profile r2

# Restore (e.g. to re-derive trace-dependent measurables):
uv run corroborate restore experiments/data/<corpus> --profile r2

# Purge LOCAL copies of cloud-archived files (manifest preserved,
# `restore` stays available). NEVER `rm` cloud-backed files
# directly — `purge` validates the manifest first. (No preflight:
# purge is local-only.)
uv run corroborate purge experiments/data/<corpus>
```

### Cache mechanics

- **Per-corpus measurements** (`<corpus>/measurements.parquet`):
  column-additive. Each `@measurable` is hashed (closure +
  reads + name); the `measurements.hashes.json` sidecar tracks
  which hashes are current. Drifted / missing measurables are
  recomputed on `--ingest`.
- **Per-hypothesis cache** (`cache/<hyp>.parquet`): a
  `diagonal_relaxed` concat over the per-corpus stores, filtered
  by the hypothesis's `MODULE_SCOPE`. Rebuilt atomically each
  directory-walk ingest. Bridges scope on its columns.
- **Cache-sources sidecar** (`cache/<hyp>.sources.json`): per-
  corpus input provenance — `data_root`, `remote_root`, and an
  append-only `ingested_at` audit trail. Mutates lockstep with
  the cache parquet (build wire-in updates entries on each
  `--ingest`; `evict` drops entries). Read via
  `runner.check_cache_sources(cache_path)` or the input-side
  section of `--check`. Distinguishes pre-sidecar caches
  (`NO_SIDECAR_RECORD` status) from drift (`DRIFTED`) from
  evicted-not-mirrored (`STALE_SIDECAR_ENTRY`).
- **Corpus stamp**: cells carry a `corpus` column derived from
  the leaf directory name — `<name>` for top-level corpora,
  `<parent>/<name>` for nested sub-corpora (parent dir has its
  own `runs.parquet`). Distinguishes sub-corpora that share leaf
  names across different parents.
- **Trace eviction (CI7)**: trace columns are heavy (~MB per
  cell) and cloud-recoverable. Local `traces.parquet` is evicted
  post-measurable-compute when the cloud manifest confirms a
  sha256-matching archived copy.
- **`.in_progress` sentinel**: a sweep mid-flight drops this
  file; `--ingest-all` walks skip the corpus until it lands
  (invariant CI1).

### When to reach for which command

| situation | command |
|---|---|
| "what corpora do I have, locally and in cloud?" | `corroborate catalogue` (above) |
| "I added a new @measurable, refresh the cache" | `corroborate hypothesis <module> --ingest <corpus>[,…]` or `--ingest-all <root>` |
| "I deleted a sweep dir, the cache still has its cells" | `corroborate hypothesis <module> --evict <corpus>` |
| "drift check, no work" | `corroborate hypothesis <module> --check` (reports both measurable-side AND input-side drift) |
| "which caches depend on corpus X? did the source drift?" | `cat experiments/data/cache/*.sources.json \| jq` or `runner.check_cache_sources(<cache_path>)` |
| "trace cols missing on a new measurable" | runner auto-restores from cloud; ensure `.env` is sourced |
| "free disk on a cloud-backed corpus" | `corroborate purge` (NEVER `rm`) |

The full internal design docs — `CACHE_ARCHITECTURE.md`,
`CACHE_ADDITIVITY.md` (the named-ingest contract), and
`CORPUS_INTEGRITY.md` (the CI1–CI8 invariants the runner
enforces) — are frozen on the `submission` branch.

## Status

Pre-v0. `main` is the clean framework repo: `src/corroborate`
(the framework), `src/corroborate_rl` (the in-tree DQN substrate),
tests, and the four docs listed below.

The acceptance test is a DDQN-vs-vanilla study reproducing the
`mechanism HELD ↛ outcome HELD ↛ link HELD` verdict pattern
across the canonical 12-env panel. That study — 3 hypothesis
packages, ~77 bridges, 13 findings, data manifests, cached
verdict snapshots, and the full internal design-doc set — is
frozen verbatim on the **`submission`** branch for paper
reproduction. Every `experiments/...` path in this README refers
to that branch's tree (or to the same conventional layout in
your own consuming project).

## Documentation

On `main`:

- `CLAUDE.md` — typing discipline, vocabulary, canonical
  analyses, contributor instructions.
- `HYPOTHESIS_AS_GRAPH.md` — the framework's organizing
  principle. Why a hypothesis IS a causal graph (not a claim),
  the bridge-naming + refutation-cluster + scope-as-extent
  authoring discipline. Pair-read with the Findings section
  above.
- `REPRODUCIBILITY.md` — what same-seed actually buys you:
  bitwise vs scientific reproducibility under XLA configuration
  (determinism flag, TF32, CUDA-graph capture), and the
  cross-mode analysis discipline it forces.

On the **`submission`** branch (frozen with the study): the
internal design docs (`ANALYSIS_RECIPE.md`, `SCOPE_SEARCH.md`,
`LIFECYCLE.md`, `CACHE_ARCHITECTURE.md`, `CACHE_ADDITIVITY.md`,
`CACHE_BUILD.md`, `CORPUS_INTEGRITY.md`, ...), the historical
findings log (`FINDINGS.md`), and the deferral list
(`FUTURE_WORKS.md`).
