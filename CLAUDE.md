# corroborate — contributor instructions

## Purpose

`corroborate` finds the *scope* of an authored mechanism claim,
then verifies the *causal chain* that explains it. The chain
runs `env feature → invariance gap → mechanism activation →
outcome`; the invariance gap (residual of the theorem's premise,
measured per env) is the load-bearing node — both the
scope-defining feature and the causal mediator.

The framework provides three composable capabilities on top of
a `@claim`-decorated executable program: (a) intervention study,
(b) falsification, (c) causal discovery. These are substrate, not
application — orchestrators (dialectic loops, audits, data-mining)
compose them differently. Authoring an invariant per mechanism
claim is the substrate-author's primary commitment. See
`README.md`.

**Organizing principle**: a hypothesis is a causal graph, not a
claim. Bridges contribute edges; the hypothesis is the graph.
Cluster-shaped causal claims (a refutation triple, a sibling
mean/median pair) live at the graph level — queried by extent
identity `(source, target, extent_hash)` over the post-evaluated
graph, NOT decided by a central aggregator. See
`docs/HYPOTHESIS_AS_GRAPH.md` for the authoring discipline this
entails (bridge naming, refutation clusters via shared scope
predicates, scope-as-extent).

**Design-doc citations.** Code comments and docstrings cite
internal design docs by name (`CACHE_ADDITIVITY.md`,
`SWEEP_PERSISTENCY.md`, `ANALYSIS_RECIPE.md`, `FINDINGS.md`
revisions, ...). The methodology set is preserved on the
`submission` branch; the rest are stable historical anchors,
not in-tree paths.

## Typing discipline (load-bearing)

The framework's contribution is logical strictness applied to
scientific claims. Its own code must be the same shape — strict
types reject spurious connections and information loss before they
reach a smoke run. This is a **load-bearing** principle, not a
cosmetic one.

**The rule.** No type erasure. C++ template-mind discipline: no
`void*` equivalents anywhere; type information flows end-to-end
through call chains.

### Disallowed unless polymorphism truly requires

- `Any` — opts out of type checking entirely. Almost never
  required; nearly always a sign that the API needs redesign.
- `object` (as a parameter or container element) — accepts
  anything but provides nothing. Forces narrowing at the use
  site, which usually re-introduces erasure. Only acceptable
  inside a *generic upper-bound* container where the polymorphism
  is intrinsic (e.g. a trace holding heterogeneous Claim[P, T]
  calls — `list[Claim[..., object]]` is the upper bound at the
  container boundary, never at element use sites).
- `cast()` — circumvents the checker. Last resort with a
  one-line comment explaining the runtime invariant the type
  system can't express.
- `# type: ignore[...]` — same. Last resort with rationale.
- `getattr` / `setattr` on typed values — these return `Any` /
  accept `object` and erase types. **Redesign the API to expose
  typed attributes on a class instance** rather than dynamic
  attributes on a function. If you find yourself writing
  `setattr(fn, '__is_claim__', True)` followed by
  `getattr(fn, '__is_claim__', False)`, return a typed wrapper
  class from the decorator — its instance attributes are
  statically resolved.

### Required where polymorphism truly applies

- **PEP 695 type parameters** (`def f[T](x: T) -> T:`,
  `class Foo[**P, T]:`). Use these aggressively. ParamSpec
  (`**P`) preserves caller signature through generic wrappers;
  TypeVarTuple (`*Ts`) preserves heterogeneous tuple shape.
- **Protocol** for structural typing — capturing "anything that
  has these typed attributes" without inheritance. Read-only
  attributes via `@property` are how Protocols match
  frozen-dataclass fields (writable Protocol fields don't match
  immutable concrete fields).
- **PEP 742 `TypeIs[T]`** — the narrowing primitive of choice.
  Returns `bool` at runtime but tells pyright to narrow the
  argument both in the True AND False branches:
  ```python
  def is_mapping(v: object) -> TypeIs[Mapping[str, object]]:
      return isinstance(v, Mapping)

  if is_mapping(x):
      # x narrowed to Mapping[str, object]
      ...
  else:
      # x narrowed to (object & not Mapping[str, object])
      ...
  ```
  Prefer `TypeIs` over `TypeGuard` (one-sided narrowing) and
  over `cast` (no narrowing). It's how runtime predicates avoid
  type erasure.
- **Concrete generic types** in containers — `list[Claim[P, T]]`,
  `dict[str, RunRow]`, never bare `list` or `dict`.
- **Frozen dataclasses with `slots=True`** for typed records —
  attribute access is statically resolved; no `getattr` paths.

### Enforcement

- `pyright` strict mode runs on `src/` AND `tests/`. Configured
  in `pyproject.toml`. Both must pass before commit.
- `pytest` covers behavioral invariants. Both green is the gate.
- Standard pre-commit (manual for now): `uv run pyright && uv run pytest`.

### Heuristic before adding a `# type: ignore`

1. Can a Protocol or generic express the constraint?
2. Can a typed wrapper class replace the dynamic-attribute pattern?
3. Is the value's true type a `Literal` or `TypeGuard`-narrowable
   union?
4. If none of the above, write a one-line comment explaining
   exactly what runtime invariant justifies the escape.

If the answer to (1)-(3) is "no" three times in a row, the API
probably needs redesign before adding the ignore.

## Style

- Modern Python only. Target 3.13+; use PEP 695 generics, PEP 692
  TypedDict-Unpack where applicable.
- Frozen dataclasses by default for records. `slots=True` for hot
  paths.
- Standard library first; reach for an external dep only when
  absolutely required (cf. framework-subtraction discipline).
- Docstrings explain WHY, not WHAT — well-named identifiers do
  the WHAT.

## When to introduce a framework primitive

The framework's contribution is logical strictness applied to
scientific claims. Each typed primitive should earn its keep
against plain Python composition. See `PRIMITIVES_AUDIT.md`
(frozen on the `submission` branch) for the full audit +
meta-pattern; the four-question test before adding a new one:

A primitive (typed dataclass, decorator, Protocol, enum) is the
right answer when it:

1. **Encodes a typed contract** the substrate / bridge author
   should obey — `Hypothesis` Protocol, `Bridge`, `Intervention`.
2. **Provides runtime narrowing** that gives pyright real
   information — `runtime_checkable` Protocols, `TypeIs[T]`,
   frozen dataclasses with typed fields.
3. **Does real work beyond labeling** — extracts signature
   metadata, registers in a typed registry, walks the partial
   tree (`@claim_bridge`, `@analysis`, `walk_paths`).
4. **Hits a performance floor** Python composition can't reach —
   polars expressions for parquet-column filters.

A primitive is unnecessary ceremony when it:

1. **Wraps a tuple-of-tuples without adding behavior.** A
   would-be `DoEffect`-like wrapper around already-meaningful
   data — the tuple IS the structure.
2. **Re-exports constants under a class namespace.**
   `class Foo: BAR = ...` is enum-mimicry without enum-purpose.
   Use a plain module file (`foo.py` with `BAR = ...`).
3. **Caches a derived value that's cheap to recompute.**
   `Bridge.params` was the canonical example —
   `inspect.signature` is sub-ms; `@property` is honest.
4. **Expresses composition where Python operators already do
   it.** `And(check_a, check_b)` is what `(check_a, check_b)`
   (tuple) or `expr_a & expr_b` (polars / numpy) already are.

The patterns to **prefer**: tuple-`+` and polars-`&` for
composition; free functions over methods when there's no shared
state; `runtime_checkable` Protocols for substrate-extensible
shapes; module-level constants composed via operators
(the `_FOURROOMS_REGIME` pattern in the frozen study's
`experiments/findings/dqn_bridges.py`, `submission` branch);
frozen dataclasses with `@property` for derived access.

If a candidate primitive doesn't pass the four-question test,
the answer is to leave it as plain Python.

## Vocabulary (paper-aligned prose, compatible identifiers)

**Prose follows the paper.** The published paper deliberately
uses a small plain-word vocabulary — *mechanism, claim, bridge,
verdict, scope, hypothesis, finding* — plus: **condition** (one
algorithm variant under study), **evaluation window** (a block of
greedy eval episodes; the code names these `burst`), **run set**
(the logged collection of runs; the code says corpus), **seeded
run** at an environment–condition pair (the code says cell), and
**the implementation under study** (the code and CLI say
"substrate"). New docs, docstrings, README text, and PR prose use
the paper's words, mentioning the code name once in parentheses
where the reader will meet it (`eval_best_burst_raw_mean`).

**Identifiers and column names do NOT churn.** `arm_key`,
`burst`, corpus paths, `--substrate`, `@measurable` are load-
bearing in parquet schemas, the CLI surface, and the frozen
`submission` archive; renaming them buys prose purity at the
cost of breaking every logged run set. The paper itself bridges
the gap in its terminology paragraph — follow that pattern.

The framework speaks of two kinds of measurables:

1. **Claim outputs** — what running the configured composition
   emits at run time (the record dict's entries: `reward[t]`,
   `loss[t]`, `td_error[t]`, …). Author-named flat keys.
2. **Leaf scalar claims** — non-recursive nodes in the graph of
   claims, observed at *composition time* by walking the bound
   `partial(...)` via `signature.walk_paths`. Dotted topology
   paths (`gamma`, `optimizer.inner.lr`,
   `bootstrap.greedification`).

RL practice calls (2) "hyperparameters". The framework does NOT.
`regime='leaf'`, `aggregate.leaf_signature`, `walk_paths`,
`flatten_leaves` — never `hp_*`. "Leaf" generalises beyond RL
configuration; "HP" leaks domain jargon into framework semantics.

Substrate code is welcome to say "HP" in its own docs/comments
(it's the reader's vocabulary). Framework code does not.

## Two primitive shapes

Substrate authoring uses two shapes; which one applies depends on
whether the entity has theoretical content and whether it carries
construction-time configuration:

1. **Free Claim** — a top-level `@claim`-decorated function. The
   `FnClaim` wrapper auto-records calls; the function IS the
   theoretical operation, no class needed. Bake leaves at
   composition time via `functools.partial`. The default shape
   for everything that's a pure operation:
   *Examples: `bootstrap`, `double_greedify`, `semi_gradient`,
   `uniform_sample`, `linear_epsilon`, `epsilon_greedy`,
   `mlp_forward`, `cnn_forward`, `adam`, `rmsprop`,
   `warmed_update`, `periodic_copy`, `squared_error`.*
   `dqn` itself (a `@claim`'d free function) sits here too.

2. **Config bundle** — a frozen-dataclass that's NOT a Claim.
   Carries construction-time leaves as fields + slot Claims as
   fields + mechanics methods (allocation / state-update glue
   with no theorem attached). The walker surfaces its fields as
   topology leaves; mechanics methods are plain methods, no
   `record_call`. Used when stateful mechanics need to be paired
   with the configuration that parameterises them.
   *Examples: `Replay` (`capacity` / `batch_size` leaves +
   `sample` slot Claim + `init` / `add` / `sample_batch`
   mechanics); `MLP` / `CNN` (`hidden` / `obs_shape` leaves +
   `init` mechanics + `__call__` delegating to the
   `mlp_forward` / `cnn_forward` Free Claim where Hornik 1989
   attaches).*

The discriminator: **does this entity bundle stateful mechanics
that need to be paired with construction-time HPs?** If yes →
config bundle (the methods are the mechanics, the fields are the
HPs and slot Claims). If no, it's just a function → Free Claim
(decorate with `@claim`, configure via `partial`).

The `@claim` decorator is the **single marker** for "this carries
a theorem and records itself." Plain functions (no decorator) are
mechanics — paired with a Claim through delegation (`MLP.__call__`
calls `mlp_forward`) or through bundle methods (`Replay.add`).
Decorator absence IS the negation; there's no `@mechanics` marker.

A method on a config bundle is just a method — it isn't a Claim
even though it's callable. Theoretical content lives on the slot
Claims that the bundle holds (e.g., `Replay.sample` is a field
pointing at `uniform_sample`, which IS a Free Claim — that's
where Lin 1992 attaches) or on the Free Claim that
`__call__` delegates to (e.g., `MLP.__call__` calls
`mlp_forward`, which IS the Hornik 1989 Claim). The bundle is
mechanical organisation around a Claim or slot of Claims.

**Escape hatch (rare).** Substrate authors who genuinely need a
class-based Claim — stateful `__call__` with a theorem attached
directly to the instance, not delegated — write a frozen
dataclass exposing `name: str` and call `record_call(self,
args, kwargs, result)` inside `__call__`. The class structurally
satisfies `Claim[P, T]` without inheritance. Unused in the
current substrate; documented in `claim.py` and tested in
`tests/test_claim.py::test_manual_dataclass_with_record_call`.

## Substrate-author exploration via `Panel`

Before `@claim_bridge` authoring, substrate authors probe data
to decide scope predicates + cluster shape. The framework's
exploration entry point is `Panel` (`corroborate.data.Panel`)
— a typed, frozen dataclass wrapping
`cells: pl.DataFrame` + `scope_chain` + `stratify_by` +
`sources`. Four constructors:

- `Panel.from_corpus(dir)` — one `runs.parquet` + sidecars.
- `Panel.from_corpora([dir_a, dir_b, ...])` — diagonal-relaxed
  union with per-corpus provenance.
- `Panel.from_cache(hyp_module)` — load per-hypothesis cache +
  populate `sources` from `<hyp>.sources.json` sidecar.
- `Panel.from_dataframe(df, ...)` — adapt an externally-built
  DataFrame.

Chaining: `narrow(expr)` extends `scope_chain`; `derive(spec)`
returns per-stratum aggregates via the shared kernel;
`with_measurables(names)` fills nulls from the `@measurable`
registry; `with_traces(cols)` joins trace columns lazily;
`diagnostics` returns typed per-stratum facts (cell counts,
corpora-per-stratum, finite-fraction, nonunique-config heterogeneity).

Closing the loop: `panel.to_cache(hyp_module)` writes the
parquet + `<hyp>.sources.json` (appending an ingested_at
timestamp to each source's audit trail), so an exploration
Panel can be promoted into the production cache without
re-ingest. The cache parquet IS `panel.cells` on disk —
there's no separate Panel format.

`@analysis` primitives consume `pl.DataFrame` directly (not
the `Panel` wrapper); pass `panel.cells` when calling them from
exploration code. The `panel: Panel` bridge fixture pattern
was tried and deleted as theatre — bridges consume @analysis
results by typed parameter name, period. The exploration that
*precedes* bridge authoring is where Panel earns its keep.

Worked example: the frozen study's
`experiments/findings/hasselt_clean/_exploration.py`
(`submission` branch).

## Findings — cluster-shaped claims on the post-eval graph

A `Finding` (in `corroborate.core.finding`) is a typed subgraph
of a Hypothesis's evaluated causal graph that asserts an
aggregate verdict. The Protocol mirrors `Hypothesis`: module-
level attributes (`EXPECTED`, `BRIDGES`, `BLOCKED_ON`, `__name__`),
no callable. The framework derives the verdict via
`composed_verdict(g, bridges=f.BRIDGES)` — every named bridge
admits → SUPPORTED; any refutes → REFUTED; mix admit /
unevaluated → UNDERPOWERED; all members admit zero cells →
EMPTY_EXTENT (corpus can't distinguish them).

Authoring conventions:
- Finding lives at `experiments/findings/<hypothesis>/finding_*.py`.
- Parent `Hypothesis` declares `FINDINGS = (finding_*, ...)` —
  the framework's discovery surface.
- `EXPECTED` pins the EMPIRICAL state, not the theoretical
  claim. If data hasn't caught up, pin to the actual current
  verdict + set `BLOCKED_ON` to a non-`None` string naming the
  gap. The renderer surfaces `[blocked]` for the pinned-pending
  state and `← DRIFT` only when the verdict actually changes.
- Prose claim lives in the module docstring (renderer quotes the
  first line on drift).
- `_validate_hypothesis` enforces `Finding.BRIDGES ⊆
  Hypothesis.BRIDGES` at startup; runtime composed-verdict never
  sees a "missing bridge" path.
- `Finding.BLOCKED_ON` non-None paired with terminal `EXPECTED`
  (SUPPORTED / REFUTED) is author contradiction — renderer flags
  `!! CONTRADICTION` (the author likely forgot to clear
  `BLOCKED_ON` after data landed).

The cluster vs envelope vs chain shape distinction is NOT a
framework primitive — the renderer surfaces structural counts
(`N bridges, M distinct extents`) without naming the pattern.
Cluster integrity, chain composition, etc. are queryable via the
graph operations in `corroborate.graph.causal`; framework doesn't
classify shape, authors don't author it, and pyright doesn't
check shape conformance. Three hand-rolled examples at the
frozen study's `experiments/findings/ddqn/finding_*.py`
(`submission` branch).

## Persistence shape (typed × open)

Each row store splits into two surfaces:

- **Framework-typed** — closed-set enums (`Verdict`,
  `RefutationClass`), lineage IDs (`id`, `parent_id`, `cycle_id`,
  `treatment_arm_id`, …), framework-controlled provenance
  (`timestamp`). Typed dataclass fields. Stable across substrates.
- **Open** — `measurements: Mapping[str, MeasurementLeaf]` where
  `MeasurementLeaf = str | int | float | bool`. Path-keyed
  scalars, substrate-shaped. The substrate decides what's in here.

Two stores join by UUID:

- **Trace store** (`TraceRow`) — per-cell raw observation. Outputs
  (1) as 1-D `list[float]` columns + leaves (2) as scalar columns.
- **Row store** (`RunRow`) — provenance + framework verdict +
  `measurements`. The cross-arm aggregate
  (`HypothesisComparisonRow`) is materialised on demand from
  RunRows via `from_cells`; it has no on-disk persistence pair.

A topology sidecar `<corpus>/graphs.json` is written alongside
the row stores. It carries one `ComputationGraph` per `arm_key`
so post-hoc consumers can recover the per-arm Claim-to-Claim
data flow that ran each cell.

**Hard rule: no JSON-wrapped struct columns in parquet.** Every
heterogeneous-keyed dict (HPs, derived measurables) is flattened
to top-level path-keyed columns at the parquet boundary. Polars
null-pads heterogeneous keys across rows; readers skip nulls. The
benefit is `df.filter(pl.col('optimizer.inner.lr') < 1e-3)` works
at the dataframe level — JSON wrapping kills this.

The path-keyed convention is collision-free by construction:
leaves use **dotted topology paths** (`replay.batch_size`),
trajectories use **flat author-chosen keys** (`reward`),
registered measurables use **bare names** (`jensen_gap`,
`eval_best_burst_mean`). The framework's namespace filter is the
measurable registry itself — `aggregate.leaf_signature` excludes
`registered_names()` from the configurational fingerprint.

## Cache + cloud operator discipline

The framework has two storage layers and two corpus roots; the
operational rules below catch the most common mistakes. (The
corpus roots follow the study convention — `main` ships no
corpora; the frozen study tree with its data manifests lives on
the `submission` branch, and consuming projects use the same
layout.)

- **Two corpus roots**. `experiments/data/` is canonical sweep
  output; `experiments/probes/` is ad-hoc pilots. **Both carry
  corpora.** Any inventory / catalogue walk that omits
  `experiments/probes/` is wrong by construction (single-root
  walks have surfaced 60% false-orphan-rate on the live tree).
- **Discovery via `corroborate catalogue`**. Reach for
  `corroborate catalogue experiments/data
  experiments/probes --remote-prefix s3://<your-bucket>/`
  BEFORE inventing your own glob / walk. The catalogue carries
  `parent`, `name`, `status` (CLOUD_AND_LOCAL / CLOUD_EVICTED /
  LOCAL_ONLY / IN_PROGRESS_SCAFFOLD), and `--leaves` /
  `--leaves-wide` give the per-(corpus, arm) configurational
  fingerprint. Use it to find which corpus actually carries the
  cells a bridge needs.
- **`--ingest <name>` resolution**. The CLI prefixes relative
  names with `experiments/data/`. Probes corpora need
  ABSOLUTE paths (`--ingest "$PWD/experiments/probes/<corpus>"`)
  or they error out. The catalogue's `parent/name` discriminator
  is the canonical address.
- **Corpus stamping** (`runner._corpus_stamp`). Top-level corpora
  stamp `corpus = sub.name`; nested sub-corpora (parent dir has
  its own `runs.parquet`) stamp `corpus = parent.name/sub.name`.
  The parent/leaf form prevents the silent eviction that bit when
  two distinct sub-corpora share a leaf name across parents.
- **Cache-sources sidecar** (`cache/<hyp>.sources.json`). Pairs
  with the existing `<hyp>.hashes.json` (output-side measurable
  provenance) by tracking the INPUT side: which corpora's cells
  contributed, with `data_root`, `remote_root`, and an
  append-only `ingested_at` audit trail. Mutates lockstep with
  the cache parquet via the `--ingest` and `--evict` paths.
  Query via `runner.check_cache_sources(cache_path)` (typed
  `tuple[SourceDrift, ...]`) or the input-side section of
  `corroborate hypothesis <module> --check`. Statuses: `MATCHED`, `DRIFTED`,
  `MISSING_LOCAL`, `NO_SIDECAR_RECORD` (cache pre-dates
  sidecar), `STALE_SIDECAR_ENTRY` (cache evicted but sidecar
  not mirrored — surfaces orphans). Pre-sidecar caches keep
  working; the first re-ingest creates the entry.
- **`purge`, never `rm` on cloud-backed files**. The
  `corroborate purge <corpus>` command validates each file is in
  the manifest before deletion (preserving the manifest so
  `restore` stays available). Direct `rm` on archived files is
  silent data loss the next time you need to ingest.
- **`--recompute-measurables`** (`corroborate.corpus.measurements.
  recompute_corpus_measurables`). Opt-in per-corpus recompute
  from **LOCAL** inputs only — fills in newly-registered
  measurables when `runs.parquet` (and optionally local
  `traces.parquet`) already carry the transitive reads. The flag
  walks each `--ingest` target before the cache merge; the
  rebuilt `measurements.parquet` is then the projection
  source-of-truth. Measurables whose reads aren't satisfied
  locally are reported `unsatisfiable` and NOT computed
  (overwriting finite values with NaN would be silent data
  loss). For cloud-evicted traces, the normal `--ingest` path
  already handles new-measurable computation via
  `_load_one_corpus`'s sidecar-current check; reach for
  `--recompute-measurables` when you've got local traces and
  don't want a cloud round-trip.
- **Cloud credentials via botocore's standard chain.** No
  `.env` auto-loading. Three accepted sources, in order of
  preference: (1) `~/.aws/credentials` + `~/.aws/config` with
  per-profile `endpoint_url` for R2; (2) env vars
  (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
  `AWS_ENDPOINT_URL`) — if kept in `.env`, the variables must use
  these exact AWS_* names (botocore doesn't read provider-specific
  names like `R2_*`); source manually
  (`set -a && . .env && set +a`); (3) IAM role (EC2/ECS).
  Every cloud-touching CLI entry runs a `preflight` check
  (`_internals.cloud_auth.preflight`) that emits a typed
  `CloudAuthError(stage, message, hint)` — stages:
  `no_credentials`, `auth_failed`, `bucket_missing`, `network`.
  Covered entries: `corroborate {archive, restore, ls,
  catalogue --remote-prefix, hypothesis, sweep run}` (`hypothesis`
  gated on `--ingest*` + `--no-restore` not set + any
  `_remote.json` 2 levels deep under the ingest scope; `sweep
  run` gated on the substrate's `sweep.archive_remote`). The
  `scripts/run_hypothesis.py` back-compat shim forwards to the
  same `corroborate.cli.hypothesis.main`. Each
  entry accepts `--profile <name>` for explicit profile
  selection and `--skip-preflight` to opt out (the `--profile`
  export to `AWS_PROFILE` still runs when skipped, so downstream
  fsspec inherits it). Library callers opt into preflight by
  importing it explicitly (no implicit pre-flight on every
  cloud op).

## Sweep + trace discipline

**Sweep entry**: `corroborate sweep run --substrate <module>
<yaml>`. The CLI is substrate-agnostic; the substrate plugs in
via a top-level `SWEEP_ENTRY_POINTS: SweepEntryPoints[S]` (and
optional `SWEEP_CLI_EXTENSIONS: SweepCliExtensions`) module-level
export. The in-tree DQN substrate exports these from
`corroborate_rl.dqn_sweep` (the lightweight entry — NOT under
`corroborate_rl.dqn.*` which eagerly imports JAX). The framework
knows nothing about JAX; substrate's `pre_import_setup(args)`
stamps env vars (`JAX_PLATFORMS`) BEFORE any heavy-import
callable fires. Use:

```bash
# export cloud creds — botocore reads the AWS_* names, so a .env
# must define AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
# AWS_ENDPOINT_URL (an .env with provider-specific names, e.g.
# R2_*, must be mapped to AWS_* before this works):
set -a && . .env && set +a
uv run --package corroborate_rl corroborate sweep run \
    --substrate corroborate_rl.dqn_sweep --device gpu \
    experiments/configs/<sweep>.yaml
```

`scripts/run_sweep.py` was removed; the CLI is feature-equivalent.

**Worktree-discipline for long sweeps.** Long sweeps create
untracked `experiments/data/<corpus>/<arm>/tmp/` directories
that survive only as long as the working tree isn't subject to
operations that delete untracked content (`git clean -fd`,
`git stash --include-untracked`, manual `rm -rf`). Standard
git operations (`pull`, `merge`, `checkout`, `reset` even with
`--hard`) do NOT touch untracked files — those are safe during
a running sweep. If you need to run parallel work that might
include the dangerous commands above, use a separate
`git worktree add ../sweep-worktree main` so the running sweep's
filesystem is independent. The runner also re-`mkdir(exist_ok)`
each cell's parent dir before write as a belt-and-suspenders
defense (commit `39ffb09`) — recovers automatically from a
transient deletion of `tmp/` between cells, but the cell that
was in-flight at the time of deletion is still lost.

Sweeps write per-arm sub-corpora under `<out_dir>/<arm>/` and
merge to top-level `<out_dir>/{runs,traces}.parquet`. The merged
corpus is what the runner ingests; per-arm subdirs are scratch
and the substrate's `dispatch_sweep` removes them post-merge.

- **`.in_progress` sentinel**: dropped at sweep start, removed
  on successful completion. `--ingest-all` walks skip any corpus
  carrying the sentinel (corpus-integrity invariant CI1; the
  CI1–CI8 catalogue is `CORPUS_INTEGRITY.md` on the `submission`
  branch). A crashed
  merge leaves the sentinel up so subsequent ingests don't pick
  up a half-built parent.
- **Cloud manifest mirror**: `archive()` mirrors `_remote.json`
  to `<remote_root>/MANIFEST.json` so cloud-side recovery works
  without local manifest state. `cloud.recover_local_manifest`
  + `cloud.list_archives` are the recovery / discovery
  primitives.
- **Trace eviction (CI7)**: per-step / per-burst trace columns
  are the heavyweight part (~MB per cell). The runner reads
  them only when computing measurables, then evicts the local
  `traces.parquet` if cloud-recoverable. CI8 refuses
  contaminated traces (cell-id mismatches between runs and
  traces).
- **Trace inspection**: `scripts/trace_schema.py <traces.parquet>
  [<bridges_module>]` lists trace columns + which measurables
  the corpus's trace schema satisfies — no decompressed data
  materialised. Use it to diagnose "this corpus's traces don't
  carry what my new measurable needs" before committing to a
  re-walk.
- **OOM-prone trace compute**: `corpus.measurements.
  compute_trace_measurables_streaming` is the opt-in row-group-
  iterating variant of the measurable computation path. Drops
  peak RAM ~10× for trace-heavy corpora; use when the default
  full-join path won't fit.
- **Sweep merge OOM**: `stream_concat_parquets` adapts
  `chunk_size` based on per-file decompressed-size estimates
  (default budget 8 GB). When chunk_size resolves to 1, routes
  through pyarrow row-group streaming so peak RAM is bounded
  by the largest single row group (~256 MB) rather than the
  whole file.

## Canonical analyses (use these, don't reimplement)

The framework provides typed analysis primitives that should be
the default surface for mech / link / outcome analyses on
substrate corpora. Reimplementing these inline in experiments
duplicates logic that's already centralized; **prefer the
primitives** below.

### Per-cell vs per-burst — when to choose

- **Per-cell scalar** (`paired_g`, `meta_regression_paired_g`):
  pair vanilla / DDQN at the trajectory-averaged level. Fast,
  reliable when training is uniphase. **Fails silently** when
  Q dynamics are non-monotone (e.g., Q-explosion-prone envs):
  the trajectory-averaged Δ_jens combines causally opposite
  phases (early bias-correction vs late Q-explosion), washing
  the signal to ~0.

- **Per-burst stratum-Δ** (`stratum_delta_link_dowhy`,
  `meta_regression_per_burst`): compute Δ per (env, burst) by
  pooling seeds INDEPENDENTLY within each arm at each stratum.
  Default for any env where Q dynamics aren't monotone. The
  panel makes phase structure visible and corroborable.

The study's FourRooms time-series analysis and SpaceInvaders
late-burst attenuation history establish per-burst as the
canonical form for any analysis on Q-explosion-prone or
phase-transition envs. Cross-burst link cancellation is real; per-burst unmasks it.

**Seed-paired analyses (`paired_g`, `paired_g_per_burst`,
`paired_link_per_burst`, `mundlak_paired_g_per_burst`) are
off-limits in RL substrate bridges.** They pseudo-replicate
seeds across strata, and on
algebraically-related predictor/target pairs they expose
near-tautological correlations (e.g., `corr(Δ_jens, Δ_MC)` on
Acrobot γ=0.999 reported r ≈ +0.998 with partial r given Δ_Q
= +1.000 — Δ_MC on both sides because `jens = Q − MC` by
definition). Valid only in synthetic SCM analytic tests
(`tests/analytic/lg_scm/`) and single-stratum smoke checks.

### Mech / link / outcome separation

Three verdicts are kept independent (the study's central
methodological claim). The corpus's `jensen_gap` measurable is
clamped to `max(0, mean(Q − MC))` — `0` does NOT mean
"unbiased": pair with the `jensen_dormancy_gap` measurable to
distinguish "true zero" from "underestimating (mech dormant)".
Link verdicts on dormant-mech cells are UNTESTABLE, not NULL.

### Conditioning rule

Link analyses MUST condition on `mech HELD` (Δ_jens < 0 with
the mechanism active, not just `jensen_gap > 0`). Otherwise
"link null" claims silently mix mech-dormant (bias premise
inactive) with mech-active-but-link-broken cells. The two are
different verdicts — the framework refuses to collapse them.

### Concrete primitives

| analysis | use for |
|---|---|
| `paired_g` | **synthetic SCM tests only** — scalar Δ pair-by seed. Off-limits in RL substrate bridges. |
| `paired_g_per_burst` | **synthetic SCM tests only** — per-burst paired Δ. Off-limits in RL substrate bridges. |
| `paired_link_per_burst` | **synthetic SCM tests only** — per-burst r(Δ_target, Δ_predictor). Off-limits in RL substrate; use `stratum_delta_link_dowhy` instead. |
| `phase_link_consistency` | **synthetic SCM tests only** — derived from `paired_link_per_burst`. |
| `stratum_delta_link_dowhy` | **canonical RL-substrate link**: per-(env, burst) Δ via independent-samples seed pooling per arm, DoWhy backdoor / placebo / RCC on the panel. Mech→outcome inference goes through this. |
| `meta_regression_paired_g` | per-stratum Δ regressed on covariates |
| `meta_regression_per_burst` | per-(stratum, burst) panel meta-regression |
| `stratified_arm_diff_pooled` | per-stratum **independent-samples** Cohen's d → DL random-effects pool with heterogeneity-flagged verdict (HELD / HELD_WITH_SCOPE_FLAG / NO_EFFECT / POWER_INSUFFICIENT). Pair with `meta_regression` sibling on the same scope for the scope-cluster pattern (docs/HYPOTHESIS_AS_GRAPH.md §3b). Use **this** for cross-env / cross-config pooling — NOT `paired_g_pooled`, which pseudo-replicates by seed (see its module docstring). |
| `stratum_effect_panel_per_burst` | per-(env, burst) **independent-samples** Cohen's d panel. Walks per-burst NDArray source (same shape as `paired_g_per_burst`) but pools treatment / baseline seeds independently within each (env, burst) → Cohen's d via simple-mean-variance form. Canonical migration target for per-burst phase-consistency bridges that can't use `paired_g_per_burst` under the RL substrate seed-pairing rule. |
| `mundlak_paired_g_per_burst` | **synthetic SCM tests only** — paired form. Off-limits in RL substrate. |
| `partial_spearman` | **Canonical mediation primitive — unified JCI (partial-)Spearman**. Subsumes five legacy variants (`stratified_spearman` / `stratified_partial_spearman` / `stratified_partial_spearman_multi` / `per_burst_jci_spearman` / `per_burst_partial_jci_spearman`). Single result type `PartialSpearmanResult`. Granularity detected from input types: `x/y: str` → per-cell; `x/y: Measurable[..., NDArray]` → per-burst (one observation per (cell, burst) — preserves phase structure). `conditioning: tuple[..., ...] = ()` — empty for marginal, single entry for closed-form first-order partial, k entries for multi-Z OLS-residual partial. Internal dispatch picks the right `graph.discovery` primitive per k. |
| `dynamic_partial_spearman` | **Trajectory-resolved mediation** on per-burst measurables. Sibling to `partial_spearman` for per-burst List-typed columns: iterates one ρ per BURST INDEX (not one observation per (cell, burst)) and returns `Mapping[Stratum, DynamicMediationResult]` carrying the per-burst ρ trajectory + `TimeAggregationStatus` enum (`SIGN_FLIP_DETECTED`, `WEAK_TIME_VARYING`, `CONSISTENT_DIRECTION`, `UNDERPOWERED_BURSTS`). `mediator_per_burst` / `outcome_per_burst` accept `str` column names OR `Measurable[..., NDArray]` instances (mirrors `partial_spearman`'s lazy-evaluation pattern via `evaluate_per_burst_source`); `mediator_per_burst` may also be a **tuple** of columns/Measurables for multi-mediator depth-≥2 conditioning (parallel to static `partial_spearman`'s `conditioning` parameter — k=1 closed-form, k≥2 multi-Z OLS-residual; df_offset shifts to `3 + k`). Result's `mediator_names: tuple[str, ...]` records the conditioning set. Both FE Fisher-z (fixed-effects, n-weighted) and DerSimonian-Laird (random-effects) pools are exposed via the result fields `rho_marginal_pooled` / `rho_partial_pooled` (FE) + `dl_marginal` / `dl_partial: FisherZDLPool` (DL). DL is the canonical aggregate — its τ²/I² quantify the heterogeneity that `aggregation_status` flags qualitatively (SIGN_FLIP → I² ≈ 1.0, large τ²; WEAK_TIME_VARYING → I² ∈ [0.5, 1.0]; CONSISTENT_DIRECTION → I² ≈ 0). The FE pool is NaN'd when `SIGN_FLIP_DETECTED` (sign-opposing bursts make the n-weighted z-average a Simpson's-paradox artifact); the DL pool is **never** NaN'd by the diagnostic gate — its heterogeneity statistics are the typed surface for the same pathology. Pool weights match the sibling primitives: marginal `(n_b − 3)` (`stratified_spearman_rho`), partial `(n_b − 4)` (`stratified_partial_spearman_rho`). DL exposes `tau2` / `i2` / `q` (z-units), `rho_pooled` / `rho_pi_lo` / `rho_pi_hi` (inverse-Fisher-z to ρ-units; PI bounds NaN at G < 3), `se_pooled` (z-units), `n_bursts_used`, `assumption_violations` (small-G DL regime warnings). DL gives parametric PI bounds (over-confident under within-cell autocorrelation); `n_bootstrap > 0` adds **cluster-bootstrap empirical CI** which is assumption-free under any within-cell autocorrelation structure (cells = resampling unit; recommended `n_bootstrap=1000` for publication-grade CIs). `bootstrap_marginal` / `bootstrap_partial: ClusterBootstrapInterval | None` populate when `n_bootstrap > 0`; default 0 keeps the fast path bit-identical. `sign_flip_min_abs_rho: float = 0.05` is the noise-floor magnitude below which a per-burst ρ is treated as sampling noise rather than structural signal — opposing-sign bursts at noise level don't trigger SIGN_FLIP, and the noise-level bursts are dropped from the `weak_time_varying_ratio` `max/min` comparison so a single near-zero burst doesn't drive WEAK classification. Per-burst alignment is "ragged tail": `n_bursts = max trajectory length` and `n_per_burst[b]` shrinks as shorter cells drop off (vs truncate-to-min, which would discard every burst past the shortest cell's tail). Mediation on RL training trajectories where burst dynamics may be non-monotone (Q-explosion phases, sign-flipping marginals, mid-training mediation peaks) MUST go through this primitive; static `partial_spearman` on per-burst data risks aggregation artifacts. |
| `dynamic_pc_adjacency` | **Trajectory-resolved PC-style mediation** on per-burst measurables. Sibling to `dynamic_partial_spearman` from a different identification path: at each burst runs Fisher-z partial-correlation CI tests (the same machinery `corroborate.graph.discovery.discover_adjacency` uses for PC edge removal) and reports per-burst edge presence + counts. Returns `Mapping[Stratum, DynamicPCResult]` with per-burst `p_marginal[b]` (depth-0 marginal Spearman CI), `p_conditional[b]` (depth-1 closed-form partial Spearman CI, df = n − 4), `rho_marginal[b]`, `rho_partial[b]`, and three boolean trajectory counts: `n_bursts_marginal_edge` (where marginal CI rejects at α), `n_bursts_mediator_dseparates` (marginal edge present AND conditional edge absent → full mediation at burst b), `n_bursts_direct_edge` (both present → partial mediation or direct effect). Consumers decide trajectory-shape thresholds (mostly-mediated / rarely-mediated); the framework declines to prescribe a meta-aggregator. Shares the `TimeAggregationStatus` classifier with `dynamic_partial_spearman` (driven by `rho_marginal[b]`). Also exposes `dl_marginal` / `dl_partial: FisherZDLPool` — the DerSimonian-Laird random-effects pool over the per-burst (ρ, n) trajectory (same shape as the sibling). DL is the canonical aggregate — its τ²/I² quantify the heterogeneity that `aggregation_status` flags qualitatively (the PC primitive doesn't expose an FE Fisher-z pool because its primary output is per-burst CI-test edge presence, not a pooled magnitude). DL gives parametric PI bounds (over-confident under within-cell autocorrelation); `n_bootstrap > 0` adds **cluster-bootstrap empirical CI** which is assumption-free under any within-cell autocorrelation structure (cells = resampling unit; recommended `n_bootstrap=1000` for publication-grade CIs). `bootstrap_marginal` / `bootstrap_partial: ClusterBootstrapInterval | None` populate when `n_bootstrap > 0`. The integer edge-count triple ALSO gets a cluster-bootstrap CI via `bootstrap_edge_counts: ClusterBootstrapEdgeCounts | None` — answers "is the edge classification robust to which cells we sampled?" (wide CI on dsep means a few outlier cells flip per-burst CI decisions across resamples). Conceptually distinct from the ρ-pool CIs (which answer "what's the average magnitude under resampling?"); both populate together at `n_bootstrap > 0`. `min_n_per_burst: int = 20` defaults higher than the partial-Spearman sibling's 5 — PC's CI tests need more samples for stable α-level control. Cross-validates `dynamic_partial_spearman` from a different identification path; discrepancies (one says "mediator d-separates" while the other reports nonzero partial ρ) are diagnostic of non-linearity or identification failure. **Multi-mediator depth-≥2**: `mediator_per_burst` accepts a tuple of columns/Measurables — the `n_bursts_mediator_dseparates` count generalises to "the JOINT mediator set d-separates"; CI test dispatches to `partial_spearman_rho_multi` per-burst with df = n − 3 − k. Result's `mediator_names: tuple[str, ...]` records the conditioning set; bootstrap edge-count CIs work identically at any k. |
| `partial_spearman_rho` (graph.discovery) | underlying single-Z closed-form first-order partial Spearman (`(rxy − rxz·ryz) / sqrt((1−rxz²)(1−ryz²))`). The `partial_spearman` analysis primitive dispatches single-Z conditioning here for verdict-stability reasons (boundary-case ρ differs from the multi-Z OLS-residual form). |
| `partial_spearman_rho_multi` / `stratified_partial_spearman_rho_multi` (graph.discovery) | underlying multi-Z OLS-residual primitive. The `partial_spearman` analysis primitive dispatches k≥2 conditioning here. |
| `stratum_panel_jci_spearman` | per-stratum-panel Spearman for mediation FALSIFICATION (marginal-vs-stratified ρ comparison per stratum). Distinct shape from `partial_spearman` (which iterates observations, not strata) — kept separate. |
| `dowhy` | DoWhy backdoor / refutation on a typed causal graph |
| `mediation_dowhy` | DoWhy two-stage mediation with typed `LinearityStatus` diagnostic on the result (`RELIABLE` / `SIGN_FLIPPED` / `OUT_OF_BOUNDS` / `UNIDENTIFIED` / `POWER_INSUFFICIENT`). **Diagnostic, NOT magnitude estimator** — surfaces whether the linear-mediation assumption is defensible on this corpus. Pair with `partial_spearman` as a HYPOTHESIS_AS_GRAPH §3b scope-cluster: both HELD → mediation survives BOTH rank-based AND linear identifications. Magnitudes remain unreliable without prior power + topology gating per the recipe below; the `linearity_status` field makes the failure modes first-class. |
| `factorial_2x2` | 2×2 factorial interaction Δ |
| `tautology_audit` | three-check audit (HP shadow / partial-correlation / convergence) |
| `verdict_distribution` | corpus-level verdict count / class breakdown |
| `universe_scope` | universal scope analysis primitive |

When proposing an analysis: **check this list first**. New
inline analyses only when none of the above fits — and even then,
prefer to extend an existing primitive (or add a sibling) rather
than copying logic.

### Moderation vs mediation (question-shape clarification)

The above primitives ask **mediation** questions ("does the
intervention work THROUGH X to affect Y?"). A structurally
distinct question is **moderation**: "does the intervention
CHANGE the strength or direction of the X→Y relationship?"
docs/HYPOTHESIS_AS_GRAPH.md §3b's scope-cluster pattern is
moderation-shaped (the meta-regression coefficient on an env
feature IS a moderation test: does the effect size differ by
context?). The mediation answer comes from
`partial_spearman_rho` / `mediation_dowhy`; the moderation
answer at the OUTCOME level comes from `meta_regression_*` on
the scope-cluster, and at the LINK level from
`stratum_link_moderation_dowhy` (currently UNCONSUMED — kept
provisionally for a future moderation-asking bridge).

If a bridge would ask "does the intervention break the link
between X and Y in some envs but not others?" — that's
moderation, not mediation. Reach for the moderation primitive,
NOT a partial-Spearman over Δ-projection (which is a mediation
question on a different sample shape).

### Mediation recipe (load-bearing — read before authoring mediation bridges)

Mediation magnitudes are slippery. The framework's
`proportion_mediated` was deprecated for documented structural
reasons (ratio explodes, lands outside [0, 1] under suppression,
first-difference identification ≠ population slope). The
ported-forward `mediation_dowhy` (DoWhy two-stage backdoor +
OLS) is similarly fragile under multicollinear mediators —
on the FR γ-WHY corpus (n=120, mediators ρ ≈ 0.93) it produced
direct ATE = −57 alongside total ATE = +1023 (a sign-flip
multicollinearity artifact), with indirect proportion = +106%
(outside [0, 1]).

The case-study prescription, refined through empirical
reproduction of the failure mode: **never read
mediation magnitudes without prior power-gate + topology-gate.**
The pipeline:

1. **Power gate the TOTAL ATE.** Run `dowhy.backdoor_ate` +
   `placebo_refutation` + `random_common_cause_refutation` on
   `(treatment, outcome)`. If placebo doesn't drop to ≈ 0 OR
   RCC drift > tolerance, the total ATE is not reliable enough
   to decompose. STOP.

2. **Topology gate via PC.** Run
   `corroborate.graph.discovery.discover_adjacency` with depth
   2 on the full variable set. If PC does NOT remove the
   treatment-outcome edge under the proposed mediator
   separating set, the posited DAG is suspect. Either re-DAG
   to match PC, or STOP.

3. **Mediation via partial-Spearman (canonical).** Use
   `stratified_partial_spearman` (single mediator) or
   `stratified_partial_spearman_multi` (joint mediators) to
   compute ρ(X, Y \| Z) on the SAME panel. Rank-based +
   multicollinearity-robust + bounded-output → reliable
   mediation evidence.

4. **`mediation_dowhy` as DIAGNOSTIC** via the typed
   `LinearityStatus` field on the result. Surfaces sign-flips
   (direct/total opposite signs) and proportions outside [0, 1]
   as first-class enum values (`SIGN_FLIPPED` / `OUT_OF_BOUNDS`)
   rather than runtime gotchas. RELIABLE means linear
   decomposition's coherent range; the other failure-mode
   statuses flag "linear assumption broken on this corpus —
   `partial_spearman` is the trustworthy answer." The
   diagnostic-sibling bridge pattern (a `partial_spearman`
   bridge paired with its linearity sibling at the same scope)
   forms a HYPOTHESIS_AS_GRAPH §3b scope-cluster — both HELD →
   robust mediation under both rank-based AND linear
   identifications. The frozen study's ddqn hypothesis
   (`submission` branch) carries the two canonical instances.

5. **Refutations on the total.** Placebo + RCC corroborate the
   foundation; mediation magnitude doesn't inherit
   reliability beyond what stage 1 and 3 establish.

Bridges that just emit `mediation_dowhy.indirect_proportion`
without the gating pipeline are NOT trusted. The empirical
example documented in `mediation_dowhy.py`'s module docstring
(FR × MLP × unshaped × baseline at γ=0.999) is the reproducible
failure mode of skipping this discipline; the typed
`linearity_status = SIGN_FLIPPED` IS the surfaced flag at that
scope.

`proportion_mediated` was the v9-era ratio-of-noisy-means
mediation primitive; deleted 2026-05-18 — statistical
deprecation case: (1) ratio explodes near zero; (2) lands
outside [0, 1] under suppression; (3) first-difference
identification doesn't recover population slopes under
seed-coupled noise (the same critique that puts `paired_g`
off-limits in RL substrate bridges). `partial_spearman` and
the salvaged `mediation_dowhy` diagnostic jointly cover the
surface.

## Test iteration

Tests that compile a JAX kernel and run DQN end-to-end on
CartPole are marked `@pytest.mark.slow`. Defaults:

- `uv run pytest tests/` → fast cohort only (~9 s, 213 tests).
- `uv run pytest tests/ -m slow` → slow only (~92 s, 22 tests).
- `uv run pytest tests/ -m ''` → full suite (~95 s, 235 tests).

`addopts = "-q --strict-markers -m 'not slow'"` in pyproject. The
empty `-m ''` overrides addopts to include both.

## Test principle

Tests assert framework output against an **analytical closed
form** within a **sampling-distribution-derived bound**. Four
rules, in order of importance:

1. **Closed-form, not arbitrary**. The expected value is computed
   from substrate parameters (`expected_g = mu_x · sqrt(n_steps) /
   sigma_x · c_4(n_pairs)`), not plucked. Bound size names what
   it absorbs (sample-SD CV at n=30 → 15%; cluster-robust SE
   inflation → t-critical at df=n-p; etc.). `g > 0.8` when
   structural g is 28 is a 35× slack — no.

2. **Substrate-grounded over synthetic-input**. Cells flow through
   `RunRow.as_dict()` → analysis (the production data path). The
   `tests/analytic/lg_scm/` substrate (Linear-Gaussian SCM with
   `@claim` Free Claims + frozen-dataclass config bundles) and
   `src/corroborate_rl/tests/analytic/deadly_triad/` (FQI / Q-
   divergence) make closed-form expectations possible without
   mocks. Synthetic-input tests (hand-built dicts → primitive →
   assertion) are unit-level coverage of internals only.

3. **Z-score bounds replace "doesn't reject null"**.
   `|coef / SE| < 2.5` against the framework's reported SE catches
   both inflated estimates AND collapsed SEs; `p > alpha + CI
   covers zero` passes garbage estimators with overconfident CIs.
   When framework SE comes from a CI (no `.se` field), invert
   with the framework's own t-critical (`scipy.stats.t.ppf(1-α/2,
   df=n-p)`), not `1.96`.

4. **No substrate-redundant assertions**. If a test reads back
   what it stamped (count of stamped verdicts, sum of stamped
   values), the assertion is tautological. The framework's logic
   isn't being verified. Either delete or replace with an
   assertion on transformation logic the framework actually does
   (case-folding, classification, dominance resolution).

Persistence tests pair: `tests/test_persistence.py` covers the
write/read CONTRACT (round-trip equality on hand-built rows);
`tests/analytic/lg_scm/test_parquet_round_trip.py` proves
**closed-form analyses still recover the structural answer**
after a real parquet round-trip. Both shapes are needed.

**Empirical coverage check.** `mutmut` is wired in
`pyproject.toml [tool.mutmut]` to mutate framework analysis
primitives and run only the analytic suite. Surviving mutants
are coverage gaps (closed-form bound too loose, or the line
isn't exercised). Run with `uv run mutmut run`; list all results
with `uv run mutmut results --all true` (default omits killed);
inspect a single mutant with `uv run mutmut show <name>`.

**Sharp edge:** mutmut wraps each function with a trampoline
that materializes default-arg values at the wrapper level, so
mutations of parameter defaults (e.g., `arm_field='arm_key'` →
`'XX...XX'`) never propagate through the call. Treat default-arg
mutations as wrap-broken, not real survivors.

**Workflow.** When fixing surfaced gaps:

1. Sample survivors per file: `uv run mutmut results --all true |
   grep "<file>.*survived" | head`.
2. Inspect each: `uv run mutmut show <name>`. Categorize as
   wrap-broken / equivalent / real-gap.
3. For real-gap clusters (e.g., Fisher-z formula never reached
   because tests give r=±1 short-circuit; IVW weighting never
   exercised because tests use uniform SE), add a closed-form
   test that **specifically constructs cells putting the
   framework on the unexercised code path**. Examples:
   `tests/analytic/lg_scm/test_paired_link_fisher_z.py` (moderate-r
   construction with independent ε per arm), `test_random_effects_ivw.py`
   (heterogeneous SE via varied n_pairs + mu_x extremes).
4. Re-run `uv run mutmut run`; verify the targeted mutants flip
   from `survived` to `killed`.

## Acceptance criteria

`v0` is acceptance-tested by reproducing the DDQN study frozen
on the `submission` branch — mechanism HELD ↛ outcome
HELD ↛ link HELD across the multi-environment panel, with the
methodological contribution living in keeping these three
verdicts separate.

The framework's primary distinguishing feature lives at the
verdict layer: `POWER_INSUFFICIENT` is a first-class verdict
distinct from `HELD` and `NO_EFFECT`. Treating an underpowered
test as "no effect" smuggles methodological problems past the
reader; the framework refuses that smuggle.
