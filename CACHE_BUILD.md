# Cache build — design contract

The cache layer is the framework's **most-touched data path**:
every bridge evaluation reads from a per-hypothesis cache; every
new measurable forces a partial rebuild; every analysis on a
real corpus depends on the cache being current. A killed-mid-
build cache loses ~all the work; a slow cache build blocks every
downstream analysis.

This doc names the invariants the cache layer must satisfy after
a total refactor, the architectural choice that drives them, and
the implementation order.

## Five invariants

### C1. Determinism — cache is a pure function of (bridges, corpora)

```
cache_build(bridges, corpora) → DataFrame
```

Running `cache_build` twice on the same inputs MUST produce
byte-identical output (modulo per-column timestamp metadata in
the sidecar). Determinism implies:

- Cell ordering is determined by a fixed key (e.g. sorted by
  `id`), not by iteration order over a dict / filesystem.
- Measurable computation order doesn't matter for the result —
  measurables are pure functions of the cell record.
- Floating-point reduction order is fixed (no parallel-reduce
  with non-deterministic accumulators).

The current code mostly holds this — the failure mode is on the
floating-point edge (reductions in `apply_trace_reductions`,
sums across non-sorted iterables) which lurks but hasn't bitten
yet.

**Substrate-stamp vs post-hoc-recompute consistency** (related to
Phase 3): when `runs_df` carries a measurable column natively
(substrate stamped at sweep time via `RunRow.measurements`) AND
`measurements.parquet` has the same column from a prior post-hoc
build, the framework explicitly prefers the substrate stamp —
existing-store value is dropped at the join. See
`build_measurements`'s overlap-drop step. This makes the
authoritative source unambiguous: substrate ≻ post-hoc-recompute.
Floating-point disagreement between the two sources is therefore
NOT a determinism violation — it's an explicit precedence rule.
Substrate authors should know that mixing sweep-time stamp with
post-hoc compute for the same measurable name will quietly
prefer the stamp on every rebuild.

### C2. Atomicity — a crashed build leaves either old or new state

The cache parquet (`<cache>.parquet`) and the closure-hash
sidecar (`<cache>.hashes.json`) are read by every analysis. A
killed-mid-write process must NOT leave a torn parquet at the
canonical path — consumers crash with `pl.exceptions.ComputeError`
or, worse, silently read truncated data and produce wrong
verdicts.

**The rule.** Every cache-side write goes through tmp+rename:
write to `<path>.partial`, then `os.rename` (atomic on POSIX).
The sidecar must be updated AFTER the parquet rename so a
half-updated state has a stale-but-readable parquet + a stale
sidecar (consumer sees pre-build state) rather than a fresh
sidecar pointing at a torn parquet (consumer reads garbage).

This is the same pattern from `SWEEP_PERSISTENCY.md` I4. The
helper exists (`stream_concat_parquets` writes to `.partial`);
the cache write path doesn't use it.

### C3. Incrementality — adding a measurable rebuilds only the affected columns

Adding a new measurable to a bridge file MUST NOT force
recomputation of unrelated columns. The current
`_invalidate_drifted` path is half-right: it drops drifted
columns, then the row-loop recomputes only those + brand-new
ones. The remaining gap is at the per-cell evaluator —
`compute_missing_columns` walks every cell even when ALL
columns are already populated (the `pending` filter is empty
but `to_dicts()` still materializes the whole frame).

**The rule.** When `pending` is empty, return the input frame
unchanged without iterating cells. When `pending` is non-empty
but small, the per-cell loop runs ONLY over the pending
measurables, not all registered ones. (Already true in the
inner loop; the outer "skip when pending is empty" early-out
is the cheap fix.)

The deeper incrementality question: when measurable X depends
on measurable Y, and Y's closure hash drifts, X's column should
also be invalidated — currently it isn't (the drift detection
is per-column, not transitive). For the present cache size
this is rarely the bottleneck; documented as a sharp edge to
fix when it bites.

### C4. Orphan eviction — removed measurables don't accumulate as dead columns

If a user removes a measurable from a bridge file (or renames
it), the cache should drop the orphan column on the next build.
The current `_invalidate_drifted` walks REQUIRED measurables
only — orphans persist forever, growing the cache file and
slowing every read.

**The rule.** Drift detection is two-way: drop cache columns
whose closure hash differs from the manifest entry (existing),
AND drop cache columns whose name is no longer in the registered
set produced by `measurable_names_for_bridges(bridges)`. The
provenance tags (`id`, `arm_key`, `env_name`, etc.) and raw
record keys are preserved — only registered-measurable columns
are subject to eviction.

### C5. Observability — cache build emits per-corpus + per-step progress

Long-running cache builds (~50 corpora × ~30s each = 25 min)
currently emit one line per corpus when restoring + the drift
warning. No progress bar, no rate/ETA, no per-step timing.
A killed-mid-build user can't tell how far in they were, what
was the slow corpus, or whether to wait or kill.

**The rule.** Cache build emits, per corpus:
- `[i/N] <corpus>: restoring (need: traces.parquet)`
- `[i/N] <corpus>: loading (84 cells, 12 trace cols)`
- `[i/N] <corpus>: computing measurables (24 columns) — 2.3 s`
- `[i/N] <corpus>: done (84 cells × 47 cols, 1.4 MB)`

Plus a final `cache built: 2,400 cells × 89 cols across 50 corpora in 23 min`.
Structured enough that the framework can later parse it for the
post-run report.

## Architectural choice — two-level cache

The current design has ONE cache per hypothesis. Multiple
hypotheses sharing the same corpora + same measurables compute
those measurables redundantly. With ~6 active bridge files and
~50 corpora, the redundancy is 5-10× on the shared measurables.

The refactor introduces a **per-corpus measurement store** as
the primary cache, and makes the per-hypothesis cache a cheap
projection over it:

```
experiments/data/<corpus>/
  runs.parquet          (raw substrate output — never modified)
  traces.parquet        (raw traces, optional, cloud-backed)
  measurements.parquet  (NEW — every measurable ever computed for this corpus)
  measurements.hashes.json  (NEW — closure-hash sidecar)
  _remote.json          (existing — cloud archive manifest)

experiments/data/cache/
  <hypothesis>.parquet  (projection: cells from <corpora_in_scope> × cols required)
```

**Building `measurements.parquet`** for one corpus:
1. Load `runs.parquet` + (optionally) `traces.parquet`.
2. For each registered measurable, check its closure hash
   against `measurements.hashes.json`. Already-current ones
   skip; drifted/missing ones are computed.
3. Stream-write the updated `measurements.parquet` (tmp+rename).
4. Update `measurements.hashes.json` (tmp+rename).

**Building `cache/<hypothesis>.parquet`** for one hypothesis:
1. Read each in-scope corpus's `measurements.parquet` (selecting
   only required columns).
2. Concat via `stream_concat_parquets` (already atomic, already
   stream-only).
3. Done. No measurable computation at this layer.

Trade-offs:

- **Win**: shared computation across hypotheses. Adding a new
  hypothesis that uses already-computed measurables is a
  ~seconds-not-minutes operation (just the concat + projection).
- **Win**: each per-corpus build is parallelizable independently
  (one process per corpus, each holds its own traces in scope —
  predictable memory budget).
- **Win**: `measurements.parquet` is per-corpus, so a killed-
  mid-build only loses one corpus's progress, not the whole
  hypothesis cache.
- **Cost**: another file per corpus — modest, since `measurements`
  is just scalar columns (~10s of KB to MB per corpus, vs. GB
  for `traces.parquet`).
- **Cost**: more state to keep consistent. The closure-hash
  manifest moves from per-hypothesis to per-corpus, which is
  actually MORE robust — you can't have a hypothesis-cache
  manifest stale relative to a corpus's measurement contents.

## Audit table — current vs. target

| invariant | held by current code? | gap |
|---|---|---|
| **C1 Determinism** | mostly (cell ordering by `id`; FP-reduction-order isn't pinned but rarely bites) | document, add regression test on a known-determinable corpus |
| **C2 Atomicity** | broken — `merged.write_parquet(cache_path)` direct write | tmp+rename in cache write + manifest write |
| **C3 Incrementality** | YES — early-out at `measurable.py:611-619` when `pending` is empty (no per-cell `to_dicts()`) | — |
| **C4 Orphan eviction** | broken — orphans persist forever | extend `_invalidate_drifted` to drop unregistered cache columns |
| **C5 Observability** | minimal — one line per corpus on restore | structured per-corpus progress + final summary |

## Implementation order

The two-level architecture is a substantial refactor; do it in
phases so each lands independently and the cache stays usable
between phases.

**Phase 0 — atomicity + incrementality + observability**:
land all five invariants on the CURRENT one-level cache architecture.
Each is a small, independent fix:

1. **C2** (~15 LoC): tmp+rename in `_ingest_and_compute`'s parquet
   write at line 439 + manifest write at line 441 + the
   `_enrich_cache_in_place` mirror at line 464. Reuse the pattern
   from `stream_concat_parquets`.

2. **C3 early-out** (~5 LoC): in `compute_missing_columns`, if
   `pending` is empty after the dedup loop, return `df`
   immediately without materializing `to_dicts()`. The fast path
   most cache reads land on.

3. **C4 orphan eviction** (~10 LoC): in `_invalidate_drifted`,
   compute the set of unregistered columns vs. the required
   set + a known-preserved provenance allowlist; drop them.
   Print a loud warning naming the orphans.

4. **C5 observability** (~30 LoC): structured per-corpus progress
   in `_load_directory` + final summary line. Use stderr and a
   simple counter; no `rich`/`tqdm` dep.

5. **Per-corpus parallelism** (~50 LoC): wrap `_load_directory`'s
   per-subdir block in a `ProcessPoolExecutor.map`. Bound
   workers to N_CPUS // 2 (each worker holds traces in scope,
   memory-budget per worker). Restore + parquet read + measurable
   computation runs in parallel across corpora.

After Phase 0, the cache is **safe by construction** at the
existing one-level architecture, AND ~4× faster on the network-
restore-bound path (for sweeps with cloud-only traces).

**Phase 1 — per-corpus measurements.parquet**:
build the new per-corpus cache layer.

1. Define `experiments/data/<corpus>/measurements.parquet` schema:
   `id` + every measurable column + `__measurable_hashes` sidecar.

2. New module `corpus/measurements.py` exposing:
   - `build_measurements(corpus_dir, *, required) -> Path`:
     reads runs+traces, computes required measurables, writes
     `measurements.parquet` atomically, updates sidecar.
   - `load_measurements(corpus_dir, *, columns) -> DataFrame`:
     pure read.

3. Migration: existing per-hypothesis caches stay valid. New
   `run()` invocations use `build_measurements` per corpus
   (cheap on already-built corpora) then concat into the
   per-hypothesis cache via `stream_concat_parquets`.

**Phase 2 — per-hypothesis cache becomes a projection**:
gut `_ingest_and_compute` to be a pure projection.

1. `_ingest_and_compute` now:
   - Calls `build_measurements` per corpus (each is fast or no-op
     given Phase 1's invariants).
   - Concats `measurements.parquet`s into the per-hypothesis
     cache via `stream_concat_parquets`.
   - No more in-process Python row-loops at this layer.

2. `<cache>.hashes.json` is gone (closure-hash manifest moved
   to per-corpus). The per-hypothesis cache's drift detection
   becomes "are all required measurables present in the
   per-corpus measurements?" — which is a column-set check, not
   a hash compare.

3. Migration: scripts/`migrate_cache_to_two_level.py` reads
   each existing per-hypothesis cache, re-emits per-corpus
   `measurements.parquet`s by partitioning on `corpus` column.
   One-time op.

**Phase 3 — sweep-time measurement** (optional, substrate-side):
Phase 3 is **not** a framework change. The existing
`run_intervention(..., measurables: tuple[Measurable[R, object],
...] = (), ...)` parameter is already the sweep-time hook: the
substrate's per-cell runner evaluates these `Measurable`
instances against the cell result `R` and stamps them onto
`RunRow.measurements`. They ship inside `runs.parquet` as
path-keyed columns at archive time. The "Phase 3 destination" is
substrate authors using this hook more aggressively — registering
the scalars they know they'll need analysed before each run —
rather than relying on Phase 2.1 to compute them lazily on the
first analysis pass.

Why no framework change: the user-facing decision ("which
measurables are 'always needed' for this study") is intrinsic to
the substrate / study design. Adding a framework-level
`eager_measurables` parameter would duplicate the existing hook
without earning its keep against the four-question test
(CLAUDE.md § "When to introduce a framework primitive"). The
framework's role here is to keep the path open and correct,
which Phases 0-2 do.

## Migration

Phase 0 is fully backwards-compat — same cache files, same
manifest schema, same call sites. Just safer + faster.

Phase 1 introduces `measurements.parquet` per corpus but doesn't
require it. Hypothesis runs that don't see one fall back to the
in-cache computation path. New runs populate it.

Phase 2 deprecates `<cache>.hashes.json` in favor of per-corpus
sidecars. A one-shot migration script handles the conversion.

## What this doc is not

A roadmap with deadlines. It's the design contract; anyone
editing `runner.py`, `measurable.py`, or the new
`corpus/measurements.py` should be able to point at the
invariant their change preserves.

If a future change can't preserve all five, the change needs a
written waiver in this doc.
