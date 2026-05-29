# Cache-ingest optimisation — design (implement AFTER the canonical-migration agent lands)

**Do not implement while an ingest agent is running** — the targets
below (`runner.py`, `corpus/measurements.py`, `corpus/cloud.py`) are
the exact files a live `--ingest` executes. Mid-run edits = the
shifting-ground collision the `fix-dont-detour` discipline warns
against. This doc is the plan; apply it on a stable tree.

## Observed bottleneck

The 2026-05-29 canonical migration: `asterix_g099_canonical_n_eps20_ckpt`
(30 cells, n_eps=20, `mc_return_from_step` / `predicted_q_per_step`
shape `(20, 1000)`) sat compute-bound ~10 min — near the agent's
`timeout 600` wrapper. snake_3M (60 cells, n_eps=5, `(5, 4000)`) was
272 s. The cost is the per-cell measurable computation on heavy 2-D
trace arrays, run **sequentially**.

## Current path (committed state)

`compute_trace_measurables_streaming` (`corpus/measurements.py` ~615):
- Batches row-groups by decompressed-byte budget (512 MiB), or drops
  to a per-`id` lazy scan for single-huge-RG files.
- `for group in rg_groups:` → `read_row_groups` → `join runs` →
  `compute_missing_columns(joined, list(required))` → keep measurable
  cols. **Sequential.** One CPU core regardless of machine.
- Computes the FULL `required` list per batch — no per-column skip of
  measurables already current in the corpus store (`5d27e1e` chose
  "load all reads, compute all" for correctness).

Cache merge (`runner.py` ~1213): each `--ingest` does
`read_parquet(cache)` → `concat(diagonal_relaxed)` → `atomic_write`.
One full read+write of the cache per corpus → O(N²) over a migration.

## Optimisations, prioritised

### 1. Parallelise the streaming compute (biggest, universal win)
`compute_missing_columns` is pure per-row (numpy, JAX-free) → cell-
independent → embarrassingly parallel. Run the `rg_groups` batches
(and the per-`id` fallback's cells) across a `ProcessPoolExecutor`
of `min(os.cpu_count()-2, n_batches)` workers; concat the per-worker
measurement frames at the end (the accumulators are small: id +
scalar/per-burst cols, never the heavy trace inputs).
- Care: workers must import the substrate measurable module so the
  `@measurable` registry is populated (pass the module path; re-import
  in `initializer`). Inputs per task: the batch's `(runs_rows,
  row_group_indices, traces_path, required)` — traces re-read per
  worker from `traces_path` by row-group (cheap, avoids pickling
  multi-GB arrays across the process boundary).
- Expected: Asterix ~10 min → ~1.5 min on an 8-core box; snake/pacman
  3M ~4.5 min → ~40 s. Helps EVERY ingest, first-time included.
- Gate behind `CORROBORATE_INGEST_WORKERS` (default = cpu-2, set 1 to
  restore the deterministic sequential path for debugging).

### 2. Compute + stream only NOT-CURRENT measurables (re-ingest win + cleaner invariant)
`5d27e1e` loads the full read set and recomputes everything. Narrow
to `to_compute = required measurables NOT present-and-current in the
corpus's measurements.parquet` (= drifted ∪ missing-from-store), then:
- pass `required=to_compute` to the streaming compute (skip recompute
  of current ones — `build_measurements`' partial-nullity branch
  already passes them through from the store);
- stream `reads(to_compute) ∩ schema` (narrower than full).
This keeps the null-bug fix (missing axis-derived measurables ARE in
`to_compute`, so their reads ARE streamed) AND, crucially, **skips
loading the heavy `(20,1000)` / `(5,4000)` arrays when every
measurable that reads them is already current** — the Asterix case,
where the sweep-time store already held most cols. Invariant:
`streamed_trace_reads ⊇ reads(to_compute) ∩ schema`. Requires a
store-aware `_measurables_to_compute(sub, required)` helper (drift +
absent), which also fixes the `_drifted_or_missing_measurables`
"absent ≠ drifted" gap noted in the 5d27e1e diagnosis.

### 3. One cache write per multi-corpus ingest
Decouple restore from ingest so `--ingest a,b,c` (or `--ingest-all`)
computes each corpus's per-corpus `measurements.parquet`, then merges
+ writes the cache ONCE at the end instead of per corpus. Removes the
O(N²) read+concat+write. Minor at 5 MB today; matters as the cache
grows and is the right shape regardless.

### 4. Operator affordance: projected restore as a CLI flag
The disk-constrained recipe (recover_local_manifest → `restore_columns`
with the 26-col read-set → `--ingest --no-restore` → rm traces) is
hand-scripted per corpus today. Fold into the CLI:
`corroborate hypothesis ... --ingest <corpus> --restore-projected`
auto-derives the read-set from `REQUIRED_MEASURABLES`' transitive
reads, restores only those columns (row-group-rechunked, already done
in `7ae09d0`), ingests, and evicts the projected traces. Turns a
5-step manual dance into one reproducible command — directly serves
the "reproducible, extensible" cache goal.

## CORRECTION (2026-05-29): parallelism granularity + the thread-pool misfire

Empirical A/B on the real ingest path (60-cell breakout, full streaming
compute) settled the parallelism question:

```
workers=1: 409.4s   workers=8: 424.4s   speedup 1.0x
same shape: True (60,96)   scalar-col mismatches: 0
```

The within-corpus **thread pool** (#1, committed `2fe504d`) is correct
(0 mismatches) but gives **1.0×** — `compute_missing_columns` is
GIL-bound *Python* (per-cell `to_dicts()` loop + `_resolve_one`
recursion over 73 measurables), not GIL-released numpy. Threads can't
parallelise it. **→ revert `2fe504d`.**

The RIGHT parallelism is **per-corpus across processes**, and it
**already exists**: `runner._load_directory` runs a fork-based
`ProcessPoolExecutor` over corpora (line ~2580), bounded by disk via
`_estimate_max_workers`. `fork` inherits the `@measurable` registry by
copy-on-write — no per-worker re-import, no registry pickling. It fires
for BOTH `--ingest a,b,c` (via `_load_data` → `_load_directory(
corpus_dirs=...)`) and `--ingest-all <root>`. So a multi-corpus ingest
is already GIL-free-parallel; the thread pool was both the wrong
primitive (threads vs processes) and the wrong granularity (within- vs
across-corpus).

**Residual gap**: a SINGLE large corpus ingested alone runs in one
process → single-threaded compute (the Asterix ~5 min case). The
across-corpus pool has only one task then. Closing it needs a
within-corpus **fork ProcessPool** over cell-batches (fork already
solves the registry problem; only the small per-batch result frames
cross back). Secondary — multi-corpus ingests sidestep it by
parallelising across corpora. Profile compute-bound vs I/O-bound
before building it.

Levers #1b (vectorise hot reductions) and #2 (compute only not-current
measurables) reduce the per-process Python WORK and stack with the
existing process parallelism — those are the durable wins.

## What already exists (don't reinvent the DAG)

`compute_missing_columns` (`corpus/measurables/measurable.py` ~718)
already implements dependency-ordered, shared-work computation:
- **`_topo_sort_pending`** orders pending measurables so a dep
  computes before its dependents (handles both param-injection and
  `record.get(...)` dep styles).
- **`_resolve_one(..., cache=per_cell_cache)`** memoizes each
  measurable + its transitive deps per cell → a shared intermediate
  (`mc_return_raw_episodes` feeding the 4 raw-outcome measurables)
  is reconstructed ONCE per cell, reused by every consumer.
- The streaming path loads `cols_to_load = ∪ reads` ONCE per batch
  (shared reads at the I/O layer); `_resolve_one`'s
  record-as-precomputed-cache reuses persisted values, never
  recomputes them.

So "build a DAG / compute shared-read measurables at once" is
already the design at the per-cell granularity. The remaining cost
is the loop SHAPE, not missing dependency analysis:

```
for cell in cells:            # per-cell Python loop (N cells)
    per_cell_cache = {}        # rebuilt per cell (correct: cells independent)
    for name, m in pending:    # per-measurable Python dispatch (73)
        evaluate_with_measurables(m.fn, cell, cache=per_cell_cache)
```

An N_cells × N_measurables single-threaded Python double-loop. The
heavy `(20,1000)` / `(5,4000)` numpy work inside each call, ×N cells,
on one core, is the Asterix stall.

### 1b. Vectorise hot reductions across cells (complements #1)
The cheapest-but-most-frequent measurables are pure column
reductions — axis-derived `*__mean_axis_-1` / `*__std_axis_-1`,
simple per-burst scalar means. These can run as ONE polars/numpy
expression over the whole batch's column instead of N per-cell
Python calls. Biggest per-measurable speedup, but it's per-measurable
effort (each needs a batched form), so scope it to the hottest
handful (the axis-derived family + the per-burst outcome). Keep the
per-cell path as the fallback for measurables without a batched form.
Orthogonal to #1: parallelism multiplies cores, vectorisation removes
Python dispatch — stack both on the hot measurables.

## Recommended order
1 (parallel compute) → 2 (not-current narrowing) → 4 (CLI affordance)
→ 3 (single write). Each is independently shippable; isolate per
re-derivation cycle (`[[isolate-substrate-fixes]]`) and confirm the
cache is bit-stable for an unchanged corpus after each. Add a timing
regression (assert the 1-RG per-cell path on a synthetic N-cell trace
uses >1 worker and stays under a wall-clock bound) alongside #1.
