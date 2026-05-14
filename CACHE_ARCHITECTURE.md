# Cache architecture — full picture

This doc names the framework's cache layers, their interactions,
and the failure modes that have bitten in practice. It's the
synthesis that `CACHE_BUILD.md` (invariants) and
`CACHE_ADDITIVITY.md` (lifecycle) don't quite cover end-to-end.

For invariant-level guarantees → `CACHE_BUILD.md`.
For ingest lifecycle / when-to-rebuild discipline → `CACHE_ADDITIVITY.md`.

## Two layers

```
PER-CORPUS STORES                      PER-HYPOTHESIS CACHE
─────────────────                      ────────────────────
experiments/data/<corpus>/             experiments/data/cache/<hyp>.parquet
  ├── runs.parquet                       (e.g. ddqn.parquet)
  ├── measurements.parquet              + <hyp>.hashes.json (closure-hash sidecar)
  ├── traces.parquet
  ├── measurements.hashes.json
  └── _remote.json                     experiments/findings/<hyp>.run.json
                                         (verdict snapshot)
```

**Per-corpus stores** are the source of truth: one directory per
sweep run, three parquet files + per-corpus measurement sidecar.
Column-additive (CI2/CI6, closure-hash drift detection, partial-
nullity recompute).

**Per-hypothesis cache** is a derived projection: a wide
`diagonal_relaxed` concat over the per-corpus stores, filtered by
the hypothesis's `MODULE_SCOPE` at ingest time. NOT
column-additive at its own layer — it's atomically rewritten on
every directory-walk ingest. The sidecar (`<hyp>.hashes.json`)
records the closure-hash of each measurable in the rebuilt cache,
used to detect drift on subsequent runs.

## Per-corpus file roles

| file | shape | source of truth for |
|---|---|---|
| `runs.parquet` | one row per cell (UUID-keyed), wide scalar columns | substrate-author-supplied measurement columns, env config, arm_key, seed, gamma, etc. |
| `traces.parquet` | one row per cell with list-column trace data (per-step / per-burst arrays) | trace-store data: `mc_return_from_step`, `target_max_q_per_step`, episode lengths, etc. Heavy — multi-GB per corpus on long-trajectory envs. |
| `measurements.parquet` | one row per cell with scalar + small-array columns | trace-derived measurables computed post-hoc (jensen_gap, bootstrap_gap_magnitude, effective_horizon, etc.) Includes `measurements.hashes.json` sidecar carrying closure-hash for each column. |
| `_remote.json` | manifest | cloud-archive state: which files are pushed to S3, sha256s, row_id sets per file |

The split `runs ⊕ traces ⊕ measurements` is intentional:
- `runs.parquet` carries substrate-stamped scalars (lightweight, always loaded).
- `traces.parquet` carries the heavyweight per-step trajectories (~MB/cell). Loaded only when computing trace-derived measurables; evicted locally after computing (CI7).
- `measurements.parquet` caches the post-hoc measurable values (cheap to load, expensive to recompute).

## The ingest pipeline

```
   sweep run                  cell_runner / sweep_loop
       ↓                          ↓
   writes per-arm parquet     stitches into corpus/{runs,traces}.parquet
       ↓                          ↓
   archive() pushes to S3     local files preserved (or evicted)
       ↓
   (later) restore + ingest   →   build_measurements per corpus
                                  → writes corpus/measurements.parquet
                                  → updates measurements.hashes.json sidecar
       ↓
   per-hypothesis cache       _ingest_and_compute walks all corpora
                              → diagonal_relaxed concat over corpus measurements
                              → filters by MODULE_SCOPE
                              → atomic rewrite of cache/<hyp>.parquet
                              → updates <hyp>.hashes.json
```

`scripts/run_hypothesis.py <module> --ingest-all experiments/data`
is the canonical user-facing trigger for this pipeline.

`scripts/run_hypothesis.py <module>` (no flags) is read-only:
loads cache, runs bridges, emits report.

## Measurable resolution + transitive_reads

A bridge declares `source`, `target` (measurable names) and an
analysis fixture (e.g. `paired_g`, `stratum_panel`). Each
measurable has `reads: tuple[str, ...]` — the trace-column names
it depends on.

The runner walks the bridge graph at ingest:
1. Collect the set of measurables ALL bridges read (`transitive_reads`)
2. For each per-corpus measurement store, compute closure-hash
   per measurable; identify drifted/missing
3. Restore trace columns needed by drifted measurables from S3
4. Run `build_measurements` to recompute drifted measurables
5. Concat to per-hypothesis cache

**REQUIRED_MEASURABLES** is the escape hatch: a per-hypothesis
attribute (`Hypothesis.REQUIRED_MEASURABLES: tuple[str, ...]`)
declaring measurables that aren't consumed by any bridge but
should be precomputed anyway. Used for prep work / scope
predicates that read cell-level values without going through a
bridge. The runner unions this into the `transitive_reads` set.

## Cloud archive + restore

The framework pushes per-corpus parquets to `s3://corroborate-archive/<corpus>/`.
Each push updates `_remote.json` (local) and `MANIFEST.json`
(cloud mirror at `<remote_root>/MANIFEST.json`).

**Restore modes:**

- `restore(sweep_dir, files=...)`: full-file restore. Verifies
  sha256 against manifest. Used for `runs.parquet`,
  `measurements.parquet`, sidecars.
- `restore_columns(sweep_dir, file_columns=...)`: column-projected
  restore for `traces.parquet`. ~19× faster than full restore
  on multi-GB archives, but writes a column-subset locally that
  doesn't verify against sha256.

The runner uses `restore_columns` for traces because trace files
are heavy and only a column subset is usually needed. The runner
narrows the requested column set to those needed by
drifted-only measurables (`_drifted_or_missing_measurables`).

**`_missing_for_restore`** decides what to restore:
- `runs.parquet`: restore if not locally present (size > 1KB)
- `traces.parquet`: restore if either (a) not locally present OR
  (b) **partial**: present but missing one of the columns in
  `trace_reads` (the column-subset of drifted measurables)

The (b) partial-detection was added 2026-05-14 in response to a
bug where a prior `restore_columns` left a column-subset local
file that the runner falsely treated as complete on the next
run. See `findings_acrobot_archive_bug` memory for the
diagnosis.

## Drift detection

Each per-corpus `measurements.parquet` has a sidecar
`measurements.hashes.json` recording the closure-hash of each
computed column. The framework's `current_signatures(sub)` reads
this; `_measurable_signature(name)` computes the current
registry hash for the measurable; mismatch = drift.

A drifted column triggers:
- Trace columns the measurable reads need to be in the local
  `traces.parquet` (or restored if missing)
- `build_measurements` recomputes the value for all cells
- The closure-hash sidecar is updated

The same machinery exists at the per-hypothesis cache layer via
`<hyp>.hashes.json`. The runner checks both layers.

## Cache invariants (cross-reference CACHE_BUILD.md)

- **C1 Determinism**: cache is a pure function of (bridges, corpora).
- **C2 Closure-hash drift**: changing a measurable's body
  invalidates the column; the framework recomputes.
- **C3 Idempotent fast-path**: if every measurable is current,
  no recompute happens (loading is cheap).
- **CI2 Column additivity**: adding a new measurable column to
  the registry doesn't invalidate other columns.
- **CI6 Partial-nullity recompute**: a column with some NaN cells
  triggers recompute (the framework treats it as drift). Users
  who want to retry after restoring missing inputs can
  `rm <corpus>/measurements.parquet` to invalidate fully.
- **CI7 Trace eviction**: local `traces.parquet` may be deleted
  after measurables are computed if cloud-recoverable.
- **CI8 Trace contamination check**: refuses cell-id mismatches
  between `runs.parquet` and `traces.parquet`.

## Common failure modes (and diagnostic protocol)

### 1. Archive incompleteness — corpus has no cloud manifest

Symptom: `fetch_remote_manifest(remote_root)` returns None.
Cause: archive was triggered with partial completion (e.g.,
CI7 trace eviction ran before archive completed); cloud lacks
MANIFEST.json + traces.parquet.

Diagnostic: `corroborate.corpus.cloud.fetch_remote_manifest` →
None means corpus is unreachable via cloud restore.

Fix: re-run the sweep (cell hashes will produce new UUIDs) OR
remove the corpus entirely if other corpora cover the env.

### 2. Archive incomplete — corpus has manifest but partial files

Symptom: local `traces.parquet` exists but has fewer columns
than expected; cloud `traces.parquet` exists with full columns.
Cause: prior `restore_columns` wrote a column-subset locally;
subsequent runs that needed different columns silently read
the partial file.

Diagnostic:
```python
import polars as pl
local_cols = set(pl.scan_parquet(traces_path).collect_schema().names())
needed_cols = {'target_max_q_per_step', 'mc_return', ...}
missing = needed_cols - local_cols
```

Fix: post-2026-05-14, `_missing_for_restore` detects this and
forces re-restore. Pre-fix workaround: `rm <corpus>/traces.parquet`
then `--ingest-all`.

### 3. `--ingest-all` silently dropping previously-computed values

Symptom: a bridge that previously fired HELD now fires
POW_INSUF; per-stratum panel shows fewer strata or smaller n.
Cause: backfill recomputed measurables from traces that don't
have the needed columns; reconciliation with per-corpus
`measurements.parquet` (which doesn't have the values either)
overwrote cache values with NaN.

Diagnostic protocol:
1. Check `cloud.fetch_remote_manifest` for affected corpora —
   None means archive is broken.
2. For corpora with manifests, check whether `traces.parquet`
   schema covers the required columns. Use `scripts/trace_schema.py`.
3. For corpora with partial traces locally:
   `rm <corpus>/traces.parquet && --ingest-all`.

Pre-2026-05-14: this could silently corrupt the cache without
warning. Post-2026-05-14: `_missing_for_restore` flags partial
trace files and forces re-restore.

### 4. NaN propagation through measurables

Symptom: trace-derived measurable is all-NaN on certain corpora.
Cause: trace columns the measurable `reads` aren't present in
the corpus's `traces.parquet` (either never computed, or
archive incomplete).

Diagnostic: `scripts/trace_schema.py <traces.parquet>` lists
trace columns + which measurables the corpus can compute.

Fix: re-run sweep to regenerate trace columns OR purge the
corpus entirely.

## Substrate-author workflow notes

**Backfilling a new measurable**: declare a bridge that consumes
it (transitive_reads pulls it in automatically), OR add to
`Hypothesis.REQUIRED_MEASURABLES`. Then run `--ingest-all
experiments/data` and the runner populates the cache.

**Trace-dependent measurables**: when authoring, the local cache
must have the trace columns. The first `--ingest-all` after
declaring a bridge that reads new trace cols will trigger a
column-projected restore from S3 (if cloud manifest carries
them). Subsequent runs use the cached values.

**Closure-hash discipline**: changing a measurable's body
silently invalidates all cached column values on next ingest.
The framework recomputes from trace cols. If trace cols are
absent, the recomputed value is NaN. The framework should
prefer the (stale-but-non-NaN) cached value over the
(recomputed-but-NaN) value in this case, but currently treats
the closure-hash mismatch as authoritative drift. Users
modifying measurable bodies should verify trace availability
before reingest.

**Workflow rule**: never `rm` cloud-backed files directly.
Use `corroborate.corpus.cloud.purge(sweep_dir, files=...)`
which validates manifest membership before deletion. Memory
`feedback_use_purge_not_rm` records this discipline.

## Related docs

- `CACHE_BUILD.md` — five invariants (C1–C5) for cache build.
- `CACHE_ADDITIVITY.md` — lifecycle contract; directory-walk vs
  legacy data paths; when-to-rebuild rules.
- `CORPUS_INTEGRITY.md` — per-corpus integrity guarantees (CI1–CI8).
- `HYPOTHESIS_AS_GRAPH.md` — bridge graph + cluster verdicts on
  top of the cache.
