# CORPUS_INTEGRITY.md — design contract for the corpus boundary

## What and why

A "corpus" is a directory under `experiments/data/<name>/` that
holds the raw cells of a sweep — `runs.parquet`, `traces.parquet`,
`measurements.parquet`, `_remote.json`, optionally per-burst
reductions and graphs sidecar. The framework reads it at analysis
time (`runner._load_one_corpus`) and writes it at sweep time
(`sweep.run_intervention` + `cloud.archive`).

Today the directory shape is **implicit**: a corpus is whatever
the filesystem happens to contain. Substrate authors do ad-hoc
operations as sweeps evolve — extending samples, resuming
interrupted runs, adding envs, replicating arms — and each
operation can drift the corpus shape without the framework
catching it. The result, observed in the wild on the
`ddqn_universe` corpus tree:

| Failure | What we found |
|---|---|
| 4 sub-corpora share one `s3://…/minatar_sync_curve/` cloud root | Each push overwrote the prior. 3 of 4 cloud copies are now ddqn_sync3k's data. |
| `minatar_sync_curve_resume/measurements.parquet` had 245,760 rows over 120 unique ids | 2048× duplication accumulated across rebuilds; left-joins exploded. |
| `vanilla_sync1k` is bit-identical to `ddqn_sync1k`'s baseline subset | 30 unique runs replicated 3× — `_dedup_by_content` couldn't see it because `env` carries an object repr with per-session memory address. |
| `minatar_sync_curve/{ddqn_sync1k, ...}/`, `polyak_tau_intervention/{polyak_tau_X}/` | Runner walks one level, silently skips nested sub-corpora. |
| `action_dim_wide/traces.parquet` (0 bytes), `reward_scale_sweep/traces.parquet` (0 bytes) | Sweep-time merge interrupted; archive_unarchived.py blindly pushed the empty placeholder. |
| 14 GB of pre-existing local `traces.parquet` files unreclaimed | Eviction policy fires only on "just-restored," ignored locally-cached. |

These aren't bugs in the substrate code — they're operational
gaps that the framework didn't catch because **nothing enforces
the corpus-shape contract**. This document is that contract.

## Vocabulary

- **Corpus**: a directory `<root>/<name>/` with at least
  `runs.parquet`. Identified by its `<name>` (the directory
  basename); has at most one `_remote.json` recording its cloud
  root.
- **Corpus components**: `runs.parquet` (mandatory),
  `traces.parquet` (optional but heavyweight),
  `measurements.parquet` + `measurements.hashes.json` (Phase 1
  per-corpus store), `_remote.json` (cloud manifest — I5 from
  SWEEP_PERSISTENCY.md), `graphs.json` (per-arm computation graph
  topology), per-burst reductions.
- **Cell**: a `RunRow` — one row in `runs.parquet`. Identified by
  its UUID `id`. The atomic unit of a sweep.
- **Cloud root**: the `remote_root` field in `_remote.json`,
  e.g. `s3://corroborate-archive/minatar_sync_curve/`.

## Seven invariants

### CI1. A corpus is a leaf in the directory tree

A directory containing `runs.parquet` MUST NOT have a
subdirectory that ALSO contains `runs.parquet`. Corpora don't
nest.

**Rule.** At ingest time, the runner refuses any corpus dir
whose subtree (depth ≥ 1) contains `runs.parquet`. Either flatten
the layout or move the parent's top-level files to a sibling
position.

**Why.** The runner's directory walk is one-level-deep by
design — recursing arbitrarily would conflate sweep extensions
with separate experiments. Nesting today silently drops the
inner corpora; this invariant makes the silent drop loud.

**Failure caught.** `minatar_sync_curve/{ddqn_sync1k, ddqn_sync3k,
vanilla_sync1k, vanilla_sync3k}/`, `minatar_sync_curve_pt2/
ddqn_sync1k/`, `minatar_sync_curve_resume/ddqn_sync3k/`,
`polyak_tau_intervention/{polyak_tau_X}/`, `polyak_tau_asterix/
{polyak_tau_X}/`. All silently dropped pre-fix.

### CI2. Cell `id`s are unique within a corpus's stores

In a corpus's `runs.parquet` and `measurements.parquet`, every
`id` MUST appear exactly once. The `measurements.parquet` rows
MUST be a subset of `runs.parquet` ids (no orphans).

**Rule.** `build_measurements` validates `existing.height ==
existing['id'].n_unique()` at entry; mismatch triggers
rebuild-from-scratch with a stderr warning. Future: promote to
a hard error after the existing corruption is cleaned up.

**Why.** A duplicate id in the store causes the next runs_df ←
existing left-join to Cartesian-multiply: 2048× duplication on
the next rebuild compounds doubling each run.

**Failure caught.** `minatar_sync_curve_resume/measurements.parquet`
(245,760 rows / 120 unique ids).

### CI3. A cloud root is owned by exactly one local corpus

Each `_remote.json`'s `remote_root` MUST be unique across all
local corpora. Two sibling corpus dirs writing to the same cloud
root is forbidden.

**Rule.** `archive(sweep_dir, remote_root)` consults a global
index of currently-claimed `remote_root`s (built by walking
local `_remote.json` files); raises `RemoteRootCollision` if
the root is already claimed by a different sweep_dir.

**Why.** Same cloud root = same cloud key = silent overwrite on
every push. The first three sub-corpora's data is gone the
moment the fourth one archives.

**Failure caught.** `s3://corroborate-archive/minatar_sync_curve/`
claimed by all four `minatar_sync_curve/*` sub-corpora; only
`ddqn_sync3k`'s upload survives on cloud.

### CI4. Content-equality dedup ignores runtime-volatile fields

`_dedup_by_content` MUST exclude object-typed columns whose
`repr()` varies per Python session even when the underlying value
is conceptually equal. Dotted-leaf scope columns (`env`, `claim`,
substrate construct instances) typically fall in this set.

**Rule.** Extend `_PROVENANCE_TAGS` to include any column with
polars dtype `Object` AND any string column whose values look
like `"<…\sobject\sat\s0x[0-9a-f]+>"`. The content-equality
check operates only on stable-content columns.

**Why.** Without this, two cells with identical (env_name, arm,
seed, all HPs) but different `<env-instance>` repr addresses are
treated as distinct. The substrate's `vanilla_sync*` corpora
literally replicated `ddqn_sync*`'s baseline cells — but
`_dedup_by_content` couldn't tell, leaving 90 baseline cells
when there were really 30.

**Failure caught.** `vanilla_sync1k` (60 cells) + baseline
subset of `ddqn_sync1k` (30 cells) → 90 cells visible to
analysis when they're really 30 unique runs replicated 3×.

### CI5. Archive refuses trivially-broken files

`cloud.archive()` MUST refuse to upload:
1. A parquet file smaller than some minimum (~1 KiB).
2. A parquet file lacking the `PAR1` magic footer.
3. Any other file whose size is suspiciously zero.

**Rule.** Add a pre-upload validation step in `archive()` that
calls `_file_present()` on each selected file (which already
does PAR1-footer + min-size checks) and raises a clear
`ArchivePrecondition` error before issuing `_fs.put_file`. The
substrate caller can opt-in to override via an explicit
`force_corrupt=True` flag — never silently.

**Why.** A 0-byte `traces.parquet` placeholder, archived
without protest, becomes the cloud's authoritative copy.
Restoring later silently materializes the empty file and
trace-dependent measurables fail KeyError on every cell.

**Failure caught.** `action_dim_wide/traces.parquet` (0 bytes,
sha256 `e3b0c44298…` = canonical empty-file hash),
`reward_scale_sweep/traces.parquet` (same).

### CI6. Per-corpus stores stay in sync with their parent runs.parquet

`measurements.parquet` rows MUST correspond 1:1 with
`runs.parquet` rows by `id`. Stale orphan rows (from removed
cells) get dropped on every rebuild.

**Rule.** At `build_measurements` entry, after the duplicate-id
check (CI2), filter `existing` to only the rows whose `id` is
present in `runs_df['id']`. Drop any orphan; emit a stderr
warning with the count.

**Why.** Sweep extensions and partial reruns can leave stale
ids in `measurements.parquet` after their cells get removed
from `runs.parquet`. The framework should reconcile rather than
accumulate.

**Failure caught.** Latent on the `_resume` corpora — could
manifest as off-by-many counts in pooled analyses if a partial
rerun ever drops cells.

### CI7. Local trace files are reclaimed proactively

After the runner finishes computing measurables for a corpus,
its `traces.parquet` (often GB-scale) MUST be evicted IF it's
recoverable from cloud (i.e., `_remote.json` lists it with a
matching sha256). Locally-only trace files (no cloud copy) stay
local — eviction would lose data permanently.

**Rule.** Extend the eviction logic in `_load_one_corpus` to
fire on every locally-recoverable trace file at the end of the
per-corpus block, not only the just-restored subset. Track each
corpus's "recoverable" status from its manifest at load time.

**Why.** Pre-existing local traces accumulate over multiple
rebuilds. With 14+ GB sitting locally and the next corpus
needing to restore another 3-15 GB, disk pressure can OOM-kill
the rebuild.

**Failure caught.** `ddqn_better_hp` (3.4 GB), `fourrooms_1m`
(3.2 GB), three `polyak_tau_intervention_*` corpora (2.7 GB
each) — 14 GB unreclaimed, contributing to disk-full failures
at corpus 22-23 in three consecutive rebuilds.

## Audit table — current vs. target

| Invariant | Held by current code? | Gap |
|---|---|---|
| **CI1** Corpora are leaves | NO — silent skip on nested | Refuse loudly at ingest |
| **CI2** Per-corpus id uniqueness | YES (Phase 0, just landed) | Promote warning to error after cleanup |
| **CI3** Cloud-root uniqueness | NO — silent overwrite | Global remote_root registry; raise `RemoteRootCollision` |
| **CI4** Content-dedup strips volatile | NO — `env` repr defeats dedup | Extend `_PROVENANCE_TAGS`; dtype-aware exclusion |
| **CI5** Archive refuses trivial files | NO — 0-byte pushes succeed | Pre-upload `_file_present` check in `archive()` |
| **CI6** Stores in sync with parent | PARTIAL — column-level orphan eviction (C4 in CACHE_BUILD.md), no row-level | Add row-level orphan drop in `build_measurements` |
| **CI7** Disk-pressure reclaim | PARTIAL — only just-restored | Extend to all cloud-recoverable traces |

## Implementation order

Phased so each lands independently and the corpus tree stays
usable between phases.

**Phase 0 — already landed defensively** (no behavior change for
correct data):
- CI2 duplicate-id detect + rebuild-from-scratch in
  `build_measurements`. Pinned by
  `test_build_measurements_rebuilds_from_scratch_on_duplicate_ids`.

**Phase 1 — fail loud at I/O boundaries**:
1. **CI1**: `_load_directory` walks one extra level to detect
   nested `runs.parquet`; raises `NestedCorpusError` with the
   path of the offender.
2. **CI3**: `archive()` consults a per-call index of local
   `_remote.json` `remote_root`s; raises `RemoteRootCollision`
   when two corpora claim the same root.
3. **CI5**: `archive()` runs `_file_present()` on each selected
   file before upload; raises `ArchivePrecondition` on 0-byte
   or non-PAR1 parquet.

Each gets a regression test that constructs the exact failure
shape from the inventory and verifies the error fires.

**Phase 2 — content-aware defenses**:
4. **CI4**: extend `_PROVENANCE_TAGS` with object-typed columns
   detected at runtime; rerun the existing
   `_dedup_by_content` test corpus with a substrate that
   carries `env` to verify dedup catches replicates.
5. **CI6**: row-level orphan eviction in `build_measurements`.

**Phase 3 — resource hygiene**:
6. **CI7**: broader eviction. Tag each restored trace file
   with its cloud sha256 at load time; eviction at end of
   `_load_one_corpus` fires when sha matches manifest entry
   regardless of whether THIS run downloaded it.

## Migration

Phase 0 is purely defensive — silent on correct data. Phase 1
introduces three new errors; the existing data tree currently
violates CI1 and CI3. A one-shot `migrate_corpus_layout.py`
script:

1. Walks `experiments/data/`, identifies CI1 violators
   (nested corpora) and proposes flatten renames.
2. Walks all `_remote.json`, groups by `remote_root`, identifies
   CI3 collisions and proposes either:
   - Distinct cloud roots (re-archive operation), or
   - Local-only operation (drop `_remote.json`).
3. Optionally executes the proposed fixes after user confirms.

The action-dim_wide / reward_scale_sweep CI5 violations are
historical (already in cloud); the migration script offers to
either remove them from cloud or mark them with a sentinel
file `_corrupt.json` documenting the known empty trace.

## What this doc is not

A roadmap with deadlines. It's the design contract; anyone
editing `runner.py`, `cloud.py`, `measurements.py`, or
`sweep.py` should be able to point at the invariant their
change preserves.

If a future change can't preserve all seven, the change needs a
written waiver in this doc.
