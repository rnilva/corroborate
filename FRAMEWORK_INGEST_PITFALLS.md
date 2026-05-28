# Framework measurable-ingestion pitfalls (handover doc)

Compiled from hitting these during 2026-05-28 expansion of
`hasselt_clean.REQUIRED_MEASURABLES`. Each section is a distinct
issue worth a principled framework-level fix. Some have obvious
fixes; others may need design discussion.

The pitfalls are listed in rough order of how badly they bit during
the session. The numbered cross-references identify load-bearing
files / functions to read first.

---

## P1. `reads=()` lies about trace dependencies

**Symptom.** `eval_late_burst_raw_mean` declares
`@measurable(name='eval_late_burst_raw_mean', reads=())` — but its
function signature takes `mc_return_raw_episodes` (a list-typed
per-burst column) as a parameter. The framework's
`transitive_reads()` then classifies `mc_return_raw_episodes` as a
required read via parameter-name injection. The `_missing_for_restore`
runner check treats `mc_return_raw_episodes` as a trace-store column
and demands `traces.parquet` to be locally restored before recompute.
When traces are cloud-evicted, the measurable is "skipped — transitive
reads not available locally" even though the data is right there in
`measurements.parquet` as a list col.

**Hit at.** SpaceInvaders (`g099_SpaceInvaders-MinAtar` corpus):
`eval_late_burst_raw_mean` never landed in the cache despite
multiple ingests; report L3b table had to mark SI as `NaN marg ρ`
and document the cache gap.

**Files.**
- `src/corroborate/runner/runner.py` — `_missing_for_restore`,
  `_load_one_corpus`'s sidecar-current check
- `src/corroborate/measurables/measurable.py` — `transitive_reads`
- `src/corroborate_rl/corroborate_rl/dqn/measurables.py:eval_late_burst_raw_mean`
  (and other reads=() + parameter-injection measurables)

**Principled fix candidates.**
- Make `reads` reflect every column the framework will read at
  computation, including parameter-injected list columns. The
  measurable IS a function of `mc_return_raw_episodes`; pretending
  it has no reads is the lie.
- OR have `_missing_for_restore` distinguish trace-store columns
  from list columns in `measurements.parquet`/`runs.parquet`. A
  column that's already a populated `List[Float]` in cache doesn't
  need traces.parquet restored.
- OR auto-derive reads from the function signature so authors
  can't desync the two.

---

## P2. Stale-NaN column drop on every ingest

**Symptom.** Every ingest emits lines like:

  measurements: dropped 10 stale-NaN registered column(s) from
  experiments/data/<corpus>/measurements.parquet (sweep-time
  stamps with no live recompute path): ddqn_bootstrap_gap,
  ddqn_bootstrap_gap_late, ..., reward_nonzero_frac,
  target_staleness_early, target_staleness_late

The "stale-NaN" guard drops registered cols that are entirely NaN
locally. Re-ingest replaces them via the recompute path — IF the
reads are satisfied locally. When traces aren't restored, they
just stay dropped, so each ingest erodes the per-corpus store.

**Hit at.** Many corpora — the dropped list above is from each
canonical γ=0.99 corpus on every ingest call. Repeated ingests
caused col counts to shrink (e.g. SI went to 34 cols across the
session).

**Files.**
- `src/corroborate/corpus/measurements.py` — `build_measurements`,
  the stale-NaN-column drop logic

**Principled fix candidates.**
- Don't drop stale-NaN columns unless the local data CAN satisfy
  the recompute. If reads aren't satisfied, leave the NaN (and
  declare the measurable explicitly stale rather than silently
  dropping).
- OR add a sidecar-level "permanently-unsatisfiable on this
  corpus" marker so repeated ingests don't re-attempt + re-drop.

---

## P3. `--ingest-all` REPLACES cache rather than adding

**Symptom.** Ran `corroborate hypothesis ... --ingest-all
experiments/data/minatar_gamma_sweep_k1_v2` expecting CACHE_ADDITIVITY
behavior (append). The resulting cache had ONLY the 4 MinAtar
sub-corpora — all 11 MLP envs from the prior ingest disappeared.

**Hit at.** Mid-session, had to redo the multi-corpus ingest
sequentially after losing the MLP envs.

**Files.**
- `src/corroborate/runner/runner.py` — main `hypothesis` entry,
  `--ingest-all` path
- `CACHE_ADDITIVITY.md` — claim is contradicted by observed behavior

**Principled fix candidates.**
- Either rename to `--ingest-all-only` if replacement is intended;
  add a non-replacing `--ingest-discover ROOT` variant.
- OR make `--ingest-all` genuinely additive (most users expect
  this; matches the documented contract).

---

## P4. Sidecar-current fast-path skips even with `--force-recompute`

**Symptom.** `--force-recompute eval_late_burst_raw_mean` should
recompute the named measurable regardless of sidecar state. It
DOES write `wrote 30 cells × 85 measurable cols` to the per-corpus
store, but the actual values stay NaN/stale because the upstream
restore-from-cloud step decided traces weren't needed (sidecar
said the measurable was current).

**Hit at.** SI debugging — had to manually delete the sidecar
entry before the framework would attempt a real recompute. Even
then the issue overlapped with P1 (transitive reads thought to be
trace-only).

**Files.**
- `src/corroborate/runner/runner.py` — `_load_one_corpus`'s
  fast-path check; interaction between `force_recompute` and
  `_measurements_sidecar_current`

**Principled fix candidates.**
- `--force-recompute NAME` should remove NAME from the sidecar
  check (and from `_missing_for_restore`'s skip set) before any
  fast-path decision.

---

## P5. Cache assembly OOM at corpus 12/15

**Symptom.** Running `--rebuild --ingest CORPUS1,...,CORPUS15`
gets killed (SIGKILL, exit 137) reproducibly at the asterix corpus
(index 12). 64 GB RAM, 43 GB free — should be plenty. Workaround:
ingest the 11 MLP corpora first with `--rebuild`, then add the 4
heavier corpora one-at-a-time.

**Hit at.** Three separate attempts; each killed at the same
spot. Two-batch workaround was the only way to assemble the full
cache.

**Files.**
- `src/corroborate/runner/runner.py` — the ingest loop's memory
  growth pattern
- `src/corroborate/corpus/measurements.py` — `build_measurements`
  (suspect: holding all per-corpus dataframes in memory)
- See also `stream_concat_parquets` in `runner/sweep.py` (already
  streaming for sweeps; should be re-used for cache assembly)

**Principled fix candidates.**
- Stream the cache write per-corpus (append to parquet) instead
  of concatenating all in memory at end.
- OR drop the per-corpus DataFrame after writing its rows to the
  cache (currently they appear to accumulate).

---

## P6. Per-corpus measurements.parquet diverge in column count

**Symptom.** After identical `--ingest` calls against the same
REQUIRED_MEASURABLES list, per-corpus stores have wildly different
col counts:
- `acrobot_g099_canonical_n_eps20_ckpt/measurements.parquet`: 85 cols
- `g099_SpaceInvaders-MinAtar/measurements.parquet`: 34 cols
- `asterix_g099_canonical_n_eps20_ckpt/measurements.parquet`: 85 cols

The framework's "skipping N drifted/missing measurable(s)" tells
us why per-corpus, but there's no top-level reconciliation: the
cache assembly silently joins the inconsistent stores. The cache
ends up with columns NaN at some envs (e.g., MinAtar) and finite
at others (MLP). L3b's all-finite filter then drops every cell at
the gappy env.

**Hit at.** L3b's per-env candidate auto-detection had to be
inside the env loop, not global. Originally my script used the
global candidate union and got all-NaN cells for MinAtar/Jumanji.

**Files.**
- `src/corroborate/runner/runner.py` — the per-corpus measurement
  compute + cache assembly steps
- `src/corroborate/data/panel.py` — `Panel.from_cache` (consumer
  of the inconsistent state)

**Principled fix candidates.**
- Add a `panel.diagnostics()` surface that reports per-env
  measurable availability so consumers can decide whether to
  drop / skip / impute.
- OR enforce uniformity at cache assembly time: any measurable
  that's missing at SOME corpora gets dropped from ALL corpora
  in the cache (loud failure, not silent NaN).
- OR keep the current behavior but make `Panel.from_cache`
  surface a per-env-per-measurable matrix so script authors
  can branch on availability without inspecting cells manually.

---

## P7. Scope predicate vs corpus stamping drift

**Symptom.** `CANONICAL_G099_CORPORA` (in
`experiments/findings/hasselt_clean/_scope.py`) had
`'g099_Asterix-MinAtar'` (the sub-corpus stamp from
`minatar_gamma_sweep_k1_v2`) but I'd been ingesting the newer
`asterix_g099_canonical_n_eps20_ckpt` (top-level corpus with full-Q
canonical traces + broad measurable set). The scope filter
silently excluded the newer corpus → Asterix shows up under the
older stamp, missing the broad mediator measurables.

This was a major source of confusion. Asterix's marg ρ = +0.73 is
the strongest in the panel and SHOULD have surfaced PC mediators —
but PC was reporting "underpowered" because the old corpus has
30+ NaN candidates.

**Hit at.** Took ~10 turns of back-and-forth to diagnose. Fixed by
swapping `g099_Asterix-MinAtar` → `asterix_g099_canonical_n_eps20_ckpt`
in CANONICAL_G099_CORPORA.

**Files.**
- `experiments/findings/hasselt_clean/_scope.py` — CANONICAL_G099_CORPORA
- `src/corroborate/runner/runner.py` — `_corpus_stamp` logic (parent/leaf form)
- All env-class canonical scopes; this issue likely replicates at
  `ddqn_sweeps`, `ddqn_three_conditions`, etc.

**Principled fix candidates.**
- Author corpus stamps as STABLE identifiers (e.g., `{env}_g{γ}_canonical`)
  that don't change when a newer sweep replaces an older one. The
  hypothesis-author authors against the stable name; the runner
  resolves to whatever corpus carries the stamp.
- OR provide a `scope.canonical_for(env, gamma)` helper that
  resolves to the freshest corpus stamp at lookup time, rather
  than hardcoding the list. The Catalogue already knows this.
- OR add a `--verify-scope` mode that runs through every entry in
  `CANONICAL_*_CORPORA` and reports which are present vs missing
  in the cache, so misalignment is loud.

---

## P8. Disk pressure on cloud-trace restore

**Symptom.** SI's `traces.parquet` is ~4 GB in cloud. Restoring it
filled /workspace to 100% (64 GB total), the restore failed
mid-write, and subsequent ingests couldn't proceed. Had to
`corroborate purge` several other corpora's traces to free space.

**Hit at.** SI debugging. The framework's auto-restore + auto-evict
pattern (CI7) doesn't guard against insufficient disk for the
single biggest trace file.

**Files.**
- `src/corroborate/corpus/cloud.py` — `restore`, `restore_columns`

**Principled fix candidates.**
- Pre-flight check available disk space vs trace-file size before
  starting the restore; abort cleanly with "needs X GB free".
- OR restore into a temp scratch dir and atomically swap.
- OR support column-projected restore even more aggressively so
  we don't pull 4 GB to read 2 columns (`restore_columns` exists
  but the upstream caller wasn't using it for this measurable).

---

## What I'd consider the principled top-3

1. **P1 + auto-derived reads (P1)**: fix the empty-`reads=()` lie
   by having the `@measurable` decorator auto-extract reads from
   the function signature (or at least validate that declared
   reads ⊇ parameter names other than `record`). Closes P1 + P4
   together.

2. **Cache assembly streaming (P5)**: re-use `stream_concat_parquets`
   for the cache-write step. Closes the OOM at any panel size.

3. **Per-env measurable availability surface (P6)**: expose
   `panel.measurable_availability_matrix()` (env × measurable
   booleans). L3b and similar scripts can then branch cleanly
   without auto-detecting per env via `is_finite().sum()` heuristics.

P3 is a documentation fix (or a tiny behavior change). P7 is a
substrate-author discipline issue + helper. P8 is a robustness
nice-to-have.

---

## Verification checklist after fixes

A clean re-ingest from scratch should:

- [ ] `--rebuild --ingest <full canonical list>` completes in one
      command without OOM at 12/15
- [ ] Every per-corpus `measurements.parquet` ends up with the same
      column set (or a sidecar declares which are unsatisfiable per
      corpus, no silent drift)
- [ ] `Panel.from_cache('experiments.findings.hasselt_clean')` returns
      12 envs × 60 cells each (PacMan + LL exception) with
      `eval_late_burst_raw_mean` finite at SI
- [ ] `papers/g099_mediation/scripts/03b_per_env_best_mediator.py`
      reports PC-discovered mediators at SI without falling through
      to "outcome variance" degeneracy
- [ ] `Panel.diagnostics()` (new surface) lists any unsatisfiable
      measurables explicitly

The `papers/g099_mediation` scripts are the live consumer; they're
the integration test.
