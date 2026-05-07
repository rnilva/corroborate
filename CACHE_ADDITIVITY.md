# CACHE_ADDITIVITY.md — design contract for cache lifecycle

## What and why

The data model is already additive at every layer:

- **Per-cell**: cells appear once in `runs.parquet`, never re-emit.
  Each has a UUID; ingest appends.
- **Per-column**: `compute_missing_columns` adds missing columns
  to existing cells. CACHE_BUILD.md C3+CI2+CI6 keep this honest
  (drift hash, partial-nullity recompute, orphan eviction).
- **Per-corpus**: the per-corpus `measurements.parquet` (Phase 1
  of CACHE_BUILD.md) is the source of truth; the per-hypothesis
  cache is a projection.

But the CLI surface obscures this. The historical `--data <root>`
flag walks the **entire** corpus tree on every analysis run, even
when nothing's changed. Most analysis iterations that should be
~seconds-fast accidentally trigger a 30-45 min walk because the
flag is in every documented example.

This doc is the principle: **cache is additive; walking is
deliberate, never automatic**.

## Principle

A user runs the analysis loop in three modes:

1. **Read-only** — "what does the cache say?" Most iterations
   fall here (edit a bridge, re-evaluate; tweak a scope predicate;
   inspect a verdict). No new data, no recompute.

2. **Ingest** — "I just produced new data — pull it in." A
   specific named corpus, OR a substrate-side measurable edit
   that drifted some column.

3. **Rebuild** — "I don't trust the cache; start fresh." Rare,
   explicit, opt-in.

The framework should make these three modes obvious in the CLI.
The first should be the default; the others should require an
explicit flag naming **what** to ingest or **why** rebuild.

## Five invariants

### CA1. Read-only is the default

`run_hypothesis.py <module>` (no extra args) reads the existing
per-hypothesis cache and runs bridges. Zero directory walks,
zero parquet reads beyond the cache itself.

**Why.** The 80% case is iterating on bridges or analyses
without touching data. Making the slow path the default has
trained users (and AI agents) to reflexively pay the cost.

### CA2. Ingest is per-corpus and named

`run_hypothesis.py <module> --ingest <corpus>[,<corpus>...]`
processes only the named corpora. Each is loaded, joined with
its `traces.parquet` if needed, run through `build_measurements`,
and appended to the per-hypothesis cache. Other corpora's cells
in the cache stay untouched.

**Why.** Sweeps produce one new corpus at a time. The natural
unit of ingest is one corpus. Walking 70 dirs to find the one
that changed wastes work, hides the user's intent, and slows
iteration.

For the rare "I don't know what's new" case:
`--ingest-all <root>` walks the full tree. This is today's
`--data <root>` behavior, renamed to make the cost-vs-need
tradeoff visible.

### CA3. Per-column drift-detect, never recompute-all

A new or edited substrate measurable drifts via closure hash;
`build_measurements` detects it per-corpus and recomputes ONLY
that column for cells whose hash differs. Other columns / other
cells preserved. New measurables added to a bridges file get
their column appended for in-scope cells.

**Why.** Already true today (Phase 0+1+2 of CACHE_BUILD.md).
Pinning it as an invariant prevents future "fixes" that wipe-
and-recompute-all.

### CA4. Rebuild is the explicit nuke

`run_hypothesis.py <module> --rebuild` is the only path that
wipes the cache. Never silent, never automatic — even after a
substrate refactor that drifts every column.

**Why.** Wipe-and-recompute is a 30-45 minute operation that
loses information about the prior state. Asking the user to
opt in protects both their time and their auditability.

### CA5. Drift detection is observable without work

`run_hypothesis.py <module> --check` (planned) compares the
current registry's closure hashes against the cache's manifest
and reports which columns drifted, without computing anything.
Lets the user decide whether to `--ingest-all` or live with the
cache.

**Why.** Without this, cache-only mode produces stale verdicts
silently when the substrate has been edited but the cache hasn't
been refreshed. `--check` makes drift visible without forcing
work.

## Audit table — current vs. target

| Invariant | Current code | Gap |
|---|---|---|
| **CA1** Read-only default | PARTIAL — `run(data=None)` works internally; CLI examples reflexively pass `--data` | Make `--data` opt-in via rename to `--ingest-all`; default = no walk |
| **CA2** Per-corpus named ingest | NOT YET — `--data <root>` walks everything | Add `--ingest <corpus>[,<corpus>...]` |
| **CA3** Per-column drift-detect | YES (CACHE_BUILD.md Phases 0-2) | — |
| **CA4** Rebuild explicit | YES (`--rebuild` exists, opt-in) | — |
| **CA5** Drift visibility without work | NOT YET | Add `--check` |

## Implementation order

**Phase 1 — CLI surface (cheap, ~30 min)**
1. Add `--ingest <corpus>[,<corpus>...]` flag. Resolves each name
   to `experiments/data/<name>/`, processes only those, appends.
2. Rename `--data` → `--ingest-all` (alias `--data` for back-
   compat in scripts; deprecate it in docs).
3. Default = no walk. CLI prints the cache-cell-count when
   running read-only so the user sees what's loaded.

**Phase 2 — `--check` mode (~1 hour)**
1. Reuse `_measurable_signature` to compute current hashes for
   every required measurable.
2. Compare against the per-corpus measurements sidecars (or
   per-hypothesis manifest as fallback).
3. Report: "5 columns drifted across 12 corpora; run
   `--ingest-all` to refresh OR `--ingest corpus_a,corpus_b` to
   refresh only the affected ones."
4. No work done — pure read.

**Phase 3 — Per-corpus mtime tracking (deferred)**
1. Sidecar `<cache>.ingested.json` records per-corpus
   `last_ingested_at_mtime`.
2. `--ingest-all`: skip corpora whose `runs.parquet.mtime` ≤
   stored mtime. Cuts the walk to changed corpora only.
3. Premature optimization until the corpus tree is much larger
   (or the iterdir-loop becomes the bottleneck).

## Migration

Phase 1 is a CLI rename + deprecation, not a behavior change.
Scripts that pass `--data <root>` continue to work via the alias.
New docs / examples only mention `--ingest` and `--ingest-all`.

After 1-2 weeks of use, the alias can be removed if no scripts
break. Or kept indefinitely — the user-facing surface that
matters is the new flag names.

## Connection to other manifests

- **CACHE_BUILD.md** (C0-C5 + Phases 0-3): the *mechanics* of
  cache building, atomicity, drift detection. This doc reuses
  those mechanics; it's about the *user surface* and *default*.
- **CORPUS_INTEGRITY.md** (CI1-CI8): integrity invariants at the
  corpus boundary. The `.in_progress` sentinel lives there.
- **SWEEP_PERSISTENCY.md** (I1-I5): sweep-time persistence layer.
  Independent.

This doc shouldn't duplicate any of those — only the
**default-is-additive** principle that they collectively make
possible but don't currently surface in the CLI.

## What this doc is not

A roadmap with deadlines. It's the design contract; anyone
editing `scripts/run_hypothesis.py` or `runner.run()`'s data
parameter should be able to point at the invariant their change
preserves.

If a future change can't preserve all five, the change needs a
written waiver in this doc.
