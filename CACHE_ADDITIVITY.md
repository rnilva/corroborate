# CACHE_ADDITIVITY.md — design contract for cache lifecycle

## What and why

The runner has **two distinct data paths** depending on how
`run()` is called:

- **Directory-walk path** (`data=<dir>`): Phase 2.2 of
  CACHE_BUILD.md — per-corpus
  `<corpus>/measurements.parquet` stores are the source of
  truth; the per-hypothesis cache parquet is rebuilt as a wide
  diagonal_relaxed concat over them. Per-corpus stores ARE
  column-additive (closure-hash drift, partial-nullity recompute,
  CI2/CI6 enforced). The per-hypothesis cache is NOT
  column-additive at its own layer — it's a derived projection,
  atomically rewritten each ingest, with the sidecar unlinked.
- **Legacy path** (`data=None` or a single DataFrame/file):
  reads the existing per-hypothesis cache parquet, applies
  `_invalidate_drifted` against `<cache>.hashes.json` (legacy
  manifest, only present after a non-directory ingest), then
  `_enrich_cache_in_place` calls `compute_missing_columns`
  against the cache. CACHE_BUILD.md's C3 fast-path
  (`measurable.py:611-619`) early-outs without per-cell
  materialisation when nothing's missing — so this path is
  cheap on a current cache.

The CLI surface today obscures these distinctions: `--data <root>`
walks the entire corpus tree on every analysis run, even when
nothing's changed at the per-corpus level. Users (and AI agents)
have learned to reflexively pay the 30-45 min cost because every
documented example passes `--data`.

This doc is the principle: **a cell's content identity is stable
across all reads; ingest is named, deliberate, and never
automatic; rebuild is opt-in**.

## Principle

A user runs the analysis loop in three modes:

1. **Read-only** — "what does the cache say?" Most iterations
   fall here (edit a bridge, re-evaluate; tweak a scope predicate;
   inspect a verdict). No new data, no per-corpus recompute, no
   directory walk.

2. **Ingest** — "I just produced new data, or edited a measurable
   that drifts a column — pull it in." A specific named corpus
   (selective), or the full root (when the user doesn't know
   what's new).

3. **Rebuild** — "I don't trust the cache; start fresh." Rare,
   explicit, opt-in. Wipes the per-hypothesis projection only;
   per-corpus stores are NOT touched (their drift detection is
   the real source of truth).

The framework should make these three modes obvious in the CLI.
The first should be the default; the others should require an
explicit flag naming **what** to ingest or **why** rebuild.

## Three invariants

### CA1. Cell `id` is content-stable across all paths except `--rebuild`

A cell's UUID `id` is assigned at sweep dispatch and persists
unchanged through every subsequent operation. `--ingest` of a
corpus may recompute its measurable columns (drift), but never
re-assigns `id`. `_dedup_against_cache` keys on `id`.

This is the load-bearing additivity invariant: **a cell once
written keeps its `id` for life, modulo a substrate-level
re-collection** (which produces fresh cells with fresh `id`s,
caught by CI4 content-dedup).

**Why.** Without this, every other "additivity" claim fragments.
A substrate edit that recomputed cells with fresh `id`s would
defeat dedup, double-count cells, and make the cache parquet
unbounded. Pinned by `_dedup_against_cache` at runner.py + CI4
content-equality dedup at `_dedup_by_content`.

**Pinned by.** *Future test*: `test_ingest_preserves_cell_ids`
— invoke `--ingest <corpus>` twice; assert no `id` changed
between runs.

### CA2. Read-only is the default; no walk, no per-corpus compute

`run_hypothesis.py <module>` (no extra flags) takes the **legacy
path** — reads the existing per-hypothesis cache parquet, runs
bridges. Specifically:

- No `_load_directory` invocation.
- No `_load_one_corpus` invocation.
- No `build_measurements` call (per-corpus stores untouched).
- `_enrich_cache_in_place` calls `compute_missing_columns` which
  early-outs via CACHE_BUILD.md's C3 fast-path
  (`measurable.py:611-619`) when no required column is missing.
  Cheap.
- Per-corpus drift detection only fires under `--ingest`/
  `--ingest-all`; the legacy path trusts the cache as-is.

The current `runner.run(data=None)` path already delivers all of
this; the gap is CLI surface (the user has been trained to pass
`--data` reflexively). Making it the obvious default is a
documentation + flag-rename change, not a code-behavior change.

**Why.** Most analysis iterations don't change data. Walking 70
corpora to confirm "yep, nothing changed" is wasted work.

**Pinned by.** *Future test*:
`test_run_no_walk_when_data_none` — pass `data=None`, assert
`_load_directory` is never called and the cache parquet's
mtime is unchanged after the run.

### CA3. Ingest is explicit, named, and subordinate to CORPUS_INTEGRITY

`run_hypothesis.py <module> --ingest <corpus>[,<corpus>...]`
processes only the named corpora. Each is loaded, joined with
its `traces.parquet` if needed, run through `build_measurements`
(per-column drift detection at the per-corpus layer); the
per-hypothesis cache parquet is rebuilt by re-projecting every
in-scope corpus's measurements store.

For the rare "I don't know what's new" case:
`--ingest-all <root>` walks the full tree (today's
`--data <root>` behavior, renamed for honesty).

**Subordinate to existing integrity invariants.** `--ingest`
routes through the same `_load_one_corpus` as `--ingest-all`,
so all CORPUS_INTEGRITY checks fire identically: CI1
nested-refusal (skip on `.in_progress` sentinel), CI3
cloud-root collision at archive-time (named-ingest doesn't
escape collision detection if it triggers a re-archive), CI5
0-byte-archive refusal, CI8 traces-id-subset.

**Edge cases:**
- `--ingest <name>` resolves to `experiments/data/<name>/` if
  relative, used as-is if absolute. To ingest a single
  `runs.parquet` file (legacy use), use the explicit
  `--ingest-file <path>` flag (Phase 1 below).
- Named corpus with no `_remote.json` and no local
  `runs.parquet`: raises `FileNotFoundError` from the existing
  `_load_data` path — not silent skip.
- `--ingest a,b` order doesn't affect content (each rebuild of
  the per-hypothesis projection re-reads every in-scope corpus's
  per-corpus store).

**Why.** The natural unit of ingest is one corpus (one sweep
finishing). Walking 70 dirs to find the one that changed wastes
work, hides intent, slows iteration. Naming what to ingest is
the user's intent, made explicit.

**Pinned by.** *Future test*:
`test_ingest_only_named_corpus` — pass `--ingest a,b`, assert
`_load_one_corpus` called for a and b only, never for c.

## Audit table — current vs. target

| Invariant | Current code | Gap |
|---|---|---|
| **CA1** Cell-id stability | YES (sweep dispatcher mints UUID per cell; no re-assignment path) | Add the named regression test |
| **CA2** Read-only default | YES at runtime (CACHE_BUILD.md C3 fast-path is implemented; `data=None` already cheap) | CLI surface only — deprecate `--data`, make no-flags the obvious default |
| **CA3** Per-corpus named ingest | NOT YET — `--data <root>` walks everything; no per-corpus name resolution | Add `--ingest <corpus>[,<corpus>...]`; rename `--data` → `--ingest-all`; deprecate `--data` |

Three rows, three concrete deltas. Invariant count tracks
substantive contract additions, not "reminders that
CACHE_BUILD.md exists."

## Implementation order

**Phase 1 — CLI surface (~30 min)**

C3 fast-path is already implemented (`measurable.py:611-619`),
so this phase is purely CLI-side.

1. Add `--ingest <corpus>[,<corpus>...]` to
   `scripts/run_hypothesis.py` and `runner.run()`. Resolves each
   name to `experiments/data/<name>/` if relative, used as-is if
   absolute.
2. Add `--ingest-file <path.parquet>` for the rare single-file
   ingest case (today's `--data path/to/single.parquet` shape).
3. Rename `--data` → `--ingest-all`. Keep `--data` as a hidden
   alias for back-compat in scripts; deprecate in docs.
4. Make `run_hypothesis.py <module>` (no `--ingest`/`--ingest-all`
   /`--rebuild`) print a one-line cache state on entry:
   `'cache: <N> cells, <M> measurable cols, last updated <ts>'`.
   Then run bridges. No walk.

**Phase 2 — `--check` mode (~1 hour, optional)**

Drift visibility without work, sourced from per-corpus
`measurements.hashes.json` files (the per-hypothesis manifest
no longer exists post-Phase-2.2). Reports:
- For each in-scope corpus: which columns drifted (current
  closure hash ≠ stored), which required columns are missing.
- A union summary: "12 corpora have drifted columns; refresh
  with `--ingest-all`" or "3 corpora carry only stale columns;
  refresh with `--ingest a,b,c`".

Skip if Phase 1 + the side-effect drift logging in `--ingest`
turns out to handle 95% of the use case (most likely outcome).

**Phase 3 — Per-corpus mtime tracking (deferred)**

When `--ingest-all <root>` runs, the per-corpus `build_measurements`
fast-path skip handles "this corpus is up-to-date" correctly,
but only AFTER reading `runs.parquet` and joining traces. A
pre-walk mtime guard could skip the parquet read entirely for
corpora whose `runs.parquet.mtime` ≤ stored last-ingest mtime.
Premature optimization until `--ingest-all` is the bottleneck;
note that this is a separate sidecar from the closure-hash
sidecar (different invariant: filesystem mtime vs registry
hash), so they layer rather than converge.

## Migration

Phase 1 is a CLI rename + addition. The behavior change is:
running `run_hypothesis.py <module>` with no flags becomes
read-only (today: requires data + walks tree).

Concrete edits (`grep -rn --include="*.{py,md}" "\-\-data" .` is
the audit lever):
- `scripts/run_hypothesis.py` — flag definitions, deprecation
  notice on `--data`.
- `README.md` line ~130 — update example.
- Auto-memory `MEMORY.md` `reference_runner_flow` entry —
  documents `--data experiments/data/` as canonical; needs
  rewrite so future agents reproduce the new form.
- `experiments/findings/*.md` and `*.run.json` failure hints —
  audit + rewrite. The snapshot regression test
  (`tests/test_run_report_snapshot_regression.py`) prints a
  `--no-restore` recommendation on failure; that wording's fine.
- Module docstrings in `experiments/findings/*.py` — usage
  examples mention `--data`.
- `CLAUDE.md` and any reference doc that names `--data`.

After 1-2 weeks of use, the `--data` alias can be removed if no
scripts break. Or kept indefinitely — the user-facing surface
that matters is the new flag names.

## Connection to other manifests

- **CACHE_BUILD.md** — the *mechanics* of cache building
  (atomicity, drift detection at the per-corpus level, two-level
  architecture). This doc reuses those mechanics. **Per-column
  drift-detect** is enforced there (closure-hash invalidation,
  partial-nullity recompute, orphan eviction); not re-stated as
  a CACHE_ADDITIVITY invariant.
- **CORPUS_INTEGRITY.md** — integrity invariants at the corpus
  boundary (CI1-CI8). CA3 is *subordinate* to all of them.
- **SWEEP_PERSISTENCY.md** — sweep-time persistence layer
  (I1-I5). Independent.

**Existing flags worth preserving but not numbered as CACHE_ADDITIVITY
invariants:** `--rebuild` is the explicit-nuke flag (wipes the
per-hypothesis projection only — per-corpus stores survive their
own drift detection); `--check` for drift visibility without
work is a Phase 2 future feature, mostly redundant with the
side-effect drift logging that `--ingest` already produces.

This doc shouldn't duplicate any of those — only the
**default-is-additive** principle that they collectively make
possible but don't currently surface in the CLI.

## What this doc is not

A roadmap with deadlines. It's the design contract; anyone
editing `scripts/run_hypothesis.py` or `runner.run()`'s data
parameter should be able to point at the invariant their change
preserves.

If a future change can't preserve all three, the change needs a
written waiver in this doc.
