# Sweep persistency — design principles

The persistency layer is the framework's contract with **sweep
artifacts that outlive a single Python process**: per-cell
`runs.parquet` / `traces.parquet` shards, manifest files for cloud
archives, and the merged corpus parquets that downstream analyses
consume. This doc names the invariants the layer must hold so that
sweep restarts, multi-intervention sweeps, and re-merges from cloud
can all be **safe by construction**, not by careful authoring.

The doc is principle-led, not bug-led. It came out of two
concrete failures (last writer wins on shared S3 paths; per-call
merge clobbers the prior call's local file), but the answer is not
"patch those two paths" — the answer is to articulate the
properties the layer should have, then audit every persistency
edge against them.

## Five invariants

### I1. Identity — every persisted artifact has a unique address

**Vocabulary.** The YAML's `interventions:` field holds
`InterventionConfig` records (one named contrast each:
HPs in `base`, multi-arm slot deltas in `arms`). Each entry
dispatches to a `DoEffect(arms=cfg.arms)` via `run_intervention`.
These are NOT framework `Hypothesis` Protocol-conformers — the
Protocol carries `INTERVENTION + BRIDGES + __name__` and is
verdict-time; `InterventionConfig` is sweep-time only and decomposes
into a Protocol-conformer + a `base` callable at dispatch.

**The collision.** A persisted artifact's location is uniquely
determined by the provenance tuple

```
(sweep_name, intervention_name, env_name, arm_key, chunk_id)
```

If two distinct cells map to the same path on S3 *or* on disk,
the path scheme is broken — not the upload code. Every dimension
of variation in `dispatch_sweep` MUST appear in the relpath.

The current `tmp/cell{NNN}__{env}__{arm_tag}` includes env and
arm but **omits intervention name**. dispatch_sweep already
creates a per-intervention local directory: `h_out_dir =
sweep.out_dir / cfg.name`. The local layout is namespaced. The
remote upload, however, is built relative to `out_dir` and
prepended with the SHARED `archive_remote` — stripping the
`cfg.name` prefix the local path included. Two interventions
producing identical rp_rels
(`tmp/cell001__Freeway-MinAtar__baseline__runs.parquet`) upload
to the same S3 object — last writer wins.

**The rule.** Local-vs-remote composition must be symmetric:

```python
# In dispatch_sweep — pass an intervention-namespaced archive_remote.
archive_remote = f'{sweep.archive_remote}/{cfg.name}'   # mirrors h_out_dir
```

That alone closes the collision. The intervention name comes from
the YAML — there's no other source of truth — and the framework
need not reach for `claim_graph_signature` or invent a structural-
signature concept. If two YAML interventions share a name, that's
an authoring bug detectable at YAML-load time
(`assert len({cfg.name for cfg in sweep.interventions}) == len(sweep.interventions)`),
not a persistency-layer concern.

**Optional hardening (CellAddress).** If we want a typed surface
that makes collisions impossible at compile time AND admits
future per-cell dimensions (`n_step`, `gamma`, etc. that may vary
within a single intervention), introduce a discriminator-typed
address. dispatch_sweep populates the discriminator from whatever
varies per cell within an intervention; the address renders to a
deterministic relpath:

```python
@dataclass(frozen=True, slots=True)
class CellAddress:
    intervention_name: str        # YAML name, e.g. 'ddqn_sync1k'
    arm_key: str                # treatment / baseline
    chunk_id: int
    discriminator: tuple[tuple[str, str], ...] = ()
    # Sorted (key, value) pairs for every per-cell-varying field
    # within the intervention (env, wrappers, n_step if swept, …).

    @property
    def relpath(self) -> str:
        disc = '__'.join(f'{k}={v}' for k, v in self.discriminator)
        suffix = f'__{disc}' if disc else ''
        return (
            f'{self.intervention_name}/tmp/'
            f'cell{self.chunk_id:03d}__{self.arm_key}{suffix}'
        )
```

`dispatch_sweep` builds the `CellAddress` set up front; if two
addresses produce the same relpath, the sweep refuses to start.
The CURRENT sync_curve_resume case has
`intervention_name='ddqn_sync1k'`, `arm_key='baseline'`,
`discriminator=(('env', 'Freeway-MinAtar'),)`. A future sweep
adding n_step variation within one intervention would extend
to `discriminator=(..., ('n_step', '3'))` — no schema change.

The minimal one-line fix (passing a namespaced `archive_remote`)
ships invariant I1 today; `CellAddress` is the typed
follow-up if future dimensions push the discriminator complexity
beyond what `cfg.name + tag` strings can express.

### I2. Idempotency — re-running a sweep cannot lose data

The persistency layer is append-only. Re-running a cell that has
already produced an artifact MUST:

1. Detect the prior artifact via manifest sha256.
2. Skip the upload if content matches (current behaviour, fine).
3. **Refuse + report** if content differs at a path the current
   manifest doesn't know about.

The current `archive()` only checks the LOCAL manifest's prior
entry. Two manifests in different intervention dirs claiming the
same S3 path is invisible to this check — both happily upload
because each manifest's `prior` is `None` for that path. This
is how the sync_curve_resume corpus lost data.

**The rule.** Before any upload, perform a remote head-object
check (or cross-manifest scan). If the object exists with a
different sha256 than this cell's, raise — the user must
explicitly opt into overwrite. The upload codepath should be:

```python
def archive_cell(addr: CellAddress, local: Path, ...) -> None:
    remote_uri = address_to_uri(addr)
    existing = head_object(remote_uri)  # None if absent
    if existing is not None:
        if existing.sha256 == sha256_of(local):
            return  # idempotent skip
        raise ConflictingArchive(remote_uri, existing, ...)
    upload(local, remote_uri)
```

The cost is one HEAD request per archive — cheap on S3, free in
the local-only path.

### I3. Cache invariance — manifest is the source of truth

A merged corpus parquet (`<corpus>/runs.parquet`) MUST be a pure
function of the manifest:

```
merge: Manifest → Path
re-merge from manifest at any time → byte-identical output
```

The current per-call merge in `run_intervention` reads from
`archived_runs_uris`, a list populated only with this call's
iteration cells. When the same `out_dir` is targeted by multiple
calls (paired sweeps with shared intervention names), each merge
clobbers the prior one with a SUBSET of the corpus's cells.

**The rule.** Merge takes the manifest as input, period. No
call-local state. If the manifest accumulates entries across
multiple `run_intervention` calls (which it should, given a
shared `out_dir`), the merge picks them all up:

```python
def merge_corpus(corpus_dir: Path) -> Path:
    manifest = load_manifest(corpus_dir)
    runs_uris = [
        archived_uri(manifest.remote_root, f.relpath)
        for f in manifest.files
        if f.relpath.startswith('tmp/') and f.relpath.endswith('runs.parquet')
    ]
    return stream_concat_parquets(runs_uris, corpus_dir / 'runs.parquet')
```

Re-merging from manifest becomes idempotent and consumer-safe.
The standalone re-merge script written for the sync_curve_resume
incident becomes the same code path as the in-loop merge.

### I4. Atomicity — partial writes are detectable

A sweep that crashes mid-merge MUST leave the corpus in either
its pre-merge or post-merge state, never a half-written
`runs.parquet`. The manifest itself uses tmp+rename (already
atomic, line 162-164 in `cloud.py`), but the merged parquets do
not — `stream_concat_parquets` writes directly to the destination.

**The rule.** Merge writes go through tmp+rename: write to
`<out>.partial`, then `os.rename` to `<out>`. A partial write on
crash leaves no `.partial` file at the consumer's path; consumers
never see torn parquets.

This is a one-line change inside `stream_concat_parquets`. The
cost is zero — the rename is filesystem-atomic on POSIX.

### I5. Provenance — every row traces to its shard

A row in the merged `runs.parquet` should carry enough metadata
to identify its origin shard. Currently `RunRow.id` (UUID per
cell run) is sufficient if the manifest records cell→shard
mapping — but the manifest doesn't materialise that link.

**The rule.** The manifest entry for each tmp shard records the
list of `RunRow.id`s it contains, so a downstream investigator
can trace `id → shard → cell address → sweep YAML`. Cost: a
small manifest field (~60 bytes per cell × ~hundreds of cells per
sweep = ~50 KB total).

This invariant pays off when debugging: "where did this anomalous
row come from?" becomes a manifest lookup, not a trace through
sweep logs.

## Audit table — current state vs. target

| invariant | held by current code? | gap |
|---|---|---|
| **I1 Identity** | partial — env+arm in path, intervention namespace missing on remote | mirror `out_dir` composition: `archive_remote=f'{sweep.archive_remote}/{cfg.name}'` |
| **I2 Idempotency** | per-dir only | add cross-manifest / head-object check on upload |
| **I3 Cache invariance** | broken — merge from call-local state | switch merge to manifest-driven |
| **I4 Atomicity** | broken for parquet writes | tmp+rename in `stream_concat_parquets` |
| **I5 Provenance** | partial — UUID exists, manifest link missing | record per-shard `id`-list in manifest |

## Implementation order

The audit table also implies a dependency order:

1. **I3 first** (manifest-driven merge). Smallest blast radius
   — it's a pure refactor of the merge step. Once merge is a
   manifest function, the re-merge script we needed for the
   sync_curve_resume incident becomes redundant — `dispatch_sweep`
   can call it post-hoc.
2. **I4 next** (tmp+rename). One-line change. Defends future
   sweeps against crash mid-merge regardless of the I3 fix.
3. **I1 third** (intervention namespace mirror on remote). Forward-
   compatible: existing corpora keep their flat layout; new
   sweeps get nested. Migration script copies old paths to new
   on S3 (one-time, optional).
4. **I2 fourth** (head-object check on upload). Requires I1 to
   be deployed first — without I1 the head-object check would
   raise on legitimate same-path-different-content uploads
   (i.e., the existing buggy behaviour). With I1 holding, a
   head-object hit IS a real conflict.
5. **I5 last** (provenance manifest field). Schema bump on the
   manifest; backward-compatible by treating the new field as
   optional.

Each step is independently testable: a regression test of the
form "two `run_intervention` calls targeting the same out_dir
with the same archive_remote produce a final `runs.parquet`
containing all cells from both calls" exercises I3+I4 together
and would catch the present bug. A test that "two hypotheses
sharing archive_remote upload to non-colliding S3 paths"
exercises I1.

## What this doc is not

It is not a roadmap with deadlines. It is the **design contract**
the persistency layer must satisfy when any of those edges is
touched. Anyone editing `cloud.py`, `sweep.py`, or the merge
primitives in `persistence.py` should be able to point at the
invariant their change preserves.

If a future change can't preserve all five, the change needs a
written waiver in this doc. The bugs that surfaced in the
sync_curve_resume corpus existed because no such contract was
written down — the persistency edges were authored in isolation,
each looking right on its own, none of them composing safely.
