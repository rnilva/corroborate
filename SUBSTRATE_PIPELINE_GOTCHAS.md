# Substrate pipeline gotchas — open design debt

**Scope**: substrate-side data pipeline (sweep → archive → ingest → cache → analysis), NOT the framework's claim/bridge/verdict layer (which has its typed-invariant discipline locked in).

**Status**: design debt log. Not a fix-now task. Hand off to a dedicated worktree.

## Why this doc exists

Each interaction with the substrate pipeline surfaces "gotchas" — corner-case behaviors where the local behavior is reasonable but compositions trip the user. CLAUDE.md has accumulated paragraph-warnings for each one. A paragraph in a contributor doc is much weaker than a typed primitive that makes the invariant unforgeable. This doc collects observed instances + proposes structural fixes.

## Observed gotchas (2026-05-26 session, LL 2M sweep + ingest)

### G1 — Discounted vs raw eval default

**Observed**: plotted "learning curve" for LL 2M; γ=0.99 showed plateau at +30 (looks like agent failed to learn). User questioned. Investigation: `mc_return` is *discounted* (`Σ γ^t r_t`); `eval_best_burst_mean` uses it directly. The raw episodic return (which actually answers "did the agent solve LL?") is reconstructed by a derived measurable `mc_return_raw_episodes` via the telescope identity:
```
raw[e] = mc_from_step[e, 0] + (1 − γ) · Σ_{t≥1} mc_from_step[e, t]
```
At γ=0.99, V's actual best raw is +245 (above LL's solved threshold of +200) — but eyes go to the discounted +30 because that's the column without the `_raw` suffix.

**Files**: `src/corroborate_rl/corroborate_rl/dqn/measurables.py:4440-4524` (raw-reconstruction logic, four `*_raw_mean` siblings); `src/corroborate_rl/corroborate_rl/dqn/eval.py:52` (`mc_return: jax.Array  # () — Σ γ^t r_t over the episode`).

**Why it's a gotcha**: the substrate has both metrics; the naming convention puts discounted as the default and raw as a `_raw_mean` sibling. For Q-function-coherent analysis the discounted metric IS the right answer (you're estimating discounted return). For "is the agent learning" the raw metric is what humans intuit. The current default is correct for the former and silently wrong for the latter.

**Proposed fix**: rename the discounted metrics to `eval_*_disc_mean` and have `eval_*_mean` autocomplete to the raw form. OR require explicit `discounted=True/False` argument when constructing the measurable. Substrate-wide rename; backwards-incompat with existing bridges referring to the old names.

### G2 — Sweep-time vs ingest-time derived-measurable computation

**Observed**: after the LL 2M sweep finished, inspected `runs.parquet` directly:
```
eval_best_burst_raw_mean: NaN  (V and D, both γ slices)
eval_final_raw_mean: NaN
eval_full_auc_raw_mean: NaN
eval_late_burst_raw_mean: NaN
```
All four are NaN because the @derived_measurable injection (`mc_return_raw_episodes`) only fires at INGEST time, not at sweep cell-run time. After running `corroborate hypothesis ... --ingest <path>`, the cache parquet has finite values for `eval_best_burst_raw_mean` but the runs.parquet still shows NaN.

**Worse**: even after ingest, `eval_final_raw_mean` STAYS NaN at LL — the injection populates some siblings (`best`) but not others (`final`). Source data (`mc_return_from_step`) is present in traces. The failure is silent.

**Files**: `src/corroborate_rl/corroborate_rl/dqn/measurables.py:4450,4490,4513,4549` (the four raw siblings). The injection point is `corroborate/...` ingest pipeline — need to trace.

**Why it's a gotcha**: a user who sees `eval_*_raw_mean` columns in `runs.parquet` with NaN values has no obvious indicator that "this gets populated at ingest." They either:
(a) Think the measurable is broken
(b) Don't notice the NaN and use the discounted sibling instead

And the silent partial-success at ingest (`best` populated, `final` not) is the failure mode that hides longest.

**Proposed fix**: (1) Either compute derived measurables at sweep-time too (slow if it requires loading mc_return_from_step into the sweep's record) OR explicitly stamp them as "pending ingest" with a sentinel value distinct from NaN. (2) The ingest pipeline should FAIL LOUDLY when a derived-measurable injection produces NaN for a column whose source data is finite. Currently it silently writes NaN.

### G3 — `--ingest` dedupes on parent dir

**Observed**:
```bash
corroborate hypothesis experiments.findings.hasselt_clean \
    --ingest "$PWD/experiments/data/lunarlander_2M_30seeds_cpu/g099" \
    --ingest "$PWD/experiments/data/lunarlander_2M_30seeds_cpu/g0999"
```
Result: only `g0999` was ingested. Runner log: `"ingesting 1 corpora from /workspace/corroborate/experiments/data/lunarlander_2M_30seeds_cpu"`. Both --ingest targets resolved to the same parent; the runner walked once, found the first sub-corpus, ingested it, stopped.

Workaround: run two separate `corroborate hypothesis` invocations, one per sub-corpus. The second ingest brought in `g099` correctly. Cache now has all 4 LL strata.

**Files**: `corroborate/cli/hypothesis.py:dispatch` + `corroborate/cell_runner.py` (or wherever `--ingest` is resolved to corpora walks).

**Why it's a gotcha**: the CLI silently does the wrong thing. No warning that "you passed two paths and only one was ingested." No discoverable error path.

**Proposed fix**: (1) error on duplicate parent paths; OR (2) when --ingest targets are sub-corpora of a common parent, walk all sub-corpora rather than deduplicating. The first is safer.

### G4 — Mid-merge disk-exhaustion leaves half-state

**Observed**: LL 2M sweep ran 6 cells × 2 γ slices to completion. Top-level merge:
- ✓ `runs.parquet` merged (48KB)
- ✗ `traces.parquet` merge SKIPPED — disk-pressure detection
- `.in_progress` sentinel left UP
- Per-arm sub-corpora intact (g099/, g0999/, each ~9.5GB traces)

Warning emitted: `"WARNING — top-level traces.parquet merge skipped (insufficient disk in experiments/data). Per-intervention sub-corpora ... are intact and usable directly for analysis / ingest. To finish the top-level merge later: archive sub-corpora, free disk, then concat their traces.parquet via stream_concat_parquets. The sweep .in_progress sentinel stays UP — --ingest-all will skip the parent dir until merged or removed."`

The recovery path documented in the warning ("archive, free disk, then stream_concat") doesn't work when the reason merge failed in the first place is "no disk to fit the merged file" — you'd need to restore the just-archived files locally before you can concat, which puts you back in the same disk state.

**Files**: `corroborate/sweep/runner.py` (or wherever the merge happens); `corroborate/cli/sweep.py`.

**Why it's a gotcha**: the substrate offers a half-built state ("parent has runs but no traces") with NO type-system enforcement that this state is invalid. `--ingest-all` walks SKIP this corpus (sentinel-aware) but a user who copies the path manually can ingest it — and the framework happily processes the parent's runs.parquet without realizing traces are at the sub-corpus level.

The framework's policy is implicit: "sub-corpora are the canonical ingest unit when sentinel is up." The sentinel is the ONLY indicator.

**Proposed fix**: 
1. Make "complete sweep" a typed property — a sweep's parent dir EITHER has both runs+traces and no sentinel, OR has no runs.parquet at parent + sub-corpora are independent corpora + sentinel removed.
2. The mid-merge state ("parent runs.parquet exists, no parent traces.parquet, sentinel up") should be unreachable by construction — the runner either completes the merge or atomically reverts the parent runs.parquet write.
3. Or: explicit `repair_sweep(corpus, policy={'commit_to_sub_corpora', 'finish_merge'})` recovery primitive with typed policy enum, instead of "remove the sentinel manually."

### G5 — Default plot used discounted MC silently

**Observed**: my initial learning curve plot read `mc_return` directly from traces (per-episode discounted return), averaged over 5 episodes per burst. User saw plateau at +30 and asked why. I had to be told.

**Why it's a gotcha**: this is a SUBSET of G1 — the same naming convention bites the plotting code too. The substrate exposes the discounted form via the most-natural-named column; consumers reach for it.

**Proposed fix**: subsumed by G1.

## The meta-pattern

The substrate has accumulated invariants over time:
- `.in_progress` sentinels for crash-recovery.
- Eviction for storage cost (trace files get archived + deleted locally after ingest).
- Parent/leaf stamping for leaf-name collision avoidance.
- @derived_measurable for adding new metrics to old corpora.
- Discounted MC as the canonical Q-target (correct for theorem-checking).
- Scope filters with column-existence checks (correct for forward-compat).
- Cloud manifests + `_remote.json` for distributed durability.
- Trace eviction (CI7), in-progress refusal (CI1), cell-id contamination guard (CI8).

**Each invariant is correct in isolation.** Each was introduced to fix a specific failure observed in practice.

**They don't compose ergonomically.** The COMPOSITIONS surface as gotchas: eviction × merge-failure (G4); derived-measurable × sweep-vs-ingest timing (G2); parent-resolution × multi-sub-corpus ingest (G3); discounted-MC × default-naming-convention (G1, G5).

**The framework has typed-primitive discipline; the substrate has runtime-convention discipline.** The framework refuses to construct an ill-typed Verdict, Bridge, or PartialSpearmanResult. The substrate happily produces a sweep state where parent.runs.parquet exists, parent.traces.parquet doesn't, and an `.in_progress` sentinel is the only indicator that this is mid-stream rather than final.

The honest statement: the substrate's failure modes that COULD be `TypeIs[CompleteCorpus]` / frozen-dataclass-with-typed-state / "you cannot construct a half-merged state by construction" are instead documentation conventions enforced at runtime via warnings + manual recovery.

## Proposed structural fixes (ranked by value)

### F1 — Typed "complete sweep" + recovery primitive (high value, scoped work)

Replace `.in_progress` sentinel with a typed `SweepState` frozen dataclass:

```python
@dataclass(frozen=True, slots=True)
class SweepState:
    parent_path: Path
    has_parent_runs: bool
    has_parent_traces: bool
    sub_corpora: tuple[SubCorpus, ...]
    
    @property
    def status(self) -> SweepStatus:  # enum
        # COMPLETE_TOP_LEVEL, COMPLETE_SUB_CORPORA_ONLY, 
        # MID_MERGE_HALF_STATE, ABANDONED
        ...

def repair_sweep(state: SweepState, policy: RepairPolicy) -> SweepState:
    """Typed recovery. RepairPolicy enum: 
       COMMIT_TO_SUB_CORPORA | FINISH_MERGE | ROLL_BACK_TO_FRESH"""
```

Eliminates G4 entirely. The mid-merge half-state becomes representable but unreachable post-`repair_sweep`.

### F2 — Substrate-wide rename: `_disc` suffix, raw as default (high value, breaking)

Rename all discounted measurables from `eval_*_mean` to `eval_*_disc_mean`. The bare `eval_*_mean` becomes raw. Forces every consumer to pick explicitly. Breaks every existing bridge — must be batched with bridge rewrite + cache invalidation. High-value if done; high-effort.

### F3 — Loud failure on derived-measurable NaN injection (medium value, contained scope)

The ingest pipeline currently writes NaN when a @derived_measurable injection fails mid-batch. Change to: track per-cell injection success, fail the whole ingest if any cell's derived measurable produces NaN when source data is finite. Distinguish "source data unavailable" (silent NaN OK) from "source available, injection failed" (loud error).

Eliminates the silent G2 partial-success.

### F4 — --ingest collision detection (low value, easy)

Error on duplicate parent paths. One-line change to the CLI.

### F5 — Derived-measurable computation at sweep-time (low value, perf-risky)

Move @derived_measurable execution into the cell-runner so `runs.parquet` is finite-valued the moment the sweep finishes. Risk: slows cell execution; some derived measurables depend on across-cell aggregates (won't work).

Probably NOT worth doing; F3 + better docs about "derived measurables only finite after ingest" is the cheaper fix.

## What I would do if I had a worktree

A worktree-scoped pass on the substrate pipeline:

1. **Audit invariants**. Read every place in the substrate that writes `.in_progress`, every place that checks for it, every place that handles partial-merge state. Build a state diagram.
2. **Implement F1**. The typed SweepState + RepairPolicy primitive. Migrate all sentinel-based logic to use it. Test on the LL 2M corpus as the recovery test case (it's currently in COMPLETE_SUB_CORPORA_ONLY state; framework should recognize that).
3. **Implement F3**. Loud failure on derived-measurable NaN. Test on LL: `eval_final_raw_mean` should error rather than silently NaN.
4. **Defer F2 + F4 + F5**. F2 is too breaking for an in-flight study; F4 is trivial and can wait; F5 is probably not worth doing.

## Worktree handoff context

Pick this up in a fresh worktree (`git worktree add ../substrate-pipeline main`). Critical not-to-break:
- Existing in-flight sweeps (any with `.in_progress` sentinel) — F1 migration must handle them in a backwards-compat way (read sentinel state, present as SweepState, no behavior change).
- The cache (`experiments/data/cache/*.parquet`) — F3 might force re-ingest if it detects historical silent-NaN cells. Test on a non-canonical cache first.
- The DDQN study results (`experiments/findings/ddqn/*`, `experiments/findings/hasselt_clean/*`) — F2 would break every bridge that names `eval_*_mean`. Plan: deferred until after publication.

The LL 2M corpus at `experiments/data/lunarlander_2M_30seeds_cpu/` is a useful test case for F1: it's currently in `COMPLETE_SUB_CORPORA_ONLY` state (parent.runs.parquet exists, no parent.traces.parquet, no sentinel as of this session's cleanup, sub-corpora's `_remote.json` cloud archives present, traces evicted locally).

## Memory cross-references

- [[findings-canonical-asterix-g0999-bit-deterministic-replicates]] — touches substrate determinism stamping (`cmdbuf provenance reader is misleading`, Task #144). Similar shape to F3: typed-property reader silently disagrees with runtime state.
- CLAUDE.md "Cache + cloud operator discipline" — the warnings paragraph that this doc proposes replacing with typed primitives.
- CLAUDE.md "Sweep + trace discipline" — same.

## Open questions for the worktree pass

- Is the `_remote.json` + `MANIFEST.json` pair the right abstraction or should it be a single typed `CloudArchive` primitive?
- Should `corpus_stamp` (`sub.name` vs `parent.name/sub.name`) be a property of the SweepState rather than computed each time?
- Should there be a `SubstrateInvariantViolation` exception type that the runner raises at the boundary, instead of warnings + manual recovery? (Currently warnings are easy to miss in long logs.)
