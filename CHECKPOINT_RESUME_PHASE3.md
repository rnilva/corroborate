# Checkpoint resume — Phase 3 design

Design document for landing the remaining seven resumable fields of
`DQNState` on top of `InitOverride` (Phase 1+2, commits `ff0cc6e`
+ `a4a9ebd`). **No implementation in this document** — decisions
and rationale only. See `CHECKPOINT_RESUME_DESIGN.md` for the
Phase 1+2 architectural backdrop.

## 1. What Phase 1+2 landed

- `InitOverride` frozen-dataclass with two fields:
  `online_params: Params | None` + `target_params: Params | None`.
  Registered as JAX pytree; flows through `jax.vmap` via
  pytree-structured `in_axes`.
- `init_q_checkpoint_path_template` + `init_q_checkpoint_load_target`
  YAML flags on `DQNSweep`.
- `QCheckpoint` msgpack schema: `{online_params, target_params,
  burst, global_step}`. Loader is `load_batched_init_override(
  template, seeds, *, load_target)`.
- `init_state` resolves online from override-or-fresh; target from
  override-or-mirror-online.

The remaining nine `DQNState` fields (`state.py:30-76`) stay
fresh-init regardless of `init_override`:

| Field | Today's init |
|---|---|
| `opt_state: OptState` | `optimizer.init(online)` — Adam moments zero |
| `replay: ReplayState` | `Replay.init(obs_shape)` — empty buffer |
| `pending_n_step: PendingNStepState` | `init_pending_n_step(obs_shape)` — empty window |
| `env_state: EnvState` | `env.reset(env_key)` — first observation |
| `obs: jax.Array` | (same — paired with env_state) |
| `step: jax.Array` | `jnp.int32(0)` — fresh step counter |
| `rng_key: jax.Array` | (split from input rng_key) |
| `ep_return: jax.Array` | `jnp.float32(0.0)` — fresh episode return |
| `state_hash_count: jax.Array` | `jnp.zeros((K,))` — fresh visit counts |

The running `asterix_g0999_init_v_burst25_continue_ddqn` sweep
operates entirely on Phase 1+2 — DDQN's clip operator applied to
V's parameters with everything else cold-start. That answers
"steady-state operator from V's parameters" cleanly. Phase 3 adds
the dial for "V's full training trajectory" — bit-exact
continuation from V's burst-25 state.

## 2. Pressure: which fields actually matter?

Not all nine are equally load-bearing for continuation semantics.
Per-field analysis:

### High value, low cost

- **`opt_state`** — Adam's `(mu, nu, count)` triple plus the warmup
  wrapper's count. Same pytree shape as `online_params` (~32k
  float32 leaves for the canonical Asterix CNN). Fresh-init means
  the first ~100 gradient steps post-resume run with `m=v=0` →
  Adam's full step size (no momentum damping). That's a substantive
  destabiliser, NOT a wash-out — V's 500k-step trajectory built up
  specific moment estimates that get discarded. msgpack growth: ~2×
  the current ~128KB per seed (negligible). **Highest
  value-per-cost ratio of the nine.**

- **`step`** — `() int32`. Trivial size. But: `linear_epsilon`
  reads `state.step`; loading `step=500_000` with `anneal_steps=
  100_000` puts ε at `eps_final`, while fresh `step=0` re-anneals
  from `eps_init`. `total_steps` countdown also reads `step`. The
  *experimental semantic* of "continue training" vs "extend
  training budget" is determined by whether step is loaded. Phase 3
  forces authors to be explicit.

- **`rng_key`** — `uint32[2]`. Trivial size. Bit-exact replay of
  V's RNG choices for the post-resume continuation. Most
  experiments don't need this (random ε-greedy branches at
  ε=0.05 fire <5% of the time; replay-batch sampling is high
  entropy regardless). Phase 3 supports it as an opt-in but doesn't
  promote it.

### Mid value, low cost

- **`pending_n_step`** — six small arrays (a partial-window
  aggregate). At `n_step=1` this is a no-op (window emits every
  step). At `n_step=3` (LL γ=0.999 canonical) and higher, a
  burst-boundary mid-window means losing up to `n_step-1` partial
  transitions. Fresh-init keeps the buffer correct but loses the
  partial aggregate. Trivial msgpack size.

- **`state_hash_count`** — `(K,) int32` where K is the env-specific
  hash cardinality (typically <10k). Only matters when
  `count_weight_alpha > 0` (the count-weighted-loss intervention).
  Otherwise the field is unused. Phase 3 includes it for
  completeness but it's the lowest-priority field.

### Mid value, mid cost

- **`env_state` + `obs` + `ep_return`** — these three are an
  *atomic unit*: gymnax's `env.reset(key)` returns
  `(obs, env_state)` paired; `ep_return` is the running per-episode
  sum that resets on `done`. Loading any one without the others is
  inconsistent (V's `obs` doesn't match a freshly-reset
  `env_state`). gymnax env_state pytree size varies — Asterix is
  ~10 fields × ~50 bytes per seed; bigger envs are <1KB per seed.
  Bundle them as a single `EnvResumeBundle` frozen-dataclass so
  authors can't opt half in.

### High value, HIGH cost

- **`replay: ReplayState`** — six parallel arrays of `(capacity,
  ...)` shape. At canonical Asterix (capacity=100k, obs=(10,10,4)):
  - `obs`: float32 × 100k × 400 = 160MB
  - `next_obs`: 160MB
  - `action`: int32 × 100k = 400KB
  - `reward`: float32 × 100k = 400KB
  - `done`: float32 × 100k = 400KB
  - `size`: 4 bytes
  - Total: **~321MB per seed**.

  Per-burst per-seed sidecar files at this size are
  archive-killing: a 40-burst × 15-seed corpus would carry
  `40 × 15 × 321MB ≈ 192GB` of replay data. The Phase 1+2 msgpack
  files at ~128KB are 6000× cheaper.

  Semantic value: V's burst-25 replay contains V's most-recent 100k
  transitions (the FIFO window over V's training experience).
  Continuing DDQN on V's transitions means DDQN's clip operator
  re-bootstraps V's collected data for the first ~capacity post-
  resume steps before the buffer turns over. That's the "DDQN
  applied to V's training distribution" semantic. Without it
  (current sweep), DDQN must accumulate fresh transitions before
  any gradient updates fire — a brief pure-rollout phase followed
  by DDQN-collected-and-trained.

  **Conclusion**: replay needs its own storage tier. Inline-msgpack
  is wrong by 3+ orders of magnitude.

### Substrate-forward-compat (not currently in `DQNState`)

- **Batch-norm / EMA statistics** — distributional sub-state on
  layers that don't have learnable parameters. Not in this
  substrate today (CNN + MLP without batchnorm). Forward-compat
  obligation: `InitOverride`'s field-by-field shape makes adding a
  new field one-line (dataclass field + init_state branch). No
  Phase 3 effort required for this — the architecture is already
  there.

## 3. Storage layout: two tiers

A single msgpack-per-checkpoint can't hold replay without exploding
the corpus footprint. Three options considered:

### 3a. All inline in msgpack — rejected

One msgpack per (cell, seed, burst) carrying every field including
replay. Average size: ~321MB per file. Per-burst frequency × 40
bursts × 15 seeds = ~192GB per sweep. Cloud archive cost alone
makes this unviable. Local disk is uncomfortable on dev machines.

### 3b. Two-tier (msgpack + sidecar parquet) — recommended

- **msgpack tier** (everything except replay): `q_checkpoints/
  cellNNN_S_burstBB.msgpack`. Carries the seven small/medium fields
  (~256KB per seed including Adam moments). Schema mirrors
  `InitOverride` 1:1.
- **Replay tier**: `q_checkpoints/cellNNN_S_burstBB_replay.parquet`.
  Six columns matching `ReplayState`: `obs`, `next_obs` as
  list-of-float (one row per transition slot); `action`, `reward`,
  `done` as scalars; `size` as a metadata scalar (could be a
  single-row separate column or part of parquet metadata).
  100k-row parquet × 6 columns. Zstd compression on the
  obs/next_obs columns drops ~3× (sparse one-hot encodings
  compress hard; dense float32 arrays modestly).
- **Lazy load**: replay parquet is only read when
  `load.replay=True`. Other Phase 3 fields are read every load
  (cheap regardless).

This matches the substrate's existing two-tier persistence
discipline: `runs.parquet` (small, lineage-keyed) + `traces.parquet`
(heavy, lazy-evicted). The CI7 trace-eviction machinery transfers
1:1 — replay parquets can be cloud-evicted once a corpus is done
training and rehydrated only for downstream continuation
experiments.

### 3c. All separate files per field — rejected

Maximum granularity (one file per field per cell per burst per
seed) means 9× the inode count. Compounds with the existing
trace-column file proliferation. Filesystem overhead and
manifest-bookkeeping cost outweigh the locality win.

**Decision: tier 3b.** msgpack stays the unit of "everything
small"; replay is its own tier.

## 4. Schema discoverability: presence-of-field, no version flag

Three options for handling Phase 1+2 vs Phase 3 msgpack differences:

1. **Version field** (`{"_version": 3, ...}`) — explicit. Loader
   reads version first, dispatches to per-version parser. Couples
   version progression to schema changes; new field requires
   version bump.

2. **Presence-of-field check** — loader probes
   `if 'opt_state' in raw_dict`. Missing key → None on
   `QCheckpoint`. No version book-keeping. New field is one line in
   the writer + one line in the loader.

3. **Sidecar manifest** — `q_checkpoints/MANIFEST.json` declares
   schema per file. Centralised but additional state to keep in
   sync.

**Decision: option 2.** Each field's optionality is local to its
own load path; missing-field semantics are `None on the dataclass`
in every case. `QCheckpoint` becomes:

```
@dataclass(frozen=True, slots=True)
class QCheckpoint:
    online_params: Params           # always present (required)
    target_params: Params | None    # Phase 2+
    burst: int                      # always present
    global_step: int                # always present
    opt_state: OptState | None      # Phase 3a+
    step: jax.Array | None          # Phase 3b+ (note: redundant w/ global_step at int vs Array)
    rng_key: jax.Array | None       # Phase 3b+
    pending_n_step: PendingNStepState | None   # Phase 3c+
    state_hash_count: jax.Array | None         # Phase 3c+
    env_resume: EnvResumeBundle | None         # Phase 3d+
    replay_path: Path | None        # Phase 3e+ (path to sidecar parquet; not the data itself)
```

`step` and `global_step` look redundant. They're not: `global_step`
is the int written for sweep-state bookkeeping (which burst was
this saved at); `step` is the jax.Array threaded through scan
(reloaded into `state.step` if `load.step=True`). Phase 3b can
deduplicate them — `state.step = jnp.int32(global_step)` is the
load path, no separate `step` field needed in msgpack.

## 5. Authoring surface: YAML opt-in via typed-flags dataclass

Phase 2 added `init_q_checkpoint_load_target: bool = False`. Phase
3 adds seven more independently-opt-inable behaviours. Three
options:

### 5a. Seven independent bool fields — rejected

```yaml
init_q_checkpoint_load_target: true
init_q_checkpoint_load_opt_state: true
init_q_checkpoint_load_step: false
init_q_checkpoint_load_rng_key: false
init_q_checkpoint_load_pending_n_step: false
init_q_checkpoint_load_state_hash_count: false
init_q_checkpoint_load_env: false
init_q_checkpoint_load_replay: false
```

Linearly scaling YAML schema. No grouping; each field is its own
top-level key. Authors read seven lines to know what's resumed.

### 5b. Enum recipe — rejected

```yaml
init_q_checkpoint_resume_level: WARM_OPT  # PARAMS_ONLY | PARAMS_AND_TARGET | WARM_OPT | BIT_EXACT
```

Constrains to pre-defined recipes. New experimental combination
needs a new enum value (e.g., "params + replay but not opt_state"
isn't expressible). Forces the framework to taxonomise
combinations that don't have stable names.

### 5c. Typed-flags frozen-dataclass — recommended

Mirror `InitOverride`'s field-by-field shape one level up:

```python
@dataclass(frozen=True, slots=True)
class QCheckpointLoadFlags:
    target_params: bool = False
    opt_state: bool = False
    step: bool = False
    rng_key: bool = False
    pending_n_step: bool = False
    state_hash_count: bool = False
    env: bool = False           # gates env_state + obs + ep_return as a unit
    replay: bool = False

@dataclass(frozen=True, slots=True)
class QCheckpointResume:
    path_template: str
    load: QCheckpointLoadFlags = field(default_factory=QCheckpointLoadFlags)
```

YAML shape:

```yaml
init_q_checkpoint:
  path_template: experiments/.../cell000_{seed}_burst25.msgpack
  load:
    target_params: true
    opt_state: true
    step: false
    rng_key: false
    replay: false
    env: false
```

`DQNSweep.init_q_checkpoint: QCheckpointResume | None = None`
replaces the current pair of flat fields. Authors who write
`load:` skip silently get full defaults (params-only). Pyright
sees field types; no `getattr` semantics; no stringly-typed field
lookup. **Decision: 5c.**

`online_params` is NOT in the flags — if a path_template is
configured, online_params is always loaded (the kwarg's existence
*is* the "load online params" decision). `online_params: bool` on
the flags would be redundant noise.

## 6. `init_state` body: uniform per-field branch

Each field reads:

```
opt_state = (
    override.opt_state
    if override is not None and override.opt_state is not None
    else optimizer.init(online)
)
replay_state = (
    override.replay
    if override is not None and override.replay is not None
    else replay.init(obs_shape)
)
pending = (
    override.pending_n_step
    if override is not None and override.pending_n_step is not None
    else init_pending_n_step(obs_shape)
)
step = (
    override.step
    if override is not None and override.step is not None
    else jnp.int32(0)
)
state_hash_count = (
    override.state_hash_count
    if override is not None and override.state_hash_count is not None
    else jnp.zeros((state_hash_cardinality,), dtype=jnp.int32)
)
if override is not None and override.env_resume is not None:
    obs = override.env_resume.obs
    env_state = override.env_resume.env_state
    ep_return = override.env_resume.ep_return
else:
    obs, env_state = env.reset(env_key, env_params)
    ep_return = jnp.float32(0.0)
final_rng = (
    override.rng_key
    if override is not None and override.rng_key is not None
    else run_key
)
```

Empty-dict guards (Phase 2 pattern at `dqn.py:127-146`) apply to
each pytree-typed field that admits a meaningful "empty" shape
(`opt_state` truthy check passes for any populated optax state;
`replay` truthy check on `state.size > 0` is more honest than
`is not None`).

The pattern is uniform: one block per field, no cross-field
interactions except the env-bundle three. `EnvResumeBundle`
collapses the three-field atomicity into one decision point.

## 7. vmap mechanics: stack-then-pytree-vmap

`opt_state` and `replay` are nested pytrees. JAX vmap handles them
natively when the per-seed instances are stacked along axis 0:

```python
def _stack_pytrees[T](trees: Sequence[T]) -> T:
    return jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *trees)

# In load_batched_init_override:
opt_states_per_seed = [load_full(p).opt_state for p in per_seed_paths]
batched_opt_state = _stack_pytrees(opt_states_per_seed)
```

vmap call gains nested-pytree `in_axes`:

```python
jax.vmap(
    by_seed_with_override,
    in_axes=(0, InitOverride(
        online_params=0, target_params=0, opt_state=0,
        replay=0, pending_n_step=0, state_hash_count=0,
        env_resume=EnvResumeBundle(env_state=0, obs=0, ep_return=0),
        step=0, rng_key=0,
    )),
)(seeds_arr, init_override_batched)
```

JAX walks the pytree structure of both args and in_axes spec in
lockstep. Every leaf gets its axis annotation; None-fields are
broadcast. `InitOverride` and `EnvResumeBundle` must be pytree-
registered via `jax.tree_util.register_dataclass` (one line each).
`ReplayState` and `PendingNStepState` are NamedTuples (already
pytree-native, per `replay.py:55,75`). `OptState` is opaque to the
framework but optax's transformations register their own state
types.

## 8. Save side: `keep_q_checkpoint_full_state` opt-in

Today `keep_q_checkpoint_final` and `keep_q_checkpoint_per_burst`
gate online+target msgpack writes. Phase 3 adds:

```python
keep_q_checkpoint_full_state: bool = False
keep_q_checkpoint_replay: bool = False
```

- `False / False` (default) — Phase 1+2 size. ~128KB per file.
- `True / False` — Phase 3a-d fields included. Adam moments
  dominate; ~256KB per file.
- `True / True` — replay sidecar parquet alongside each msgpack.
  ~321MB per file. Opt-in only; cloud-archived per CI7.

Independent flags so authors can write the "warm opt_state but no
replay" combination (Phase 3a + 3b without 3e) common in
continuation experiments that don't need V's transition stream.

`dqn`'s @claim signature gains both as exogenous kwargs:

```python
keep_q_checkpoint_full_state: Annotated[bool, Exogenous] = False
keep_q_checkpoint_replay: Annotated[bool, Exogenous] = False
```

Substrate's existing `__q_checkpoint__<arm>__<role>__<param_key>`
sentinel-key convention extends to:

```
__q_checkpoint__online__final__<param_key>     # existing
__q_checkpoint__target__final__<param_key>     # existing
__q_checkpoint__opt_state__final__<leaf_path>  # new — flat path keys
__q_checkpoint__step__final                    # new — scalar
__q_checkpoint__rng_key__final                 # new — uint32[2]
__q_checkpoint__pending__final__<field>        # new — per-NamedTuple-field
__q_checkpoint__shc__final                     # new — (K,) int32
__q_checkpoint__env_state__final__<leaf_path>  # new — env-specific pytree
__q_checkpoint__obs__final                     # new — obs array
__q_checkpoint__ep_return__final               # new — scalar
__q_checkpoint__replay__final__<field>         # only if keep_replay
```

Cell runner's sentinel-key parser at `q_checkpoint.py:93-100`
extends with new arm values (`opt_state`, `step`, `pending`,
`shc`, `env_state`, `obs`, `ep_return`, `replay`). Each routes to
its own writer. Replay routes to the parquet writer rather than the
msgpack.

This keeps the in-record sentinel-key boundary as the framework's
single seam between dqn's record dict and persistence. No new
hook needed in `train_with_eval`.

## 9. Migration order

Each Phase 3 field is independently landable. Recommended order
by value-per-cost:

### Phase 3a — `opt_state` (one commit)

- Extend `QCheckpoint` msgpack writer/loader.
- Add `opt_state: OptState | None` to `InitOverride`.
- Add `opt_state: bool = False` to `QCheckpointLoadFlags`.
- `init_state`'s `optimizer.init(online)` branch.
- Test: closed-form. Run 100 training steps from V's burst-25
  with `load.opt_state=True` vs `False`. Loaded-path's first
  gradient update direction matches V's burst-26 first-update
  direction (closed-form: compute V's actual gradient at burst-25
  parameters and assert update step matches under same data).
- ~150 LOC delta.

### Phase 3b — `step` + `rng_key` (one commit)

- Add both to `QCheckpoint`, `InitOverride`, `QCheckpointLoadFlags`.
- `init_state` branches.
- `dqn`'s `total_steps` countdown reads from loaded step.
- Tests:
  - `linear_epsilon(step=500_000, anneal=100_000, eps_init=1.0,
    eps_final=0.05) == 0.05` (closed-form ε post-anneal).
  - With matching seed, `rng_key` loaded vs `rng_key` fresh produce
    different first-action samples but identical sequences within
    each path.
- ~100 LOC delta.

### Phase 3c — `pending_n_step` + `state_hash_count` (one commit)

- Same pattern; both are small NamedTuples / arrays.
- Test: at `n_step=3`, mid-window burst-boundary load preserves
  the partial aggregate (assert `state.pending_n_step.acc_reward`
  bit-equals msgpack's after load).
- ~80 LOC delta. Can be co-bundled into Phase 3b if both are
  ready together.

### Phase 3d — `env_resume` (env_state + obs + ep_return) (one commit)

- New `EnvResumeBundle` frozen-dataclass, JAX-registered.
- `InitOverride.env_resume: EnvResumeBundle | None`.
- `QCheckpointLoadFlags.env: bool` (gates all three as a unit).
- `init_state` branch.
- Tests:
  - Loaded `env_state` matches msgpack's (env-pytree round-trip).
  - Loaded `ep_return` resumes mid-episode return.
  - Sanity: at burst boundary on a `done=True` step, the next
    rollout step matches a fresh-env scenario regardless of load.
- ~120 LOC delta.

### Phase 3e — `replay` (one commit, biggest)

- Sidecar parquet writer: `replay.save_parquet(path, state)`.
- Sidecar parquet loader: `replay.load_batched_parquet(template,
  seeds) -> ReplayState`.
- `QCheckpointLoadFlags.replay: bool`.
- `init_state` branch reading `override.replay`.
- Substrate's `archive_remote` walker recognises the
  `_replay.parquet` extension for cloud-eviction.
- Tests:
  - Closed-form: write a `ReplayState` with known transitions,
    load via parquet, assert per-slot bit-equality.
  - End-to-end: load V's replay, sample one batch, assert
    transitions match V's (transition-hash invariant).
- ~250 LOC delta (the parquet schema + lazy-eviction wiring is
  the bulk).

**Total Phase 3: ~700 LOC across 5 commits.** Each commit is
independently testable; sweeps can adopt any subset.

## 10. Test strategy: closed-form per field

Each field's test asserts a SPECIFIC behaviour the load enables.
Generic "field is not None after load" tests are tautological —
they only verify the loader didn't drop the field, not that
`init_state` uses it correctly.

The pattern (per CLAUDE.md "Test principle"):

- **Build a controlled checkpoint** with known field values (write
  a synthetic msgpack — no V-training-trajectory needed for unit
  tests; the round-trip is the contract).
- **Compute the expected init_state slot** by closed form
  (e.g. `expected_eps = eps_final` when loading step past anneal;
  `expected_target_params = msgpack.target_params` not online).
- **Call `init_state` via the full path** with an `InitOverride`
  built from the loaded `QCheckpoint`.
- **Assert** the relevant `DQNState` field bit-equals (or
  within-eps-equals for float comparisons) the closed-form
  expectation.

Slow tests (`@pytest.mark.slow`) exercising the end-to-end
"load V's checkpoint and continue training" path are belt-and-
suspenders — the closed-form tests are the real coverage.

## 11. Phase 4 forward-compat: `Learner` bundling

`state.py:18` comments forward to a Phase 4 in this substrate:
"bundle `params+opt_state` into a `Learner` sub-state when that
lands." Phase 3 should leave room for this refactor:

- `InitOverride.online_params + target_params + opt_state` would
  collapse to `InitOverride.learner: LearnerInit | None` where
  `LearnerInit` bundles all three.
- Phase 3's `QCheckpointLoadFlags.target_params + opt_state` would
  collapse to `learner_full: bool` (single switch loading all
  three or none).
- `init_state`'s three independent branches collapse to one
  conditional.

Phase 3 doesn't pre-empt this — it lands the fields flat and lets
Phase 4 consolidate. The migration cost is bounded: dataclass
rename + field re-arrangement, no semantic change.

The same pattern applies to other potential bundles:
- `EnvResumeBundle` is Phase 3's explicit precedent (env_state +
  obs + ep_return are already bundled, anticipating that future
  authors won't want to opt half in).
- `state_hash_count` could be bundled with future per-state
  visit-frequency machinery if that grows.

The Phase 4 hint is the reason `InitOverride` shouldn't be
parameterised generically (no `dict[str, jax.Array]` payload
field). Bundling is structural, not by-name.

## 12. What we DEFER beyond Phase 3

- **Multi-substrate `InitOverride`**. Each substrate (DQN, future
  PPO, future SAC) defines its own `InitOverride` with its own
  fields. No cross-substrate inheritance; the framework's
  `Intervention` primitive accepts any `Annotated[T, Exogenous]`
  kwarg type. Future substrates derive their own
  `init_override.py` from the same recipe.
- **Schema migration tools**. If a Phase 3 field name changes
  later, msgpack readers fail loudly on the old key. We rely on
  the presence-of-field discipline to keep additions
  backward-compatible; renames are a separate maintenance task.
- **Cross-burst loading**. The current YAML uses
  `{seed}` as the only template placeholder. A `{burst}`
  placeholder would let a sweep load N different bursts of V to
  test "which V-stage's Q is the closest fixed-point". Out of
  scope; the `path_template: str` API already supports any
  python-format key the YAML author chooses, so this is a
  YAML-level extension only.
- **Network re-architecting on resume**. Loading V's CNN params
  into a DIFFERENT-sized network requires per-layer
  mapping/projection. Not a state-resume question — a
  transfer-learning question. Out of scope.
- **Soft-init mixing**. `online = α·V + (1-α)·random_init`. Out
  of scope; the convex-mix experiments operate at the operator
  level (`dampened_double_greedify(α)`) not at init.

## 13. Decision rule: which sub-phases to land first

The cost of full Phase 3 is ~5 commits + 8 closed-form tests +
`InitOverride` growing to 9 fields. The benefit is bit-exact
continuation semantics for downstream experiments.

Phase 3a (`opt_state`) pays off the moment any experiment asks
"continue V's training trajectory" — and the running
`asterix_g0999_init_v_burst25_continue_ddqn` sweep is approximately
that question, modulo the Adam-cold-start contamination. Phase 3a
removes the contamination. **Recommend landing 3a immediately
after the running sweep returns**, ideally before authoring any
finding that depends on the continuation semantic being faithful.

Phase 3b (`step` + `rng_key`) only pays off when an experiment
makes an explicit "extend budget" vs "resume budget" claim and
needs the ε-schedule to match. Land when a sweep YAML wants to
toggle this.

Phase 3c-d (`pending_n_step` / `state_hash_count` / env_resume)
are insurance against specific experimental shapes
(`n_step > 1` + mid-burst resume; count-weighted-loss
interventions; mid-episode env continuation). Land per-demand.

Phase 3e (`replay`) is the heaviest. Defer until a bridge / finding
explicitly demands V's actual transitions in DDQN's update stream
— "DDQN clip applied to V-collected data" semantic. The current
running sweep doesn't need this (it's the "DDQN clip applied to
DDQN-collected data starting from V's parameters" semantic).
Concrete trigger: a "training-data-distribution-vs-operator"
decomposition bridge.

**Default plan**: land 3a opportunistically; defer 3b-e until a
specific experiment requires each.
