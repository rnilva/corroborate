# Checkpoint resume — extensibility design

Design document for evolving the current `init_online_params` plumbing
(commits a89fe86 + ef2b889) into a properly extensible "resume training
from saved state" API. **No implementation in this document** — design
choices and rationale only.

## 1. What the current implementation does

- `dqn()` and `init_state()` accept a single `init_online_params:
  Params | None` kwarg, Exogenous-marked.
- `cell_runner.run_dqn_arm` accepts `init_online_params_batched: Params
  | None`, threads a seed-axis-0 stack through `jax.vmap(..., in_axes=
  (0, 0))`.
- `DQNRunner` reads `grid_point['init_online_params_batched']` if
  present.
- YAML sweep config exposes `init_q_checkpoint_path_template: str |
  None`; `dispatch_sweep` materialises one batched pytree per chunk
  via `load_batched_online_params`.
- `q_checkpoint.QCheckpoint` already carries `online_params`,
  `target_params`, `burst`, `global_step` on disk — but the loader
  drops `target_params` on the floor.

## 2. The pressure: five more "resumable" state fields are coming

A complete "resume training" semantic needs to plug in:

1. **target_params** — currently mirrored from online; should reflect
   the source-trajectory's actual target (which lags online by
   `sync_period` steps).
2. **Replay buffer** — V's burst-25 replay contains 100k transitions
   of V's recent experience; fresh-empty buffer is the wrong
   continuation start.
3. **Optimizer state** — Adam's moment estimates after 500k steps
   are not equivalent to `optimizer.init(online)`.
4. **Step counter / RNG state** — continuation should resume
   `global_step` at V's value; ε-schedule extrapolation reads
   `state.step`.
5. **Batch-norm / EMA stats** — not used in this substrate but a
   forward-compatibility commitment for other DRL substrates.

Continuing the one-kwarg-per-field pattern (`init_target_params`,
`init_replay`, `init_opt_state`, `init_step`, `init_rng_key`) would
balloon `dqn()`'s signature to 5+ optional kwargs that are
operationally always set together. That's not a design — that's
accumulated drift.

## 3. Recommended pattern: typed `InitOverride` frozen dataclass

A single optional kwarg on `dqn()` and `init_state()`:

```
init_override: Annotated[InitOverride | None, Exogenous] = None
```

where `InitOverride` is a frozen-dataclass record:

```
@dataclass(frozen=True, slots=True)
class InitOverride:
    online_params: Params | None = None
    target_params: Params | None = None
    opt_state: OptState | None = None      # opaque optax handle
    replay: ReplayState | None = None      # opaque per-Replay impl
    pending_n_step: PendingNStepState | None = None
    step: jax.Array | None = None
    rng_key: jax.Array | None = None
    state_hash_count: jax.Array | None = None
```

Every field is `T | None` — the default `InitOverride()` (or `None`
itself) preserves freshly-init semantics. Authors who want to override
ONLY `target_params` write `InitOverride(target_params=t)`. The init
body in `init_state` reads each field once:

```
online = (override.online_params if override and override.online_params
          is not None else q_network.init(init_key, obs_shape, n_actions))
target = (override.target_params if override and override.target_params
          is not None else online)
# ... etc
```

### Four-question test (CLAUDE.md "When to introduce a framework primitive")

1. **Typed contract** — yes. Each field's type is the same shape
   `DQNState` carries (`Params`, `ReplayState`, `jax.Array`). No
   `dict[str, Any]`, no opaque payload. Pyright checks field types
   at every call site.
2. **Runtime narrowing** — yes. Frozen-dataclass instance attributes
   are statically resolved; the field-by-field `is None` checks
   inside `init_state` narrow each field to its non-None type in the
   "override active" branch. A would-be `dict[str, object]` payload
   would force narrowing at every use site.
3. **Real work beyond labeling** — yes. The dataclass holds the
   batched pytree (under `jax.vmap`, every field auto-batches along
   the seed axis when `in_axes=InitOverride(...)` matches), and
   serves as the single point where new resumable fields get added.
   Adding a new field is one line on the dataclass + one branch in
   `init_state` — no surgery on `dqn()`'s signature, no surgery on
   `cell_runner`'s vmap.
4. **Performance floor** — neutral. The dataclass is allocated once
   per cell (constant cost). JAX doesn't materialise None branches.
   `slots=True` keeps attribute access O(1).

Passes 3/4 with 4 neutral. Recommended.

### Why NOT a `ResumeSpec` Protocol with `.apply(state) -> state`

Considered and rejected:

- **Lifecycle mismatch.** `init_state` constructs DQNState from
  scratch via `q_network.init`, `replay.init`, etc. A `.apply(state)
  -> state` Protocol would require constructing a full default state
  first, then over-writing fields — wasteful (re-allocates Q-net
  params that are about to be replaced) and lossy (the `q_network.
  init`'s init_key seed differs from the source's, so even the
  "discarded" init walked the RNG forward).
- **Polymorphism cost.** Each substrate would need its own
  ResumeSpec subclass; the test code path quadruples (substrate-
  generic apply + substrate-specific subclass + per-field override
  logic + composition). The `InitOverride` record collapses to "if
  field is not None, use it".
- **No real polymorphism.** Every conceivable ResumeSpec for DQN
  would touch the same fields (DQNState's). The "subclass per
  intervention" surface is theatre — there's no `ResumeFromRandom`
  that's structurally different from `ResumeFromCheckpoint`.

### Why NOT a capability registry (field-name → loader function)

Considered and rejected:

- **Two-pass parse.** Registry-based selection means YAML carries
  a `fields: [online_params, target_params]` list, and a runtime
  table maps each name to a loader. That's `getattr` semantics in
  framework code — exactly what CLAUDE.md's typing discipline forbids
  ("getattr / setattr on typed values — these return `Any` /
  accept `object` and erase types").
- **The fields ARE known statically.** `DQNState` is a NamedTuple
  with a fixed shape per-substrate; the override surface is the
  field cross-product. A typed dataclass mirrors this cleanly; a
  registry buys nothing.

## 4. How target_params decoupling falls out (concrete next step)

The current `init_state` line 117:

```
target_params=online,
```

becomes:

```
target_params=(
    override.target_params
    if override is not None and override.target_params is not None
    else online
),
```

The msgpack `QCheckpoint` already carries `target_params` — the
`load_batched_online_params` helper just needs a sibling
`load_batched_init_override(template, seeds)` that returns a single
batched `InitOverride` carrying BOTH `online_params` AND
`target_params` along the seed axis.

YAML stays the same shape (`init_q_checkpoint_path_template`), but a
new optional sibling flag opts into using BOTH:

```
init_q_checkpoint_load_target: true    # default false (back-compat)
```

When `true`, the loaded `InitOverride` populates `.target_params`
from the same msgpack file. Backward-compatible default: target
mirrors online (current behaviour).

The vmap call in `cell_runner` then changes from:

```
jax.vmap(by_seed_with_params, in_axes=(0, 0))(seeds_arr,
                                              init_params_batched)
```

to:

```
jax.vmap(by_seed_with_override,
         in_axes=(0, InitOverride(
             online_params=0, target_params=0,
             opt_state=None, replay=None, ...))
        )(seeds_arr, init_override_batched)
```

where `in_axes` is itself an `InitOverride` whose fields are 0 (seed-
batched) or None (broadcast / absent). JAX's vmap supports per-leaf
in_axes via pytree-structured in_axes specs — frozen dataclasses are
valid JAX pytrees when registered via `jax.tree_util.register_dataclass`
(or via `@jax.tree_util.register_pytree_node_class` historically; the
newer dataclass-aware path is one line on the InitOverride class).

## 5. Migration from `init_online_params` to `init_override`

**The current kwarg is one cell tree wide.** Three layers (dqn,
init_state, cell_runner.run_dqn_arm + DQNRunner) plus the test plus
the YAML loader. Migration in one PR:

### Phase 1 (back-compat shim)

1. Add `InitOverride` dataclass + `init_override` kwarg.
2. Keep `init_online_params` kwarg as a thin deprecated shim:
   ```
   if init_online_params is not None and init_override is None:
       init_override = InitOverride(online_params=init_online_params)
   elif init_online_params is not None and init_override is not None:
       raise ValueError('pass one of init_online_params or '
                        'init_override, not both')
   ```
3. `cell_runner.run_dqn_arm` keeps `init_online_params_batched` as a
   shim → constructs `InitOverride(online_params=batched)`.
4. Existing YAML config + running sweep continue to work unchanged.
5. Existing test (`test_init_from_q_checkpoint.py`) keeps passing
   against the shim path; add a new test for the
   `init_override`-direct call path.

### Phase 2 (target_params decoupling)

1. Add `init_q_checkpoint_load_target: bool = False` to `DQNSweep`.
2. Add `load_batched_init_override(template, seeds, *,
   load_target: bool)` helper next to `load_batched_online_params`.
3. `dispatch_sweep` chooses helper based on the new flag.
4. New test: `test_target_params_decouples_when_load_target_true`
   — assert init_state's DQNState.target_params matches msgpack's
   target_params (not online_params).

### Phase 3 (replay / opt_state / step / rng_key)

1. Extend msgpack format: add `replay_state`, `opt_state`,
   `step`, `rng_key` fields to `QCheckpoint` (back-compat: missing
   fields stay None on read).
2. Extend `InitOverride` fields + `init_state` branches.
3. Per-field test (each new field tests a closed-form recovery: e.g.
   resuming step=500_000 advances ε-schedule from the post-anneal
   value, not from step=0).

Phase 1 is one commit, ~150 LOC delta. Phase 2 is another commit.
Phase 3 lands per-field as the substrate needs each capability.

## 6. Trade-offs vs simpler alternative: `init_target_params` ad-hoc

The simpler alternative: add `init_target_params: Params | None` as
a sibling kwarg to `init_online_params`. No new dataclass. Total
diff: ~30 LOC.

Pros of the ad-hoc kwarg:
- Smaller code surface NOW.
- No vmap-in_axes-pytree complication.
- Easier to read at the call site (every field is a named kwarg).

Cons of the ad-hoc kwarg:
- After 3-4 fields, the signature is unreadable (`init_online_params=
  ..., init_target_params=..., init_opt_state=..., init_replay=...,
  init_step=..., init_rng_key=...`).
- Every new field needs surgery on `dqn()`, `init_state()`,
  `cell_runner.run_dqn_arm`, `cell_runner.run_dqn_cell`, `DQNRunner`,
  YAML loader. Six places per field.
- The "config bundle" rule in CLAUDE.md ("does this entity bundle
  stateful mechanics that need to be paired with construction-time
  HPs?") points squarely at "yes — InitOverride bundles override
  fields that are paired together at init time". The framework's
  vocabulary names this pattern.
- vmap `in_axes` for many kwargs gets ugly — `in_axes=(0, 0, 0, 0,
  None, None)` is unreadable; `in_axes=(0, InitOverride(...))` is
  self-documenting.

**Decision rule.** If the substrate's expected next-2-years
trajectory is "we'll add 1 more field (target_params) and stop",
the ad-hoc kwarg wins. If it's "we'll add 3-5 fields as
state-resumption matures", the dataclass wins by Phase 2.

The probability-weighted forecast is closer to the second
(target_params is the only confirmed near-term need, but replay /
opt_state are realistic mid-term). Recommend `InitOverride` from
the start — Phase 1's shim absorbs the migration cost.

## 7. Concrete next-step ordering

1. **Land Part 1 review fixes (small, before any design work):**
   - Move/rephrase the stale `trace_context` comment in
     `cell_runner.py:499-503`.
   - Reject empty `Params` dict at `_is_params` (sweep.py) and
     `init_state`'s `init_online_params is not None` (dqn.py)
     boundaries.
   - Drop the redundant `init_params_batched = init_online_params_batched`
     rebind at `cell_runner.py:487`.
   - Document the relative-path-resolves-to-CWD invariant on the
     YAML field's helptext (it's in the dataclass comment now;
     also belongs in the loader's error message when a path
     doesn't resolve).

2. **Land Phase 1 `InitOverride` shim** (this design's deliverable).
   New kwarg + shim from current kwarg + dataclass + JAX pytree
   registration. Existing tests pass unchanged. New unit test for
   the direct-`InitOverride` path. ~150 LOC.

3. **Land Phase 2 target_params decoupling.** New YAML flag,
   sibling loader, init_state branch. New closed-form test:
   assert that after `init_state` with `load_target=True`,
   `state.target_params` bit-equals the ckpt's `target_params`
   (NOT the ckpt's `online_params`).

4. **Defer Phase 3** until the substrate has a concrete experiment
   that requires step / rng_key / replay resumption. Each adds one
   field; the architecture stays the same.

5. **Author note for Phase 2:** the running
   `asterix_g0999_init_v_burst25_continue_ddqn` sweep produces
   the SAME outcome under load_target=False as the current
   implementation (target mirrors online). Switching to
   load_target=True changes the experiment's semantic
   subtly — the "DDQN's early dynamics vs steady state" question
   becomes "DDQN's steady-state operator applied to V's
   online + target". Be explicit in the new YAML config which
   regime is being tested.
