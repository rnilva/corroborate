# Claim unification — single Claim shape, LCA locality, init↔partial isomorphism

## Status

Open design.

## Motivation

The substrate currently uses two primitive shapes (CLAUDE.md § Two
primitive shapes):

1. **Free Claim** — a `@claim`-decorated function. Pure operation;
   configured at composition time via `functools.partial`. e.g.,
   `mlp_forward`, `bootstrap`, `epsilon_greedy`, `uniform_sample`.
2. **Config bundle** — a frozen dataclass holding configuration
   fields, slot Claims, and mechanics methods. e.g., `Replay`,
   `MLP`, `CNN`, `LinearEpsilon`.

This split has accumulated structural tensions:

- **Mechanics methods aren't Claims.** `Replay.init`, `MLP.__call__`,
  `Replay.add` carry behavior but bypass the @claim graph. They're
  invisible to `walk_paths`, `transitive_reads`, and the
  trace-context computation graph. The substrate-author has to
  remember which methods are real Claims and which are mechanical
  glue. Reading the substrate, you can't tell from the shape alone.

- **`MLP.__call__` delegates to `mlp_forward`** purely to participate
  in the trace. The dataclass instance is structurally a Claim, but
  the implementation requires a layer of indirection: dataclass
  with method that calls a separate @claim function.

- **`init` is composition pretending to be a method.**
  `MLP.init(rng, obs_shape, n_actions) → params` builds state from
  config (`hidden`) plus runtime (`rng_key`, `obs_shape`,
  `n_actions`). The output is then plugged into
  `mlp_forward(params, obs)` as a kwarg. This is a partial Claim
  composition — but the substrate spells it as a dataclass + method.

- **`partial(mlp_forward, params=...)` is a functor.** The dataclass
  instance `MLP(...)` (init + call methods + state) is also a
  functor: configured + ready to be called. They're structurally
  identical — both are "callable closing over some bound kwargs,
  exposing the rest." The framework treats them as different things.

- **Locality is decided manually.** Each kwarg's "where does it
  live" depends on author judgment. `n_actions` ended up at dqn
  top-level and was passed down to consumers; `sync_period` lives at
  both `dqn` top-level AND inside `target_sync`. Manual placement →
  drift → silent inconsistency (the gates we just landed caught one
  of these; others are still latent).

- **`partial(warmed_update, inner=partial(adam))`** — two layers of
  partial nesting because `default_optimizer` is itself nested. The
  walker has to special-case partials-of-partials. The leaf path
  `optimizer.inner.lr` leaks the wrapping into the structural
  fingerprint; if RMSProp is swapped in (which has no `inner`
  layer), the path becomes `optimizer.lr` and the arm-key changes
  shape.

The principle we want to express:

> **Every entity in the @claim graph is a Claim. Configuration is
> just kwargs. "Init" is just a Claim returning a partial.
> "Functor" and "configured callable" are the same thing as a
> partial Claim. Locality is the closest common ancestor of
> consumers — derived, not decided.**

This collapses three currently-separate concerns (config bundles,
init/forward methods, top-level kwarg placement) into one recursive
structure.

## Principle: single Claim shape

```python
@claim
def f(
    *,
    config_kwarg_1=...,
    config_kwarg_2=...,
    runtime_kwarg_1,
    ...
) -> T: ...
```

Every entity is `@claim`-decorated. Kwargs split by *binding moment*,
not by primitive type:

- **Config kwargs** have defaults. They bind at composition time via
  `partial`.
- **Runtime kwargs** have no defaults. They bind at call time when
  the runner provides them.

The Claim's body composes these into the result. The result type may
be:

- **Data** (an array, a scalar, a record dict) — the Claim is a
  *pure operation*. Examples: `mlp_forward`, `bootstrap`,
  `squared_error`.
- **A callable** (a partial Claim, a functor) — the Claim is a
  *factory*. Examples: `mlp` (returns `partial(mlp_forward,
  params=...)`), `adam` (returns optax handle).
- **A state** (a record dict / pytree threaded through other
  Claims) — the Claim is an *init*. Examples: `replay_init`
  (returns ReplayState), `init_state` (returns DQNState).

These are not different primitive shapes. They are different return
types of the same shape. The trace context records each Claim call
the same way; `walk_paths` recurses uniformly.

## LCA: locality from topology

For each kwarg `x` consumed by some Claim(s) in the @claim graph,
`x` should live at the closest @claim that's a common ancestor of
every consumer. That's the LCA.

- If `x` has a single consumer: LCA = the consumer. `x` lives in
  that Claim's signature.
- If `x` has multiple consumers: LCA = their lowest shared ancestor
  in the call tree.

Anywhere `x` is bound *above* its LCA is a leak: a place where the
kwarg is declared but not consumed at this level — pure pass-through
plumbing.

The framework can detect leaks mechanically by walking the @claim
graph and checking each Claim's kwargs against its descendants'
consumption.

### Examples in the current substrate

| kwarg | consumers | LCA | currently lives at | leak? |
|---|---|---|---|---|
| `gamma` | `bootstrap`, `eval_burst` | `dqn` (their LCA) | `dqn` top-level | no |
| `sync_period` | `target_sync.periodic_copy` | `target_sync` | `dqn` AND `target_sync` | **yes — duplicated above LCA** |
| `n_step` | `bootstrap` (n-step return) | `bootstrap` | `dqn` AND `bootstrap` | **yes — duplicated above LCA** |
| `n_actions` | `mlp_init` / `cnn_init` (output dim) | `mlp` / `cnn` | `dqn` top-level | **yes — leaked all the way up** |
| `obs_shape` | `mlp_init` / `cnn_init` (input dim) | `mlp` / `cnn` | `dqn` top-level | **yes** |
| `replay.capacity` | `replay_init` (buffer size) | `replay_init` | `Replay` bundle field | acceptable* |

*`replay.capacity` at the bundle level is acceptable because the
bundle is the parent of `replay_init` (the only consumer). But it
could equally live in `replay_init`'s signature directly — same LCA.

The first three are real leaks; locality fixes are mechanical.

## init↔partial isomorphism

This is the deeper unification.

### The MLP case — current shape

```python
@dataclass(frozen=True, slots=True)
class MLP:
    hidden: tuple[int, ...] = (64, 64)

    def init(
        self,
        rng_key: jax.Array,
        obs_shape: tuple[int, ...],
        n_actions: int,
    ) -> dict[str, jax.Array]:
        # Glorot init; param dict shaped by hidden + obs_shape + n_actions.
        ...

    def __call__(
        self, params: dict[str, jax.Array], obs: jax.Array,
    ) -> jax.Array:
        return mlp_forward(params=params, obs=obs)
```

Two methods (`init`, `__call__`) plus one delegated `@claim`
(`mlp_forward`). The instance carries:
- state-of-config (`hidden`)
- a path to produce state-of-runtime (`init` → `params`)
- a path to consume state-of-runtime (`__call__` → `mlp_forward`)

### The MLP case — unified shape

```python
@claim
def mlp(
    *,
    rng_key: jax.Array,
    obs_shape: tuple[int, ...],
    n_actions: int,
    hidden: tuple[int, ...] = (64, 64),
) -> Callable[[jax.Array], jax.Array]:
    """Glorot-initialised MLP. Returns a partial of `mlp_forward`
    with `params` bound."""
    params = _glorot_init(rng_key, obs_shape, n_actions, hidden)
    return functools.partial(mlp_forward, params=params)
```

One Claim. Returns a callable. The callable IS a partial of
`mlp_forward`. From the caller's perspective:

```python
# Old:
net = MLP(hidden=(64,))
params = net.init(rng_key, obs_shape, n_actions)
q = net(params, obs)             # delegates to mlp_forward(params, obs)

# New:
forward = mlp(rng_key=k, obs_shape=os, n_actions=n, hidden=(64,))
q = forward(obs)                 # forward IS partial(mlp_forward, params=...)
```

Same semantics. Less ceremony.

### Why this matters for the @claim graph

`forward = mlp(...)` is a Claim call: it fires under
`trace_context`, records itself in the graph. The result is a
partial of `mlp_forward`, which is also a Claim. When `forward(obs)`
is later called, that fires `mlp_forward` under the trace.

The trace records both:
1. `mlp` (the factory Claim) — config + setup-runtime kwargs
2. `mlp_forward` (the operation Claim) — `params` (closure-captured
   by partial) + `obs` (runtime)

The graph has two nodes: `mlp → mlp_forward` with edge
`params ← <return>`. Structurally honest: the graph captures both
the configuration step AND the per-call invocations.

Under the OLD shape, `MLP.__call__` records `mlp_forward` with each
call but doesn't record `MLP.init` (it's a method, not a @claim) —
the graph is missing one node. The unification surfaces it.

### Functor ≡ partial Claim

The substrate's class-based-Claim escape hatch (CLAUDE.md § Escape
hatch) already says "a frozen dataclass with `name: str` and
`record_call(...)` is structurally a Claim." A functor (configured
callable) is structurally a partial: closes over config-bound
kwargs, exposes runtime kwargs.

`functools.partial(claim_fn, **config)` is a Claim by structural
Protocol if the resulting partial has a `name` attribute and records
on call. The framework already handles partials as Claim-shaped
structurally (`signature.py`'s `_walk_partial`, `claim.py`'s
`canonical_str`). The unification doesn't require new framework
primitives — it requires expressing the substrate in a way that
exploits what's already there.

## Stateful operations (init → state, not callable)

`Replay` is the test case for "what about multi-operation bundles
where init produces state, not a callable?"

### Current shape

```python
@dataclass(frozen=True, slots=True)
class Replay:
    capacity: int = 10000
    batch_size: int = 64
    sample: SampleClaim = uniform_sample  # slot Claim

    def init(self, obs_shape) -> ReplayState: ...
    def add(self, state, transition) -> ReplayState: ...
    def sample_batch(self, state, key) -> Batch:  # delegates
        return self.sample(state, key, batch_size=self.batch_size)
```

Three operations, shared state (`ReplayState`), shared config
(`capacity`, `batch_size`). One slot Claim already factored
(`sample`).

### Unified — bundle of Claims

```python
@claim
def replay_init(*, capacity: int = 10000, obs_shape: tuple[int, ...]) -> ReplayState:
    return ReplayState(buf=zeros((capacity, *obs_shape)), ...)

@claim
def replay_add(state: ReplayState, transition: Transition) -> ReplayState: ...

@claim
def replay_sample(
    state: ReplayState, *, rng_key: jax.Array, batch_size: int = 64,
) -> Batch:
    """Default uniform sample; swap-in via the `sample` slot if needed."""
    return uniform_sample(state, rng_key, batch_size=batch_size)


@dataclass(frozen=True, slots=True)
class Replay:
    """Group of three stateful Claims sharing ReplayState. No
    bundle-level config — each Claim owns the kwargs it reads."""
    init: ReplayInit = replay_init
    add: ReplayAdd = replay_add
    sample: ReplaySample = replay_sample
```

The bundle stays — it's a grouping convenience — but its fields are
all Claims (no methods). The bundle has **zero config fields**:
config lives where it's read.

`walk_paths` traversal: `replay → init → capacity` (path
`replay.init.capacity = 10000`); `replay → sample → batch_size`
(path `replay.sample.batch_size = 64`). Slightly longer paths but
locality-honest.

Intervention surface stays clean: `replay.sample = expectile_sample`
swaps just the sample Claim; config moves with it.

### Optional shape — flat (no bundle)

If the bundle adds nothing beyond grouping:

```python
@claim
def dqn(
    *,
    replay_init: ReplayInit = replay_init,
    replay_add: ReplayAdd = replay_add,
    replay_sample: ReplaySample = replay_sample,
    ...
)
```

Three slots in dqn. Tradeoff: ergonomically less grouped, but slot
substitution is direct (no `Replay(sample=...)` constructor).

**Recommendation**: keep `Replay` bundle for grouping ergonomics;
document that all bundle fields are Claims (no methods, no config).

## Substrate transformation table

Mapping current → unified:

| current | shape | unified shape |
|---|---|---|
| `MLP(hidden, init, __call__)` | dataclass + method delegation | `mlp` @claim returning `partial(mlp_forward, params=...)` |
| `CNN(hidden, init, __call__)` | same | `cnn` @claim, same shape |
| `Replay(capacity, batch_size, init, add, sample, methods)` | dataclass + methods + slot | `Replay` bundle with 3 Claim fields, zero config |
| `Adam(lr, b1, b2, eps, weight_decay)` | dataclass wrapping optax | `adam` @claim returning optax handle |
| `WarmedUpdate(inner, warmup_steps)` | dataclass wrapping factory | `warmed_update` @claim taking factory, returning factory |
| `LinearEpsilon(eps_init, eps_final, anneal_steps)` | dataclass | `linear_epsilon` @claim |

### Optimizer: the partial-of-partial collapses

Current default:
```python
default_optimizer = partial(warmed_update, inner=partial(adam))
```

`walk_paths(default_optimizer)` produces nested paths
`optimizer.inner.lr`. If RMSProp swaps in (no `inner` layer), the
path becomes `optimizer.lr` — arm-key shape mismatch.

Unified:
```python
@claim
def adam(*, lr: float = 1e-3, b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8) -> optax.GradientTransformation:
    return optax.adam(lr, b1=b1, b2=b2, eps=eps)

@claim
def warmed_update(
    *,
    inner: OptimizerFactory = adam,
    warmup_steps: int = 1000,
) -> optax.GradientTransformation:
    return optax.chain(optax.warmup_constant_schedule(warmup_steps), inner())
```

Single layer. `walk_paths(warmed_update)` produces
`optimizer.inner.lr` because `inner = adam` is a Claim with `lr`
kwarg. Symmetric across optimizer swaps: `optimizer = adam`
produces `optimizer.lr`; `optimizer = warmed_update(inner=adam)`
produces `optimizer.inner.lr`. Different optimizers, different
shapes — but each shape is honest about what it's wrapping.

(The asymmetry between `optimizer = adam` and
`optimizer = warmed_update(...)` paths is intrinsic — different
compositions have different leaf surfaces. The arm-key changing
across these is correct: they're structurally different
optimizers.)

## Framework consequences

### `walk_paths` simplifies

Current: walker has separate branches for `_walk_fn` (function /
Claim), `_walk_dataclass` (config bundle), `_walk_partial`
(bake-in). Three shapes → three branches.

Under unification:
- `_walk_dataclass` either vanishes (if all bundles disappear) or
  becomes "walk dataclass fields as Claims" (no config to descend
  into).
- `_walk_fn` and `_walk_partial` remain, handle Claims and partials
  uniformly.

Net: fewer branches, more uniform recursion.

### Regime detection works again

The current `Annotated[..., Exogenous]` forward-ref bug surfaces
because `dqn` imports `Env` under `TYPE_CHECKING`. Under
unification, the Claim that needs `Env` is the env-resolution Claim
(or `mlp` if it takes env directly) — and it can import `Env`
unconditionally because it's runtime-required. The forward-ref
disappears; `typing.get_type_hints()` resolves cleanly; regime
detection actually works.

### Registry uniformity

Currently: `_FN_CACHE` (Claims) + measurable registry + analysis
registry. Under unification: same three registries, but every entity
that lives in `_FN_CACHE` is a Claim of one of three return-shapes
(data / callable / state). Measurables are Claims that read
`record` and produce scalars — same shape. Analyses are Claims that
take `cells` and produce verdict-typed results — same shape.

### LCA leak detection

A new framework primitive becomes mechanical:

```python
def lca_leaks(claim: Claim) -> tuple[Leak, ...]:
    """For each kwarg in the @claim graph rooted at `claim`,
    check that its consumers' LCA equals the binding site. Return
    (claim, kwarg, declared_lca, actual_lca) tuples for leaks."""
    ...
```

Wired into substrate-CI: a hypothesis fails to land if its claim
graph has leaks above tolerance.

## Worked example: dqn under unification

Sketch of dqn's signature post-unification:

```python
@claim
def dqn(
    *,
    # Per-cell author primitives (LCA = dqn body):
    env_name: Annotated[str, Exogenous],
    seed: Annotated[int, Exogenous] = 0,
    wrappers: Annotated[tuple[EnvWrapper, ...], Exogenous] = (),
    # Cross-cutting algorithmic (LCA = dqn — multiple consumers):
    gamma: float = 0.99,
    total_steps: int = 50_000,
    eval_every: int = 5_000,
    n_episodes: int = 20,
    # Slot Claims:
    q_network: QNetworkFactory = mlp,           # was MLP(...)
    action_select: ActionSelect = epsilon_greedy,
    replay: ReplayBundle = Replay(),            # bundle of 3 Claims, 0 config
    bootstrap: Bootstrap = bootstrap,           # owns n_step internally
    loss_fn: LossFn = squared_error,
    target_sync: TargetSync = periodic_copy,    # owns sync_period internally
    optimizer: OptimizerFactory = warmed_update,  # 1-layer factory
) -> dict[str, jax.Array]:
    """..."""
    rng_key = jax.random.PRNGKey(seed)
    env, env_params = gymnax.make(env_name)
    for w in wrappers:
        env = w.wrap(env)
    n_actions = int(env.action_space(env_params).n)
    obs_shape = env_spec(env_name).observation_shape

    # Configure q_network: factory call returns a callable.
    init_key, run_key = jax.random.split(rng_key, 2)
    q_forward = q_network(
        rng_key=init_key, obs_shape=obs_shape, n_actions=n_actions,
    )  # partial(mlp_forward, params=...)

    # Replay state init.
    replay_state = replay.init(capacity=..., obs_shape=obs_shape)

    # ... rest of body uses q_forward, replay.add, replay.sample,
    # bootstrap, target_sync, etc.
```

Tree (post-unification, key paths only):

```
dqn
├── env_name [exogenous]
├── seed [exogenous]
├── wrappers [exogenous]
├── gamma
├── total_steps, eval_every, n_episodes
├── q_network = mlp
│   └── hidden = (64, 64)              ← only kwarg now
├── action_select = epsilon_greedy
│   └── schedule = linear_epsilon
│       ├── eps_init = 1.0
│       ├── eps_final = 0.05
│       └── anneal_steps = 10000
├── replay = Replay
│   ├── init = replay_init
│   │   └── capacity = 10000           ← was at bundle level
│   ├── add = replay_add
│   └── sample = replay_sample
│       └── batch_size = 64            ← was at bundle level
├── bootstrap = bootstrap
│   ├── greedification = max_greedify
│   ├── gradient_rule = semi_gradient
│   └── n_step = 1                     ← moved here from dqn top-level
├── loss_fn = squared_error
├── target_sync = periodic_copy
│   └── sync_period = 100              ← only place sync_period lives
└── optimizer = warmed_update
    ├── inner = adam
    │   ├── lr = 0.001
    │   ├── b1 = 0.9
    │   ├── b2 = 0.999
    │   ├── eps = 1e-8
    │   └── weight_decay = 0.0
    └── warmup_steps = 1000
```

Top-level kwargs go from ~22 → 14. Locality clean. Optimizer has one
layer of factory nesting (was two).

### cell_runner adapts

```python
def run_dqn_arm(env_spec, seeds, claim, arm_key, measurables, *, cycle_id=None):
    # No env-build, no cell_kwargs. dqn does its own setup.
    configured = partial(claim, env_name=env_spec.name)  # arm-level
    leaf_measurements = _leaf_measurements(configured)

    def by_seed(seed): return configured(seed=seed)
    seeds_arr = jnp.asarray(seeds, dtype=jnp.uint32)
    with trace_context() as records:
        batched = jax.vmap(by_seed)(seeds_arr)
    graph = build_computation_graph(records)
    # ... rest same ...
```

Three lines of substrate-specific glue (env_spec.name binding)
instead of the current ~25.

## Migration phases

Independent, incremental. Each phase lands one part of the
unification.

### Phase 0: design doc (this doc)

Land this design. Reference from CLAUDE.md.

### Phase 1: locality fixes (mechanical, pre-unification)

Substrate-only, no framework changes. Drop leaked kwargs:

- `n_actions` / `obs_shape` removed from `dqn` signature; bound into
  `MLP` / `CNN` config field (intermediate step before full
  unification — keeps current dataclass shape, just adds the field).
- Top-level `sync_period` and `n_step` dropped from `dqn`; owned by
  `target_sync` / `bootstrap` slots only.
- Move env-build inside dqn body; drop `env`, `env_params`,
  `eval_episode_cap`, `state_hash` from dqn signature.

Tree shrinks; arm-key paths simplify; gates surface fewer false
positives.

### Phase 2: collapse `MLP` / `CNN` into Claims-returning-partials

Per-class:
- Author `_glorot_init` (private helper or @claim Claim).
- `mlp` @claim takes `(rng_key, obs_shape, n_actions, hidden)`,
  returns `partial(mlp_forward, params=...)`.
- Drop `MLP.__call__`, `MLP.init` methods.
- Update callers: `forward = mlp(...)` then `forward(obs)`.

Substrate test passes confirm semantic identity. Graph shows the
new `mlp → mlp_forward` edge.

### Phase 3: collapse `Replay` to bundle-of-Claims

- Author `replay_init`, `replay_add` as @claims (alongside existing
  `uniform_sample`).
- `Replay` dataclass keeps three Claim fields, drops `capacity` /
  `batch_size` (config moves into the Claims).
- Update dqn body: `replay.init(capacity=..., obs_shape=...)`,
  `replay.add(state, t)`, `replay.sample(state, rng_key,
  batch_size=...)`.

### Phase 4: optimizer cleanup

- `adam` @claim returns optax handle directly; takes lr / b1 / b2 /
  eps / weight_decay as kwargs.
- `warmed_update` @claim takes `inner: OptimizerFactory` + warmup
  kwargs.
- Drop `partial(warmed_update(partial(adam)))` two-layer nesting.

### Phase 5: framework — LCA leak detection

- Implement `lca_leaks(claim) → tuple[Leak, ...]` walking the @claim
  graph.
- Wire into substrate-CI: a test that asserts
  `lca_leaks(corroborate_rl.dqn.dqn) == ()`.

### Phase 6: forward-ref fix in `walk_paths`

Now that `Env` doesn't need to be a top-level kwarg of dqn (env
resolution is in the body), the `TYPE_CHECKING` import disappears or
the kwarg vanishes. Either way, `Annotated[..., Exogenous]` regime
detection works at runtime — `regime='exogenous'` actually fires.

The gate's `_claim_leaves` can simplify to just `regime='leaf'`
(exogenous values are runtime-derived, no longer in the signature).

## Open questions

### Q1: Bundles or flat slots for stateful groups?

Replay bundle (`Replay(init, add, sample)`) groups three Claims by
state. Alternative: flat slots (`replay_init`, `replay_add`,
`replay_sample`) at dqn top-level.

Bundle wins on grouping ergonomics. Flat wins on locality
(intervention surface is exactly one slot). For Replay specifically
(3 ops, common state, well-known cluster) → bundle.

### Q2: What about Claims with mutable state?

None in current substrate (JAX-side is functional; params and state
re-created each step). If a future substrate needs mutable state
(e.g., Python-side replay buffer with side effects), the class-based
Claim escape hatch (CLAUDE.md § Escape hatch) covers it.

### Q3: Do all helpers become Claims?

Not necessarily. Pure helpers (`canonical_str`, `_leaf_scalar`,
`_glorot_init`) can stay plain functions. The threshold is "does
this carry theoretical content / appear in a paper claim?" If yes,
Claim. If pure scaffolding, plain function. The escape hatch flows
the other way too.

### Q4: How does this interact with the @measurable layer?

Measurables are already a Claim-shape variant: `@measurable(reads=...)`
decorates a function that reads `record` keys (and other
measurables) and produces a scalar. Under unification, measurables
remain a separate decorator (different return-type contract: scalar,
not callable / data) but share the same registry mechanics.

A possible follow-up: collapse `@measurable` into `@claim` with a
read-spec annotation. But that's orthogonal to this design.

### Q5: Does the cell_runner generalise?

Once dqn does its own env-build + state init, the cell_runner
becomes a generic "vmap any @claim over seeds, capture trace"
harness. Lift it to the framework? Then NLP / biology substrates can
reuse it.

That's a separate framework refactor. Track as follow-up.

### Q6: Backwards compatibility?

Substrate refactor changes intervention authoring (`q_network =
MLP(hidden=(128,))` becomes `q_network = partial(mlp, hidden=(128,))`
or `mlp` directly with the new shape). Existing experiments in
`experiments/findings/*.py` need migration.

Per-phase migration guide in each phase's commit. The unification
isn't a one-shot rewrite; phases 2–4 land independently with
sub-phase migration.

### Q7: Persisted corpus compatibility?

`runs.parquet` columns are leaf paths. After unification, paths
shift (e.g., `replay.capacity` → `replay.init.capacity`). Existing
parquets become non-comparable to new ones at the column-name level.

Two options:
- One-shot migration script: `replay.capacity` → `replay.init.capacity`,
  etc.
- Path aliasing in the persistence reader: substrate ships an alias
  table for legacy paths.

Probably script + cutover (similar to the `arm_key` migration in
task #13).

## Relationship to other design docs

- **`ENDOGENEITY_TOPOLOGY.md`** — the gate uses
  `walk_paths(claim).leaves` for endogeneity. Under unification, the
  walk is uniform (no special-cased dataclass branch); the gate's
  contract doesn't change, but the leaf set shifts as paths
  reshape.
- **`ADMISSION_GATES_DESIGN.md`** — gate Architecture stays. Once
  the unification lands, add `lca_leaks` as an INFO gate (or as a
  static-analysis CI check before evaluation).
- **CLAUDE.md § Two primitive shapes** — needs revision to reflect
  the unification (one shape with three return types).

## Decision

Phases 1, 2, 3, 4 are substrate work; 5, 6 are framework. Each
phase is a discrete commit. Phase 1 is mechanical (the locality
fixes we already discussed). Phase 2 is the conceptual core
(collapse `MLP` to `mlp` @claim returning a partial — proves the
isomorphism in production code). Phases 3–4 follow the same pattern.
Phase 5 makes locality auditable. Phase 6 cleans up the
forward-ref bug.

Order matters: Phase 1 before Phase 2 because Phase 2 already
assumes `n_actions` etc. live where they're consumed. Phases 3 and
4 can be parallel after Phase 2 (different config bundles, no
mutual dependency).

Recommend: land Phase 1 next, audit the result, then Phase 2 as a
focused PR.
