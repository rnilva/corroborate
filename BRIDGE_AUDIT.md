# Bridge audit — converting prose to typed objects

Companion to `PRIMITIVES_AUDIT.md` and `ADMISSION_GATES_DESIGN.md`.
The findings modules (`ddqn/`, `dqn_bridges.py`) have
accumulated prose at a rate the framework hasn't kept up with.
This manifest names what the prose is, how to incarnate the
convertible parts, and where the rest goes.

First application: `experiments/findings/ddqn/` (4411
lines, 49 bridges). Why now:

- The replay ring-buffer fix means the `*_postfix` corpora are
  the valid data; pre-fix docstring numbers are drift candidates.
- Seed-paired analyses are retiring *per-bridge* — within-seed
  variance reduction is load-bearing for some contrasts (e.g.
  CLAIM 9's n-step falsification). Convert with a variance-
  reduction reading, not a global sweep.

## Two meta-principles

**(M1) Runner-acts-on-it discriminator.** Can the runner *do*
something with the prose — verdict, block, warn, roll up,
report XFAIL/XPASS? If yes, it incarnates. If no, the prose
migrates: Hypothesis-module docstring (methodological prior),
`findings_*.md` with explicit destination (research journal),
or git log (migration history).

**(M2) Complete or extend the existing primitive before
proposing a new one.** Read every site of the closest existing
primitive end-to-end first. Extend the primitive to cover the
gap; introduce a new primitive only when extension genuinely
doesn't fit. The Prediction-bundle extension of
`predicted_direction` (not a parallel primitive) is the
canonical M2 application — see `BRIDGE_PREDICTION_DESIGN.md`.

## Incarnation surfaces

| Surface | What it is | Where it lives | Action |
|---|---|---|---|
| Shape A — bridge | Unauthored claim about the corpus | `@claim_bridge` | author |
| Shape B — admission gate | Authoring rule the runner refuses / warns on | `gates=(…)` on Bridge | author when boilerplate has surfaced ≥ 3× |
| Hoist | Repeated inline expression / function | Module-level `pl.Expr` / `@measurable` / `Final[float]` | lift |
| Prediction bundle | xfail-shape extension of `predicted_direction` | new `Prediction` on `Bridge.prediction`; see `BRIDGE_PREDICTION_DESIGN.md` | extend existing primitive |

### Shape A — prose-as-bridge

Claims about the corpus the runner can produce a verdict for.
Each could be a `@claim_bridge` function whose verdict carries
the claim, making the prose evaporate.

**Calibration:** deletion-memo bullets are mixed — typically
one incarnatable bridge per memo, plus debugging diagnoses
(per-stratum leverage, weighting artifacts) that belong in
`findings_*.md`.

**Examples that incarnate:**

- CLAIM 16's bf-cluster bullet ("bf clusters at [0.98, 1.00]
  across true chain MDPs") → bridge with
  `partition_aggregate('bootstrap_fraction', by='env_name',
  op='std') < 0.01`.
- CLAIM 14's "8-of-8 sign-match is structurally forced" →
  measurement-identity bridge that fails LOUDLY if a future
  corpus violates the equivalence.

**Examples that migrate to findings:**

- "FourRooms-domination artifact (27% cell share)" → migrates
  to `findings_per_env_vs_per_cell_weighting.md`.
- "Bandit-tail leverage on bf cross-env" → covariate-leverage
  observation; migrates to the relevant findings file.

### Shape B — prose-as-admission-gate

Authoring rules learned the hard way. The slot exists
(`Bridge.gates: tuple[AdmissionGate, ...]`); the gates are
mostly unwritten. Author when the boilerplate has surfaced
≥ 3 times in past bridges.

Cleanly authorable today:

- `ShadowOfMediator(against='jensen_gap')` — "run partial-
  Spearman | Δ_jens before authoring any mediator bridge."
- `EndogenousScopeOnly` — "scope must use endogenous
  predicates, not HP-knob equality." Partial precedent in
  `ENDOGENEITY_TOPOLOGY.md`.
- `StratifiedPoolingRequired` — partially encoded as
  `VariableScope.assert_stratification_admissible`.

Descoped: `SubstrateVersionInvariance` (multi-PR schema
change; `RunRow` has no substrate-version column; defer
until the column exists).

### Hoisting (discipline, not shape)

Anything appearing ≥ 3× with a name that survives author
turnover moves to a module-level binding. Trigger raised
from 2× to avoid comfort-constants.

Three discriminators:

1. **Stable referent with a meaning.** `BOUNDED_Q_REGIME =
   finite_lt('q_divergence_score', 1.0)` survives author
   turnover. `_EXCLUDE_TWO_ENVS = ~CartPole & ~MetaMaze099`
   doesn't.
2. **Promote-to-measurable when the constant is a
   derivation.** `(mean(reward) / mean(episode_length))` is
   a `@measurable`, not a `pl.Expr`. Promote directly — don't
   hoist to constant first.
3. **Discoverable coupling for scope-that-is-also-a-claim.**
   When a hoisted constant's *value* is the threshold of an
   invariant bridge, define it as `Final[float]` imported by
   both the cohort and the invariant. *Discoverable +
   reviewable*, not mechanical: Python doesn't prevent inline
   `1.0`. When the constant's value changes during a hoist or
   revision, grep the codebase for the literal and migrate or
   document independent copies.

Default to plain Python. Promote to a typed wrapper
(`Cohort('G1', expr=...)`) only when the runner needs runtime
semantics on the name; otherwise it's enum-mimicry — see
`PRIMITIVES_AUDIT.md`'s four-question test.

### Prediction bundle (extension of `predicted_direction`)

`predicted_direction: PredictedDirection | None` already lives
on `Bridge` and is threaded through `verdict_from_paired_stats`,
`random_effects_verdict`, `paired_comparison.py`, `runner/report.
py`, and the substrate YAML loader. It's the framework's
existing xfail-shape, but:

- The `'null'` decision-tree branch is unimplemented in both
  decision functions (falls through to defensive
  `NO_EFFECT/NULL_EFFECT`).
- There's no warrant slot the runner displays when XPASS fires.
- There's no strict-escalation flag.

The extension: a `Prediction(direction, reason, strict)`
frozen dataclass; `Bridge.prediction: Prediction | None`;
`Bridge.predicted_direction` becomes a `@property` derived
from `prediction` so the 40+ existing analysis sites read it
unchanged.

**Detailed design — including the decorator-kwarg migration,
display-value `Literal` typing, mandatory-reason policy,
`verdict_from_threshold` carried-forward debt, and the
framework/substrate PR split — lives in
`BRIDGE_PREDICTION_DESIGN.md`. That doc is written after the
audit reveals which bridges survive; designing the framework
completion against a phantom bridge set is premature.**

## What NOT to incarnate

| Prose type | Destination |
|---|---|
| Methodological prior (scope warrant) | Hypothesis module top-level `__doc__` |
| Research journal (deletion-memo diagnostic bullets) | `findings_*.md` with explicit pointer |
| Migration history | git log + commit message |
| Frozen empirical readings (claim warrant) | `Prediction.reason` when the bundle lands |
| Frozen empirical readings (non-warrant) | delete |
| Section banners describing clusters | `# CLAIM N` comment; `claim_id: str` IFF audit leaves ≥ 3 clusters with ≥ 3 bridges |

"Delete to git log alone" loses content the next researcher
needs to find by topic, not by date. Migrate to a typed
destination first; git log is the *additional* record.

## Order of operations

Steps 1–3 are concrete and ready. Steps 4+ depend on the
audit's output and are sketched only.

1. **Pin and audit.** Drop a `.in_progress`-style snapshot
   marker (`CORPUS_INTEGRITY.md` CI1) so the audit reads a
   frozen cell-set even while new sweeps land. Re-run every
   bridge against the snapshot. Tag each:
   - SURVIVED — verdict matches docstring.
   - STALE — verdict drifted.
   - DEAD — claim retracted in prose.
   - POWER_COLLAPSED — n_pairs collapsed under proposed
     stratification conversion.
   - SCOPE_VACATED — scope expression matches zero cells.
   For each seed-paired bridge, also record a
   **variance-reduction reading**: does within-stratum
   pooling reproduce the within-seed contrast at acceptable
   power? If not, the seed-paired form stays.
   Deliverable: a verdict table.
2. **Cut.** Delete DEAD and SCOPE_VACATED bridges plus STALE
   bridges with retracted claims. Migrate research-journal
   prose to per-bridge `findings_*.md` destinations.
3. **Incarnate Shape A.** Author the bridges hiding in
   deletion memos and "we noticed X" comments. Mixed budget.

After step 3 produces a survivor set:

4. **Hoist.** Scan survivors + new bridges for inline
   constructions appearing ≥ 3× with stable names. Apply the
   three hoisting discriminators. Surviving bridges still
   carry the legacy `predicted_direction=` kwarg at this
   point; hoist makes no kwarg changes.
5. **Author Shape B gates** for methodology rules with ≥ 3×
   surface.
6. **Framework completion (PR 1).** Lands per
   `BRIDGE_PREDICTION_DESIGN.md` §10. Decision-tree `'null'`
   branches in both decision functions; `Prediction` bundle;
   `Bridge.prediction` field; `@property predicted_direction`
   for back-compat; decoration validation; report-layer
   display `Literal`; YAML refactor + migration script. After
   this PR, BOTH `predicted_direction=` (legacy) and
   `prediction=Prediction(...)` decorator kwargs work — see
   companion §3 option (a).
7. **Substrate migration (PR 2).** Every surviving bridge
   migrates `predicted_direction='x'` →
   `prediction=Prediction(direction='x', reason='…')`. The
   `reason` field is populated per warrant by the author,
   NOT copy-pasted from the verdict. The legacy decorator
   kwarg may be hard-cut after this PR completes — companion
   §3 option (b).
8. **Cluster split — only when navigation pain is concrete.**
   The Hypothesis Protocol permits per-module or per-class
   Hypotheses; it doesn't *prefer* one. Subjective call at
   audit time, not a numeric gate. Default is consolidated.

## Non-goals

- **No free-form description / narrative / rationale field on
  `Bridge`** except `Prediction.reason` (framework-typed via
  decoration-time validation per the companion design doc).
- **No new typed primitives** beyond the `Prediction` bundle
  (and the report-layer `Literal` for derived display values —
  see companion).
- **No global mechanical conversion of seed-paired bridges.**
  Per-bridge variance-reduction reading.
- **No promise on substrate-version invariance until the
  corpus carries the column.**
- **No backward-compat for the YAML flat `predicted_direction`
  key.** Hard-cut with a one-shot migration script.
- **No second pass.** If another large prose blob accumulates,
  one of the surfaces was missed; identify it.
