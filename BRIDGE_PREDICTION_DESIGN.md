# Bridge prediction design — extending `predicted_direction`

Companion to `BRIDGE_AUDIT.md`. Designs the framework
completion that extends the existing `predicted_direction:
PredictedDirection | None` field on `Bridge` with a
pytest-xfail-shape bundle: `Prediction(direction, reason,
strict)`.

**Status: stub.** Filled in once `BRIDGE_AUDIT.md` step 3
produces a survivor set. Designing the framework-completion
details against a phantom bridge population is premature;
this doc lists the open questions and the chosen-or-pending
answer for each.

## Background: what `predicted_direction` already does

- Type: `Literal['a_gt_b', 'a_lt_b', 'two_sided', 'null']`
  (`core/hypothesis.py:67`).
- Stored on `Bridge` (`bridge/bridge.py:247`); validated at
  decoration time (`:486-509`).
- Consumed by `verdict_from_paired_stats`
  (`stats/effect_size.py:144-194`) and
  `random_effects_verdict` (`:380-421`) to map paired stats →
  `(Verdict, RefutationClass)`.
- Threaded through `PairedComparisonResult.predicted_direction`
  (`analyses/paired_comparison.py:80`) and the run report
  (`runner/report.py:72`).
- `Bridge.predicted_direction` is the sole authoring surface.
  The substrate YAML (`InterventionConfig`) no longer carries
  this field — the inheritance pathway it implied was never
  wired (bridges always authored their own value via
  `@claim_bridge(predicted_direction=...)`), so the YAML field
  was dropped in the 2026-05-14 schema cleanup.
- `RefutationClass.SIGN_FLIP` already encodes XPASS-shape;
  `RefutationClass.NULL_EFFECT` already encodes
  XFAIL_VIOLATED-shape.

## What the design must resolve

### 1. `'null'` decision-tree branch (decided)

Both `verdict_from_paired_stats` and `random_effects_verdict`
fall through on `'null'` to a defensive `NO_EFFECT/NULL_EFFECT`.
Add a parallel `'null'` branch to both:

- adequately powered, |g| < MDE → `(HELD, None)` — null
  prediction confirmed.
- adequately powered, |g| ≥ MDE → `(NO_EFFECT,
  RefutationClass.SIGN_FLIP)` — null prediction refuted
  (XPASS).
- not adequately powered → `(POWER_INSUFFICIENT,
  RefutationClass.UNDERPOWERED)`.

### 2. `Prediction` bundle shape (decided)

```python
@dataclass(frozen=True, slots=True)
class Prediction:
    direction: PredictedDirection
    reason: str | None = None
    strict: bool = False
```

```python
@dataclass(frozen=True)
class Bridge:
    # ...existing fields unchanged...
    prediction: Prediction | None = None

    @property
    def predicted_direction(self) -> PredictedDirection | None:
        return self.prediction.direction if self.prediction else None
```

Justification:

- Runtime narrowing: `bridge.prediction is not None` narrows
  the bundle's *existence*. The directional-dependency
  (`direction='null' ⇒ reason: str`) is enforced at
  decoration time, not by pyright. Honest: the bundle's
  type-discipline contribution is narrowing-existence, not
  enforcing the dependency.
- Four-question test (`PRIMITIVES_AUDIT.md`): (1) typed
  contract ✓; (2) runtime narrowing ✓ (existence only); (3)
  real work beyond labeling ✓ (decoration-time validation +
  runner-side strict escalation); (4) perf floor N/A.

### 3. Decorator-kwarg migration (TBD — pick one)

`claim_bridge` currently takes `predicted_direction:
PredictedDirection | None` (~53 call sites). Options:

- **(a) Accept both kwargs during migration.** Legacy
  `predicted_direction='x'` wraps to `Prediction(direction=
  'x')` at decoration time. Drop the legacy kwarg after the
  audit's substrate-migration PR.
- **(b) Hard-cut on the decorator.** All 53 sites edited in
  the bundle-introduction PR. Cleaner but couples framework
  PR to a substrate edit pass.

**Recommendation:** (a) for the framework PR; flip to (b)
after the substrate PR migrates every site.

### 4. Mandatory-reason policy (TBD)

Decoration-time validation rejects `Prediction(direction=
'null', reason=None)`. Open: should *directional*
predictions also require a reason?

- Asymmetric (current sketch): `reason` framework-typed for
  null; narrative for directional.
- Universal: every prediction carries a reason; conventional
  `reason='no-warrant'` allowed where appropriate.

The asymmetry's risk: a directional bridge with `reason=
None` has no warrant the runner can display when SIGN_FLIP
fires — same gap the null case was trying to close.
Universal is cleaner; asymmetric is what the v4 sketch
proposed.

**Recommendation:** universal, with a conventional empty
token. Revisit after the audit surfaces whether warrant
authoring is heavy enough to justify the asymmetric escape
hatch.

### 5. Display-value typing (decided)

The runner derives a status from `(verdict, refutation_class,
prediction.direction)`. Value set is closed:

```python
PredictionDisplay = Literal[
    'EXPECTED', 'XPASS', 'XFAIL_VIOLATED', 'UNDETERMINED',
]
```

Plain `str` is wrong (closed set, framework-emitted,
serialized to `run.json`, downstream consumers branch). A
`StrEnum` is acceptable if `runner/report.py` consumers
benefit from `.name` access; otherwise `Literal[...]` is
sufficient and lighter.

**Decided:** `Literal[...]` at minimum. Default in v1.

### 6. `UNDETERMINED` resolution (TBD)

Two distinct states collapse:

- `bridge.prediction is None` — author silence; no claim
  about direction.
- `bridge.prediction is not None` AND verdict is
  POWER_INSUFFICIENT / INADMISSIBLE — data inadequate
  despite a claim.

Options:

- Split: `'NO_PREDICTION'` for author silence;
  `'UNDETERMINED'` for inadequate data.
- Collapse: keep `'UNDETERMINED'`, document the loss.

**Recommendation:** split. The audit-trail distinction is
why we added the bundle.

### 7. YAML loader refactor (obsolete)

Original plan: extend `HypothesisConfig.predicted_direction`
to a `prediction: Prediction` bundle on the YAML. Obsoleted by
the 2026-05-14 schema cleanup, which dropped the YAML's
predicted-direction surface entirely — `Bridge.predicted_direction`
is the canonical (and sole) home, authored at `@claim_bridge`
decoration time. If a `Prediction(direction, reason, strict)`
bundle ships, it lives on `Bridge`, not on `InterventionConfig`.

### 8. `verdict_from_threshold` carried-forward debt

Substrate helpers `_eff_h_mediation_holds_when` and
`_staleness_mediation_holds_when` (since-removed pre-migration;
see `experiments/findings/ddqn/` for current shape) operate on
`proportion_mediated.proportion` against a `dominance_floor`
— proportion-style, not paired-g. `verdict_from_paired_stats`
can't subsume them. They're ~25-line substrate-specific
verdict-from-threshold factories.

A future framework primitive:

```python
def verdict_from_threshold(
    value: float, threshold: float, *,
    direction: Literal['lt', 'gt'],
    predicted_direction: PredictedDirection | None,
    # … power / NaN / unit-interval handling …
) -> tuple[Verdict, RefutationClass | None]:
    ...
```

would shrink the helpers to one-line calls. **Not in v1
scope.** Tracked as carried-forward debt; revisit when the
helper count grows past 2-3.

### 9. Body-flip helpers DON'T change

v3 and v4 (residually) claimed the helpers "stop flipping."
Reading the helpers (now removed; previously in `ddqn_universe.py`): they
already return framework-natural HELD when prediction is
confirmed and NO_EFFECT when refuted. Nothing flips. What
changes: bridges declare `prediction=Prediction(direction=
'null', reason='…')` instead of `predicted_direction='null'`.
The helper bodies are unchanged.

### 10. PR split

- **Framework PR.** Decision-tree `'null'` branches in both
  `verdict_from_paired_stats` and `random_effects_verdict`.
  `Prediction` dataclass. `Bridge.prediction` field +
  `@property predicted_direction` for back-compat.
  Decoration-time validation per (4). Report-layer display
  Literal per (5). YAML refactor + migration script per (7).
  Decorator-kwarg compatibility per (3a).
- **Substrate PR.** Every surviving bridge migrates
  `predicted_direction='x'` → `prediction=Prediction(
  direction='x', reason='…')`. Reason fields populated by the
  author per warrant, NOT copy-pasted from the verdict.
  Optionally hard-cut the decorator kwarg per (3b).

Framework PR lands first; substrate PR follows. The
`@property` back-compat means substrate continues to work
between PRs.

## Open questions to resolve at design time

- (3) decorator-kwarg: (a) then (b), or just (b)?
- (4) mandatory-reason: asymmetric or universal?
- (6) UNDETERMINED: split or collapse?

Items 1, 2, 5, 7, 8, 9, 10 are decided above.

### 11. Seed-pairing retirement (per-fixture migration)

Scoped in 2026-05-11 after the audit's step-5 variance-reduction
reading and a follow-up methodological critique. See
`experiments/findings/BRIDGE_AUDIT_TABLE.md` "Step 5 — REVISED".

**The methodological issue.** Seed-pairing assumes within-pair
correlation reflects a shared confounder that cancels in the Δ
(A/B testing's "same unit, two treatments" shape). RL violates
this: same seed ⇏ same trajectory. From step ~1 the two arms
diverge as the treatment changes the loss, sample order, and
explored state space. The shared seed cancels init weights and
PRNG state, not the bulk of training variance. Empirically on
FourRooms n_step=1, paired SE is ~100× tighter than independent
because `cov(arm_t, arm_b) ≈ +0.9999` on a near-deterministic
env — a property of training dynamics, not a statistical
property of the comparison.

The practical inferential target in RL is the cross-init
population distribution; the seed-paired form makes within-init
effects look like population effects.

**Per-fixture migration.**

| Current fixture | Stratified analog | Exists in framework? | Action |
|---|---|---|---|
| `paired_g` | `stratified_arm_diff_pooled` | ✓ | bridge-side migration |
| `paired_link_per_burst` | `stratified_link_per_burst` | ✗ | author |
| `proportion_mediated` | `stratified_proportion_mediated` | ✗ | author |
| `partial_spearman_paired_delta` | per-stratum partial Spearman + Fisher-z pool | ✗ | author (Fisher-z + DerSimonian-Laird) |
| `paired_delta_link_dowhy` | DoWhy stratified backdoor (env as confounder, not pairing axis) | partial | refactor existing |
| `link_attenuation_dowhy` | Same | partial | refactor existing |
| `mundlak_paired_g_per_burst` | Mundlak already does between/within decomposition; consume per-stratum aggregates not per-pair Δs | refactor existing | refactor existing |
| `meta_regression_per_burst` | half-stratified: within-stratum paired via `pair_by`, across-stratum pooled. Migrate to fully-stratified per-(env, burst) independent-samples then pool | partial — author the within-stratum-stratified variant |
| `paired_link_per_env` | half-stratified: within-env paired, across-env pooled. Migrate to within-env independent-samples link r then pool | partial — author the within-stratum-stratified variant |
| `cross_config_paired_slope` | half-stratified: within-config paired, across-config Spearman. Migrate to within-config independent-samples then Spearman | partial — author the within-stratum-stratified variant |

5 fixtures need new framework analogs; 2 (`mundlak_*`,
DoWhy-family) need internal refactors; 4 are already stratified
(remove vestigial `pair_by=('seed',)` kwarg).

**Sequencing within step 6.**

1. **Vestigial cleanup (1 bridge).** Only
   `stratified_arm_diff_pooled` truly ignores `pair_by`.
   Cleanup dropped the explicit kwarg from
   `ddqn_helps_under_three_gate_scope__cross_env` (2026-05-11);
   framework default still applies but the primitive ignores
   it. The other 4 bridges originally classed as "vestigial"
   are actually *half-stratified* (within-stratum pair via
   `pair_by`, across-stratum pool); they migrate alongside the
   load-bearing bridges in step 4 below.
2. **Author missing stratified analogs** (3 new fixtures:
   `stratified_link_per_burst`, `stratified_proportion_mediated`,
   per-stratum partial Spearman). Frame-level: each must
   produce a `(d, se, pooled_p)` shape compatible with
   `verdict_from_paired_stats`'s `'null'` branch so the
   Prediction bundle's xfail semantics apply uniformly.
3. **Refactor existing fixtures** (`mundlak_*`,
   DoWhy-family). The Mundlak decomposition already separates
   between- and within-env variance; the refactor is to source
   per-stratum aggregates as input. DoWhy fixtures already
   support per-stratum backdoor; rework call sites to pass env
   as a confounder rather than as a pairing axis.
4. **Bridge-side migration** (17 bridges from §11's table
   classified as "load-bearing seed-paired" in the audit).
   Each bridge swaps its fixture import to the stratified
   analog. Substrate PR (4b in BRIDGE_AUDIT.md's order of
   operations).
5. **Re-run the audit** under stratified methodology. Expect
   SURVIVED → POWER_COLLAPSED shifts on bridges whose verdict
   bodies use p-value gates rather than pure effect-size
   thresholds.

**Risk of expansion.** The "complete `predicted_direction`"
work was originally scoped to a `'null'` branch + Prediction
bundle + 2-3 supporting changes. Adding the stratified-analog
authoring (items 2-3) is genuinely new framework work. This
section is the entry where step 6 grows from "framework
completion" to "framework completion + RL-correct stratified
analysis primitives." The growth is justified — the methodology
was load-bearing for the substrate's correctness — but should
land as a separate PR sequence from the Prediction bundle PR
to keep code review tractable.

## Anti-list

- **No `PredictionMatch` typed enum.** Display-only
  `Literal[...]` per (5).
- **No `Threshold` newtype** for `Final[float]` coupling.
  Discoverability + grep + code review; not mechanical
  enforcement. See `BRIDGE_AUDIT.md` hoisting discriminator
  (3).
- **No `verdict_from_threshold` in v1.** Carried-forward
  debt per (8).
- **No retroactive substrate-version invariance.** Future
  substrate work.
