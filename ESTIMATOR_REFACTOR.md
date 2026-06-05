# Estimator refactor — a single-edge probe on the bootstrap evaluator

## What this is really about

The published result (in review) is a **single-edge causal probe**,
not a new algorithm. Double-DQN's construction supplies the
intervention: its second estimator is a *time-delayed copy of the
online network* rather than an independently trained one, so its
decoupling of selection from evaluation is incomplete — a gap
noted by `nagarajan2025double` (Deep Double Q-learning). Holding
the selector and the acting policy intact and **swapping only the
evaluator** — from the own time-delayed target to an independently
trained network's time-delayed target (`DDQN-indp`) — drives the
residual overestimation at Asterix γ=0.999 from large-positive
through zero to slightly negative (signed bias **+126.9 → −4.9**,
n=30), and removes the harm (discounted return recovers to **15.7**
vs vanilla **12.4**). The bias change is categorical (over- → slight
under-estimation) and travels with the disappearance of the harm,
locally and causally implicating the target-side coupling at this
single environment.

This refactor's job is therefore narrow and precise: make the
bootstrap **evaluator a swappable edge**, so that
`DDQN → DDQN-indp` is *exactly one property changed*, with
everything else — selector, acting policy, target-net stationarity
— held byte-identical. The crude implementation does the right
experiment but hides the edge behind a name-marker + untyped
phases routing + always-packed state; the principled version
relocates the edge to where it varies and deletes the four hacks
as a corollary.

> **Scope discipline (mirrors the paper).** We make **no** claim
> that an independent evaluator is a generally better design, **no**
> claim about other environments, and **no** account of why the
> long horizon is the trigger. The deliverable is the *localisation*
> — where observation could not pin the harm's mechanism, a
> single-edge intervention does. The plan must not over-author
> beyond this (see §5).

> **Design history (rejected dead ends — do not revive).** Earlier
> drafts proposed an `Estimator` / `Gradient` / `UpdateRule`
> dataclass layer, a `paired_double_q` bundle, and elevated a
> "pure 2-net" form as theory-preferred. All rejected: the framework
> shape is *one `@claim` (`dqn`) configured by `partial`,
> intervention = slot/edge swap*, with **no** new top-level bundle
> and **no** slot-key strings. The 2-net is a *confound*, not the
> pure form — see §4d.

---

## 1. The theorem (and its published anchor)

Van Hasselt 2010 and 2016 are two points on one axis: **the
decorrelation quality of the evaluator relative to the selector.**

Shared premise (`bootstrap.py:26`): vanilla DQN's max-bootstrap
overestimates because `max` is convex in `Q`, scaling with the
Jensen floor `σ_Q · √(2 log |A|)` — the invariance gap the
framework measures per env (`jensen_dormancy_gap`,
`hasselt_covariance_gap`). DDQN attacks it by decoupling selection
(online) from evaluation (target), but **the target is a
time-delayed copy of the online net**, so the two estimators stay
*correlated* and overestimation persists. `nagarajan2025double`
names this exactly: Double DQN "explicitly trains only a single
action-value function and does not fully decouple its estimators."
The independent evaluator closes that gap by construction.

The intervention is therefore one edge in the bootstrap target
graph — *which net evaluates the selected action* — and nothing
else. That edge today is hidden; §3 surfaces it.

---

## 2. Why the crude implementation is theory-dishonest

Four hacks (`bootstrap.py:96`, `dqn.py:262`, `phases.py:576` /
`:713`, `state.py:78`) are symptoms of one bug: the edge that
actually changes is invisible to the fingerprint, and a fake one
is fingerprinted instead.

1. **The fingerprint lies.** `double_greedify_indep` is
   byte-identical to `double_greedify`; `arm_key` attributes the
   variation to the *greedification* node, where the computation
   is unchanged. The real edge (evaluator A⁻ → B⁻, plus the
   machinery that trains B) lives in `phases.py` routing that
   never enters `canonical_str`.
2. **Theory reconstructed from a name.** `dqn.py:262` sniffs
   `…greedification is double_greedify_indep` into
   `select_with_independent: bool` — the negation of "`@claim` is
   the single marker that carries a theorem."
3. **Theorem-free branches.** The `if select_with_independent:`
   blocks carry the design as imperative control flow keyed off
   the sniffed bool.
4. **Inert mechanics on every run.** `online_params_2 /
   target_params_2 / opt_state_2` are always packed + init'd so
   the single-net scan stays uniform, then ride dead on every
   vanilla / DDQN run.

---

## 3. The principled intervention — the evaluator as a swappable edge

`dqn` is the single ground. The change is one new hook: the
bootstrap's **evaluator becomes a provided net-source, defaulting
to the unit's own target**:

```python
train_phase(state, ..., evaluator_params=None)   # None → state.target_params (own target)
```

- **vanilla / DDQN-2016** — don't supply it → evaluator = own
  target. Fingerprint **frozen** (default branch, byte-identical).
- **DDQN-indp** — the evaluator is supplied as the time-delayed
  target of an independently trained partner net.

That is the whole edge. `arm_key` reads the change because the
arm binds a non-default evaluator source; vanilla/DDQN bind
nothing new and stay frozen. No name-marker, no slot-key string,
no `Estimator`/`Gradient`/`paired_double_q` bundle.

The selector stays the online net; the acting policy stays the
online net; the evaluator stays a *time-delayed target* — only
*whose* target (own vs independent) changes. One property moves.

---

## 4. How `DDQN-indp` runs (and why it is the 4-net)

### 4a. The edge is single only if stationarity is held constant

DDQN's evaluator A⁻ has two properties: **(i) correlated** with the
selector (it is the online net's own lagged self) and **(ii)
stationary** (frozen for the target-net-update interval). The probe
must move **exactly one**:

| evaluator | (i) correlation | (ii) stationarity | edges moved |
|---|---|---|---|
| DDQN: `A⁻` | correlated | stationary | — |
| **DDQN-indp: `B⁻`** (independent net's **target**) | **independent** | stationary | **1 ✓** |
| 2-net: live `B` (independent net's **online**) | independent | **non-stationary** | 2 ✗ (confound) |

So the evaluator must be the independent partner's **time-delayed
target** `B⁻`, not its live online net. This is exactly the
published `DDQN-indp`, and it is why the experiment needs four nets
(A, A⁻, B, B⁻): the fourth net B⁻ exists *to hold stationarity
constant* while independence is varied.

### 4b. Shared agent, one rollout, one buffer

Critically — A is the **only acting net**, there is **one replay
buffer**, and B is a non-acting learner drawing *independent
minibatches from the same buffer* (`phases.py` already records the
intent: "drawn from the SAME replay buffer … the independence is in
the SAMPLE + init + optimizer, not a separate buffer"). Two full
`dqn_step`s would each run their own rollout + buffer — wrong, and
it would change the result. The correct host shares the agent and
reuses the phase claims:

```python
# PairedDQNState = shared agent (env, obs, replay, rng, step)
#                + A's (online, target, opt) + B's (online, target, opt)
def paired_step(state, ...):
    state, roll = rollout_phase(state, ...)                 # A acts → fills the SHARED replay
    # both learners sample the SAME buffer, independent RNG → independent minibatches
    state = train_a(state, evaluator_params=state.b_target) # y_A = Q_{B⁻}(s', argmax Q_{A_online})
    state = train_b(state, evaluator_params=state.a_target) # y_B = Q_{A⁻}(s', argmax Q_{B_online})
    state = sync_a(state); state = sync_b(state)            # A⁻←A, B⁻←B (periodic_copy)
    return state, roll                                       # behaviour + eval = A only
```

B's extra nets live in `PairedDQNState`, **not** in the single-net
`DQNState` — so the single-net path (vanilla / DDQN) carries zero
inert B-state. The always-packed sin is gone precisely because
`DDQN-indp` is honestly its own program with its own state shape.
Selector held intact (A's online selects in `train_a`); acting
policy held intact (A acts/evals); stationarity held intact (B⁻ is
a periodic copy). Only the evaluator's *source* differs.

### 4c. Relationship to DDQL (`nagarajan2025double`)

This `DDQN-indp` is structurally **DN-DDQL with Double-DQN-style
targets** (`DN-DDQL_DoubleDQN` in their Appendix D.2 / Table 6):
two separate networks, shared buffer with distinct minibatches,
both updated, *online-net selects / partner's target evaluates*.
The variant choices that differ from their tuned `DN-DDQL_DQN` —
online-select (not target-select) and A-only behaviour (not the
½Q₁+½Q₂ average) — are **required by the single-edge design**, not
oversights: target-select would move the selector edge too, and
averaged behaviour would move the acting-policy edge too. We cite
DDQL for the incomplete-decoupling gap and name our variant
precisely; we are not proposing it as a better *algorithm*.

### 4d. The 2-net `train_target` form is a confound, not the pure form

A standalone, single-`dqn` "online + target both trained, evaluate
each other **live**" form is appealing (one slot swap,
`target_sync = train_target`), but per §4a it changes *two*
properties at once (independence **and** stationarity). It is
therefore **not** a clean probe of the evaluator edge and is **not**
the theory-pure improvement an earlier draft claimed. DDQL's
Appendix D.2 independently argues the same point — target-net
stationarity is load-bearing for reciprocal training. Keep the
2-net only as a documented, *separately-scoped* two-edge variant if
ever wanted; it is not part of this probe.

### 4e. Validity note — approximate independence is enough (reviewer rebuttal)

The full double-estimator bias-reduction *guarantee* (tabular van
Hasselt 2010) needs the evaluator statistically **independent** of
the selector's noise, secured there by **dataset partitioning**
(disjoint experience per estimator). The shared buffer + distinct
minibatches here (DDQL's default approximation) gives only
**approximate** independence: A and B see overlapping transitions,
so their errors stay partially correlated and the bias reduction is
*partial*, not the idealized limit.

That does **not** invalidate the probe, because the probe asserts
no limit. It needs only that **B⁻ is more decorrelated from A's
selector than A⁻ is** — which holds with a shared buffer (B has
independent init, independent minibatch draws, its own trajectory).
The edge is "increase evaluator–selector decorrelation"; the result
is the empirical consequence of moving along that axis. The slight
*over*shoot in the result (+126.9 → **−4.9**, through zero to mild
under-estimation rather than exactly zero) is itself consistent
with *partial* — not perfect — decorrelation plus B⁻'s lag. Within
A's step B⁻ is stop-gradient (correct — no cross-gradient leak);
across training B is updated as often as A, so it is a fully
co-trained estimator, not a stale one. The strict-partition
(separate-buffer) DDQL variant (§8 D3) is the knob that would
tighten approximate → exact independence — a future extension, not
a precondition.

### Summary

| arm | selector | acting | evaluator | nets | host |
|---|---|---|---|---|---|
| vanilla | online | online | own target `A⁻` (`max`) | online, target | `dqn_step` |
| DDQN-2016 | online | online | own target `A⁻` (`double`) | online, target | `dqn_step` |
| **DDQN-indp** | online | online | **independent target `B⁻`** | A, A⁻, B, B⁻ | `paired_step` |

Vanilla / DDQN bind nothing new → frozen. `DDQN-indp` binds the
evaluator edge → its own honest `arm_key`.

---

## 5. Deliverable — the localisation, and only that

The principled refactor's deliverable is the **single-environment
causal localisation** that the paper reports: the categorical bias
flip (+126.9 → −4.9) and harm removal (return 15.7 vs 12.4) at
Asterix γ=0.999, attributed to the evaluator edge by a clean
single-edge intervention. That is the result.

The plan must **not** over-author beyond it. In particular, the
earlier speculative cross-env bridge ("where `hasselt_covariance_gap`
is high, an independent evaluator closes more of the Jensen
residual") is **out of scope** — it is precisely the
general-design / cross-env claim the paper disclaims. The framework
*enables* such bridges later; this work does not author them.

What the framework contribution buys here is the methodological
shape: a single-edge structural intervention pins a harm that
observational analysis (Jensen-gap correlations, scope mining)
could not localise. That is the corroborate thesis, applied.

---

## 6. Implementation steps (ordered)

1. **The evaluator hook** — add `evaluator_params: Params | None =
   None` to the bootstrap / `train_phase` path; `None` →
   `state.target_params`. Assert vanilla / DDQN-2016 byte-identical
   (default branch) and `canonical_str` byte-frozen.
2. **greedify role clarity** — confirm `double_greedify` reads
   `estimator(s')[argmax policy(s')]` cleanly against the hook; no
   fingerprint change for `double_greedify`.
3. **`PairedDQNState`** — shared agent (env / obs / replay / rng /
   step) + A's (online, target, opt) + B's (online, target, opt).
   Its own type; the single-net `DQNState` is **not** widened.
4. **`paired_step`** — one `rollout_phase` (A acts → shared
   buffer), two `train_phase` passes (A, B) with cross-injected
   `evaluator_params` and independent minibatch RNG, two
   `sync_phase` (periodic_copy). Behaviour + eval read A only.
   **Hard constraint (D1):** `paired_step` maximally reuses the
   existing `rollout_phase` / `train_phase` / `sync_phase` `@claim`s
   and primitives — it orchestrates them over `PairedDQNState`, it
   does **not** reimplement any phase body.
5. **Delete the four hacks** — `double_greedify_indep`
   (`bootstrap.py:96`), the `dqn.py:262` sniff, the `phases.py`
   `select_with_independent` branches, the `state.py:78` `_2`
   triple. The single-net path becomes inert-B-free.
6. **Measurable migration** — single-net readers
   (`state.online_params` / `target_params`) stay valid; add
   per-unit accessors for `PairedDQNState`; confirm the signed-bias
   / Jensen measurables resolve on the A unit.
7. **Tests** — `uv run pyright && uv run pytest`; LG-SCM /
   deadly-triad analytic suites; frozen-arm_key assertion for
   vanilla / DDQN-2016; the default-hook byte-identical check.

---

## 7. Validation gates

- **Single-edge property holds:** `DDQN → DDQN-indp` changes the
  evaluator source and nothing else — selector, acting policy, and
  target-net stationarity byte-identical (B⁻ is a periodic copy,
  not a live net). Asserted structurally + in test.
- `vanilla` / `DDQN-2016` byte-identical (eval + RNG) vs current;
  `canonical_str` byte-frozen.
- `pyright` strict + `pytest` green on `src/` and `tests/`.
- **Reproduces the published localisation.** `paired_step` at
  Asterix γ=0.999, n=30, reproduces signed bias **+126.9 → −4.9**
  and discounted return **15.7** (vanilla **12.4**) within seed
  noise. Non-negotiable — the result is in review.
- No new bundle, no `isinstance` / `is`-marker dispatch, no `Any` /
  `getattr` on typed values, no always-packed inert state on the
  single-net path (grep clean) — corollaries of §3–§4.
- **No over-authoring:** no cross-env / general-design bridge is
  added; the deliverable is the single-env localisation (§5).

---

## 8. Decisions (settled — planning closed)

- **D1 — `paired_step` location + reuse.** Lives in a sibling
  module `dqn_paired.py`, off the single-net path. **Must maximally
  reuse** the existing `rollout_phase` / `train_phase` /
  `sync_phase` `@claim`s and primitives — orchestration only, no
  phase-body reimplementation. (This is the acceptance condition,
  not just a preference.)
- **D2 — `evaluator_params` plumbing.** Explicit kwarg on the
  bootstrap / `train_phase` path, `default None` (→
  `state.target_params`, own target). Not carried on the bootstrap
  partial. The `None` default keeps vanilla / DDQN byte-identical.
- **D3 — B's selector + buffer (reproduce-first, extend later).**
  The published variant: B selects with its **online** net
  (DoubleDQN-style), evaluated by A⁻; shared buffer, distinct
  minibatches. **Reproduce this exactly first.** Supporting the
  full DDQL family afterward — target-select (`DDQL_DQN`), separate
  buffers (strict partitioning, §4e), double-head architecture — is
  a planned follow-on, gated on the reproduction landing.
- **D4 — Behaviour/eval policy.** A-only (holds the acting policy
  intact — required for the single edge, identical to
  vanilla/DDQN). Do **not** adopt DDQL's ½Q₁+½Q₂ average; that
  moves a second edge.
- **D5 — `B⁻` not live `B` (code comment, load-bearing).** The
  evaluator is the independent net's **time-delayed target** `B⁻`,
  never its live online net — this holds stationarity constant so
  the edge stays single (§4a). The `train_a` call site carries an
  explicit comment to this effect; a "cleanup" to `state.b_online`
  would silently become the 2-net confound and break reproduction.
  *Author's-call recommendation (paper, not code):* tighten the
  prose to "the time-delayed target of an independently trained
  network" so a reviewer can't read it as the live-net confound.
