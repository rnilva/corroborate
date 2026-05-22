# Abstract — `corroborate` (Finding the Frame workshop, RLC 2026)

**Target venue:** [Finding the Frame](https://sites.google.com/view/findingtheframe/home), an RLC 2026 workshop on the philosophy, practice, and formalisms of reinforcement learning. Submission deadline 2026-05-22 (AoE).

**Current draft (v11):**

A theorem's premise can be measured per cell. Once you do, *where does the mechanism fire?* has an empirical answer that scalar-benchmark practice routinely smuggles past. The field's verdicts on RL algorithms — modest, mixed, ablation-conditional — are downstream of this collapse: scalar summaries cannot distinguish where the premise admits from where the chain to outcome breaks.

We present `corroborate`, a framework that operationalises per-cell premise verification by codifying mechanism claims as falsifiable Python programs — falsifiable not only against future data but against the authors' own subsequent analyses, as multiple in-flight walk-backs during the case study attest. Theoretical bounds become first-class upstream edges in a verified causal chain; premise non-activation is the failure of an edge, not a noisy null at outcome. Verdicts compose under a power-aware trichotomy (HELD / REFUTED / POWER_INSUFFICIENT) that propagates underdetermination through chains rather than silently collapsing it. Where the field's per-paper methodologies are non-comparable, the framework provides a shared substrate — registered analyses, common measurables, common verdict semantics — making per-claim findings structurally commensurable.

Double-DQN is the exemplar. Across ten environments and intervention sweeps, the case study replaces the modest-or-universal verdict with a structural account. The Jensen-bias floor (Hasselt 2010) functions empirically as the premise indicator: per-cell premise-activation types each environment as where the bound bites or where it is dormant, and DDQN's bias-reduction mechanism corroborates the theorem exactly where it should. The bias-correction→outcome link is genuinely env-conditional, scoped by the bootstrap fraction; goal-reaching and survival environments exhibit opposite mediator signs under the same intervention. As one concrete curiosity the framework surfaces, in sparse-reward regimes DDQN's outcome direction sign-aligns with a within-episode loop-reduction channel — pointing to mediator candidates beyond the canonical bias-clip path.

---

## Revision history

- **v1**: Lead-cited Chang21/Pearl/Bareinboim; emphasised pre-registration discipline; HP-mixing artifact in body as methodology demo.
- **v2-v5**: Iterative tightening; word count walked from ~330 down to ~290.
- **v6 (broad-investigation pivot)**: Project scale brought in (196 bridges, 56 findings, 13-revision FINDINGS.md); shifted from session-results-only to project-scope framing.
- **v7**: Softened opening; "falsifiable Python programs" replaces "typed claims"; theoretical bound reframed as upstream edge (not scope predicate); four-stage workflow made explicit.
- **v8**: Engaged the modest-or-universal field binary directly; "Jensen-floor functions empirically as the premise indicator" sentence promoted.
- **v9**: Dropped n-step clause; deduplicated Jensen-floor reference between framework claim and empirical anchor; methods parenthetical removed.
- **v10**: Pipeline comma cleaned; `bootstrap_fraction` → "the bootstrap fraction" prose.
- **v11 (post-adversarial-review pivot)**: Inverted structure — paradigmatic claim leads, framework is its operationalisation, case study is evidence. Walkbacks beat merged into the "falsifiable Python programs" sentence with two-directionality ("falsifiable not only against future data but against the authors' own subsequent analyses"). "Modest-or-universal" softened to "modest, mixed, ablation-conditional" (descriptive of literature spread, not strawman binary). Commensurability answer made explicit (registered analyses + common measurables + common verdict semantics). Loop-channel closing light-touched — no p-value, no causal-mediation overreach.

## Author notes on framing choices

- **Chang et al. 2021** is the closest analytical sibling (algorithmic-causal-model-of-learning, modularity criterion via d-separation on ACML). The framework was NOT built from Chang21; citation belongs in §1/§2 framing, not the opening.
- **Pre-registration** is supported by the framework (`BLOCKED_ON`, `EXPECTED` drift detection) but most bridges were authored iteratively rather than strict-pre-registered. Don't promote as headline.
- **Three-verdict trichotomy** (HELD / REFUTED / POWER_INSUFFICIENT) is framework-original structural commitment — `_stamp_level` dispatches on it, `cluster_verdict` composes it. Novelty pins to *propagation through chains* (TOST already exists at single-test level; what's new is composed propagation refusing silent collapse).
- **Theoretical bound as upstream edge** is the *principled framing* of what the framework currently implements as scope predicate (`ddqn_refuted_when_dormancy_fires`). The current implementation is functionally equivalent; the abstract presents the principled form the in-flight refactor will converge on. Novelty pins to *runnable operationalisation* (per-cell measurable + auto-stamp verdict), not framing — prior causal-RL and offline-RL work types assumptions but doesn't operationalise them this way.
- **Loop-reduction channel finding** light-touched in the closing sentence — sign-alignment at 7/8 envs is correlational evidence with a fragile scope-filter dependency (different filter today gives p=0.145). Mediator-candidate framing is the honest level.
- **σ_Λ_a / HP-mixing-artifact walk-back** is folded into the walkbacks beat in P2 rather than detailed in the abstract — specific in-flight walk-backs don't headline well, but framework-caught-its-own-walkbacks does.
- **Falsifiable two-directionality** ("not only against future data but against the authors' own subsequent analyses") is the conceptual core of the framework-caught-its-own-walkbacks beat — the framework's typing makes the *author's own past claims* falsifiable across revisions of their own analyses.

## Open questions for the next pass

1. Title not yet locked. Candidates:
   - *Falsifiable mechanism claims as code: typed causal-graph verification for reinforcement-learning algorithms*
   - *The chain is the claim: causal-graph verification of RL mechanism claims, applied to Double-DQN*
   - *From scalar benchmarks to structural accounts: corroborate, a framework for empirical claim verification in reinforcement learning*
   - *Where does the mechanism fire? Per-cell premise verification for RL algorithmic claims*

2. Whether to keep the "ten environments" anchor or drop. Currently kept once.

3. Whether to mention agent-enablement (authoring-labor collapse via coding assistants) in the abstract or only in §4 / Discussion. Currently held back.

4. Whether to compress "registered analyses, common measurables, common verdict semantics" to two items for cleaner cadence.
