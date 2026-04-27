# corroborate

Framework for executable scientific claims — intervention study,
falsification, causal discovery.

A scientific claim is an honest, executable program. `corroborate`
provides the substrate for three composable capabilities on top of
that program:

- **(a) Intervention study** — `partial(theory, mechanism=alternative)`
  literally re-runs the system with a swapped `@claim`. Active
  intervention, not observational reconstruction.
- **(b) Falsification** — verdict trichotomy (`HELD`, `NO_EFFECT`,
  `POWER_INSUFFICIENT`) with explicit MDE + power tracking.
  "Below MDE" is a first-class verdict, distinct from both
  confirmation and refutation.
- **(c) Causal discovery** — measurement-graph structure from a
  trace, with Pearl-tier and direction.

These three are substrate, not application. A dialectic loop
(Improve + Falsify across cycles) is one composition; a one-shot
reproducibility audit is another; a pure data-mining run on an
existing corpus is another. All three reuse the same primitives.

## Status

Pre-v0. The repo's first acceptance test is the DDQN study
documented in `PAPER_NOTES.md` §3 — mechanism HELD ↛ outcome
HELD ↛ link HELD across 17 envs, with the methodological
contribution living in keeping these three verdicts separate.
