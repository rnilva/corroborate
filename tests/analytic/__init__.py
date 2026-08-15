"""Analytic implementations for testing corroborate's framework primitives
against closed-form ground truth.

Two siblings (built incrementally; only `lg_scm/` is present at the
first slice):

- `lg_scm/` — Linear-Gaussian SCM. Closed-form means and Δ on
  paired-arm contrasts. Used to assert analysis primitives
  (`paired_g`, `paired_link_per_burst`, `meta_regression_*`,
  `tautology_audit`, `dowhy`, ...) recover structural coefficients
  within an analytical SE bound.

- `mdp/` — closed-form 2–3-state MDP (planned). Q* and MC computable
  in closed form; two arms (vanilla bias `b` vs faithful Q*). Used
  to assert the framework's three-verdict architecture
  (mech / link / outcome) and `POWER_INSUFFICIENT` discrimination
  behave as the contributors intend.

Pattern: implementation produces `RunRow` instances whose `measurements`
are closed-form functions of a small set of structural coefficients;
tests run framework primitives on `[r.as_dict() for r in rows]` and
assert against the closed form within an analytical SE bound. When a
test fails, the message names the analytical truth so the bug points
clearly at the framework code path under test, not at the substrate.
"""
