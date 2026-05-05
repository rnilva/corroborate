"""Verdict — the primary outcomes of a falsifiable test.

The trichotomy HELD / NO_EFFECT / POWER_INSUFFICIENT encodes the
field-methodologically-critical distinction between "tested and
held," "tested and refuted," and "couldn't tell at this n." This
is the framework's primary contribution at the verdict layer:
**"below MDE" is a first-class verdict**, distinct from
corroboration and refutation. PAPER_NOTES.md §3.4 makes this
load-bearing — at standard published n=3-10 seeds, most outcome
tests are POWER_INSUFFICIENT, and treating that as "no effect" or
"inconclusive" smuggles methodological problems past the reader.

HELD_WITH_SCOPE_FLAG is a refinement of HELD: the population-
level pool excludes zero in the predicted direction (so the claim
corroborates) BUT the random-effects I² indicates effects are
heterogeneous across strata. v9's aggregation reframing makes
this the natural input for empirical scope discovery — meta-
regression on per-stratum g identifies the cleavage axes.
Treating heterogeneous HELD as plain HELD smuggles a uniform-
population claim that the data doesn't support.

INVARIANT_VIOLATION is the orthogonal fifth: a tautological-
tagged invariant rejected, meaning the claim's mechanism didn't
operate (axiom 18: invariants are theorem-direct, not proxy-via-
assumption). The outcome test under that condition is out of
scope, NOT refuted.

RefutationClass is an optional sub-classification — it says WHICH
KIND of NO_EFFECT or POWER_INSUFFICIENT. The framework offers it;
orchestrators may attach or ignore."""
from __future__ import annotations

from enum import Enum


class Verdict(Enum):
    """The primary verdicts of a falsifiable test."""
    HELD = 'held'
    HELD_WITH_SCOPE_FLAG = 'held_with_scope_flag'
    NO_EFFECT = 'no_effect'
    POWER_INSUFFICIENT = 'power_insufficient'
    INVARIANT_VIOLATION = 'invariant_violation'
    INADMISSIBLE = 'inadmissible'

    def is_terminal(self) -> bool:
        """True iff the verdict doesn't require more data to
        resolve. HELD / HELD_WITH_SCOPE_FLAG / NO_EFFECT /
        INVARIANT_VIOLATION / INADMISSIBLE are terminal;
        POWER_INSUFFICIENT is the only verdict that explicitly
        says 'rerun at higher n'."""
        return self is not Verdict.POWER_INSUFFICIENT

    def is_corroboration(self) -> bool:
        """True iff the verdict is positive evidence the claim
        holds at population level. HELD and HELD_WITH_SCOPE_FLAG
        both qualify — the latter additionally signals
        heterogeneity for downstream cleavage discovery."""
        return self in (Verdict.HELD, Verdict.HELD_WITH_SCOPE_FLAG)

    def is_uniform(self) -> bool:
        """True iff the verdict attests to a uniform population-
        level effect (low I²). Distinguishes plain HELD from
        HELD_WITH_SCOPE_FLAG, which corroborates with the caveat
        that effects vary across strata."""
        return self is Verdict.HELD

    def is_refutation(self) -> bool:
        """True iff the verdict is positive evidence the claim
        does NOT hold under conditions where it should. Only
        NO_EFFECT — POWER_INSUFFICIENT means we cannot tell;
        INVARIANT_VIOLATION means the test was out of scope."""
        return self is Verdict.NO_EFFECT


class RefutationClass(Enum):
    """Optional sub-classification of NO_EFFECT or
    POWER_INSUFFICIENT. Orthogonal refinement that says WHY the
    primary verdict landed where it did. Orchestrators may attach
    these or leave them None."""
    # NO_EFFECT refinements (test had adequate power):
    NULL_EFFECT = 'null_effect'           # observed effect ≈ 0
    SIGN_FLIP = 'sign_flip'               # observed effect opposite to predicted

    # POWER_INSUFFICIENT refinements (test could not resolve):
    UNDERPOWERED = 'underpowered'         # n too small for the observed magnitude
    TIME_BUDGET_DORMANT = 'time_budget_dormant'  # effect grows with training; need longer runs
