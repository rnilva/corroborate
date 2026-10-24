"""Hypothesis — the verdict-time contract for an authored evidence graph.

A module or class exposes ``BRIDGES``, ``FINDINGS`` and ``__name__``.
Each bridge declares its own source: an observational quantity or a
``DoEffect`` contrast. A hypothesis can therefore combine observations
and several interventions without choosing one global intervention.

Executing experiments is a separate contract. ``run_intervention``
takes an explicit ``DoEffect`` and a row runner; evaluating the resulting
records does not execute that experiment. Existing modules may retain
an ``INTERVENTION`` constant for their own sweep code, but verdict-time
consumers neither require nor consult it.

Measurable dependencies belong to bridges and the measurable registry.
The hypothesis need not repeat them; optional ``REQUIRED_MEASURABLES``
adds quantities requested for exploration or future claims.
"""
from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Literal,
    Protocol,
    runtime_checkable,
)

from corroborate._internals.canonical import canonical_str

if TYPE_CHECKING:
    from corroborate.bridge.bridge import Bridge
    from corroborate.core.finding import Finding


__all__ = ['Hypothesis', 'PredictedDirection', 'canonical_str']


type PredictedDirection = Literal['a_gt_b', 'a_lt_b', 'two_sided', 'null']
"""Author-declared *prior* sign of the predicted treatment-vs-
baseline effect.

- `'a_gt_b'` predicts the intervention's arm exceeds the baseline.
- `'a_lt_b'` predicts below.
- `'two_sided'` predicts non-zero in either direction.
- `'null'` predicts no effect — the pytest-`xfail` analog. The
  bridge author declares "I expect the data NOT to show an
  effect"; HELD then means the null prediction was confirmed
  (small |g|, no detectable arm effect). NO_EFFECT means an
  effect WAS observed when none was predicted (the unexpected-
  pass / xpass analog). This is the canonical encoding for
  link-broken bridges (mech HELD ↛ outcome HELD) and for
  mechanism-attenuation bridges (e.g., DDQN no longer reduces
  bias when n_step → MC).

Convention: `HELD` always means "prediction confirmed" — the
mapping between `(predicted_direction, observed g)` and Verdict
is uniform across all four directions. The reader scanning
HELD/NO_EFFECT doesn't have to track which-prediction-which-
direction; the bridge body's threshold logic encodes it once.

Per-bridge metadata: `Bridge.predicted_direction` carries it for
the analysis the bridge consumes. Distinct from
`graph.causal.Direction` — that's the *observed* sign (DIRECT /
INVERSE) inferred post-hoc from a stat's value."""


@runtime_checkable
class Hypothesis(Protocol):
    """The framework's typed verdict-time hypothesis contract.

    Conforming objects expose three attributes:

    - `BRIDGES: tuple[Bridge, ...]` — the authored verdict
      declarations.
    - `FINDINGS: tuple[Finding, ...]` — cluster-shaped claims
      authored against the post-evaluated graph (empty tuple if
      none). See `corroborate.core.finding.Finding`.
    - `__name__: str` — the Python identity attribute. Modules
      carry their dotted path; classes carry their bare name. The
      runner uses it for cache-path defaults and display; both
      module and class shapes carry it for free.

    Modules and classes both satisfy the Protocol structurally
    via attribute access. The framework's verdict-time runner
    reads `BRIDGES`. An experiment runner receives its actual
    intervention separately; bridges need not share one contrast.

    Optional `CLAIM` attribute (`Claim[..., object] | None`):
    the implementation's outermost @claim function — the structural
    truth for endogeneity gating (cf. ENDOGENEITY_TOPOLOGY.md).
    The runner reads it via `getattr(h, 'CLAIM', None)` and
    threads to `evaluate(..., claim=...)`. Hypotheses that omit
    it fall back to `None`; the gates short-circuit on the
    endogeneity check (still correct for typo/contract-shape
    gates). New implementation hypothesis modules should declare
    `CLAIM = dqn` (or their implementation's outermost claim) at
    module level.

    Optional `MODULE_SCOPE` attribute (`pl.Expr | None`): a
    hypothesis-module-level scope filter that AND-combines with
    each bridge's own `scope=` at evaluation time. Used to
    encode universe-level exclusions — e.g., "every cross-env
    bridge in this file excludes bsuite diagnostic chains
    (DiscountingChain bandit-by-step-0, DeepSea one-arrow
    optimum, etc.) because the chain-amplifier theory doesn't
    apply to those env shapes". The runner reads it via
    `getattr(h, 'MODULE_SCOPE', None)` and threads to
    `evaluate(..., module_scope=...)`. Hypotheses that omit it
    pass None → each bridge's scope stands alone. Bridges that
    legitimately violate the module-level filter must live in a
    different hypothesis module — there's no per-bridge opt-out
    by design (a hypothesis module's scope universe is
    file-level, not bridge-level).

    Optional `REQUIRED_MEASURABLES` attribute
    (`tuple[str, ...]`): explicit hatch for "compute these
    measurables during `--ingest` even though no bridge names
    them yet." The runner unions this set with the
    bridge-derived `required` (from `measurable_names_for_bridges`)
    so the persisted `measurements.parquet` carries them at the
    next build. Use cases the bridge-only required-set doesn't
    cover:

    - Exploration: look at a measurable's distribution before
      authoring the bridge that consumes it (chicken-and-egg
      otherwise — bridges define what's required, but you need
      the data to know which bridge makes sense).
    - Pre-population: stage measurables for future analyses so
      re-ingest cost doesn't gate the work.
    - Per-burst PC / discovery: PC walks a panel of candidate
      variables; the variables don't have bridges yet by
      definition.

    Names listed here MUST be registered `@measurable` functions
    — `_validate_hypothesis` raises if any are unknown to the
    registry. Silently dropping (the behavior for unrecognized
    bridge names) is wrong here: an explicit author declaration
    should fail loud on typos."""

    __name__: str
    BRIDGES: 'tuple[Bridge, ...]'
    FINDINGS: 'tuple[Finding, ...]'
    """Cluster-shaped claims authored against the hypothesis's
    post-evaluated graph. Each Finding declares which subset of
    `BRIDGES` it claims about + an expected verdict; the
    framework's `evaluate_finding(f, g)` returns the actual verdict
    via `composed_verdict`, and drift surfaces in the
    `run_hypothesis.py` rollup. Required on the Protocol so
    pyright catches "I forgot to declare FINDINGS"; declare `()`
    if the hypothesis has no findings yet."""
