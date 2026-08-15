"""Core — implementation-author daily imports.

The framework's most-imported types and decorators:

- `core.claim` — `@claim` decorator, `FnClaim` wrapper, `Claim`
  Protocol, `record_call`, `trace_context`, `CallRecord`,
  `is_claim`.
- `core.signature` — walker (`walk`, `walk_paths`,
  `flatten_leaves`, `flatten_exogenous`), `ClaimSignature` /
  `KwargInfo` records, `Exogenous` marker.
- `core.hypothesis` — `Hypothesis` typed-record + the
  `PredictedDirection` literal.
- `core.intervention` — `Intervention` (do-effect surgery),
  `DoEffect`, `Replacement` runtime union.
- `core.loop` — `Loop` Protocol + `iterate` driver +
  `python_loop` (non-JAX reference).

Public surface re-exported here: decorators / types /
walker / Hypothesis / Intervention / Loop. Internal helpers
(`apply_interventions`, `combined_arm_key`, `is_replacement`,
`canonical`, `Regime`) and the lower-level walker primitives
live on the submodule path."""
from corroborate._internals.canonical import canonical_str
from corroborate.core.claim import (
    CallRecord,
    Claim,
    FnClaim,
    claim,
    is_claim,
    record_call,
    trace_context,
)
from corroborate.core.hypothesis import (
    Hypothesis,
    PredictedDirection,
)
from corroborate.core.intervention import (
    ArmRole,
    DoEffect,
    Intervention,
    Replacement,
)
from corroborate.core.loop import (
    Loop,
    iterate,
    python_loop,
)
from corroborate.core.signature import (
    ClaimSignature,
    Exogenous,
    KwargInfo,
    flatten_exogenous,
    flatten_leaves,
    walk,
    walk_paths,
)

__all__ = [
    'ArmRole',
    'CallRecord',
    'Claim',
    'ClaimSignature',
    'DoEffect',
    'Exogenous',
    'FnClaim',
    'Hypothesis',
    'Intervention',
    'KwargInfo',
    'Loop',
    'PredictedDirection',
    'Replacement',
    'canonical_str',
    'claim',
    'flatten_exogenous',
    'flatten_leaves',
    'is_claim',
    'iterate',
    'python_loop',
    'record_call',
    'trace_context',
    'walk',
    'walk_paths',
]
