"""Core — substrate-author daily imports.

The framework's most-imported types and decorators live here:

- `core.claim` — `@claim` decorator, `FnClaim` wrapper, `Claim`
  Protocol, `record_call`, `trace_context`, `CallRecord`,
  `is_claim`.
- `core.signature` — walker (`walk`, `walk_paths`, `flatten_*`),
  `ClaimSignature` / `KwargInfo` records, `Exogenous` marker.
- `core.hypothesis` — `Hypothesis` typed-record + the
  `PredictedDirection` literal.
- `core.intervention` — `Intervention` (do-effect surgery), the
  `Replacement` runtime union, `combined_arm_key`,
  `apply_interventions`, `DoEffect`, `is_replacement`.
- `core.loop` — `Loop` Protocol + `iterate` driver +
  `python_loop` (a non-JAX reference loop).

Consumers `from corroborate.core import X` for the substrate-
author surface."""
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
    DoEffect,
    Intervention,
    Replacement,
    apply_interventions,
    combined_arm_key,
    is_replacement,
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
    Regime,
    canonical,
    flatten_exogenous,
    flatten_leaves,
    walk,
    walk_paths,
)

__all__ = [
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
    'Regime',
    'Replacement',
    'apply_interventions',
    'canonical',
    'claim',
    'combined_arm_key',
    'flatten_exogenous',
    'flatten_leaves',
    'is_claim',
    'is_replacement',
    'iterate',
    'python_loop',
    'record_call',
    'trace_context',
    'walk',
    'walk_paths',
]
