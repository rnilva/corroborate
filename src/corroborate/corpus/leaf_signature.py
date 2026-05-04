"""Leaf-signature projection — the configurational fingerprint
used as a group-by key.

A leaf-regime kwarg is a non-recursive scalar claim of the
configured composition, observed at composition time. The
fingerprint filters out registered-measurable outputs, framework-
typed metadata, and substrate-supplied exogenous keys; what
remains is the configurational leaves at their dotted topology
paths. RL practice calls these hyperparameters; the framework
name is `leaf` (HP leaks domain jargon into framework semantics).

Hashable; suitable as a group-by key for cross-arm aggregations
that need to partition on identical configuration."""
from __future__ import annotations

from collections.abc import Mapping

from corroborate.corpus.schema import MeasurementLeaf


# Legacy-corpus exclusion. Pre-Phase-6 RunRows carried an
# `intervention_name` column (the old substrate-chosen arm-name
# string); new corpora use the framework-typed `arm_key`
# attribute on `RunRow` instead, and `arm_key` lives outside
# `measurements` so it never reaches `leaf_signature`. This
# exclusion stays as a guard for legacy parquets that still
# carry the column inside `measurements`.
_FRAMEWORK_EXCLUDED_KEYS: frozenset[str] = frozenset({
    'intervention_name',
})


def leaf_signature(
    measurements: Mapping[str, MeasurementLeaf],
    *,
    exogenous_keys: frozenset[str] = frozenset(),
) -> tuple[tuple[str, str], ...]:
    """The configurational fingerprint — leaf-only subset of
    `measurements` as a sorted (path, str-canonical-value) tuple.
    Hashable; suitable as a group-by key.

    Filters out:
    - Every registered-measurable name (the registry is the
      single source of truth post-Phase-5; substrate-paper-
      narrative prefixes are gone).
    - Legacy `intervention_name` (excluded for old parquets that
      still carry the column; new corpora use `arm_key`).
    - Substrate-supplied exogenous keys: keys the substrate
      declared via `Annotated[T, Exogenous]` on its `@claim`'s
      kwargs. Caller passes those names as `exogenous_keys` (e.g.
      `frozenset({'env_name', 'seed', 'total_steps'})` for the RL
      substrate). The framework does NOT hardcode RL key names.

    What remains is the configurational leaves at their dotted
    topology paths. "Leaf" rather than "HP": a leaf-regime kwarg
    is a non-recursive scalar claim of the configured composition,
    observed at composition time."""
    from corroborate.measurables import registered_names
    excluded = (
        _FRAMEWORK_EXCLUDED_KEYS
        | exogenous_keys
        | frozenset(registered_names())
    )
    return tuple(sorted(
        (k, str(v))
        for k, v in measurements.items()
        if k not in excluded
    ))


__all__ = ['leaf_signature']
