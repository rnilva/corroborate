"""Test fixture: a module that registers a measurable ONLY via
`register_as` (no plain `@measurable`).

Mirrors `corroborate_rl/dqn/trace_reductions.py` — the lone
implementation module with this shape — to exercise the
`registry_source_modules()` completeness gap found reviewing
`fix/ingest-fork-deadlock`: a `register_as` alias carries the
factory's `fn.__module__` (`corroborate.measurables.reductions`),
NOT this module, so without recording the aliasing caller this
module is absent from the forkserver / spawn re-import set and its
alias null-pads in a fresh worker. Importing this module registers
the alias as a side effect (the registration discipline)."""
from __future__ import annotations

from corroborate.measurables import from_key, register_as
from corroborate.measurables.reductions import reduce_axis

# A stable hand-picked name bound to a factory composition — the
# exact shape of `trace_reductions.py`'s q-per-step aliases.
ALIAS_NAME = '_fork_fixture_register_as_only_alias'
SOURCE_KEY = '_fork_fixture_per_action_signal'

register_as(
    reduce_axis(from_key(SOURCE_KEY), axis=-1, op='mean'),
    name=ALIAS_NAME,
)
