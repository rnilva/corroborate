"""Override pytree for `init_state` resumption from a saved cell.

The substrate accumulates "resumable state" fields one at a time:
online_params first (a89fe86 + ef2b889), with target_params, replay
buffer, optimizer state, step counter, and RNG state queued by
CHECKPOINT_RESUME_DESIGN.md §2. Threading each as a separate
optional kwarg on `dqn()`, `init_state()`, `run_dqn_arm`, and
`DQNRunner` balloons the signature with kwargs that operationally
always travel together — and the vmap `in_axes` spec grows
unreadable past three fields.

`InitOverride` is the typed bundle the design selected: a frozen
dataclass with `T | None` fields per resumable. The default
`InitOverride()` and `None` itself both preserve fresh-init
semantics — every field-presence check is one `is not None` branch
inside `init_state`. Adding a new resumable in Phase 3 is one line
on this dataclass + one branch in `init_state`; no surgery on
the cell_runner / DQNRunner / YAML loader.

Registered as a JAX pytree via `jax.tree_util.register_dataclass`
so `jax.vmap(..., in_axes=InitOverride(online_params=0, ...))`
threads each field's seed axis (or None for broadcast / absent
fields) — the canonical batched-resume call shape.

Four-question test (CLAUDE.md "When to introduce a primitive"):

1. Typed contract — yes. `online_params: Params | None` etc. are
   the same shapes `DQNState` carries; pyright checks at every
   call site.
2. Runtime narrowing — yes. Frozen-dataclass instance attributes
   are statically resolved; `is None` narrows each branch.
3. Real work beyond labeling — yes. Holds the batched pytree under
   `jax.vmap`, single point where new resumables get added.
4. Performance floor — neutral; one allocation per cell."""
from __future__ import annotations

from dataclasses import dataclass

import jax

from corroborate_rl.dqn.claims.q_network import Params


@dataclass(frozen=True, slots=True)
class InitOverride:
    """Bundle of "use this instead of fresh init" leaves for
    `init_state`. Each field is `T | None`; `None` means "use the
    fresh-init code path." The default constructor (all-None) is
    equivalent to passing no override.

    Phase 1 covers `online_params` (the existing
    `init_online_params` path) plus `target_params` (which Phase 2
    decouples from online — the design's only confirmed near-term
    need). Phase 3 will extend with `opt_state`, `replay`, `step`,
    `rng_key` as the substrate needs each capability."""
    online_params: Params | None = None
    target_params: Params | None = None


# Registered as a JAX pytree so vmap's `in_axes` accepts
# `InitOverride(online_params=0, target_params=0)` for seed-axis
# threading. Both fields are data (potential `jax.Array` leaves);
# no meta-fields (the dataclass carries no static markers).
_ = jax.tree_util.register_dataclass(
    InitOverride,
    data_fields=['online_params', 'target_params'],
    meta_fields=[],
)


__all__ = ['InitOverride']
