"""DQN claims — paper-honest implementations, one per slot role.

Each module contains the primitives that fill one slot in
`theory.dqn_step`. Adding a new alternative implementation is a
single new function in the matching module; the @claim decorator
+ Protocol contract in `types.py` ensure pyright catches signature
mismatches at the swap site.

Module layout:
- `q_network` — parameter init, forward pass.
- `action_select` — rollout policy (ε-greedy + schedule).
- `bootstrap` — Bellman target. **Slot DDQN swaps**.
- `loss` — per-sample TD-error loss.
- `target_sync` — target-network update rule.
- `replay` — FIFO replay buffer init/add/sample.

Re-exports below let consumers `from corroborate.rl.dqn.claims
import mlp_q` rather than the longer module-qualified path."""
from corroborate.rl.dqn.claims.action_select import (
    epsilon_greedy,
    linear_epsilon,
)
from corroborate.rl.dqn.claims.bootstrap import (
    ddqn_bootstrap,
    vanilla_bootstrap,
)
from corroborate.rl.dqn.claims.loss import squared_error
from corroborate.rl.dqn.claims.q_network import init_mlp, mlp_q
from corroborate.rl.dqn.claims.replay import (
    buffer_add,
    buffer_init,
    buffer_sample,
)
from corroborate.rl.dqn.claims.target_sync import periodic_copy

__all__ = [
    'buffer_add',
    'buffer_init',
    'buffer_sample',
    'ddqn_bootstrap',
    'epsilon_greedy',
    'init_mlp',
    'linear_epsilon',
    'mlp_q',
    'periodic_copy',
    'squared_error',
    'vanilla_bootstrap',
]
