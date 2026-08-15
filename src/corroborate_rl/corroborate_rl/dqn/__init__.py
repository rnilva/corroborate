"""DQN implementation — `@claim`-decorated free functions + frozen-
dataclass config bundles composed by `dqn` / `dqn_step`.

The side-effect import below is the implementation's contract: importing
`corroborate_rl.dqn` registers the implementation's named measurables
(jensen_gap, log_mc_variance_per_burst, etc.) into the framework's
measurable registry, so analyses targeting them by name resolve at
call time. Implementation consumers who import the implementation get
registration for free; the framework's implementation-neutral analyses
no longer reach into `corroborate_rl.dqn` for side-effect
registration."""
from corroborate_rl.dqn import measurables  # noqa: F401  # registers DQN measurables on implementation import

__all__ = ['measurables']
