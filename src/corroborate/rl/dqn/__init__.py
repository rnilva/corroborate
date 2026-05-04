"""DQN substrate — `@claim`-decorated free functions + frozen-
dataclass config bundles composed by `dqn` / `dqn_step`.

The side-effect import below is the substrate's contract: importing
`corroborate.rl.dqn` registers the substrate's named measurables
(jensen_gap, log_mc_variance_per_burst, etc.) into the framework's
measurable registry, so analyses targeting them by name resolve at
call time. Substrate consumers who import the substrate get
registration for free; the framework's substrate-neutral analyses
no longer reach into `corroborate.rl.dqn` for side-effect
registration."""
from corroborate.rl.dqn import measurables  # noqa: F401  # registers DQN measurables on substrate import

__all__ = ['measurables']
