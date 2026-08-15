"""Per-env jumanji factories registered into the env catalogue.

Each factory is a typed closure: it lazily imports `jumanji` (and
the env's per-env `Observation` NamedTuple) inside the closure
body so module import doesn't construct any jumanji env. Metadata
(`n_actions`, `observation_shape`, `horizon`) is passed explicitly
to `_register_jumanji` so registration likewise doesn't trigger
env construction.

**Why lazy.** Some jumanji envs (Sokoban-v0) trigger network
calls on first instantiation — Sokoban downloads its level
dataset from HuggingFace Hub via `hf_hub_download`. With the
factory called at module-import time (the prior shape), every
implementation import fired the HF Hub download regardless of whether
any jumanji cell would run. The lazy form keeps implementation import
free of jumanji's network behaviour; jumanji is only imported
when `make_env(env_spec)` actually constructs a jumanji-backed
env.

To add a new jumanji env: write a factory `_make_<env>()` that
lazy-imports jumanji at the top of its body, then call
`_register_jumanji('<env>-jumanji', factory=_make_<env>,
n_actions=K, observation_shape=(...), horizon=H, ...)`. The
`-jumanji` suffix distinguishes our backend from gymnax names.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from gymnax.environments.environment import EnvParams as GymnaxEnvParams

from corroborate_rl.env_catalogue import _register_jumanji, image_bucket_hash
from corroborate_rl.jumanji_adapter import JumanjiEnv


def _make_snake_v1() -> tuple[JumanjiEnv[object, object], GymnaxEnvParams]:
    """Snake-v1: 12×12×5 grid, 4 actions, horizon 4000.

    `Observation` is a NamedTuple of `(grid, step_count,
    action_mask)`; `obs_extract` projects to `grid` only — the
    step_count is carried in the dqn state separately, action_mask
    is unused (DQN doesn't gate by mask).

    SURVIVE polarity: episode terminates on snake-self-collision
    or wall hit; reward is +1 per fruit eaten (per-step positive),
    so longer episodes accumulate more reward.
    """
    import jumanji
    from jumanji.environments.routing.snake.types import (
        Observation as SnakeObservation,
    )

    inner = jumanji.make('Snake-v1')

    def obs_extract(obs: SnakeObservation) -> jax.Array:
        return obs.grid

    adapter: JumanjiEnv[object, object] = JumanjiEnv(
        inner=inner,
        # The dataclass is generic but the closure only sees Snake
        # observations; we erase to `object` at the container
        # boundary. The closure body remains typed against
        # `SnakeObservation` because the local import binds it.
        obs_extract=obs_extract,
        obs_shape=(12, 12, 5),
        n_actions=4,
    )
    params = GymnaxEnvParams(max_steps_in_episode=4000)
    return adapter, params


_register_jumanji(
    'Snake-jumanji',
    factory=_make_snake_v1,
    n_actions=4,
    observation_shape=(12, 12, 5),
    horizon=4000,
    r_min=0.0,
    r_max=1.0,
    reward_regime='per_step',
    solve_threshold=None,
    solve_threshold_source='no-canonical-criterion',
    solve_threshold_confidence='absent',
)


def _make_pacman_v1() -> tuple[JumanjiEnv[object, object], GymnaxEnvParams]:
    """PacMan-v1: 31×28 maze, 5 actions (NOOP + 4 dirs), time_limit=1000.

    `obs_extract` builds a 5-channel grid `(31, 28, 5)`:
      ch0 = walls (1=wall, 0=floor) — `obs.grid`
      ch1 = pacman position — scatter at (player.y, player.x)
      ch2 = ghost positions — scatter at (ghost[:, 0], ghost[:, 1])
      ch3 = remaining pellets — scatter at (pellet[:, 1], pellet[:, 0])
            because `pellet_locations` uses (col, row) = (x, y)
            convention, distinct from `player_locations.{y, x}` (the
            jumanji viewer code at routing/pac_man/viewer.py confirms
            this — pellet/power-up positions index as `[p[1], p[0]]`).
      ch4 = power-up positions — scatter at (power[:, 1], power[:, 0])

    Eaten pellets reset to (0, 0) inside `pellet_locations`; we
    leave them in the scatter (0,0 is the top-left wall corner so
    a stray scatter there is harmless even though logically wrong).
    The agent learns from the EVOLVING set of ones in ch3 — fewer
    bright cells = more pellets eaten = closer to clearing.

    SURVIVE polarity: episode ends on ghost contact (death,
    `state.dead`) or `step_count >= time_limit` (timeout). +10
    reward per pellet eaten, larger reward for power-ups + scared
    ghosts. Longer episodes accumulate more reward.
    """
    import jumanji
    from jumanji.environments.routing.pac_man.types import (
        Observation as PacManObservation,
    )

    inner = jumanji.make('PacMan-v1')
    h, w = 31, 28

    def obs_extract(obs: PacManObservation) -> jax.Array:
        walls = obs.grid.astype(jnp.float32)

        pacman = (
            jnp.zeros((h, w), dtype=jnp.float32)
            .at[obs.player_locations.y, obs.player_locations.x]
            .set(1.0)
        )

        ghost_y = obs.ghost_locations[:, 0]
        ghost_x = obs.ghost_locations[:, 1]
        ghosts = (
            jnp.zeros((h, w), dtype=jnp.float32)
            .at[ghost_y, ghost_x]
            .add(1.0)
        )

        # `pellet_locations[i]` is (x=col, y=row); see docstring.
        pellet_x = obs.pellet_locations[:, 0]
        pellet_y = obs.pellet_locations[:, 1]
        pellets = (
            jnp.zeros((h, w), dtype=jnp.float32)
            .at[pellet_y, pellet_x]
            .add(1.0)
        )

        # Power-ups follow the same convention as pellet_locations.
        power_x = obs.power_up_locations[:, 0]
        power_y = obs.power_up_locations[:, 1]
        powerups = (
            jnp.zeros((h, w), dtype=jnp.float32)
            .at[power_y, power_x]
            .add(1.0)
        )

        return jnp.stack([walls, pacman, ghosts, pellets, powerups], axis=-1)

    adapter: JumanjiEnv[object, object] = JumanjiEnv(
        inner=inner,
        obs_extract=obs_extract,
        obs_shape=(h, w, 5),
        n_actions=5,
    )
    params = GymnaxEnvParams(max_steps_in_episode=1000)
    return adapter, params


# Pacman state_hash via random projection (per
# `image_bucket_hash` docstring, recommended for high-resolution
# image obs where downsample-pool would be too coarse). 4^4=256
# buckets — same default as other image-obs envs.
_PACMAN_HASH, _PACMAN_CARD = image_bucket_hash(
    (31, 28, 5), n_proj_dims=4, n_buckets_per_dim=4,
)

_register_jumanji(
    'PacMan-jumanji',
    factory=_make_pacman_v1,
    n_actions=5,
    observation_shape=(31, 28, 5),
    horizon=1000,
    r_min=0.0,
    r_max=200.0,
    reward_regime='per_step',
    state_hash=_PACMAN_HASH,
    state_hash_cardinality=_PACMAN_CARD,
    solve_threshold=None,
    solve_threshold_source='no-canonical-criterion',
    solve_threshold_confidence='absent',
)


def _make_game2048() -> tuple[JumanjiEnv[object, object], GymnaxEnvParams]:
    """Game2048-v1: 4×4 board, 4 actions (up/down/left/right tile shift).

    `obs_extract` projects the int32 board to a single-channel float
    grid `(4, 4, 1)`. Tile values are powers of 2 (0 for empty),
    which span [0, 2^15] — we log-scale to keep the input
    well-conditioned for CNN: `log2(board + 1) / 16` clips to
    [0, ~1] for typical play. The channel-1 axis is added for
    CNN compatibility (matches Snake/PacMan shape convention).

    SURVIVE polarity: episode terminates when no valid moves
    remain; reward per step is the sum of merged tile values
    (positive). Longer games accumulate more reward. No
    time_limit — purely terminate-on-stuck.
    """
    import jumanji
    from jumanji.environments.logic.game_2048.types import (
        Observation as Game2048Observation,
    )

    inner = jumanji.make('Game2048-v1')

    def obs_extract(obs: Game2048Observation) -> jax.Array:
        # log2-scale to compress the wide tile-value range; +1
        # avoids log(0). Divide by 16 to keep most values in [0, 1].
        board = obs.board.astype(jnp.float32)
        scaled = jnp.log2(board + 1.0) / 16.0
        return scaled[..., None]

    adapter: JumanjiEnv[object, object] = JumanjiEnv(
        inner=inner,
        obs_extract=obs_extract,
        obs_shape=(4, 4, 1),
        n_actions=4,
    )
    # Game2048 has no canonical horizon; pick a generous cap.
    params = GymnaxEnvParams(max_steps_in_episode=2000)
    return adapter, params


_register_jumanji(
    'Game2048-jumanji',
    factory=_make_game2048,
    n_actions=4,
    observation_shape=(4, 4, 1),
    horizon=2000,
    r_min=0.0,
    r_max=4096.0,  # max merge reward in practice; theoretical
    # bound is 2^16 but rarely approached.
    reward_regime='per_step',
    solve_threshold=None,
    solve_threshold_source='no-canonical-criterion',
    solve_threshold_confidence='absent',
)


def _make_maze() -> tuple[JumanjiEnv[object, object], GymnaxEnvParams]:
    """Maze-v0: 10×10 maze, 4 actions, REACH polarity (reach target).

    `obs_extract` builds a 3-channel grid `(10, 10, 3)`:
      ch0 = walls (boolean → float)
      ch1 = agent position one-hot
      ch2 = target position one-hot

    REACH polarity: reward +1 on reaching target, 0 otherwise.
    Episode terminates on success or `step_count >= time_limit`.
    """
    import jumanji
    from jumanji.environments.routing.maze.types import (
        Observation as MazeObservation,
    )

    inner = jumanji.make('Maze-v0')
    h, w = 10, 10

    def obs_extract(obs: MazeObservation) -> jax.Array:
        walls = obs.walls.astype(jnp.float32)
        agent = (
            jnp.zeros((h, w), dtype=jnp.float32)
            .at[obs.agent_position.row, obs.agent_position.col]
            .set(1.0)
        )
        target = (
            jnp.zeros((h, w), dtype=jnp.float32)
            .at[obs.target_position.row, obs.target_position.col]
            .set(1.0)
        )
        return jnp.stack([walls, agent, target], axis=-1)

    adapter: JumanjiEnv[object, object] = JumanjiEnv(
        inner=inner,
        obs_extract=obs_extract,
        obs_shape=(h, w, 3),
        n_actions=4,
    )
    params = GymnaxEnvParams(max_steps_in_episode=100)
    return adapter, params


_register_jumanji(
    'Maze-jumanji',
    factory=_make_maze,
    n_actions=4,
    observation_shape=(10, 10, 3),
    horizon=100,
    r_min=0.0,
    r_max=1.0,
    reward_regime='terminal_only',
    solve_threshold=None,
    solve_threshold_source='no-canonical-criterion',
    solve_threshold_confidence='absent',
)


def _make_sokoban() -> tuple[JumanjiEnv[object, object], GymnaxEnvParams]:
    """Sokoban-v0: 10×10×2 grid, 4 actions, REACH polarity.

    `obs_extract` is the identity (uint8 → float32) — the env
    already provides a 2-channel grid (player+wall layer / box+target
    layer in jumanji's convention).

    Reward: -0.1 per step + +1 per box pushed onto target. Episode
    terminates on all-boxes-on-target or `step_count >= time_limit`.

    **Network call on first construction.** `jumanji.make('Sokoban-v0')`
    downloads its level dataset from HuggingFace Hub. The factory
    is intentionally lazy (called only by `make_env`); registration
    uses explicit metadata so implementation import doesn't trigger the
    download.
    """
    import jumanji
    from jumanji.environments.routing.sokoban.types import (
        Observation as SokobanObservation,
    )

    inner = jumanji.make('Sokoban-v0')

    def obs_extract(obs: SokobanObservation) -> jax.Array:
        return obs.grid.astype(jnp.float32)

    adapter: JumanjiEnv[object, object] = JumanjiEnv(
        inner=inner,
        obs_extract=obs_extract,
        obs_shape=(10, 10, 2),
        n_actions=4,
    )
    params = GymnaxEnvParams(max_steps_in_episode=120)
    return adapter, params


_register_jumanji(
    'Sokoban-jumanji',
    factory=_make_sokoban,
    n_actions=4,
    observation_shape=(10, 10, 2),
    horizon=120,
    r_min=-12.0,  # max -0.1 × 120 steps
    r_max=10.0,
    reward_regime='shaped',
    solve_threshold=None,
    solve_threshold_source='no-canonical-criterion',
    solve_threshold_confidence='absent',
)


def _make_sliding_tile_puzzle() -> tuple[
    JumanjiEnv[object, object], GymnaxEnvParams,
]:
    """SlidingTilePuzzle-v0: 5×5 puzzle, 4 actions, REACH polarity.

    `obs_extract` projects the int32 puzzle (tile labels 0..24) to a
    single-channel float grid `(5, 5, 1)`. Normalizes by 24 to keep
    inputs in [0, 1]. The channel axis is added for CNN compatibility.

    REACH polarity: +1 reward on solving, 0 otherwise. Episode
    terminates on success or `step_count >= time_limit`.
    """
    import jumanji
    from jumanji.environments.logic.sliding_tile_puzzle.types import (
        Observation as SlidingTilePuzzleObservation,
    )

    inner = jumanji.make('SlidingTilePuzzle-v0')

    def obs_extract(obs: SlidingTilePuzzleObservation) -> jax.Array:
        scaled = obs.puzzle.astype(jnp.float32) / 24.0
        return scaled[..., None]

    adapter: JumanjiEnv[object, object] = JumanjiEnv(
        inner=inner,
        obs_extract=obs_extract,
        obs_shape=(5, 5, 1),
        n_actions=4,
    )
    params = GymnaxEnvParams(max_steps_in_episode=500)
    return adapter, params


_register_jumanji(
    'SlidingTilePuzzle-jumanji',
    factory=_make_sliding_tile_puzzle,
    n_actions=4,
    observation_shape=(5, 5, 1),
    horizon=500,
    r_min=0.0,
    r_max=1.0,
    reward_regime='terminal_only',
    solve_threshold=None,
    solve_threshold_source='no-canonical-criterion',
    solve_threshold_confidence='absent',
)
