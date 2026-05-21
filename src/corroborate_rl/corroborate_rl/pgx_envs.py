"""Per-env pgx factories registered into the env catalogue.

Mirrors `jumanji_envs.py`: each factory lazy-imports `pgx` inside
the closure body so module import doesn't construct any pgx env
(or trigger pgx's own first-touch checkpoint downloads, where
applicable). Metadata is passed explicitly to `_register_pgx`.

To add a new pgx env: write `_make_<env>()` that lazy-imports
pgx, then call `_register_pgx('<name>-pgx', factory=_make_<env>,
...)`. The `-pgx` suffix distinguishes our backend from
gymnax / jumanji names.
"""
from __future__ import annotations

from gymnax.environments.environment import EnvParams as GymnaxEnvParams

from corroborate_rl.env_catalogue import _register_pgx, image_downsample_hash
from corroborate_rl.pgx_adapter import PgxEnv


# Match the convention used for gymnax MinAtar (Asterix / Breakout /
# Freeway / SpaceInvaders) — 3×3 spatial pool, channel-sum, 2
# buckets per cell → 2^9 = 512 state-hash buckets. Same parameters
# across the MinAtar family for cross-env-comparable repeat-rate
# diagnostics.
_SEAQUEST_HASH, _SEAQUEST_CARD = image_downsample_hash(
    (10, 10, 10), pool_size=3, n_buckets_per_dim=2, channel_agg='sum',
    feature_low=0.0, feature_high=2.0,
)


def _make_seaquest_minatar() -> tuple[PgxEnv[object], GymnaxEnvParams]:
    """MinAtar Seaquest (pgx-implemented).

    Pgx's MinAtar suite is the only JAX-native source of Seaquest
    (the gymnax MinAtar set has Asterix, Breakout, Freeway,
    SpaceInvaders only). Seaquest is structurally distinct from
    the gymnax-side envs: the agent must surface periodically to
    refill oxygen — a multi-objective tension between scoring and
    survival that the simpler MinAtar envs don't have.

    Observation: 10x10x10 (bool, cast to float32 by the adapter).
    Actions: 6 (no-op, fire, up, right, left, down).
    Horizon: MinAtar conventional ~2500 steps per episode.
    Reward: +1 per fish shot, 0 otherwise (per_step positive).
    """
    import pgx
    inner = pgx.make('minatar-seaquest')
    adapter: PgxEnv[object] = PgxEnv(
        inner=inner,
        obs_shape=inner.observation_shape,
        n_actions=inner.num_actions,
    )
    params = GymnaxEnvParams(max_steps_in_episode=2500)
    return adapter, params


_register_pgx(
    'Seaquest-MinAtar',
    factory=_make_seaquest_minatar,
    n_actions=6,
    observation_shape=(10, 10, 10),
    horizon=2500,
    r_min=0.0,
    r_max=1.0,
    reward_regime='event_triggered',
    state_hash=_SEAQUEST_HASH,
    state_hash_cardinality=_SEAQUEST_CARD,
)
