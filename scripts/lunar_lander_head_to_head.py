"""Head-to-head dynamics comparison between the JAX-native
LunarLander port and the gymnasium / Box2D reference.

Purpose: build distributional + per-trajectory evidence for the
review at LUNAR_LANDER_DYNAMICS_REVIEW.md. Not a tested module
— hand-driven script.

Outputs:
- prints summary tables
- writes figures to experiments/figures/lunar_lander/
- writes raw arrays to experiments/data/cache/lunar_lander_h2h.npz

Use:
    uv run python scripts/lunar_lander_head_to_head.py
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gymnasium

from corroborate_rl.lunar_lander_jax import (
    LunarLanderEnv,
    LunarLanderParams,
    make_lunar_lander,
)


OUT_FIG = Path("experiments/figures/lunar_lander")
OUT_FIG.mkdir(parents=True, exist_ok=True)
OUT_DATA = Path("experiments/data/cache")
OUT_DATA.mkdir(parents=True, exist_ok=True)


# ---------- rollout primitives ----------

def jax_rollout(
    env: LunarLanderEnv,
    params: LunarLanderParams,
    actions: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Roll out the JAX env for `len(actions)` steps with given seed.
    Returns (obs_traj [T+1, 8], rewards [T], dones [T], step_done_first, length).
    Stops the loop at first done; auto-reset is built into step but
    we record the boundary."""
    rng = jax.random.PRNGKey(seed)
    obs, state = env.reset(rng, params)
    obs_buf = [np.asarray(obs)]
    rew_buf: list[float] = []
    done_buf: list[bool] = []
    first_done = -1
    for t, a in enumerate(actions):
        next_obs, state, reward, done, _ = env.step(
            jax.random.PRNGKey(0), state, jnp.int32(int(a)), params,
        )
        rew_buf.append(float(reward))
        done_buf.append(bool(done))
        if bool(done) and first_done == -1:
            first_done = t
            obs_buf.append(np.asarray(next_obs))
            break
        obs_buf.append(np.asarray(next_obs))
    length = first_done + 1 if first_done >= 0 else len(actions)
    return (
        np.array(obs_buf),
        np.array(rew_buf, dtype=np.float32),
        np.array(done_buf, dtype=bool),
        first_done,
        length,
    )


def gym_rollout(
    actions: np.ndarray, seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int, str]:
    """Roll out gymnasium env (Box2D) deterministically — same seed,
    same action sequence as the JAX rollout.

    Returns (obs_traj [T+1, 8], rewards [T], dones [T], first_done, length, termination_kind).
    """
    gym_env = gymnasium.make("LunarLander-v3", enable_wind=False)
    obs, _info = gym_env.reset(seed=seed)
    obs_buf = [obs.astype(np.float32)]
    rew_buf: list[float] = []
    done_buf: list[bool] = []
    first_done = -1
    term_kind = "timeout"
    for t, a in enumerate(actions):
        next_obs, reward, terminated, truncated, _info = gym_env.step(int(a))
        done = terminated or truncated
        rew_buf.append(float(reward))
        done_buf.append(bool(done))
        obs_buf.append(next_obs.astype(np.float32))
        if done and first_done == -1:
            first_done = t
            if reward < -50:
                term_kind = "crash"
            elif reward > 50:
                term_kind = "land"
            else:
                term_kind = "bounds"
            break
    gym_env.close()
    length = first_done + 1 if first_done >= 0 else len(actions)
    return (
        np.array(obs_buf),
        np.array(rew_buf, dtype=np.float32),
        np.array(done_buf, dtype=bool),
        first_done,
        length,
        term_kind,
    )


# ---------- analyses ----------

def fixed_action_comparison() -> None:
    """Run a small set of fixed action sequences from the JAX env
    AND gymnasium with seed=0, compare obs trajectories per-axis.

    Note: the initial random impulse is seeded differently — JAX
    uses `jax.random.uniform` on PRNGKey, gymnasium uses
    `np_random.uniform`. So the initial velocities won't match
    bit-for-bit. We compare distributions, not bit-equivalence."""
    env, params = make_lunar_lander()

    print("\n=== Fixed action sequence: 100 nops (free fall) ===")
    actions = np.zeros(100, dtype=np.int64)
    j_obs, j_r, _, j_fd, j_len = jax_rollout(env, params, actions, seed=0)
    g_obs, g_r, _, g_fd, g_len, g_kind = gym_rollout(actions, seed=0)
    print(f"JAX: episode length = {j_len}, total reward = {j_r.sum():+.2f}")
    print(
        f"GYM: episode length = {g_len}, total reward = {g_r.sum():+.2f}, "
        f"termination = {g_kind}",
    )

    print("\n=== Fixed: 200 main engine ===")
    actions = np.full(200, 2, dtype=np.int64)
    j_obs, j_r, _, j_fd, j_len = jax_rollout(env, params, actions, seed=0)
    g_obs, g_r, _, g_fd, g_len, g_kind = gym_rollout(actions, seed=0)
    print(f"JAX: episode length = {j_len}, total reward = {j_r.sum():+.2f}")
    print(
        f"GYM: episode length = {g_len}, total reward = {g_r.sum():+.2f}, "
        f"termination = {g_kind}",
    )

    print("\n=== Fixed: alternating L/R side (action 1 then 3) ===")
    actions = np.tile(np.array([1, 3], dtype=np.int64), 100)
    j_obs, j_r, _, j_fd, j_len = jax_rollout(env, params, actions, seed=0)
    g_obs, g_r, _, g_fd, g_len, g_kind = gym_rollout(actions, seed=0)
    print(f"JAX: episode length = {j_len}, total reward = {j_r.sum():+.2f}")
    print(
        f"GYM: episode length = {g_len}, total reward = {g_r.sum():+.2f}, "
        f"termination = {g_kind}",
    )

    print("\n=== Fixed: 50 left side only (action 1) ===")
    actions = np.full(100, 1, dtype=np.int64)
    j_obs, j_r, _, j_fd, j_len = jax_rollout(env, params, actions, seed=0)
    g_obs, g_r, _, g_fd, g_len, g_kind = gym_rollout(actions, seed=0)
    print(f"JAX: episode length = {j_len}, total reward = {j_r.sum():+.2f}")
    print(
        f"GYM: episode length = {g_len}, total reward = {g_r.sum():+.2f}, "
        f"termination = {g_kind}",
    )


def random_policy_distributional() -> dict[str, object]:
    """N random episodes from each env, compare distributions."""
    n_episodes = 100
    horizon = 1000
    rng_np = np.random.default_rng(seed=4242)

    env, params = make_lunar_lander()

    jax_returns = []
    jax_lengths = []
    jax_term = {"crash": 0, "land": 0, "bounds": 0, "timeout": 0}
    gym_returns = []
    gym_lengths = []
    gym_term = {"crash": 0, "land": 0, "bounds": 0, "timeout": 0}

    # Per-step obs arrays (collect ALL observations encountered)
    jax_all_obs: list[np.ndarray] = []
    gym_all_obs: list[np.ndarray] = []
    jax_step_rewards: list[float] = []
    gym_step_rewards: list[float] = []

    for ep in range(n_episodes):
        # Pre-sample a long action sequence; both envs see the same
        # actions per episode (action determinism is what we control).
        actions = rng_np.integers(0, 4, size=horizon).astype(np.int64)
        seed = int(rng_np.integers(0, 2**31 - 1))

        j_obs, j_r, _, j_fd, j_len = jax_rollout(env, params, actions, seed=seed)
        g_obs, g_r, _, g_fd, g_len, g_kind = gym_rollout(actions, seed=seed)

        # Termination kind for JAX. The env auto-resets on done, so
        # `j_obs[-1]` is the post-reset obs (useless for inspection).
        # Use `j_obs[j_fd]` (the obs at the FIRST done step, before
        # auto-reset clobbered it) together with the terminal reward
        # to classify.
        if j_fd >= 0:
            last_r = float(j_r[j_fd])
            terminal_obs = j_obs[j_fd]
            x_at_done = float(terminal_obs[0])
            if last_r > 50:
                j_kind = "land"
            elif abs(x_at_done) >= 1.0:
                j_kind = "bounds"
            elif last_r < -50:
                j_kind = "crash"
            else:
                j_kind = "timeout"
        else:
            j_kind = "timeout"

        jax_term[j_kind] = jax_term.get(j_kind, 0) + 1
        gym_term[g_kind] = gym_term.get(g_kind, 0) + 1

        jax_returns.append(float(j_r.sum()))
        gym_returns.append(float(g_r.sum()))
        jax_lengths.append(j_len)
        gym_lengths.append(g_len)
        # Discard step-0 obs to avoid double-counting reset transients
        if j_obs.shape[0] > 1:
            jax_all_obs.append(j_obs[1:])
        if g_obs.shape[0] > 1:
            gym_all_obs.append(g_obs[1:])
        jax_step_rewards.extend(j_r.tolist())
        gym_step_rewards.extend(g_r.tolist())

    jax_obs_arr = np.concatenate(jax_all_obs, axis=0)
    gym_obs_arr = np.concatenate(gym_all_obs, axis=0)

    print("\n=== Random policy distributional comparison ===")
    print(f"  n_episodes = {n_episodes}, horizon = {horizon}")
    print(
        f"  JAX returns: mean = {np.mean(jax_returns):+.2f}, "
        f"sd = {np.std(jax_returns):.2f}, "
        f"min = {np.min(jax_returns):+.2f}, "
        f"max = {np.max(jax_returns):+.2f}"
    )
    print(
        f"  GYM returns: mean = {np.mean(gym_returns):+.2f}, "
        f"sd = {np.std(gym_returns):.2f}, "
        f"min = {np.min(gym_returns):+.2f}, "
        f"max = {np.max(gym_returns):+.2f}"
    )
    print(
        f"  JAX lengths: mean = {np.mean(jax_lengths):.1f}, "
        f"sd = {np.std(jax_lengths):.1f}, "
        f"min = {np.min(jax_lengths)}, "
        f"max = {np.max(jax_lengths)}"
    )
    print(
        f"  GYM lengths: mean = {np.mean(gym_lengths):.1f}, "
        f"sd = {np.std(gym_lengths):.1f}, "
        f"min = {np.min(gym_lengths)}, "
        f"max = {np.max(gym_lengths)}"
    )
    print(f"  JAX termination: {jax_term}")
    print(f"  GYM termination: {gym_term}")

    # Per-step reward comparison
    print(
        f"  JAX per-step reward: mean = {np.mean(jax_step_rewards):+.4f}, "
        f"sd = {np.std(jax_step_rewards):.4f}"
    )
    print(
        f"  GYM per-step reward: mean = {np.mean(gym_step_rewards):+.4f}, "
        f"sd = {np.std(gym_step_rewards):.4f}"
    )

    # Per-axis obs envelope
    axes = ["x", "y", "vx", "vy", "angle", "ang_vel", "leg1", "leg2"]
    print("\n  Per-axis obs distribution: [mean ± sd, min..max]")
    print(f"    {'axis':>8} {'jax':>32} {'gym':>32}")
    for i, ax in enumerate(axes):
        j_m = jax_obs_arr[:, i].mean()
        j_s = jax_obs_arr[:, i].std()
        j_lo = jax_obs_arr[:, i].min()
        j_hi = jax_obs_arr[:, i].max()
        g_m = gym_obs_arr[:, i].mean()
        g_s = gym_obs_arr[:, i].std()
        g_lo = gym_obs_arr[:, i].min()
        g_hi = gym_obs_arr[:, i].max()
        print(
            f"    {ax:>8} "
            f"{f'{j_m:+.3f} ± {j_s:.3f} [{j_lo:+.2f}..{j_hi:+.2f}]':>32} "
            f"{f'{g_m:+.3f} ± {g_s:.3f} [{g_lo:+.2f}..{g_hi:+.2f}]':>32}"
        )

    # Per-axis KS test
    try:
        from scipy.stats import ks_2samp

        print("\n  Per-axis KS distance + p:")
        for i, ax in enumerate(axes):
            stat, p = ks_2samp(jax_obs_arr[:, i], gym_obs_arr[:, i])
            print(f"    {ax:>8}: D = {stat:.4f}, p = {p:.2e}")
    except ImportError:
        pass

    # Save figures
    fig, axs = plt.subplots(2, 4, figsize=(16, 8))
    for i, ax_name in enumerate(axes):
        a = axs[i // 4, i % 4]
        bins = np.linspace(
            min(jax_obs_arr[:, i].min(), gym_obs_arr[:, i].min()),
            max(jax_obs_arr[:, i].max(), gym_obs_arr[:, i].max()),
            50,
        )
        a.hist(jax_obs_arr[:, i], bins=bins, alpha=0.5, density=True, label="JAX port")
        a.hist(gym_obs_arr[:, i], bins=bins, alpha=0.5, density=True, label="Box2D")
        a.set_title(ax_name)
        a.legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "obs_distributions.png", dpi=100)
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].hist(jax_returns, bins=30, alpha=0.5, label="JAX")
    axs[0].hist(gym_returns, bins=30, alpha=0.5, label="GYM")
    axs[0].set_title(f"Episode return (n={n_episodes})")
    axs[0].legend()
    axs[1].hist(jax_lengths, bins=30, alpha=0.5, label="JAX")
    axs[1].hist(gym_lengths, bins=30, alpha=0.5, label="GYM")
    axs[1].set_title("Episode length")
    axs[1].legend()
    axs[2].bar(
        [0, 1, 2, 3], [jax_term[k] for k in ("crash", "land", "bounds", "timeout")],
        alpha=0.5, label="JAX",
    )
    axs[2].bar(
        [0.4, 1.4, 2.4, 3.4],
        [gym_term[k] for k in ("crash", "land", "bounds", "timeout")],
        alpha=0.5, label="GYM",
    )
    axs[2].set_xticks([0.2, 1.2, 2.2, 3.2])
    axs[2].set_xticklabels(["crash", "land", "bounds", "timeout"])
    axs[2].set_title("Termination breakdown")
    axs[2].legend()
    fig.tight_layout()
    fig.savefig(OUT_FIG / "episode_distributions.png", dpi=100)
    plt.close(fig)

    return {
        "jax_returns": jax_returns,
        "gym_returns": gym_returns,
        "jax_lengths": jax_lengths,
        "gym_lengths": gym_lengths,
        "jax_term": jax_term,
        "gym_term": gym_term,
    }


def torque_asymmetry_probe() -> None:
    """Probe whether the JAX port's main-engine impulse has the same
    sign as gymnasium's at non-zero lander angles.

    Construct a state with angle != 0, fire main engine, compare dvx
    and dvy between JAX and gymnasium-Box2D directly.
    """
    print("\n=== Torque / impulse direction probe ===")

    # We control gymnasium's lander.angle via direct field mutation
    # after env.reset (Box2D supports body.angle/transform set).
    test_angles = [-0.5, -0.25, 0.0, 0.25, 0.5, 1.0]
    actions_to_probe = [
        ("main", 2),
        ("side_left", 1),
        ("side_right", 3),
    ]

    env, params = make_lunar_lander()

    for action_name, a in actions_to_probe:
        print(f"\n  Action: {action_name} (index {a})")
        print(
            f"    {'angle':>8} {'JAX (dvx, dvy, dω)':>40} {'GYM (dvx, dvy, dω)':>40}"
        )

        for angle in test_angles:
            # JAX side: build a state at rest with the given angle
            from corroborate_rl.lunar_lander_jax import (
                LunarLanderState, HELIPAD_Y, INITIAL_Y, INITIAL_X,
                LEG_REST_OUTWARD_ANGLE,
            )
            jstate = LunarLanderState(
                x=jnp.float32(INITIAL_X),
                y=jnp.float32(INITIAL_Y),
                vx=jnp.float32(0.0), vy=jnp.float32(0.0),
                angle=jnp.float32(angle), angular_vel=jnp.float32(0.0),
                leg_contact_l=jnp.float32(0.0), leg_contact_r=jnp.float32(0.0),
                leg_angle_l=jnp.float32(LEG_REST_OUTWARD_ANGLE),
                leg_angle_r=jnp.float32(LEG_REST_OUTWARD_ANGLE),
                leg_omega_l=jnp.float32(0.0), leg_omega_r=jnp.float32(0.0),
                terrain_y=jnp.full((11,), HELIPAD_Y, dtype=jnp.float32),
                prev_shaping=jnp.float32(0.0),
                crashed=jnp.bool_(False), landed=jnp.bool_(False),
                time=jnp.int32(0),
            )
            _, jnext, _, _, _ = env.step(
                jax.random.PRNGKey(0), jstate, jnp.int32(a), params,
            )
            # Apparent acceleration: dvx contains gravity contribution
            # too. We subtract gravity*dt to isolate the engine impulse.
            dt = 1.0 / 50.0
            j_dvx = float(jnext.vx) - 0.0
            j_dvy = float(jnext.vy) - (-10.0) * dt   # subtract gravity contribution
            j_dom = float(jnext.angular_vel)

            # Gymnasium side: monkeypatch lander angle, fire engine,
            # measure delta-v.
            gym_env = gymnasium.make("LunarLander-v3", enable_wind=False)
            gobs, _ = gym_env.reset(seed=999)
            lander = gym_env.unwrapped.lander
            # Zero out the random impulse so we measure engine effect.
            lander.linearVelocity = (0.0, 0.0)
            lander.angularVelocity = 0.0
            lander.angle = angle
            # Save pre-step velocity
            pre_vx, pre_vy = lander.linearVelocity
            pre_om = lander.angularVelocity
            _ = gym_env.step(a)
            post_vx, post_vy = lander.linearVelocity
            post_om = lander.angularVelocity
            g_dvx = post_vx - pre_vx
            g_dvy = post_vy - pre_vy - (-10.0) * dt
            g_dom = post_om - pre_om
            gym_env.close()

            print(
                f"    {angle:>+8.3f} "
                f"{f'({j_dvx:+.4f}, {j_dvy:+.4f}, {j_dom:+.4f})':>40} "
                f"{f'({g_dvx:+.4f}, {g_dvy:+.4f}, {g_dom:+.4f})':>40}"
            )


def landing_detection_probe() -> None:
    """Test the soft-landing / awake predicate: in JAX it's `legs in
    contact ∧ |v| < 0.5 ∧ |ω| < 0.5 ∧ |angle| < 0.2`; in gymnasium
    it's `lander.awake == False`. Test a hand-crafted gentle
    landing trajectory."""
    print("\n=== Soft-landing predicate probe ===")
    # Build a state right at the moment of touchdown in JAX, see if
    # `landed=True` triggers
    from corroborate_rl.lunar_lander_jax import (
        LunarLanderState, HELIPAD_Y, LEG_REST_OUTWARD_ANGLE,
    )
    env, params = make_lunar_lander()

    # State: lander at helipad ground level, both legs touching,
    # near zero velocity / angle. With articulated legs at rest
    # outward splay (θ_leg = 1.058), foot body-frame y = -0.538.
    # Position body so foot just touches.
    y = HELIPAD_Y + 0.50
    flat_terrain = jnp.full((11,), HELIPAD_Y, dtype=jnp.float32)
    state = LunarLanderState(
        x=jnp.float32(10.0),
        y=jnp.float32(y),
        vx=jnp.float32(0.0), vy=jnp.float32(-0.1),
        angle=jnp.float32(0.05), angular_vel=jnp.float32(0.0),
        leg_contact_l=jnp.float32(1.0), leg_contact_r=jnp.float32(1.0),
        leg_angle_l=jnp.float32(LEG_REST_OUTWARD_ANGLE),
        leg_angle_r=jnp.float32(LEG_REST_OUTWARD_ANGLE),
        leg_omega_l=jnp.float32(0.0), leg_omega_r=jnp.float32(0.0),
        terrain_y=flat_terrain,
        prev_shaping=jnp.float32(20.0),
        crashed=jnp.bool_(False), landed=jnp.bool_(False),
        time=jnp.int32(0),
    )
    _, jnext, jr, jd, _ = env.step(
        jax.random.PRNGKey(0), state, jnp.int32(0), params,
    )
    print(
        f"  JAX gentle touchdown: landed={bool(jnext.landed)}, "
        f"done={bool(jd)}, reward={float(jr):+.2f}"
    )

    # Test what happens with high lateral velocity but legs touching
    state2 = LunarLanderState(
        x=jnp.float32(10.0), y=jnp.float32(y),
        vx=jnp.float32(1.0), vy=jnp.float32(-0.1),
        angle=jnp.float32(0.05), angular_vel=jnp.float32(0.0),
        leg_contact_l=jnp.float32(1.0), leg_contact_r=jnp.float32(1.0),
        leg_angle_l=jnp.float32(LEG_REST_OUTWARD_ANGLE),
        leg_angle_r=jnp.float32(LEG_REST_OUTWARD_ANGLE),
        leg_omega_l=jnp.float32(0.0), leg_omega_r=jnp.float32(0.0),
        terrain_y=flat_terrain,
        prev_shaping=jnp.float32(20.0),
        crashed=jnp.bool_(False), landed=jnp.bool_(False),
        time=jnp.int32(0),
    )
    _, jnext2, jr2, jd2, _ = env.step(
        jax.random.PRNGKey(0), state2, jnp.int32(0), params,
    )
    print(
        f"  JAX touchdown with |vx|=1.0: landed={bool(jnext2.landed)}, "
        f"done={bool(jd2)}, reward={float(jr2):+.2f}"
    )


# ---------- entrypoint ----------

def main() -> None:
    fixed_action_comparison()
    summary = random_policy_distributional()
    torque_asymmetry_probe()
    landing_detection_probe()


if __name__ == "__main__":
    main()
