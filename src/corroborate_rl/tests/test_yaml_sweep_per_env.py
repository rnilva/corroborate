"""Schema-contract smoke for `minatar_1M.yaml` and
`ddqn_effective.yaml` per-env sweeps. The schema's contract:
each `(template, env)` resolves to a concrete `InterventionConfig`
with
- `CNN.obs_shape` substituted from `EnvSpec.public_attrs()`,
- frozen-dataclass equality on every config-bundle slot,
- stable arm keys and `claim_graph_signature`.

`build_per_env` is the substrate's per-env resolver; asserting
the contract at its output is the strongest guarantee short of
running the sweep itself."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path

import pytest

from corroborate.core.intervention import DoEffect, Intervention
from corroborate.runner.registry import Registry
from corroborate_rl.dqn.collect import EnvConfig
from corroborate_rl.dqn.config_loader import InterventionConfig
from corroborate_rl.dqn.yaml_sweep import (
    DQNSweep, build_per_env, default_dqn_registry, load_sweep,
)
from corroborate_rl.env_catalogue import get as get_env_spec
from corroborate.core.signature import claim_graph_signature


REPO_ROOT = Path(__file__).resolve().parents[3]
MINATAR_1M_PATH = (
    REPO_ROOT / 'experiments' / 'configs' / 'minatar_1M.yaml'
)
DDQN_EFFECTIVE_PATH = (
    REPO_ROOT / 'experiments' / 'configs' / 'ddqn_effective.yaml'
)


# ---------- Python-authored references ----------

def _python_minatar_1M_intervention(
    name: str, env_name: str,
) -> InterventionConfig:
    """Canonical Python recipe for the minatar_1M cohort. Used as
    the reference the YAML sweep must match structurally."""
    from corroborate_rl.dqn.claims.bootstrap import (
        bootstrap, double_greedify,
    )
    from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
    from corroborate_rl.dqn.claims.q_network import CNN
    from corroborate_rl.dqn.claims.replay import Replay
    spec = get_env_spec(env_name)
    base: dict[str, object] = {
        'total_steps': 1_000_000,
        'eval_every': 50_000,
        'n_episodes': 5,
        'gamma': 0.99,
        'replay': Replay(capacity=50_000, batch_size=32),
        'optimizer': partial(
            warmed_update,
            inner=partial(adam, lr=1e-4),
            warmup_steps=100,
        ),
        'sync_period': 100,
        'q_network': CNN(
            obs_shape=spec.observation_shape,
            channels=(16, 32), kernel_size=3, hidden=(128,),
        ),
    }
    if name == 'vanilla_dqn':
        return InterventionConfig(
            name=f'vanilla_dqn_{env_name}', base=base,
            do_effect=DoEffect(arms=((),)),
        )
    if name == 'ddqn':
        boot = partial(bootstrap, greedification=double_greedify)
        return InterventionConfig(
            name=f'ddqn_{env_name}', base=base,
            do_effect=DoEffect(arms=(
                (),
                (Intervention(slot_path='bootstrap', replacement=boot),),
            )),
        )
    raise ValueError(name)


def _python_ddqn_effective_intervention(
    name: str, env_name: str,
) -> InterventionConfig:
    """Canonical Python recipe for the ddqn_effective cohort
    (200k steps). Reference for the matching YAML sweep."""
    from corroborate_rl.dqn.claims.bootstrap import (
        bootstrap, double_greedify,
    )
    from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
    from corroborate_rl.dqn.claims.q_network import CNN
    from corroborate_rl.dqn.claims.replay import Replay
    spec = get_env_spec(env_name)
    base: dict[str, object] = {
        'total_steps': 200_000,
        'eval_every': 20_000,
        'n_episodes': 5,
        'gamma': 0.99,
        'replay': Replay(capacity=50_000, batch_size=32),
        'optimizer': partial(
            warmed_update,
            inner=partial(adam, lr=1e-4),
            warmup_steps=100,
        ),
        'sync_period': 100,
        'q_network': CNN(
            obs_shape=spec.observation_shape,
            channels=(16, 32), kernel_size=3, hidden=(128,),
        ),
    }
    if name == 'vanilla_dqn':
        return InterventionConfig(
            name=f'vanilla_dqn_{env_name}', base=base,
            do_effect=DoEffect(arms=((),)),
        )
    if name == 'ddqn':
        boot = partial(bootstrap, greedification=double_greedify)
        return InterventionConfig(
            name=f'ddqn_{env_name}', base=base,
            do_effect=DoEffect(arms=(
                (),
                (Intervention(slot_path='bootstrap', replacement=boot),),
            )),
        )
    raise ValueError(name)


# ---------- fixtures ----------

@pytest.fixture
def reg() -> Registry:
    return default_dqn_registry()


@pytest.fixture
def minatar_1M_sweep(reg: Registry) -> DQNSweep:
    s = load_sweep(MINATAR_1M_PATH, reg=reg)
    assert s.env_binding == 'per_env'
    return s


@pytest.fixture
def ddqn_effective_sweep(reg: Registry) -> DQNSweep:
    s = load_sweep(DDQN_EFFECTIVE_PATH, reg=reg)
    assert s.env_binding == 'per_env'
    return s


# ---------- minatar_1M envelope ----------

def test_minatar_1M_envelope(
    minatar_1M_sweep: DQNSweep,
) -> None:
    s = minatar_1M_sweep
    assert s.name == 'minatar_1M'
    assert s.out_dir == Path('experiments/data/minatar_1M')
    assert s.archive_remote == 's3://corroborate-archive/minatar_1M'
    assert s.envs == (
        EnvConfig('Asterix-MinAtar', n_seeds=30, chunk_size=15),
        EnvConfig('Breakout-MinAtar', n_seeds=30, chunk_size=15),
        EnvConfig('Freeway-MinAtar', n_seeds=30, chunk_size=15),
        EnvConfig('SpaceInvaders-MinAtar', n_seeds=30, chunk_size=15),
    )
    assert len(s.intervention_templates) == 2


# ---------- ddqn_effective envelope ----------

def test_ddqn_effective_envelope(
    ddqn_effective_sweep: DQNSweep,
) -> None:
    s = ddqn_effective_sweep
    assert s.name == 'ddqn_effective_cohort'
    assert s.out_dir == Path(
        'experiments/data/ddqn_effective_cohort',
    )
    assert s.archive_remote is None
    assert len(s.envs) == 5
    assert {ec.env_name for ec in s.envs} == {
        'Asterix-MinAtar', 'Breakout-MinAtar',
        'SpaceInvaders-MinAtar', 'Freeway-MinAtar',
        'MNISTBandit-bsuite',
    }
    assert len(s.intervention_templates) == 2


# ---------- per-env build contract, parametrised ----------

@pytest.mark.parametrize(
    'env_name', [
        'Asterix-MinAtar', 'Breakout-MinAtar',
        'Freeway-MinAtar', 'SpaceInvaders-MinAtar',
    ],
)
@pytest.mark.parametrize('h_name', ['vanilla_dqn', 'ddqn'])
def test_minatar_1M_per_env_contract(
    minatar_1M_sweep: DQNSweep,
    reg: Registry,
    env_name: str,
    h_name: str,
) -> None:
    """For each (env, intervention), the YAML resolves to an
    InterventionConfig whose config bundles match a reference
    Python construction, including env-specific CNN.obs_shape
    substitution."""
    built, envs_aligned = build_per_env(minatar_1M_sweep, reg=reg)
    yaml_h = _pick(built, envs_aligned, env_name, h_name)
    py_h = _python_minatar_1M_intervention(h_name, env_name)

    assert yaml_h.name == py_h.name
    assert _arm_keys(yaml_h) == _arm_keys(py_h)

    # `partial` lacks value equality; `canonical_str` is the
    # framework's value-equality contract for partial-baked claims.
    from corroborate._internals.canonical import canonical_str
    for k in ('q_network', 'optimizer', 'replay'):
        assert canonical_str(yaml_h.base[k]) == canonical_str(
            py_h.base[k],
        )

    # CNN obs_shape resolved to the env's spec attribute.
    spec = get_env_spec(env_name)
    from corroborate_rl.dqn.claims.q_network import CNN
    qn_yaml = yaml_h.base['q_network']
    assert isinstance(qn_yaml, CNN)
    assert qn_yaml.obs_shape == spec.observation_shape


@pytest.mark.parametrize(
    'env_name', [
        'Asterix-MinAtar', 'Breakout-MinAtar',
        'SpaceInvaders-MinAtar', 'Freeway-MinAtar',
        'MNISTBandit-bsuite',
    ],
)
@pytest.mark.parametrize('h_name', ['vanilla_dqn', 'ddqn'])
def test_ddqn_effective_per_env_contract(
    ddqn_effective_sweep: DQNSweep,
    reg: Registry,
    env_name: str,
    h_name: str,
) -> None:
    built, envs_aligned = build_per_env(ddqn_effective_sweep, reg=reg)
    yaml_h = _pick(built, envs_aligned, env_name, h_name)
    py_h = _python_ddqn_effective_intervention(h_name, env_name)

    assert yaml_h.name == py_h.name
    assert _arm_keys(yaml_h) == _arm_keys(py_h)
    # `partial` lacks value equality; `canonical_str` is the
    # framework's value-equality contract for partial-baked claims.
    from corroborate._internals.canonical import canonical_str
    for k in ('q_network', 'optimizer', 'replay'):
        assert canonical_str(yaml_h.base[k]) == canonical_str(
            py_h.base[k],
        )


# ---------- cross-env signature stability ----------

def test_per_env_ddqn_bootstrap_signature_stable(
    minatar_1M_sweep: DQNSweep, reg: Registry,
) -> None:
    """The DDQN partial sits on the treatment arm; its
    `claim_graph_signature` is env-independent (only CNN slot has
    env binding) — same hash across all envs."""
    built, envs_aligned = build_per_env(minatar_1M_sweep, reg=reg)
    yaml_h = _pick(built, envs_aligned, 'Asterix-MinAtar', 'ddqn')
    py_h = _python_minatar_1M_intervention('ddqn', 'Asterix-MinAtar')
    sig_yaml = claim_graph_signature(yaml_h.do_effect.arms[1][0].replacement)
    sig_python = claim_graph_signature(py_h.do_effect.arms[1][0].replacement)
    assert sig_yaml == sig_python


def test_per_env_count_matches_template_x_env(
    minatar_1M_sweep: DQNSweep, reg: Registry,
) -> None:
    """The expanded (interventions, envs_aligned) tuples have
    `n_envs * n_templates` entries — so a downstream zip lands
    one (h, env) pair per arm."""
    built, envs_aligned = build_per_env(minatar_1M_sweep, reg=reg)
    n_envs = len(minatar_1M_sweep.envs)
    n_templates = len(minatar_1M_sweep.intervention_templates)
    assert len(built) == n_envs * n_templates
    assert len(envs_aligned) == n_envs * n_templates


def test_shared_sweep_rejects_build_per_env(reg: Registry) -> None:
    """`build_per_env` is the wrong helper for a shared sweep
    and refuses early — the alternative would be silently
    iterating envs while ignoring per-env env_attrs, which
    would produce nonsense output."""
    shared_path = (
        REPO_ROOT / 'experiments' / 'configs' / 'expectile_3way.yaml'
    )
    shared = load_sweep(shared_path, reg=reg)
    assert shared.env_binding == 'shared'
    with pytest.raises(ValueError, match="env_binding='per_env'"):
        _ = build_per_env(shared, reg=reg)


def test_from_env_in_shared_mode_raises(reg: Registry) -> None:
    """A `{from_env: ...}` placeholder is only meaningful in
    per-env dispatch; trying to build interventions for a
    shared sweep that contains one fails fast with a clear
    error pointing at the schema mistake."""
    sweep = load_sweep(MINATAR_1M_PATH, reg=reg)
    with pytest.raises(ValueError, match='from_env'):
        _ = sweep.build_interventions(reg=reg)  # no env_attrs


# ---------- helpers ----------

def _arm_keys(cfg: InterventionConfig) -> tuple[str, ...]:
    return cfg.do_effect.arm_keys()


def _pick(
    built: tuple[InterventionConfig, ...],
    envs_aligned: tuple[EnvConfig, ...],
    env_name: str,
    h_name: str,
) -> InterventionConfig:
    """Find the (env, intervention-name) pair in the expanded
    per-env tuples. The expanded `cfg.name` includes the env's
    `{from_env: name}` substitution, so we match
    `<h_name>_<env_name>`."""
    target = f'{h_name}_{env_name}'
    for h, ec in zip(built, envs_aligned, strict=True):
        if ec.env_name == env_name and h.name == target:
            return h
    raise KeyError(f'no ({env_name}, {h_name}) in built tuples')


# ---------- dispatcher collision detection ----------

def test_dispatch_sweep_raises_on_cfg_name_collision(tmp_path: Path) -> None:
    """`env_binding: per_env` + multi-env intervention without a
    `{from_env: <attr>}` substitution in `name` yields multiple
    `InterventionConfig`s sharing `cfg.name`. They all write to
    `<out_dir>/<cfg.name>/runs.parquet` and the final merge silently
    concatenates the same file N times — losing all but one env's
    data. `dispatch_sweep` must refuse the collision pre-dispatch.

    Regression test for the 2026-05-11 silent-overwrite bug in
    reward_scale_sweep_postfix (CORPUS_INTEGRITY.md CI9)."""
    from corroborate_rl.dqn.yaml_sweep import dispatch_sweep

    cfg = tmp_path / 'colliding_per_env_sweep.yaml'
    cfg.write_text(
        'name: colliding\n'
        f'out_dir: {tmp_path / "out"}\n'
        'env_binding: per_env\n'
        'envs:\n'
        '  - {name: FourRooms-misc, n_seeds: 2}\n'
        '  - {name: Acrobot-v1, n_seeds: 2}\n'
        'defaults:\n'
        '  total_steps: 1000\n'
        '  eval_every: 500\n'
        '  n_episodes: 1\n'
        '  gamma: 0.99\n'
        'interventions:\n'
        '  - name: ddqn_vs_vanilla\n'  # no {from_env} substitution
        '    arms:\n'
        '      - []\n'
        '      - - slot_path: bootstrap\n'
        '          replacement:\n'
        '            fn: bootstrap\n'
        '            greedification: {fn: double_greedify}\n'
    )
    sweep = load_sweep(cfg, reg=default_dqn_registry())
    with pytest.raises(ValueError, match=r"share output paths"):
        dispatch_sweep(sweep)
    # No corpus should have been written.
    assert not (tmp_path / 'out').exists() or not list(
        (tmp_path / 'out').glob('**/runs.parquet')
    )


def test_dispatch_sweep_accepts_shared_multi_env(tmp_path: Path) -> None:
    """`env_binding: shared` packs all envs into a single config —
    no name collision, no silent overwrite. The shared path is
    one of the two valid fixes for the collision case."""
    from corroborate_rl.dqn.yaml_sweep import load_sweep
    cfg = tmp_path / 'shared_multi_env.yaml'
    cfg.write_text(
        'name: shared_multi\n'
        f'out_dir: {tmp_path / "out"}\n'
        'env_binding: shared\n'
        'envs:\n'
        '  - {name: FourRooms-misc, n_seeds: 2}\n'
        '  - {name: Acrobot-v1, n_seeds: 2}\n'
        'defaults:\n'
        '  total_steps: 1000\n'
        '  eval_every: 500\n'
        '  n_episodes: 1\n'
        '  gamma: 0.99\n'
        'interventions:\n'
        '  - name: ddqn_vs_vanilla\n'
        '    arms:\n'
        '      - []\n'
        '      - - slot_path: bootstrap\n'
        '          replacement:\n'
        '            fn: bootstrap\n'
        '            greedification: {fn: double_greedify}\n'
    )
    sweep = load_sweep(cfg, reg=default_dqn_registry())
    configs = list(sweep.build_interventions(reg=default_dqn_registry()))
    # One config carrying all envs — no collision possible.
    assert len(configs) == 1
    assert configs[0].name == 'ddqn_vs_vanilla'
    assert len(sweep.envs) == 2
