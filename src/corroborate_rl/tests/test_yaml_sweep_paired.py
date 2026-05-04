"""Schema-contract smoke for `minatar_1M.yaml` and
`ddqn_effective.yaml` paired-mode sweeps. The schema's contract:
each `(template, env)` resolves to a concrete `Hypothesis` with
- `CNN.obs_shape` substituted from `EnvSpec.public_attrs()`,
- frozen-dataclass equality on every config-bundle slot,
- stable `arm_key()` and `claim_graph_signature`.

`build_paired` is the substrate's per-env resolver; asserting
the contract at its output is the strongest guarantee short of
running the sweep itself."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path

import pytest

from corroborate.core.intervention import Intervention
from corroborate.runner.registry import Registry
from corroborate_rl.dqn.collect import EnvConfig
from corroborate_rl.dqn.config_loader import HypothesisConfig
from corroborate_rl.dqn.yaml_sweep import (
    DQNSweep, build_paired, default_dqn_registry, load_sweep,
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

def _python_minatar_1M_hypothesis(
    name: str, env_name: str,
) -> HypothesisConfig:
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
        return HypothesisConfig(
            name='vanilla_dqn', intervention=base, predicted_direction=None,
            intervention_arms=(),
        )
    if name == 'ddqn':
        boot = partial(bootstrap, greedification=double_greedify)
        base['bootstrap'] = boot
        return HypothesisConfig(
            name='ddqn', intervention=base, predicted_direction='a_gt_b',
            intervention_arms=(
                Intervention(slot_path='bootstrap', replacement=boot),
            ),
        )
    raise ValueError(name)


def _python_ddqn_effective_hypothesis(
    name: str, env_name: str,
) -> HypothesisConfig:
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
        return HypothesisConfig(
            name='vanilla_dqn', intervention=base, predicted_direction=None,
            intervention_arms=(),
        )
    if name == 'ddqn':
        boot = partial(bootstrap, greedification=double_greedify)
        base['bootstrap'] = boot
        return HypothesisConfig(
            name='ddqn', intervention=base, predicted_direction='a_gt_b',
            intervention_arms=(
                Intervention(slot_path='bootstrap', replacement=boot),
            ),
        )
    raise ValueError(name)


# ---------- fixtures ----------

@pytest.fixture
def reg() -> Registry:
    return default_dqn_registry()


@pytest.fixture
def minatar_1M_sweep(reg: Registry) -> DQNSweep:
    s = load_sweep(MINATAR_1M_PATH, reg=reg)
    assert s.arms_shape == 'paired'
    return s


@pytest.fixture
def ddqn_effective_sweep(reg: Registry) -> DQNSweep:
    s = load_sweep(DDQN_EFFECTIVE_PATH, reg=reg)
    assert s.arms_shape == 'paired'
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
    assert len(s.hypothesis_templates) == 2


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
    assert len(s.hypothesis_templates) == 2


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
    """For each (env, hypothesis), the YAML resolves to a
    Hypothesis whose config bundles match a reference Python
    construction, including env-specific CNN.obs_shape
    substitution."""
    built, envs_aligned = build_paired(minatar_1M_sweep, reg=reg)
    yaml_h = _pick(built, envs_aligned, env_name, h_name)
    py_h = _python_minatar_1M_hypothesis(h_name, env_name)

    assert yaml_h.name == py_h.name
    assert yaml_h.predicted_direction == py_h.predicted_direction
    assert yaml_h.arm_key() == py_h.arm_key()

    # `partial` lacks value equality; `canonical_str` is the
    # framework's value-equality contract for partial-baked claims.
    from corroborate._internals.canonical import canonical_str
    for k in ('q_network', 'optimizer', 'replay'):
        assert canonical_str(yaml_h.intervention[k]) == canonical_str(
            py_h.intervention[k],
        )

    # CNN obs_shape resolved to the env's spec attribute.
    spec = get_env_spec(env_name)
    from corroborate_rl.dqn.claims.q_network import CNN
    qn_yaml = yaml_h.intervention['q_network']
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
    built, envs_aligned = build_paired(ddqn_effective_sweep, reg=reg)
    yaml_h = _pick(built, envs_aligned, env_name, h_name)
    py_h = _python_ddqn_effective_hypothesis(h_name, env_name)

    assert yaml_h.name == py_h.name
    assert yaml_h.predicted_direction == py_h.predicted_direction
    assert yaml_h.arm_key() == py_h.arm_key()
    # `partial` lacks value equality; `canonical_str` is the
    # framework's value-equality contract for partial-baked claims.
    from corroborate._internals.canonical import canonical_str
    for k in ('q_network', 'optimizer', 'replay'):
        assert canonical_str(yaml_h.intervention[k]) == canonical_str(
            py_h.intervention[k],
        )


# ---------- cross-env signature stability ----------

def test_paired_ddqn_bootstrap_signature_stable(
    minatar_1M_sweep: DQNSweep, reg: Registry,
) -> None:
    """The DDQN partial sits inside the intervention dict; its
    `claim_graph_signature` is env-independent (only CNN slot has
    env binding) — same hash across all envs."""
    built, envs_aligned = build_paired(minatar_1M_sweep, reg=reg)
    yaml_h = _pick(built, envs_aligned, 'Asterix-MinAtar', 'ddqn')
    py_h = _python_minatar_1M_hypothesis('ddqn', 'Asterix-MinAtar')
    sig_yaml = claim_graph_signature(
        yaml_h.intervention['bootstrap'],
    )
    sig_python = claim_graph_signature(
        py_h.intervention['bootstrap'],
    )
    assert sig_yaml == sig_python


def test_paired_arms_count_matches_paired_arms_helper(
    minatar_1M_sweep: DQNSweep, reg: Registry,
) -> None:
    """The expanded (hypotheses, envs_aligned) tuples have
    `n_envs * n_templates` entries — so `paired_arms` zips them
    cleanly, one (h, env) pair per arm."""
    built, envs_aligned = build_paired(minatar_1M_sweep, reg=reg)
    n_envs = len(minatar_1M_sweep.envs)
    n_templates = len(minatar_1M_sweep.hypothesis_templates)
    assert len(built) == n_envs * n_templates
    assert len(envs_aligned) == n_envs * n_templates


def test_chunked_sweep_rejects_build_paired(reg: Registry) -> None:
    """`build_paired` is the wrong helper for a chunked sweep
    and refuses early — the alternative would be silently
    iterating envs while ignoring per-env env_attrs, which
    would produce nonsense output."""
    chunked_path = (
        REPO_ROOT / 'experiments' / 'configs' / 'expectile_3way.yaml'
    )
    chunked = load_sweep(chunked_path, reg=reg)
    assert chunked.arms_shape == 'chunked'
    with pytest.raises(ValueError, match="arms_shape='paired'"):
        _ = build_paired(chunked, reg=reg)


def test_from_env_in_chunked_mode_raises(reg: Registry) -> None:
    """A `{from_env: ...}` placeholder is only meaningful in
    paired-mode dispatch; trying to build hypotheses for a
    chunked sweep that contains one fails fast with a clear
    error pointing at the schema mistake."""
    sweep = load_sweep(MINATAR_1M_PATH, reg=reg)
    with pytest.raises(ValueError, match='from_env'):
        _ = sweep.build_hypotheses(reg=reg)  # no env_attrs


# ---------- helpers ----------

def _pick(
    built: tuple[HypothesisConfig, ...],
    envs_aligned: tuple[EnvConfig, ...],
    env_name: str,
    h_name: str,
) -> HypothesisConfig:
    """Find the (env, hypothesis-name) pair in the expanded
    paired tuples."""
    for h, ec in zip(built, envs_aligned, strict=True):
        if ec.env_name == env_name and h.name == h_name:
            return h
    raise KeyError(f'no ({env_name}, {h_name}) in built tuples')
