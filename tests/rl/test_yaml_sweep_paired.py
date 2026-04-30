"""Paired-mode parity smoke: `minatar_1M.yaml` and
`ddqn_effective.yaml` manifests reproduce the per-env Hypothesis
tuples that the canonical Python recipe below produces, including
env-specific CNN obs_shape resolution.

Identity contract per (template, env) pair:
- Frozen-dataclass equality on Module Claims (CNN/MLP/Adam/...).
- Per-env CNN.obs_shape matches `EnvSpec.observation_shape`.
- `arm_key()` and `claim_graph_signature` match between the
  YAML- and Python-built path.

`build_paired_hypotheses` is the substrate's per-env resolver;
asserting parity at its output is the strongest guarantee
short of running the sweep itself."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path

import pytest

from corroborate.hypothesis import Hypothesis
from corroborate.intervention import Intervention
from corroborate.rl.dqn.collect import EnvConfig
from corroborate.rl.dqn.yaml_sweep import (
    PairedManifest, build_paired_hypotheses,
    default_dqn_registry, load_manifest,
)
from corroborate.rl.env_catalogue import get as get_env_spec
from corroborate.signature import claim_graph_signature


REPO_ROOT = Path(__file__).resolve().parents[2]
MINATAR_1M_PATH = (
    REPO_ROOT / 'experiments' / 'configs' / 'minatar_1M.yaml'
)
DDQN_EFFECTIVE_PATH = (
    REPO_ROOT / 'experiments' / 'configs' / 'ddqn_effective.yaml'
)


# ---------- Python-authored references ----------

def _python_minatar_1M_hypothesis(
    name: str, env_name: str,
) -> Hypothesis[Mapping[str, object]]:
    """Canonical Python recipe for the minatar_1M cohort. Used as
    the reference the YAML manifest must match structurally."""
    from corroborate.rl.dqn.claims.bootstrap import (
        bootstrap, double_greedify,
    )
    from corroborate.rl.dqn.claims.optimizer import (
        Adam, WarmedUpdate,
    )
    from corroborate.rl.dqn.claims.q_network import CNN
    from corroborate.rl.dqn.claims.replay import Replay
    spec = get_env_spec(env_name)
    base: dict[str, object] = {
        'total_steps': 1_000_000,
        'eval_every': 50_000,
        'n_episodes': 5,
        'gamma': 0.99,
        'replay': Replay(capacity=50_000, batch_size=32),
        'optimizer': WarmedUpdate(
            inner=Adam(lr=1e-4), warmup_steps=100,
        ),
        'sync_period': 100,
        'q_network': CNN(
            obs_shape=spec.observation_shape,
            channels=(16, 32), kernel_size=3, hidden=(128,),
        ),
    }
    if name == 'vanilla_dqn':
        return Hypothesis(
            name='vanilla_dqn', intervention=base,
            bridges=(), predicted_direction=None,
            intervention_arms=(),
        )
    if name == 'ddqn':
        boot = partial(bootstrap, greedification=double_greedify)
        base['bootstrap'] = boot
        return Hypothesis(
            name='ddqn', intervention=base,
            bridges=(), predicted_direction='a_gt_b',
            intervention_arms=(
                Intervention(slot_path='bootstrap', replacement=boot),
            ),
        )
    raise ValueError(name)


def _python_ddqn_effective_hypothesis(
    name: str, env_name: str,
) -> Hypothesis[Mapping[str, object]]:
    """Canonical Python recipe for the ddqn_effective cohort
    (200k steps). Reference for the matching YAML manifest."""
    from corroborate.rl.dqn.claims.bootstrap import (
        bootstrap, double_greedify,
    )
    from corroborate.rl.dqn.claims.optimizer import (
        Adam, WarmedUpdate,
    )
    from corroborate.rl.dqn.claims.q_network import CNN
    from corroborate.rl.dqn.claims.replay import Replay
    spec = get_env_spec(env_name)
    base: dict[str, object] = {
        'total_steps': 200_000,
        'eval_every': 20_000,
        'n_episodes': 5,
        'gamma': 0.99,
        'replay': Replay(capacity=50_000, batch_size=32),
        'optimizer': WarmedUpdate(
            inner=Adam(lr=1e-4), warmup_steps=100,
        ),
        'sync_period': 100,
        'q_network': CNN(
            obs_shape=spec.observation_shape,
            channels=(16, 32), kernel_size=3, hidden=(128,),
        ),
    }
    if name == 'vanilla_dqn':
        return Hypothesis(
            name='vanilla_dqn', intervention=base,
            bridges=(), predicted_direction=None,
            intervention_arms=(),
        )
    if name == 'ddqn':
        boot = partial(bootstrap, greedification=double_greedify)
        base['bootstrap'] = boot
        return Hypothesis(
            name='ddqn', intervention=base,
            bridges=(), predicted_direction='a_gt_b',
            intervention_arms=(
                Intervention(slot_path='bootstrap', replacement=boot),
            ),
        )
    raise ValueError(name)


# ---------- fixtures ----------

@pytest.fixture
def minatar_1M_manifest() -> PairedManifest:
    reg = default_dqn_registry()
    m = load_manifest(MINATAR_1M_PATH, reg=reg)
    assert isinstance(m, PairedManifest)
    return m


@pytest.fixture
def ddqn_effective_manifest() -> PairedManifest:
    reg = default_dqn_registry()
    m = load_manifest(DDQN_EFFECTIVE_PATH, reg=reg)
    assert isinstance(m, PairedManifest)
    return m


# ---------- minatar_1M envelope ----------

def test_minatar_1M_envelope(
    minatar_1M_manifest: PairedManifest,
) -> None:
    m = minatar_1M_manifest
    assert m.name == 'minatar_1M'
    assert m.out_dir == Path('experiments/data/minatar_1M')
    assert m.archive_remote == 's3://corroborate-archive/minatar_1M'
    assert m.envs == (
        EnvConfig('Asterix-MinAtar', n_seeds=30, chunk_size=15),
        EnvConfig('Breakout-MinAtar', n_seeds=30, chunk_size=15),
        EnvConfig('Freeway-MinAtar', n_seeds=30, chunk_size=15),
        EnvConfig('SpaceInvaders-MinAtar', n_seeds=30, chunk_size=15),
    )
    assert len(m.hypothesis_templates) == 2


# ---------- ddqn_effective envelope ----------

def test_ddqn_effective_envelope(
    ddqn_effective_manifest: PairedManifest,
) -> None:
    m = ddqn_effective_manifest
    assert m.name == 'ddqn_effective_cohort'
    assert m.out_dir == Path(
        'experiments/data/ddqn_effective_cohort',
    )
    assert m.archive_remote is None
    assert len(m.envs) == 5
    assert {ec.env_name for ec in m.envs} == {
        'Asterix-MinAtar', 'Breakout-MinAtar',
        'SpaceInvaders-MinAtar', 'Freeway-MinAtar',
        'MNISTBandit-bsuite',
    }
    assert len(m.hypothesis_templates) == 2


# ---------- per-env build parity, parametrised ----------

@pytest.mark.parametrize(
    'env_name', [
        'Asterix-MinAtar', 'Breakout-MinAtar',
        'Freeway-MinAtar', 'SpaceInvaders-MinAtar',
    ],
)
@pytest.mark.parametrize('h_name', ['vanilla_dqn', 'ddqn'])
def test_minatar_1M_per_env_parity(
    minatar_1M_manifest: PairedManifest,
    env_name: str,
    h_name: str,
) -> None:
    """Build the YAML manifest's hypotheses against env_specs,
    pick the one for `(env_name, h_name)`, compare against the
    Python authoring path."""
    reg = default_dqn_registry()
    built, envs_aligned = build_paired_hypotheses(
        minatar_1M_manifest, reg=reg,
    )
    yaml_h = _pick(built, envs_aligned, env_name, h_name)
    py_h = _python_minatar_1M_hypothesis(h_name, env_name)

    assert yaml_h.name == py_h.name
    assert yaml_h.predicted_direction == py_h.predicted_direction
    assert yaml_h.arm_key() == py_h.arm_key()

    # Module Claim equality (frozen dataclasses).
    for k in ('q_network', 'optimizer', 'replay'):
        assert yaml_h.intervention[k] == py_h.intervention[k]

    # CNN obs_shape resolved to the env's spec attribute.
    spec = get_env_spec(env_name)
    from corroborate.rl.dqn.claims.q_network import CNN
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
def test_ddqn_effective_per_env_parity(
    ddqn_effective_manifest: PairedManifest,
    env_name: str,
    h_name: str,
) -> None:
    reg = default_dqn_registry()
    built, envs_aligned = build_paired_hypotheses(
        ddqn_effective_manifest, reg=reg,
    )
    yaml_h = _pick(built, envs_aligned, env_name, h_name)
    py_h = _python_ddqn_effective_hypothesis(h_name, env_name)

    assert yaml_h.name == py_h.name
    assert yaml_h.predicted_direction == py_h.predicted_direction
    assert yaml_h.arm_key() == py_h.arm_key()
    for k in ('q_network', 'optimizer', 'replay'):
        assert yaml_h.intervention[k] == py_h.intervention[k]


# ---------- cross-env signature distinguishability ----------

def test_paired_ddqn_bootstrap_signature_matches_python(
    minatar_1M_manifest: PairedManifest,
) -> None:
    """The DDQN partial sits inside the intervention dict; its
    `claim_graph_signature` must match between YAML and Python
    paths regardless of which env we look at — the bootstrap
    binding is env-independent (only CNN slot has env binding)."""
    reg = default_dqn_registry()
    built, envs_aligned = build_paired_hypotheses(
        minatar_1M_manifest, reg=reg,
    )
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
    minatar_1M_manifest: PairedManifest,
) -> None:
    """The expanded (hypotheses, envs_aligned) tuples have
    `n_envs * n_templates` entries — so `paired_arms` zips them
    cleanly, one (h, env) pair per arm."""
    reg = default_dqn_registry()
    built, envs_aligned = build_paired_hypotheses(
        minatar_1M_manifest, reg=reg,
    )
    n_envs = len(minatar_1M_manifest.envs)
    n_templates = len(minatar_1M_manifest.hypothesis_templates)
    assert len(built) == n_envs * n_templates
    assert len(envs_aligned) == n_envs * n_templates


# ---------- helpers ----------

def _pick(
    built: tuple[Hypothesis[Mapping[str, object]], ...],
    envs_aligned: tuple[EnvConfig, ...],
    env_name: str,
    h_name: str,
) -> Hypothesis[Mapping[str, object]]:
    """Find the (env, hypothesis-name) pair in the expanded
    paired tuples."""
    for h, ec in zip(built, envs_aligned, strict=True):
        if ec.env_name == env_name and h.name == h_name:
            return h
    raise KeyError(f'no ({env_name}, {h_name}) in built tuples')
