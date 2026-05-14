"""Schema-contract smoke for the `expectile_3way` YAML sweep:
every authored slot resolves to the typed handle the substrate's
Claim graph holds.

The reference Python construction below is one realisation of
the same schema — useful for catching loader regressions, but
the canonical contract is the YAML schema itself: a config-
bundle slot resolves to a frozen-dataclass instance, a slot
binding to a `partial`-of-FnClaim resolves with the right inner
FnClaim, and `claim_graph_signature` is stable across loads."""
from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path

import pytest

from corroborate.core.intervention import Intervention, combined_arm_key
from corroborate.runner.registry import Registry
from corroborate_rl.dqn.collect import EnvConfig
from corroborate_rl.dqn.config_loader import InterventionConfig
from corroborate_rl.dqn.yaml_sweep import (
    DQNSweep, default_dqn_registry, load_sweep,
)
from corroborate.core.signature import claim_graph_signature


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = (
    REPO_ROOT / 'experiments' / 'configs' / 'expectile_3way.yaml'
)


# ---------- Python-authored reference ----------

def _python_intervention(
    name: str,
) -> InterventionConfig:
    """Canonical Python recipe for the expectile_3way cohort —
    the reference that `expectile_3way.yaml` must match
    structurally."""
    from corroborate_rl.dqn.claims.bootstrap import (
        bootstrap, double_greedify, expectile_greedify,
    )
    from corroborate_rl.dqn.claims.optimizer import adam, warmed_update
    from corroborate_rl.dqn.claims.q_network import MLP
    from corroborate_rl.dqn.claims.replay import Replay
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
        'q_network': MLP(hidden=(64, 64)),
    }
    if name == 'vanilla_dqn':
        return InterventionConfig(
            name='vanilla_dqn', base=base, arms=((),),
        )
    if name == 'ddqn':
        boot = partial(bootstrap, greedification=double_greedify)
        return InterventionConfig(
            name='ddqn', base=base,
            arms=(
                (),
                (Intervention(slot_path='bootstrap', replacement=boot),),
            ),
        )
    if name == 'expectile_dqn':
        boot = partial(
            bootstrap,
            greedification=partial(expectile_greedify, tau=0.7),
        )
        return InterventionConfig(
            name='expectile_dqn', base=base,
            arms=(
                (),
                (Intervention(slot_path='bootstrap', replacement=boot),),
            ),
        )
    raise ValueError(name)


# ---------- fixtures ----------

@pytest.fixture
def reg() -> Registry:
    return default_dqn_registry()


@pytest.fixture
def sweep(reg: Registry) -> DQNSweep:
    s = load_sweep(MANIFEST_PATH, reg=reg)
    assert s.env_binding == 'shared'
    return s


@pytest.fixture
def yaml_interventions(
    sweep: DQNSweep, reg: Registry,
) -> tuple[InterventionConfig, ...]:
    return sweep.build_interventions(reg=reg)


# ---------- do_effect_arms / base_hp_kwargs semantics ----------

def test_do_effect_arms_defaults_to_single_empty_arm() -> None:
    """No `arms` declared → single empty control arm `((),)`.
    The empty arm is the Pearl-style "no intervention" baseline;
    one cell per (env, seed) at the base config."""
    cfg = InterventionConfig(
        name='vanilla_template',
        base={'n_step': 1},
    )
    assert cfg.do_effect_arms() == ((),)


def test_do_effect_arms_returns_arms_unchanged() -> None:
    """`do_effect_arms()` is identity: the field IS the canonical
    arm representation. Multi-arm contrast survives the call."""
    from corroborate_rl.dqn.claims.bootstrap import bootstrap

    iv = Intervention(slot_path='bootstrap', replacement=bootstrap)
    arms = ((), (iv,), (iv,))  # 3 arms
    cfg = InterventionConfig(
        name='triarm',
        base={},
        arms=arms,
    )
    assert cfg.do_effect_arms() == arms


def test_base_hp_kwargs_returns_base() -> None:
    """`base` IS the partial(dqn, **kwargs) kwargs map. Arm
    interventions override at dispatch via partial precedence;
    `base` itself never strips per-arm slot values (Pearl-style
    "no intervention" empty arm inherits whatever base sets)."""
    from corroborate_rl.dqn.claims.replay import Replay

    base_replay = Replay(capacity=50_000, batch_size=32)
    arm_replay_small = Replay(capacity=5_000, batch_size=32)
    arms = (
        (),  # empty control inherits base_replay
        (Intervention(slot_path='replay', replacement=arm_replay_small),),
    )
    cfg = InterventionConfig(
        name='replay_multi_arm',
        base={'gamma': 0.99, 'replay': base_replay},
        arms=arms,
    )
    hp = cfg.base_hp_kwargs()
    assert hp['gamma'] == 0.99
    assert hp['replay'] is base_replay


# ---------- envelope checks ----------

def test_sweep_envelope_fields(sweep: DQNSweep) -> None:
    assert sweep.name == 'expectile_3way'
    assert sweep.out_dir == Path(
        'experiments/data/expectile_3way',
    )
    assert sweep.archive_remote is None
    assert sweep.env_binding == 'shared'


def test_sweep_envs_tuple_matches(sweep: DQNSweep) -> None:
    expected_envs = (
        EnvConfig('Catch-bsuite', n_seeds=30, chunk_size=15),
        EnvConfig('DiscountingChain-bsuite', n_seeds=30, chunk_size=15),
        EnvConfig('MountainCar-v0', n_seeds=30, chunk_size=15),
        EnvConfig('Acrobot-v1', n_seeds=30, chunk_size=15),
        EnvConfig('FourRooms-misc', n_seeds=30, chunk_size=15),
    )
    assert sweep.envs == expected_envs


def test_sweep_intervention_count(
    yaml_interventions: tuple[InterventionConfig, ...],
) -> None:
    assert len(yaml_interventions) == 3
    assert [h.name for h in yaml_interventions] == [
        'vanilla_dqn', 'ddqn', 'expectile_dqn',
    ]


# ---------- per-intervention schema-contract checks ----------

@pytest.fixture
def intervention_pairs(
    yaml_interventions: tuple[InterventionConfig, ...],
) -> dict[str, tuple[
    InterventionConfig,
    InterventionConfig,
]]:
    yaml_by_name = {h.name: h for h in yaml_interventions}
    return {
        name: (yaml_by_name[name], _python_intervention(name))
        for name in ('vanilla_dqn', 'ddqn', 'expectile_dqn')
    }


@pytest.mark.parametrize(
    'h_name', ['vanilla_dqn', 'ddqn', 'expectile_dqn'],
)
def test_base_leaves_match(
    intervention_pairs: dict[str, tuple[
        InterventionConfig,
        InterventionConfig,
    ]],
    h_name: str,
) -> None:
    yaml_h, py_h = intervention_pairs[h_name]
    for k in (
        'total_steps', 'eval_every', 'n_episodes', 'gamma',
        'sync_period',
    ):
        assert yaml_h.base[k] == py_h.base[k], (
            f'{h_name}.base[{k!r}] differs: '
            f'yaml={yaml_h.base[k]!r} '
            f'python={py_h.base[k]!r}'
        )


@pytest.mark.parametrize(
    'h_name', ['vanilla_dqn', 'ddqn', 'expectile_dqn'],
)
def test_module_claim_slots_equal(
    intervention_pairs: dict[str, tuple[
        InterventionConfig,
        InterventionConfig,
    ]],
    h_name: str,
) -> None:
    """Slot equality across YAML- and Python-built interventions.
    Compared via `canonical_str` because `functools.partial`
    instances don't define value-equality (`partial(f, x=1) !=
    partial(f, x=1)`); the framework's canonical-string fingerprint
    IS the value-equality contract for partial-baked claims."""
    from corroborate._internals.canonical import canonical_str
    yaml_h, py_h = intervention_pairs[h_name]
    for k in ('q_network', 'optimizer', 'replay'):
        assert canonical_str(yaml_h.base[k]) == canonical_str(
            py_h.base[k],
        )


@pytest.mark.parametrize(
    'h_name', ['ddqn', 'expectile_dqn'],
)
def test_bootstrap_signature_matches(
    intervention_pairs: dict[str, tuple[
        InterventionConfig,
        InterventionConfig,
    ]],
    h_name: str,
) -> None:
    """The headline contract: `claim_graph_signature` of the
    treatment arm's slot replacement is identical across YAML-
    and Python-authored paths. If they differ, downstream corpus
    rows tagged with the signature land in different
    structural-identity buckets."""
    yaml_h, py_h = intervention_pairs[h_name]
    yaml_repl = yaml_h.arms[1][0].replacement
    py_repl = py_h.arms[1][0].replacement
    assert claim_graph_signature(yaml_repl) == claim_graph_signature(py_repl)


@pytest.mark.parametrize(
    'h_name', ['vanilla_dqn', 'ddqn', 'expectile_dqn'],
)
def test_arm_keys_match(
    intervention_pairs: dict[str, tuple[
        InterventionConfig,
        InterventionConfig,
    ]],
    h_name: str,
) -> None:
    """Pairing key for paired_comparison. Drift here would
    place YAML and Python rows in different arms."""
    yaml_h, py_h = intervention_pairs[h_name]
    yaml_keys = tuple(
        combined_arm_key(arm) for arm in yaml_h.do_effect_arms()
    )
    py_keys = tuple(
        combined_arm_key(arm) for arm in py_h.do_effect_arms()
    )
    assert yaml_keys == py_keys


def test_signatures_distinct_across_arms(
    yaml_interventions: tuple[InterventionConfig, ...],
) -> None:
    """The signature is not constant — it actually distinguishes
    the three interventions. ddqn != expectile_dqn at the bootstrap
    slot, confirming the registry-resolved partials aren't
    collapsing into the same hash."""
    by_name = {h.name: h for h in yaml_interventions}
    sig_ddqn = claim_graph_signature(
        by_name['ddqn'].arms[1][0].replacement,
    )
    sig_expectile = claim_graph_signature(
        by_name['expectile_dqn'].arms[1][0].replacement,
    )
    assert sig_ddqn != sig_expectile
