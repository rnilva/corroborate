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

from corroborate.core.intervention import Intervention
from corroborate.runner.registry import Registry
from corroborate_rl.dqn.collect import EnvConfig
from corroborate_rl.dqn.config_loader import HypothesisConfig
from corroborate_rl.dqn.yaml_sweep import (
    DQNSweep, default_dqn_registry, load_sweep,
)
from corroborate.core.signature import claim_graph_signature


REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_PATH = (
    REPO_ROOT / 'experiments' / 'configs' / 'expectile_3way.yaml'
)


# ---------- Python-authored reference ----------

def _python_hypothesis(
    name: str,
) -> HypothesisConfig:
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
    if name == 'expectile_dqn':
        boot = partial(
            bootstrap,
            greedification=partial(expectile_greedify, tau=0.7),
        )
        base['bootstrap'] = boot
        return HypothesisConfig(
            name='expectile_dqn', intervention=base, predicted_direction='a_gt_b',
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
def sweep(reg: Registry) -> DQNSweep:
    s = load_sweep(MANIFEST_PATH, reg=reg)
    assert s.arms_shape == 'chunked'
    return s


@pytest.fixture
def yaml_hypotheses(
    sweep: DQNSweep, reg: Registry,
) -> tuple[HypothesisConfig, ...]:
    return sweep.build_hypotheses(reg=reg)


# ---------- do_effect_arms semantics ----------

def test_do_effect_arms_empty_intervention_arms_yields_single_arm() -> None:
    """Empty `intervention_arms` is the chunked-mode "this template
    is one arm in a multi-template sweep" intent. `do_effect_arms()`
    returns `((),)` — a single empty arm — NOT `((), ())` (the
    legacy 2-identical-arms artifact that doubled compute and
    fingerprint-collided to two `arm_key='baseline'` cells)."""
    cfg = HypothesisConfig(
        name='vanilla_template',
        intervention={'n_step': 1},
        intervention_arms=(),
    )
    assert cfg.do_effect_arms() == ((),)


def test_do_effect_arms_nonempty_intervention_arms_yields_binary() -> None:
    """Non-empty `intervention_arms` (legacy binary shape):
    `((), intervention_arms)` — empty baseline vs treatment."""
    from corroborate_rl.dqn.claims.bootstrap import bootstrap

    iv = Intervention(slot_path='bootstrap', replacement=bootstrap)
    cfg = HypothesisConfig(
        name='binary_template',
        intervention={},
        intervention_arms=(iv,),
    )
    assert cfg.do_effect_arms() == ((), (iv,))


def test_base_hp_kwargs_strips_arm_slots_for_legacy_intervention_arms() -> None:
    """Legacy binary schema: `intervention:` self-documents the
    treatment (duplicating the arm slot). Stripping arm-slot paths
    reconstructs the vanilla baseline. The strip is necessary so
    the empty (baseline) arm falls through to dqn's default for
    that slot."""
    from corroborate_rl.dqn.claims.bootstrap import bootstrap

    iv = Intervention(slot_path='bootstrap', replacement=bootstrap)
    cfg = HypothesisConfig(
        name='binary_template',
        intervention={'gamma': 0.99, 'bootstrap': bootstrap},
        intervention_arms=(iv,),
    )
    hp = cfg.base_hp_kwargs()
    assert 'gamma' in hp
    assert hp['gamma'] == 0.99
    assert 'bootstrap' not in hp  # stripped — empty arm uses dqn default


def test_base_hp_kwargs_keeps_arm_slots_for_new_arms_schema() -> None:
    """New N-arm `arms:` schema: `intervention:` IS the base.
    Even if a slot is varied per-arm, the base value is preserved
    so the empty-tuple arm inherits it (Pearl-style "no
    intervention" control). Closes 2026-05-12 bug where
    `base_intervention.replay = Replay(50k)` got stripped because
    other arms varied `replay`."""
    from corroborate_rl.dqn.claims.replay import Replay

    base_replay = Replay(capacity=50_000, batch_size=32)
    arm_replay_small = Replay(capacity=5_000, batch_size=32)
    arms = (
        (),  # empty control — must inherit base_replay
        (Intervention(slot_path='replay', replacement=arm_replay_small),),
    )
    cfg = HypothesisConfig(
        name='replay_multi_arm',
        intervention={'gamma': 0.99, 'replay': base_replay},
        intervention_arms=(),
        arms=arms,
    )
    hp = cfg.base_hp_kwargs()
    assert hp['gamma'] == 0.99
    assert hp['replay'] is base_replay  # NOT stripped; empty arm gets this


def test_do_effect_arms_explicit_arms_takes_precedence() -> None:
    """When `arms:` is authored explicitly (new N-arm schema), it
    wins over the legacy translation regardless of
    `intervention_arms`."""
    from corroborate_rl.dqn.claims.bootstrap import bootstrap

    iv = Intervention(slot_path='bootstrap', replacement=bootstrap)
    arms_explicit = ((), (iv,), (iv,))  # 3 arms
    cfg = HypothesisConfig(
        name='triarm',
        intervention={},
        intervention_arms=(),
        arms=arms_explicit,
    )
    assert cfg.do_effect_arms() == arms_explicit


# ---------- envelope checks ----------

def test_sweep_envelope_fields(sweep: DQNSweep) -> None:
    assert sweep.name == 'expectile_3way'
    assert sweep.out_dir == Path(
        'experiments/data/expectile_3way',
    )
    assert sweep.archive_remote is None
    assert sweep.arms_shape == 'chunked'


def test_sweep_envs_tuple_matches(sweep: DQNSweep) -> None:
    expected_envs = (
        EnvConfig('Catch-bsuite', n_seeds=30, chunk_size=15),
        EnvConfig('DiscountingChain-bsuite', n_seeds=30, chunk_size=15),
        EnvConfig('MountainCar-v0', n_seeds=30, chunk_size=15),
        EnvConfig('Acrobot-v1', n_seeds=30, chunk_size=15),
        EnvConfig('FourRooms-misc', n_seeds=30, chunk_size=15),
    )
    assert sweep.envs == expected_envs


def test_sweep_hypothesis_count(
    yaml_hypotheses: tuple[HypothesisConfig, ...],
) -> None:
    assert len(yaml_hypotheses) == 3
    assert [h.name for h in yaml_hypotheses] == [
        'vanilla_dqn', 'ddqn', 'expectile_dqn',
    ]


# ---------- per-hypothesis schema-contract checks ----------

@pytest.fixture
def hypothesis_pairs(
    yaml_hypotheses: tuple[HypothesisConfig, ...],
) -> dict[str, tuple[
    HypothesisConfig,
    HypothesisConfig,
]]:
    yaml_by_name = {h.name: h for h in yaml_hypotheses}
    return {
        name: (yaml_by_name[name], _python_hypothesis(name))
        for name in ('vanilla_dqn', 'ddqn', 'expectile_dqn')
    }


@pytest.mark.parametrize(
    'h_name', ['vanilla_dqn', 'ddqn', 'expectile_dqn'],
)
def test_predicted_direction_matches(
    hypothesis_pairs: dict[str, tuple[
        HypothesisConfig,
        HypothesisConfig,
    ]],
    h_name: str,
) -> None:
    yaml_h, py_h = hypothesis_pairs[h_name]
    assert yaml_h.predicted_direction == py_h.predicted_direction


@pytest.mark.parametrize(
    'h_name', ['vanilla_dqn', 'ddqn', 'expectile_dqn'],
)
def test_intervention_leaves_match(
    hypothesis_pairs: dict[str, tuple[
        HypothesisConfig,
        HypothesisConfig,
    ]],
    h_name: str,
) -> None:
    yaml_h, py_h = hypothesis_pairs[h_name]
    for k in (
        'total_steps', 'eval_every', 'n_episodes', 'gamma',
        'sync_period',
    ):
        assert yaml_h.intervention[k] == py_h.intervention[k], (
            f'{h_name}.intervention[{k!r}] differs: '
            f'yaml={yaml_h.intervention[k]!r} '
            f'python={py_h.intervention[k]!r}'
        )


@pytest.mark.parametrize(
    'h_name', ['vanilla_dqn', 'ddqn', 'expectile_dqn'],
)
def test_module_claim_slots_equal(
    hypothesis_pairs: dict[str, tuple[
        HypothesisConfig,
        HypothesisConfig,
    ]],
    h_name: str,
) -> None:
    """Slot equality across YAML- and Python-built hypotheses.
    Compared via `canonical_str` because `functools.partial`
    instances don't define value-equality (`partial(f, x=1) !=
    partial(f, x=1)`); the framework's canonical-string fingerprint
    IS the value-equality contract for partial-baked claims."""
    from corroborate._internals.canonical import canonical_str
    yaml_h, py_h = hypothesis_pairs[h_name]
    for k in ('q_network', 'optimizer', 'replay'):
        assert canonical_str(yaml_h.intervention[k]) == canonical_str(
            py_h.intervention[k],
        )


@pytest.mark.parametrize(
    'h_name', ['ddqn', 'expectile_dqn'],
)
def test_bootstrap_signature_matches(
    hypothesis_pairs: dict[str, tuple[
        HypothesisConfig,
        HypothesisConfig,
    ]],
    h_name: str,
) -> None:
    """The headline contract: `claim_graph_signature` of the
    intervened-on slot is identical across YAML- and
    Python-authored paths. If they differ, downstream corpus
    rows tagged with the signature land in different
    structural-identity buckets."""
    yaml_h, py_h = hypothesis_pairs[h_name]
    sig_yaml = claim_graph_signature(
        yaml_h.intervention['bootstrap'],
    )
    sig_python = claim_graph_signature(
        py_h.intervention['bootstrap'],
    )
    assert sig_yaml == sig_python


@pytest.mark.parametrize(
    'h_name', ['vanilla_dqn', 'ddqn', 'expectile_dqn'],
)
def test_arm_key_matches(
    hypothesis_pairs: dict[str, tuple[
        HypothesisConfig,
        HypothesisConfig,
    ]],
    h_name: str,
) -> None:
    """Pairing key for paired_comparison. Drift here would
    place YAML and Python rows in different arms."""
    from corroborate.core.intervention import combined_arm_key

    yaml_h, py_h = hypothesis_pairs[h_name]
    yaml_keys = tuple(
        combined_arm_key(arm) for arm in yaml_h.do_effect_arms()
    )
    py_keys = tuple(
        combined_arm_key(arm) for arm in py_h.do_effect_arms()
    )
    assert yaml_keys == py_keys


def test_signatures_distinct_across_arms(
    yaml_hypotheses: tuple[HypothesisConfig, ...],
) -> None:
    """The signature is not constant — it actually distinguishes
    the three arms. ddqn != expectile_dqn at the bootstrap slot,
    confirming the registry-resolved partials aren't collapsing
    into the same hash."""
    by_name = {h.name: h for h in yaml_hypotheses}
    sig_ddqn = claim_graph_signature(
        by_name['ddqn'].intervention['bootstrap'],
    )
    sig_expectile = claim_graph_signature(
        by_name['expectile_dqn'].intervention['bootstrap'],
    )
    assert sig_ddqn != sig_expectile
