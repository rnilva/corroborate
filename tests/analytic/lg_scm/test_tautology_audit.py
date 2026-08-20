"""Closed-form assertions on `tautology_audit` over a multi-env
LG-SCM corpus.

Four mediator candidates, each engineered to fail one (and only
one) of the audit's three independent checks — or all three to
pass simultaneously (the `clean` mediator). The distinguishing
feature of the framework's audit is that each check is an
independent flag, so a real causal mediator passes ALL three
while different artifact patterns trip different flags.

The four candidates:

| name              | mechanism                              | expected flag                  |
|-------------------|----------------------------------------|--------------------------------|
| `outcome_shadow`  | reads-set overlap with outcome's reads | `flagged_outcome` (jaccard)    |
| `hp_shadow`       | value deterministic in HP `mu_x`       | `flagged_hp = ('mu_x',)`       |
| `hp_correlated`   | mu_x + iid noise — HP-driven marginal r| `flagged_no_residual_signal`   |
| `clean`           | structural mediator Z on X→Z→Y         | (none — passes all three)      |

`hp_correlated` is the canonical HP-shadow-false-positive case
mentioned in CLAUDE.md auto-memory: a measurable that *looks*
correlated with the outcome marginally, only because both depend
on a shared HP. Under stratified Spearman ρ within a fixed `mu_x`,
the residual correlation collapses to zero — and the framework
must catch that.

Implementation: 5 envs × 30 seeds × one arm. Each env has its own
`mu_x`. Mediator candidates are injected post-hoc into each
cell's measurements (the audit reads `mediator.<name>` paths)
since they're not part of the structural SCM."""
from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace

from corroborate.analyses.diagnostic.tautology_audit import tautology_audit
from corroborate.corpus.schema import MeasurementLeaf, RunRow
from corroborate.measurables.redundancy_check import TautologyReport
from corroborate.data import cells_to_dataframe

from tests.analytic.lg_scm.composition import LinearGaussianSCM
from tests.analytic.lg_scm.runner import run_arm


# Tuned so that within-env z_mean variance is large enough that
# `clean = z_mean` has R²(z_mean, mu_x) below the 0.95 threshold
# (otherwise the clean mediator would falsely trip flagged_hp).
# σ_x large + n_steps small inflates within-env noise relative
# to between-env signal.
_SIGMA_X = 1.0
_SIGMA_Z = 0.5
_SIGMA_Y = 0.5
_BETA_XZ = 0.5
_BETA_ZY = 1.5
_N_STEPS = 20
# 60 seeds per env (~300 cells total) tightens the pooled stratified
# Spearman SE on a structurally-null ρ enough that finite-sample
# fluctuations stay reliably below the audit threshold below.
_N_PAIRS = 60
_MU_X_GRID: tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 2.5)
# Stratified-ρ threshold passed to the audit. The default 0.1 is
# sensitive at n=60-per-stratum (true-null pooled SE ~1/√(n*k-3*k) ≈
# 0.07). 0.15 absorbs typical sampling fluctuation on a
# structurally-null correlation while staying tight enough that
# clean within-stratum signal (ρ near +1) clears it easily.
_STRATIFIED_RHO_THRESHOLD = 0.15


def _scm(*, mu_x: float) -> LinearGaussianSCM:
    return LinearGaussianSCM(
        mu_x=mu_x, sigma_x=_SIGMA_X,
        beta_xz=_BETA_XZ, sigma_z=_SIGMA_Z,
        beta_zy=_BETA_ZY, sigma_y=_SIGMA_Y,
        n_steps=_N_STEPS,
    )


def _hp_correlated_value(*, mu_x: float, seed: int, env_name: str) -> float:
    """Value that depends on mu_x plus deterministic per-cell noise.

    The HP carries the marginal correlation; the noise scrambles
    within-stratum dependence with the outcome. random.Random
    seeded by (env, seed) makes the noise reproducible across test
    runs while staying independent of any signal."""
    # `hash()` on tuples uses Python's per-process randomization
    # (PYTHONHASHSEED=random); using zlib.adler32 (deterministic
    # CRC) gives reproducible noise across processes so the test
    # outcome is stable across pytest invocations.
    import zlib
    seed_key = f'hp_correlated|{env_name}|{seed}'.encode()
    rng = random.Random(zlib.adler32(seed_key) & 0xFFFF_FFFF)
    return mu_x + rng.gauss(0.0, 0.5)


def _augment_with_mediators(rows: list[RunRow]) -> list[RunRow]:
    """Add four `mediator.<name>` synthetic measurements to each
    RunRow. The audit's path resolution sees the dot in the name
    and uses it as-is."""
    out: list[RunRow] = []
    for r in rows:
        m: dict[str, MeasurementLeaf] = dict(r.measurements)
        mu_x = m.get('mu_x')
        y_mean = m.get('y_mean')
        z_mean = m.get('z_mean')
        seed_v = m.get('seed')
        env_v = m.get('env_name')
        if not isinstance(mu_x, (int, float)) or isinstance(mu_x, bool):
            raise TypeError(f'cell missing scalar mu_x: {r.id}')
        if not isinstance(y_mean, (int, float)) or isinstance(y_mean, bool):
            raise TypeError(f'cell missing scalar y_mean: {r.id}')
        if not isinstance(z_mean, (int, float)) or isinstance(z_mean, bool):
            raise TypeError(f'cell missing scalar z_mean: {r.id}')
        if not isinstance(seed_v, int) or isinstance(seed_v, bool):
            raise TypeError(f'cell missing int seed: {r.id}')
        if not isinstance(env_v, str):
            raise TypeError(f'cell missing string env_name: {r.id}')
        m['mediator.outcome_shadow'] = float(y_mean)
        m['mediator.hp_shadow'] = float(mu_x)
        m['mediator.hp_correlated'] = _hp_correlated_value(
            mu_x=float(mu_x), seed=int(seed_v), env_name=env_v,
        )
        m['mediator.clean'] = float(z_mean)
        out.append(replace(r, measurements=m))
    return out


def _build_audit_corpus() -> list[Mapping[str, object]]:
    """Single-arm corpus across 5 mu_x envs × 30 seeds, plus
    injected mediator measurements."""
    rows: list[RunRow] = []
    for mu in _MU_X_GRID:
        rows.extend(run_arm(
            _scm(mu_x=mu),
            seeds=range(_N_PAIRS),
            arm_key='single',
            env_name=f'env_mu_{mu:g}',
        ))
    augmented = _augment_with_mediators(rows)
    return [r.as_dict() for r in augmented]


_MEDIATOR_SPECS: tuple[Mapping[str, object], ...] = (
    {'name': 'mediator.outcome_shadow', 'reads': ('y_per_episode',)},
    {'name': 'mediator.hp_shadow', 'reads': ('mu_x',)},
    {'name': 'mediator.hp_correlated', 'reads': ('mu_x',)},
    {'name': 'mediator.clean', 'reads': ('z_per_episode',)},
)


def _audit() -> dict[str, TautologyReport]:
    """Run the audit on the canned corpus and return the
    `by_name` lookup. All tests share the same corpus and audit
    parameters so the four mediator stories are inspectable as
    one consistent picture."""
    cells = _build_audit_corpus()
    result = tautology_audit.fn(
        cells_to_dataframe(cells),
        measurables=_MEDIATOR_SPECS,
        outcome_path='y_mean',
        outcome_reads=('y_per_episode',),
        hp_axes=('mu_x',),
        hp_stratum_axis='mu_x',
        stratified_rho_threshold=_STRATIFIED_RHO_THRESHOLD,
    )
    return {r.measurable_name: r for r in result.reports}


# ============ Outcome-shadow ============

def test_outcome_shadow_is_flagged_by_structural_jaccard() -> None:
    """`outcome_shadow.reads` overlaps `outcome_reads` — the
    structural check should flag it. The other two checks may
    or may not also flag (the mediator equals the outcome
    by construction, so HP-R² and stratified-ρ behave similarly
    to applying them to the outcome itself)."""
    by_name = _audit()
    r = by_name['mediator.outcome_shadow']
    assert r.flagged_outcome, (
        'outcome_shadow has reads matching outcome_reads exactly; '
        'jaccard = 1.0 must trip flagged_outcome'
    )
    assert r.outcome_jaccard == 1.0
    assert not r.is_clean


# ============ HP-deterministic ============

def test_hp_shadow_is_flagged_by_hp_r_squared() -> None:
    """`hp_shadow.value == mu_x` per cell — R²(mediator, mu_x) = 1.

    Structural check passes (reads=('mu_x',) doesn't overlap
    outcome_reads); HP-R² fires with `('mu_x',)`."""
    by_name = _audit()
    r = by_name['mediator.hp_shadow']
    assert not r.flagged_outcome
    assert r.flagged_hp == ('mu_x',), (
        f'hp_shadow flagged_hp = {r.flagged_hp!r}, '
        f"expected ('mu_x',) — value is mu_x exactly"
    )
    # R² should be ≈ 1.0 (mediator IS the HP).
    r2 = r.hp_r_squared['mu_x']
    assert r2 > 0.99, f'hp_r_squared[mu_x] = {r2}, expected ≈ 1.0'
    assert not r.is_clean


# ============ HP-shadow false-positive (the headline) ============

def test_hp_correlated_caught_by_stratified_rho_check() -> None:
    """`hp_correlated` correlates with the outcome MARGINALLY (both
    depend on mu_x), but within a fixed mu_x stratum the mediator's
    noise is independent of the outcome — stratified ρ collapses to
    ≈0, p ≥ α.

    This is the canonical HP-shadow false-positive case: a measurable
    that *looks* like a real predictor on the marginal corpus but is
    structurally explained by HP variation. The framework's three-
    check audit MUST catch this — silently accepting hp_correlated
    as a clean mediator would be the methodological smuggle CLAUDE.md
    flags as critical (`findings_tautology_audit`)."""
    by_name = _audit()
    r = by_name['mediator.hp_correlated']
    # R² should be moderate (mu_x explains some but not most of the
    # variance — implementation parameters are tuned so noise dominates
    # over the HP signal).
    r2 = r.hp_r_squared['mu_x']
    assert r2 < 0.95, (
        f'hp_correlated R²(mu_x) = {r2}; tuned to be below 0.95 '
        f'so the HP-R² check does NOT fire and we isolate the '
        f'stratified-ρ check as the active flag'
    )
    # The stratified-ρ check is the load-bearing one for this case.
    assert r.flagged_no_residual_signal, (
        f'hp_correlated NOT flagged by stratified-ρ — but the '
        f'mediator carries no within-stratum signal beyond noise. '
        f'rho={r.outcome_stratified_rho:.4f}, '
        f'p={r.outcome_stratified_p:.4f}'
    )
    assert abs(r.outcome_stratified_rho) < _STRATIFIED_RHO_THRESHOLD
    assert not r.is_clean


# ============ Clean ============

def test_clean_mediator_passes_all_three_checks() -> None:
    """`clean` is the structural mediator z_mean on the X→Z→Y path.
    It must pass all three checks:

    - Structural: reads=('z_per_episode',) — disjoint from
      outcome_reads=('y_per_episode',) → jaccard=0
    - HP-R²: implementation tuned so within-env z_mean noise is large
      relative to between-env mean spread → R² < 0.95
    - Stratified-ρ: within a fixed mu_x, both z_mean and y_mean
      depend on the same X(seed) realisation → ρ ≈ +1 → no flag

    A regression that conflates 'flagged' with 'clean', or that
    short-circuits the audit, would fail this test by reporting
    a structurally-clean mediator as flagged."""
    by_name = _audit()
    r = by_name['mediator.clean']
    assert not r.flagged_outcome, (
        f'clean flagged_outcome=True; jaccard='
        f'{r.outcome_jaccard:.4f}, expected 0'
    )
    assert r.flagged_hp == (), (
        f"clean flagged_hp = {r.flagged_hp!r}, expected ()"
    )
    assert not r.flagged_no_residual_signal, (
        f'clean flagged_no_residual_signal=True; '
        f'rho={r.outcome_stratified_rho:.4f}, '
        f'p={r.outcome_stratified_p:.4f}, expected ρ ≈ +1 '
        f'(z_mean and y_mean co-vary within a stratum via X(seed))'
    )
    # Within-stratum ρ should be strongly positive.
    assert r.outcome_stratified_rho > 0.5, (
        f'clean rho={r.outcome_stratified_rho:.4f}; the structural '
        f'mediator must show within-stratum correlation with the '
        f'outcome'
    )
    assert r.is_clean, (
        f'clean reported NOT clean: outcome={r.flagged_outcome}, '
        f'hp={r.flagged_hp!r}, no_residual='
        f'{r.flagged_no_residual_signal}'
    )


# ============ Aggregate sanity ============

def test_clean_names_returns_only_the_clean_mediator() -> None:
    """Sanity: `result.clean_names` should return the single
    structurally-clean mediator. Three of four candidates are
    flagged on at least one check; only `mediator.clean` survives.
    This is the bridge-level summary surface — bridges typically
    consume `clean_names` to filter their candidate list."""
    cells = _build_audit_corpus()
    result = tautology_audit.fn(
        cells_to_dataframe(cells),
        measurables=_MEDIATOR_SPECS,
        outcome_path='y_mean',
        outcome_reads=('y_per_episode',),
        hp_axes=('mu_x',),
        hp_stratum_axis='mu_x',
        stratified_rho_threshold=_STRATIFIED_RHO_THRESHOLD,
    )
    assert result.clean_names == ('mediator.clean',), (
        f'clean_names = {result.clean_names!r}, expected '
        f"only ('mediator.clean',). Three of the four mediator "
        f'candidates were engineered to fail one check each.'
    )


# ============ mediator_path_for forwarding ============


def test_tautology_audit_forwards_mediator_path_for_to_audit_panel() -> None:
    """Regression: `mediator_path_for` is a kwarg that the public
    `tautology_audit` analysis must forward verbatim to
    `audit_mediator_panel`. A typo or dropped argument would
    silently break the explicit name → key override.

    The default path resolution looks up each mediator's value at
    `record[f'mediator.{name}']` (or `record[name]` when name has
    a dot). Implementation authors whose cells expose mediator scalars
    under BARE names (no `mediator.` prefix and no dots) would
    silently get all-NaN audits without the forwarding kwarg.

    Construction: clone the standard audit corpus but RENAME the
    `mediator.clean` key to a bare `clean_z`. Without
    `mediator_path_for`, the audit looks at `mediator.clean_z` →
    not found → degenerate result. With the forwarding, the audit
    looks at `clean_z` → finds it → returns real verdicts."""
    cells = _build_audit_corpus()
    # Replace `mediator.clean` with a bare-name key.
    bare_cells: list[Mapping[str, object]] = []
    for c in cells:
        new = dict(c)
        if 'mediator.clean' in new:
            new['clean_z'] = new.pop('mediator.clean')
        bare_cells.append(new)

    bare_specs: tuple[Mapping[str, object], ...] = (
        {'name': 'clean_z', 'reads': ('z_per_episode',)},
    )

    # Without forwarding: audit looks at `mediator.clean_z`, finds
    # nothing, returns degenerate report (all NaN signals).
    no_forward = tautology_audit.fn(
        cells_to_dataframe(bare_cells),
        measurables=bare_specs,
        outcome_path='y_mean',
        outcome_reads=('y_per_episode',),
        hp_axes=('mu_x',),
        hp_stratum_axis='mu_x',
        stratified_rho_threshold=_STRATIFIED_RHO_THRESHOLD,
    )
    no_fwd = no_forward.reports[0]
    # When mediator_path_for is not provided AND the cell-key
    # convention doesn't match, audit_mediator_panel produces a
    # NaN stratified-ρ — the metric can't be computed without
    # the column.
    import math as _math
    assert _math.isnan(no_fwd.outcome_stratified_rho), (
        f'without mediator_path_for, audit should NOT find the '
        f'bare-name mediator and the stratified-ρ should be NaN; '
        f'got {no_fwd.outcome_stratified_rho:.4f}'
    )

    # With forwarding: audit reads at `clean_z` directly. Real
    # verdicts come back; the structural mediator passes all
    # three checks (matches the `mediator.clean` case in the
    # existing test_clean_mediator_passes_all_three_checks test).
    forwarded = tautology_audit.fn(
        cells_to_dataframe(bare_cells),
        measurables=bare_specs,
        outcome_path='y_mean',
        outcome_reads=('y_per_episode',),
        hp_axes=('mu_x',),
        hp_stratum_axis='mu_x',
        stratified_rho_threshold=_STRATIFIED_RHO_THRESHOLD,
        mediator_path_for={'clean_z': 'clean_z'},
    )
    fwd = forwarded.reports[0]
    assert not _math.isnan(fwd.outcome_stratified_rho), (
        f'with mediator_path_for, audit should find the bare-name '
        f'mediator and compute real stratified-ρ; got NaN. The '
        f'kwarg forwarding contract is broken.'
    )
    # Sanity: the structural mediator (z_mean) should show strong
    # within-stratum correlation with y_mean. Same as the
    # `mediator.clean` case.
    assert fwd.outcome_stratified_rho > 0.5
