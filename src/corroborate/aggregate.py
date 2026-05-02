"""Aggregate — sweep → ComparisonRow hand-off.

The framework's sweep emits `list[RunRow]`; downstream consumers
need per-(treatment, baseline) ComparisonRows. This module
provides the typed factory functions for that hand-off.

The framework is *substrate-agnostic*: the outcome path being
aggregated is supplied by the caller as `outcome_path: str` on
every entry point. The framework reads `measurements[outcome_path]`
off each run, writes `{outcome_path}.effect_size_g` / `se` /
`derived_q` / etc. on the resulting ComparisonRow. v0's RL
substrate authors `outcome.late_window_mean`; other substrates or
other outcomes pass their own keys.

Structure:

- `leaf_signature(measurements)` — the configurational fingerprint
  used as a group-by key. Filters out outcome/bridge/invariant
  paths and per-cell metadata keys, returns sorted (path, str)
  pairs. "Leaf" because each entry is a non-recursive scalar
  claim of the configured composition (RL practice calls these
  hyperparameters; the framework name is `leaf`).
- `aggregate_cell_verdict(verdicts)` — Popperian aggregation over
  per-bridge verdicts (any single refutation refutes; INVARIANT_
  VIOLATION dominates).
- `paired_comparison_from_runs(treatment_runs, baseline_runs, *,
  outcome_path, predicted_direction, ...)` — paired-by-seed Δ on
  `outcome_path`; computes Hedges' g + SE + Popperian verdict.
- `link_pearson_across_groups(mechanism_comparisons,
  outcome_comparisons, *, mechanism_path, outcome_path,
  group_label, ...)` — cross-group Pearson r between mechanism
  and outcome effect sizes (groups typically envs)."""
from __future__ import annotations

import math
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from corroborate.bridge import BridgeResult
from corroborate.hypothesis import Hypothesis, PredictedDirection
from corroborate.schema import (
    ComparisonRow,
    FactRow,
    GroupStats,
    HypothesisComparisonRow,
    MeasurementLeaf,
    RunRow,
)
from corroborate.statistics import (
    PooledStats,
    adequately_powered_paired,
    delta_i_from_q,
    derived_q_from_g_se,
    hedges_g_paired,
    random_effects_summary,
    random_effects_verdict,
    verdict_from_paired_stats,
)
from corroborate.verdict import RefutationClass, Verdict


# ============ Leaf-signature projection ============

# Output-side prefixes — paths whose values are observed at run
# time, not authored at composition. Filtered out of the
# configurational fingerprint.
_OUTPUT_PREFIXES: tuple[str, ...] = ('outcome.', 'bridge.', 'invariant.')

# Always-excluded framework-typed metadata. These are
# substrate-AGNOSTIC: every RunRow carries `intervention_name`
# (the Hypothesis name), so it's never a configurational leaf.
_FRAMEWORK_EXCLUDED_KEYS: frozenset[str] = frozenset({
    'intervention_name',
})


def leaf_signature(
    measurements: Mapping[str, MeasurementLeaf],
    *,
    exogenous_keys: frozenset[str] = frozenset(),
) -> tuple[tuple[str, str], ...]:
    """The configurational fingerprint — leaf-only subset of
    `measurements` as a sorted (path, str-canonical-value) tuple.
    Hashable; suitable as a group-by key.

    Filters out:
    - Output paths (`outcome.`/`bridge.`/`invariant.`).
    - Every registered-measurable name (registry-based filter).
      Becomes load-bearing once Phase 5 drops the substrate-paper-
      narrative prefixes; today it's a superset of the prefix
      filter (every registered measurable's column name still
      matches one of the existing prefixes), so behaviour is
      unchanged.
    - The framework-typed `intervention_name` (always excluded).
    - Substrate-supplied exogenous keys: keys the substrate
      declared via `Annotated[T, Exogenous]` on its `@claim`'s
      kwargs. Caller passes those names as `exogenous_keys` (e.g.
      `frozenset({'env_name', 'seed', 'total_steps'})` for the RL
      substrate). The framework does NOT hardcode RL key names.

    What remains is the configurational leaves at their dotted
    topology paths. "Leaf" rather than "HP": a leaf-regime kwarg
    is a non-recursive scalar claim of the configured composition,
    observed at composition time."""
    from corroborate.measurable import registered_names
    excluded = (
        _FRAMEWORK_EXCLUDED_KEYS
        | exogenous_keys
        | frozenset(registered_names())
    )
    return tuple(sorted(
        (k, str(v))
        for k, v in measurements.items()
        if not any(k.startswith(p) for p in _OUTPUT_PREFIXES)
        and k not in excluded
    ))


# ============ Cell-level verdict aggregator (shared) ============

def aggregate_cell_verdict(verdicts: tuple[Verdict, ...]) -> Verdict:
    """Cell-level verdict from per-bridge verdicts. Popperian
    aggregation: any single refutation refutes.

    Precedence (highest first):
    1. Any `Verdict.INVARIANT_VIOLATION` (a tautological-tagged
       gap exceeded its scope-commitment threshold; the run sat
       outside the theorem's domain — outcome verdicts are out of
       scope per axiom 18).
    2. Any `Verdict.NO_EFFECT` → NO_EFFECT (one bridge refuted
       is enough; the hypothesis as a whole is refuted under this
       cell — Popperian falsification).
    3. All `Verdict.HELD` → HELD.
    4. Otherwise (mixed HELD + POWER_INSUFFICIENT) →
       `POWER_INSUFFICIENT` (cannot tell).
    Empty input → `POWER_INSUFFICIENT` (no test was performed)."""
    if not verdicts:
        return Verdict.POWER_INSUFFICIENT
    if any(v is Verdict.INVARIANT_VIOLATION for v in verdicts):
        return Verdict.INVARIANT_VIOLATION
    if any(v is Verdict.NO_EFFECT for v in verdicts):
        return Verdict.NO_EFFECT
    if all(v is Verdict.HELD for v in verdicts):
        return Verdict.HELD
    return Verdict.POWER_INSUFFICIENT


# ============ Paired-by-seed comparison (v9-port stats) ============

def _resolved_timestamp(timestamp: str | None) -> str:
    return (
        timestamp
        if timestamp is not None
        else datetime.now(UTC).isoformat(timespec='seconds')
    )


def _run_intervention_name(run: RunRow) -> str:
    v = run.measurements.get('intervention_name')
    if not isinstance(v, str):
        raise TypeError(
            f"RunRow {run.id!r} missing 'intervention_name' measurement"
        )
    return v


def _pair_runs_by_key(
    treatment: Sequence[RunRow], baseline: Sequence[RunRow],
    *,
    pair_by: tuple[str, ...],
    dup_check: bool = False,
    group_label: object | None = None,
) -> tuple[
    list[tuple[MeasurementLeaf, ...]],
    dict[tuple[MeasurementLeaf, ...], RunRow],
    dict[tuple[MeasurementLeaf, ...], RunRow],
    int,
]:
    """Index treatment + baseline runs by `pair_by`-key; return
    paired keys (sorted), the two key→RunRow dicts, and the count
    of unpaired-and-dropped runs.

    `dup_check=True` raises `ValueError` on duplicate pair-keys
    within either arm — silent dedup would mask a misconfigured
    slice. Stratified aggregators set this; the marginal
    `paired_comparison_from_runs` doesn't (legacy contract)."""
    def _index(
        runs: Sequence[RunRow], side: str,
    ) -> dict[tuple[MeasurementLeaf, ...], RunRow]:
        out: dict[tuple[MeasurementLeaf, ...], RunRow] = {}
        for r in runs:
            pk = _run_pair_key(r, pair_by)
            if dup_check and pk in out:
                tag = (
                    f' for group {group_label!r}'
                    if group_label is not None else ''
                )
                raise ValueError(
                    f'duplicate pair_by={pair_by!r} key {pk!r} '
                    f'in {side}{tag}',
                )
            out[pk] = r
        return out

    t_by_key = _index(treatment, 'treatment')
    b_by_key = _index(baseline, 'baseline')
    paired = sorted(t_by_key.keys() & b_by_key.keys())
    n_dropped = (
        (len(t_by_key) - len(paired))
        + (len(b_by_key) - len(paired))
    )
    return paired, t_by_key, b_by_key, n_dropped


def _run_pair_key(
    run: RunRow, pair_by: tuple[str, ...],
) -> tuple[MeasurementLeaf, ...]:
    """Read the tuple of measurement values at `pair_by` keys.
    Used as a hashable index for paired comparisons. Loud error
    if any key is missing or non-scalar."""
    out: list[MeasurementLeaf] = []
    for k in pair_by:
        v = run.measurements.get(k)
        if v is None or not isinstance(v, (str, int, float, bool)):
            raise TypeError(
                f"RunRow {run.id!r} missing scalar pair-key "
                f"{k!r} (pair_by={pair_by!r})",
            )
        out.append(v)
    return tuple(out)


def _run_outcome(
    run: RunRow,
    outcome_path: str,
) -> float:
    """Read a scalar outcome measurement off a run; loud error if
    absent or non-numeric. The substrate authors `outcome_path` as
    the path-keyed measurement; the framework reads it back here."""
    v = run.measurements.get(outcome_path)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise TypeError(
            f"RunRow {run.id!r} missing scalar {outcome_path!r} "
            f"measurement"
        )
    return float(v)


def paired_comparison_from_runs(
    treatment_runs: Sequence[RunRow],
    baseline_runs: Sequence[RunRow],
    *,
    outcome_path: str,
    pair_by: tuple[str, ...],
    predicted_direction: PredictedDirection | None,
    alpha: float = 0.05,
    power: float = 0.8,
    cycle_id: str | None = None,
    timestamp: str | None = None,
    extra_measurements: Mapping[str, MeasurementLeaf] | None = None,
) -> ComparisonRow:
    """Paired-by-`pair_by` ComparisonRow.

    Substrate-agnostic: `pair_by` names the measurement keys that
    identify a matched (treatment, baseline) pair. For RL,
    `pair_by=('seed',)` (when grouping is per-env) or
    `pair_by=('env_name', 'seed')` (when one comparison spans
    multiple envs). For non-RL substrates, whatever the matching
    axis is.

    Pairs by tuple-of-values at `pair_by`, drops unmatched pairs,
    computes Δ_i = treatment_i − baseline_i across pairs, fits
    Hedges' g + SE on the Δ distribution, derives MDE and verdict
    via `corroborate.statistics`. Same shape as v9's paired
    comparison (dialectic/hypothesis.py:332-350).

    Treatment and baseline must share at least one pair-key value
    for a valid comparison; if the intersection is empty, returns
    a POWER_INSUFFICIENT row with NaN stats."""
    from corroborate.statistics import (
        delta_i_from_q,
        derived_q_from_g_se,
        hedges_g_paired,
        verdict_from_paired_stats,
    )

    if not treatment_runs or not baseline_runs:
        raise ValueError(
            'paired_comparison_from_runs requires non-empty '
            'treatment_runs and baseline_runs'
        )
    if not pair_by:
        raise ValueError(
            'pair_by must be non-empty — the substrate must name '
            'the measurement key(s) that identify matched pairs.'
        )

    paired_keys, treatment_by_key, baseline_by_key, _ = (
        _pair_runs_by_key(
            treatment_runs, baseline_runs, pair_by=pair_by,
        )
    )

    deltas: list[float] = [
        _run_outcome(treatment_by_key[k], outcome_path)
        - _run_outcome(baseline_by_key[k], outcome_path)
        for k in paired_keys
    ]
    n_pairs = len(deltas)

    g, se = hedges_g_paired(deltas) if n_pairs >= 2 else (
        float('nan'), float('nan'),
    )
    q = derived_q_from_g_se(g, se)
    delta_i = delta_i_from_q(q)
    verdict, refutation_class, is_powered = verdict_from_paired_stats(
        g, se, n_pairs,
        predicted_direction=predicted_direction,
        alpha=alpha, power=power,
    )

    treatment_intervention = _run_intervention_name(treatment_runs[0])

    measurements: dict[str, MeasurementLeaf] = {
        'intervention_name': treatment_intervention,
        'n_treatment': len(treatment_runs),
        'n_baseline': len(baseline_runs),
        'n_pairs': n_pairs,
    }
    if not math.isnan(g):
        measurements[f'{outcome_path}.effect_size_g'] = g
    if not math.isnan(se):
        measurements[f'{outcome_path}.se'] = se
    if not math.isnan(q):
        measurements[f'{outcome_path}.derived_q'] = q
    measurements[f'{outcome_path}.delta_i_population'] = delta_i

    if extra_measurements is not None:
        measurements.update(extra_measurements)

    return ComparisonRow(
        id=str(uuid.uuid4()),
        parent_id=None,
        cycle_id=cycle_id,
        timestamp=_resolved_timestamp(timestamp),
        treatment_arm_id='',  # paired path doesn't materialise arms
        baseline_arm_id='',
        predicted_direction=predicted_direction,
        verdict=verdict,
        refutation_class=refutation_class,
        adequately_powered=is_powered,
        measurements=measurements,
    )


# ============ Cross-group link verdict (PAPER_NOTES.md §3.5) ============

def paired_deltas_from_runs(
    treatment_runs: Sequence[RunRow],
    baseline_runs: Sequence[RunRow],
    *,
    paths: Sequence[str],
    pair_by: tuple[str, ...],
) -> dict[str, list[float]]:
    """Per-pair Δ values (treatment − baseline) for each path.

    For within-stratum mediator search: regress Δ_outcome on
    Δ_mediator across pairs to test whether DDQN's effect on the
    mediator predicts DDQN's effect on the outcome at the
    seed-pair level. The cross-env analog is
    `link_pearson_across_groups`; this is the within-env version.

    Pairs by tuple-of-values at `pair_by` (typically `('seed',)`).
    Drops unmatched pairs. For each path, drops pairs where
    either side's measurement is non-finite or missing.

    Returns a dict `{path: [Δ_pair_0, Δ_pair_1, ...]}`. Lengths
    can differ across paths if some pairs are missing for some
    paths; the caller handles index alignment if needed.

    Use `scipy.stats.pearsonr` (or spearmanr) on the resulting
    vectors to score each candidate mediator's correlation with
    the outcome's Δ."""
    if not pair_by:
        raise ValueError(
            'paired_deltas_from_runs: pair_by must be non-empty',
        )
    paired_keys, treatment_by_key, baseline_by_key, _ = (
        _pair_runs_by_key(
            treatment_runs, baseline_runs, pair_by=pair_by,
        )
    )

    out: dict[str, list[float]] = {}
    for path in paths:
        deltas: list[float] = []
        for k in paired_keys:
            t = treatment_by_key[k].measurements.get(path)
            b = baseline_by_key[k].measurements.get(path)
            if not isinstance(t, (int, float)) or isinstance(t, bool):
                continue
            if not isinstance(b, (int, float)) or isinstance(b, bool):
                continue
            tf, bf = float(t), float(b)
            if not (math.isfinite(tf) and math.isfinite(bf)):
                continue
            deltas.append(tf - bf)
        out[path] = deltas
    return out


def link_pearson_across_groups(
    mechanism_comparisons: Sequence[ComparisonRow],
    outcome_comparisons: Sequence[ComparisonRow],
    *,
    mechanism_path: str,
    outcome_path: str,
    group_by: str,
    alpha: float = 0.05,
    power: float = 0.8,
    cycle_id: str | None = None,
    timestamp: str | None = None,
    extra_measurements: Mapping[str, MeasurementLeaf] | None = None,
) -> ComparisonRow:
    """Cross-group link verdict — Pearson r between mechanism Δ
    and outcome Δ across groups.

    PAPER_NOTES.md §3.5: 'a methodological intervention's mechanism
    fingerprint should covary with its outcome fingerprint across
    [groups].' For RL, the group is `env_name`. For non-RL
    substrates, whatever single key identifies a group across
    treatment/baseline pairs.

    `group_by` is the substrate-named measurement key — caller
    passes `'env_name'` for RL or e.g. `'patient_id'` for clinical.
    Pairs `mechanism_comparisons[g]` and `outcome_comparisons[g]`
    by their `measurements[group_by]`; mismatches drop quietly
    (only fully-paired groups contribute to the Pearson r).

    Returns a ComparisonRow tagged at the cross-group granularity;
    its measurements include the Pearson r as the effect-size
    statistic, a derived MDE-style verdict on r vs 0, and the
    `group_by` key recorded as `'group_by': <key>`."""
    from corroborate.statistics import (
        adequately_powered_paired,
        delta_i_from_q,
        derived_q_from_g_se,
    )

    def _group_id(c: ComparisonRow) -> str:
        v = c.measurements.get(group_by)
        if not isinstance(v, str):
            raise TypeError(
                f"ComparisonRow {c.id!r} missing scalar string "
                f"measurement {group_by!r} for grouping",
            )
        return v

    mech_by_group = {_group_id(c): c for c in mechanism_comparisons}
    out_by_group = {_group_id(c): c for c in outcome_comparisons}
    paired_groups = sorted(mech_by_group.keys() & out_by_group.keys())

    mech_gs: list[float] = []
    out_gs: list[float] = []
    for g_id in paired_groups:
        mg = mech_by_group[g_id].measurements.get(mechanism_path)
        og = out_by_group[g_id].measurements.get(outcome_path)
        if isinstance(mg, (int, float)) and isinstance(og, (int, float)):
            if not (math.isnan(float(mg)) or math.isnan(float(og))):
                mech_gs.append(float(mg))
                out_gs.append(float(og))

    n = len(mech_gs)
    if n < 3:
        # Pearson's r needs at least 3 paired observations to be
        # well-defined; return a POWER_INSUFFICIENT row.
        underpowered_measurements: dict[str, MeasurementLeaf] = {
            'intervention_name': 'link',
            'n_paired_groups': n,
            'group_by': group_by,
        }
        if extra_measurements is not None:
            underpowered_measurements.update(extra_measurements)
        return ComparisonRow(
            id=str(uuid.uuid4()), parent_id=None, cycle_id=cycle_id,
            timestamp=_resolved_timestamp(timestamp),
            treatment_arm_id='', baseline_arm_id='',
            predicted_direction='a_gt_b',
            verdict=Verdict.POWER_INSUFFICIENT,
            refutation_class=RefutationClass.UNDERPOWERED,
            adequately_powered=False,
            measurements=underpowered_measurements,
        )

    # Pearson r with one-sided test (predicted positive).
    r = float(_pearson(mech_gs, out_gs))
    # SE under H0 (Fisher z-transform approximation): SE_r = 1/sqrt(n-3).
    se_r = 1.0 / math.sqrt(n - 3) if n > 3 else float('nan')
    q = derived_q_from_g_se(r, se_r) if not math.isnan(se_r) else float('nan')
    delta_i = delta_i_from_q(q) if not math.isnan(q) else 0.0
    is_powered = adequately_powered_paired(
        r, n, alpha=alpha, power=power, alternative='larger',
    )

    if not is_powered:
        verdict = Verdict.POWER_INSUFFICIENT
        refutation = RefutationClass.UNDERPOWERED
    elif r > 0:
        verdict = Verdict.HELD
        refutation = None
    else:
        verdict = Verdict.NO_EFFECT
        refutation = RefutationClass.SIGN_FLIP

    measurements: dict[str, MeasurementLeaf] = {
        'intervention_name': 'link',
        'n_paired_groups': n,
        'group_by': group_by,
        'link.pearson_r': r,
        'link.se': se_r,
        'link.derived_q': q,
        'link.delta_i_population': delta_i,
    }
    if extra_measurements is not None:
        measurements.update(extra_measurements)

    return ComparisonRow(
        id=str(uuid.uuid4()), parent_id=None, cycle_id=cycle_id,
        timestamp=_resolved_timestamp(timestamp),
        treatment_arm_id='', baseline_arm_id='',
        predicted_direction='a_gt_b',
        verdict=verdict,
        refutation_class=refutation,
        adequately_powered=is_powered,
        measurements=measurements,
    )


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson r — population formula. Returns 0.0 for zero-
    variance inputs (degenerate; deferred verdict)."""
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    sx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    sy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if sx == 0.0 or sy == 0.0:
        return 0.0
    return num / (sx * sy)


# ============ FactRow projection ============

def _binary_entropy(p: float) -> float:
    """H₂(p) = −(p log₂ p + (1−p) log₂ (1−p)). 0 at p=0 or p=1."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p))


def _verdict_q(natural_strength: float, verdict: Verdict) -> float:
    """Convert natural [0,1] strength + verdict → posterior P(claim
    is true).

    HELD pulls q above 0.5 by `0.5 * strength`; NO_EFFECT pulls
    below; everything else (POWER_INSUFFICIENT,
    INVARIANT_VIOLATION) sits at 0.5 (no information). The
    factor-of-2 stretch makes a maximal |ρ|=1 reach q=1 (or 0)
    → 1 bit of ΔI."""
    s = max(0.0, min(1.0, natural_strength))
    if verdict is Verdict.HELD:
        return 0.5 + 0.5 * s
    if verdict is Verdict.NO_EFFECT:
        return 0.5 - 0.5 * s
    return 0.5


def _delta_i_oriented(
    natural_strength: float, verdict: Verdict,
) -> float:
    """Verdict-oriented ΔI = 1 − H₂(q_oriented), in bits. Symmetric:
    HELD with strength 1 and NO_EFFECT with strength 1 both → 1
    bit. INVARIANT_VIOLATION → 0."""
    if verdict is Verdict.INVARIANT_VIOLATION:
        return 0.0
    q = _verdict_q(natural_strength, verdict)
    return 1.0 - _binary_entropy(q)


def natural_strength_from_stats(
    stats: Mapping[str, float | int | bool | str],
) -> float:
    """Compute a bridge's natural [0,1] strength from its
    sufficient statistic. Verdict-agnostic — interpretation
    (HELD / NO_EFFECT) happens at delta_i computation time.

    Looks for these stat keys, in order:
    - `rho` — Pearson / Spearman correlation; |ρ| capped at 1.
    - `partial_at` + `partial_bt` — paired partial correlations;
      min(|.|, 1).
    - `value` + `threshold` — a bounded-magnitude invariant; the
      strength is `1 - |value| / threshold` (margin to bound).
    - `ate` — average treatment effect; tanh(|.|) maps to (0, 1).

    Returns 0.0 when nothing matches — the caller falls back to
    evidentiary-level placeholders."""
    rho = stats.get('rho')
    if isinstance(rho, (int, float)) and not math.isnan(float(rho)):
        return float(min(abs(rho), 1.0))
    partial_at = stats.get('partial_at')
    partial_bt = stats.get('partial_bt')
    if (isinstance(partial_at, (int, float))
            and isinstance(partial_bt, (int, float))):
        return float(min(abs(partial_at), abs(partial_bt), 1.0))
    value = stats.get('value')
    threshold = stats.get('threshold')
    if (isinstance(value, (int, float))
            and isinstance(threshold, (int, float))
            and threshold > 0):
        margin = 1.0 - abs(value) / threshold
        return float(max(0.0, min(margin, 1.0)))
    ate = stats.get('ate')
    if isinstance(ate, (int, float)):
        return float(math.tanh(abs(ate)))
    return 0.0


def _evidentiary_level_from_bridge_result(r: BridgeResult) -> str:
    """Per-result evidentiary tier label. 'causal_bridged' is a
    graph-level promotion (≥2 paired admits) computed elsewhere;
    the per-result tier is at most causal_one_sided."""
    if r.verdict is not Verdict.HELD:
        return 'refuted'
    tier = r.stats.get('tier')
    if tier == 'interventional':
        return 'causal_one_sided'
    return 'correlational'


def _grouped_bridge_fields(
    run: RunRow,
) -> dict[str, dict[str, MeasurementLeaf]]:
    """Group `(bridge|invariant).<name>.<field>` measurements by
    bridge name. Shared between `_facts_from_runrow` and
    `reconstruct_bridge_results` — both walk the same prefix
    structure persisted by `_bridge_result_to_measurements`."""
    by_name: dict[str, dict[str, MeasurementLeaf]] = {}
    for k, v in run.measurements.items():
        if k.startswith('bridge.'):
            prefix_len = len('bridge.')
        elif k.startswith('invariant.'):
            prefix_len = len('invariant.')
        else:
            continue
        rest = k[prefix_len:]  # `<name>.<rest>`
        if '.' not in rest:
            continue
        name, field_path = rest.split('.', 1)
        by_name.setdefault(name, {})[field_path] = v
    return by_name


def _targets_from_fields(
    fields: Mapping[str, MeasurementLeaf],
) -> tuple[str, ...]:
    """Decode a comma-joined `targets` measurement into the bridge's
    reads-set tuple. Empty string → empty tuple. Missing targets
    field (older corpus or hand-constructed RunRow) → empty tuple."""
    targets_v = fields.get('targets')
    if not isinstance(targets_v, str) or not targets_v:
        return ()
    return tuple(targets_v.split(','))


def _stats_from_fields(
    fields: Mapping[str, MeasurementLeaf],
) -> dict[str, float | int | bool | str]:
    """Pull `stats.<k>` measurements into the BridgeResult-shaped
    stats dict. Non-MeasurementLeaf values are skipped."""
    stats: dict[str, float | int | bool | str] = {}
    for fk, fv in fields.items():
        if fk.startswith('stats.') and isinstance(
            fv, (int, float, bool, str),
        ):
            stats[fk[len('stats.'):]] = fv
    return stats


def reconstruct_bridge_results(
    run: RunRow,
) -> tuple[BridgeResult, ...]:
    """Inverse of `_bridge_result_to_measurements`.
    Walks `run.measurements` for `bridge.<name>.*` and
    `invariant.<name>.*` groups and rebuilds a tuple of
    BridgeResults.

    Lossless round-trip of the five BridgeResult fields:
    `name` (the path's middle component), `verdict` (from
    `<prefix>.verdict`), `reason` (from `<prefix>.reason`),
    `targets` (from `<prefix>.targets`), `stats` (from
    `<prefix>.stats.<k>`).

    Empty tuple if the run carries no bridge/invariant
    measurements. Invalid verdict strings or missing verdict
    fields skip the corresponding bridge silently — same defensive
    posture as `_facts_from_runrow`."""
    by_name = _grouped_bridge_fields(run)
    out: list[BridgeResult] = []
    for name in sorted(by_name):
        fields = by_name[name]
        verdict_v = fields.get('verdict')
        if not isinstance(verdict_v, str):
            continue
        try:
            verdict = Verdict(verdict_v)
        except ValueError:
            continue
        reason_v = fields.get('reason')
        reason = reason_v if isinstance(reason_v, str) else ''
        out.append(BridgeResult(
            verdict=verdict,
            reason=reason,
            stats=_stats_from_fields(fields),
            name=name,
            targets=_targets_from_fields(fields),
        ))
    return tuple(out)


def _bridge_result_to_measurements(
    result: BridgeResult,
) -> dict[str, MeasurementLeaf]:
    """Flatten a BridgeResult into path-keyed measurements.
    Canonical implementation — used by cell_runner at sweep time.
    Lives in aggregate because the bridge↔measurements transform
    IS the data-layer responsibility this module owns.

    `bridge.<name>.verdict` (or `invariant.<name>.verdict` when
    `stats['kind'] == 'tautological'`) carries the verdict;
    `<prefix>.<name>.reason` carries the free-text explanation;
    `<prefix>.<name>.targets` carries the comma-joined reads-set
    (the trace columns the bridge consumes); each scalar entry
    of `result.stats` lands under `<prefix>.<name>.stats.<key>`.

    Targets persistence is what makes `reconstruct_bridge_results`
    lossless and what lets downstream consumers (redundancy,
    register) compute reads-overlap without re-instantiating
    bridges from author code."""
    is_invariant = result.stats.get('kind') == 'tautological'
    prefix = (
        f'invariant.{result.name}' if is_invariant
        else f'bridge.{result.name}'
    )
    out: dict[str, MeasurementLeaf] = {
        f'{prefix}.verdict': result.verdict.value,
        f'{prefix}.reason': result.reason,
        f'{prefix}.targets': ','.join(result.targets),
    }
    # `BridgeResult.stats` is typed `Mapping[str, float | int |
    # bool | str]` — every value already satisfies
    # MeasurementLeaf. Forward each entry verbatim.
    for stat_key, stat_value in result.stats.items():
        out[f'{prefix}.stats.{stat_key}'] = stat_value
    return out


def _facts_from_runrow(run: RunRow) -> tuple[FactRow, ...]:
    """Reconstruct FactRows from a RunRow's flat-keyed
    measurements. Bridge results are persisted as
    `bridge.<name>.verdict` + `bridge.<name>.targets` +
    `bridge.<name>.stats.<k>` entries (see
    `_bridge_result_to_measurements`); invariants use
    `invariant.<name>.*` prefixes."""
    by_name = _grouped_bridge_fields(run)
    facts: list[FactRow] = []
    for name in sorted(by_name):
        fields = by_name[name]
        verdict_v = fields.get('verdict')
        if not isinstance(verdict_v, str):
            continue
        try:
            verdict = Verdict(verdict_v)
        except ValueError:
            continue
        stats = _stats_from_fields(fields)
        strength = natural_strength_from_stats(stats)
        delta_i = _delta_i_oriented(strength, verdict)
        # Evidentiary level: mirror the BridgeResult-based logic.
        if verdict is not Verdict.HELD:
            level = 'refuted'
        elif stats.get('tier') == 'interventional':
            level = 'causal_one_sided'
        else:
            level = 'correlational'
        facts.append(FactRow(
            name=name,
            reads=frozenset(_targets_from_fields(fields)),
            verdict=verdict,
            natural_strength=strength,
            delta_i=delta_i,
            evidentiary_level=level,
        ))
    return tuple(facts)


def _aggregate_facts_across_runs(
    runs: Sequence[RunRow],
) -> tuple[FactRow, ...]:
    """Fold per-run facts into a deduped union, one FactRow per
    bridge name.

    `verdict` is the majority verdict across runs; ties break to
    POWER_INSUFFICIENT (consistent with how aggregate_cell_verdict
    handles disagreement). `natural_strength` and `delta_i` are
    averaged. `reads` unions across runs."""
    by_name: dict[str, list[FactRow]] = {}
    for run in runs:
        for fact in _facts_from_runrow(run):
            by_name.setdefault(fact.name, []).append(fact)
    out: list[FactRow] = []
    for name in sorted(by_name):
        facts = by_name[name]
        verdicts = [f.verdict for f in facts]
        # Majority verdict; ties go to POWER_INSUFFICIENT.
        from collections import Counter
        c = Counter(verdicts)
        most_common = c.most_common()
        if len(most_common) == 1 or most_common[0][1] > most_common[1][1]:
            verdict = most_common[0][0]
        else:
            verdict = Verdict.POWER_INSUFFICIENT
        ns_mean = sum(f.natural_strength for f in facts) / len(facts)
        di_mean = sum(f.delta_i for f in facts) / len(facts)
        reads = frozenset().union(*(f.reads for f in facts))
        # Use the first fact's evidentiary_level as the
        # representative — this is best-effort; a richer
        # consumer would track per-run levels.
        out.append(FactRow(
            name=name, reads=reads, verdict=verdict,
            natural_strength=ns_mean, delta_i=di_mean,
            evidentiary_level=facts[0].evidentiary_level,
        ))
    return tuple(out)


def fact_from_bridge_result(r: BridgeResult) -> FactRow:
    """Project a `BridgeResult` into a `FactRow`.

    `reads` is `frozenset(r.targets)` — the bridge's declared
    target keys. Bridges that consume registered measurables get
    a richer reads-set when consumers call
    `Bridge.transitive_reads()` and substitute it; per-result
    bridge introspection isn't possible from `BridgeResult`
    alone (no back-reference to the Bridge that produced it),
    so this projection uses targets only."""
    strength = natural_strength_from_stats(r.stats)
    delta_i = _delta_i_oriented(strength, r.verdict)
    return FactRow(
        name=r.name,
        reads=frozenset(r.targets),
        verdict=r.verdict,
        natural_strength=strength,
        delta_i=delta_i,
        evidentiary_level=_evidentiary_level_from_bridge_result(r),
    )


# ============ HypothesisComparisonRow.from_cells ============

def _partition_runs_by(
    runs: Sequence[RunRow], group_by: str,
) -> dict[object, list[RunRow]]:
    """Partition runs by the value of measurement key `group_by`.
    Loud KeyError when a run is missing the key."""
    out: dict[object, list[RunRow]] = {}
    for r in runs:
        if group_by not in r.measurements:
            raise KeyError(
                f'run {r.id!r} missing group_by key {group_by!r}',
            )
        out.setdefault(r.measurements[group_by], []).append(r)
    return out


def _per_group_stats(
    h: 'Hypothesis[Mapping[str, object]]',
    group_value: object,
    treatment_runs: Sequence[RunRow],
    baseline_runs: Sequence[RunRow],
    *,
    outcome_path: str,
    pair_by: tuple[str, ...],
    alpha: float,
    power: float,
    predicted_direction: PredictedDirection | None = None,
) -> tuple[GroupStats | None, int]:
    """Pair within one group, compute Hedges' g + verdict +
    GroupStats. Returns (GroupStats | None, n_dropped_unpaired).
    None when no pairs survive the outcome-finite filter.

    `predicted_direction` overrides `h.predicted_direction` for
    the sign test when the caller knows a per-edge prior (e.g.
    a typed `claim_bridge.Bridge` carries `predicted_direction='a_lt_b'`
    on its mechanism edge but `'a_gt_b'` on its outcome edge). When
    None, falls back to `h.predicted_direction`.

    Raises ValueError on duplicate pair_by keys within an arm —
    silent dedup would hide a misconfigured slice."""
    paired, t_by_pkey, b_by_pkey, n_dropped = _pair_runs_by_key(
        treatment_runs, baseline_runs,
        pair_by=pair_by, dup_check=True, group_label=group_value,
    )

    a_values: list[float] = []
    b_values: list[float] = []
    deltas: list[float] = []
    for pk in paired:
        a = _run_outcome(t_by_pkey[pk], outcome_path)
        b = _run_outcome(b_by_pkey[pk], outcome_path)
        if not (math.isfinite(a) and math.isfinite(b)):
            continue
        a_values.append(a)
        b_values.append(b)
        deltas.append(a - b)

    n_pairs = len(deltas)
    if n_pairs == 0:
        return None, n_dropped

    a_mean = float(sum(a_values) / n_pairs)
    b_mean = float(sum(b_values) / n_pairs)
    a_sd: float | None
    b_sd: float | None
    if n_pairs > 1:
        a_var = sum((v - a_mean) ** 2 for v in a_values) / (n_pairs - 1)
        b_var = sum((v - b_mean) ** 2 for v in b_values) / (n_pairs - 1)
        a_sd = float(a_var ** 0.5)
        b_sd = float(b_var ** 0.5)
    else:
        a_sd = b_sd = None

    g, se = (
        hedges_g_paired(deltas) if n_pairs >= 2
        else (float('nan'), float('nan'))
    )
    q = derived_q_from_g_se(g, se)
    di = delta_i_from_q(q)
    effective_direction = (
        predicted_direction if predicted_direction is not None
        else h.predicted_direction
    )
    verdict, refutation, is_powered = verdict_from_paired_stats(
        g, se, n_pairs,
        predicted_direction=effective_direction,
        alpha=alpha, power=power,
    )

    g_safe: float | None = None if math.isnan(g) else float(g)
    se_safe: float | None = None if math.isnan(se) else float(se)
    q_safe: float | None = None if math.isnan(q) else float(q)

    return GroupStats(
        group_value=group_value,
        n_pairs=n_pairs,
        arm_a_mean=a_mean, arm_a_sd=a_sd,
        arm_b_mean=b_mean, arm_b_sd=b_sd,
        effect_size_g=g_safe, se=se_safe, derived_q=q_safe,
        delta_i=di, verdict=verdict,
        refutation_class=refutation,
        adequately_powered=is_powered,
    ), n_dropped


def hypothesis_comparison_from_cells(
    h: 'Hypothesis[Mapping[str, object]]',
    treatment_runs: Sequence[RunRow],
    baseline_runs: Sequence[RunRow],
    *,
    outcome_path: str,
    pair_by: tuple[str, ...],
    group_by: str | None = None,
    alpha: float = 0.05,
    power: float = 0.8,
    cycle_id: str | None = None,
    timestamp: str | None = None,
    baseline_h: 'Hypothesis[Mapping[str, object]] | None' = None,
    predicted_direction: PredictedDirection | None = None,
) -> HypothesisComparisonRow:
    """The canonical cross-arm aggregator. Builds one
    HypothesisComparisonRow from per-cell `RunRow`s.

    `pair_by` identifies a (treatment, baseline) pair within a
    stratum (e.g. `('seed',)` for DQN). REQUIRED for paired tests.

    `group_by` partitions runs into strata before pairing (e.g.
    `'env_name'`). When set, the row carries `per_group:
    tuple[GroupStats, ...]` AND `pooled: PooledStats` from
    DerSimonian-Laird random-effects pooling across strata; the
    top-level `effect_size_g` mirrors `pooled.pooled_g`.

    When `group_by` is None, single-group mode: per-arm stats and
    Hedges' g over the paired Δ distribution; `per_group=()`,
    `pooled=None`.

    `baseline_h` carries the typed identity of the baseline arm
    (`intervention_arms` tuple). Default None means baseline is
    the empty-arms baseline (`arm_key()='baseline'`). The two arm
    keys MUST differ — equal keys raise ValueError as a
    HPO-smuggle indicator (the comparison would be self-against-
    self, signal-free).

    `predicted_direction` overrides the hypothesis-level prior
    (`h.predicted_direction`) for the sign test. The verdict-walk
    path through `hypothesis_subgraph_verdict` passes the per-edge
    `claim_bridge.Bridge.predicted_direction` here so each edge's
    sign test uses its own prior (mechanism predicts a_lt_b,
    outcome predicts a_gt_b — they can't share one direction).
    When None, falls back to `h.predicted_direction`.

    Raises:
    - `ValueError` on empty arms or empty `pair_by`.
    - `ValueError` when treatment and baseline arm keys match
      (HPO-smuggle indicator).
    - `ValueError` on duplicate `pair_by` keys within an arm
      within a group (silent dedup would mask a misconfigured
      slice).
    - `KeyError` when a run is missing the `group_by` measurement
      key (should never happen for a correctly-typed corpus).

    Drops unmatched pairs silently; the count lands on
    `row.n_dropped_unpaired` so consumers see the gap."""
    if not treatment_runs:
        raise ValueError(
            'hypothesis_comparison_from_cells: treatment_runs empty',
        )
    if not baseline_runs:
        raise ValueError(
            'hypothesis_comparison_from_cells: baseline_runs empty',
        )
    if not pair_by:
        raise ValueError(
            'hypothesis_comparison_from_cells: pair_by must be non-empty',
        )

    treatment_arm_key = h.arm_key()
    baseline_arm_key = (
        baseline_h.arm_key() if baseline_h is not None else 'baseline'
    )
    if treatment_arm_key == baseline_arm_key:
        raise ValueError(
            f'hypothesis_comparison_from_cells: treatment and baseline '
            f'share arm_key {treatment_arm_key!r}; the comparison would '
            f'be self-against-self (HPO-smuggle indicator). Treatment '
            f'and baseline must differ in their `intervention_arms`.',
        )

    # Arm-key consistency check: every run in `treatment_runs`
    # should carry `treatment_arm_key`; same for baseline. The
    # framework's load-bearing promise: HPO variation does NOT
    # change arm_key, so mixed-arm runs in one arm-list signal a
    # pairing-data bug. The default 'baseline' value on legacy
    # RunRows passes the baseline check transparently.
    t_arms = {r.arm_key for r in treatment_runs}
    if t_arms != {treatment_arm_key}:
        raise ValueError(
            f'hypothesis_comparison_from_cells: treatment_runs carry '
            f'mixed arm_keys {sorted(t_arms)!r}; expected all to be '
            f'{treatment_arm_key!r}. Pairing inconsistency.',
        )
    b_arms = {r.arm_key for r in baseline_runs}
    if b_arms != {baseline_arm_key}:
        raise ValueError(
            f'hypothesis_comparison_from_cells: baseline_runs carry '
            f'mixed arm_keys {sorted(b_arms)!r}; expected all to be '
            f'{baseline_arm_key!r}. Pairing inconsistency.',
        )

    intervention_name = _run_intervention_name(treatment_runs[0])

    effective_direction = (
        predicted_direction if predicted_direction is not None
        else h.predicted_direction
    )

    if group_by is None:
        # Single-group mode.
        gs, n_dropped = _per_group_stats(
            h, group_value=None,
            treatment_runs=treatment_runs,
            baseline_runs=baseline_runs,
            outcome_path=outcome_path,
            pair_by=pair_by, alpha=alpha, power=power,
            predicted_direction=effective_direction,
        )
        if gs is None:
            return HypothesisComparisonRow(
                id=str(uuid.uuid4()),
                parent_id=None,
                cycle_id=cycle_id,
                timestamp=_resolved_timestamp(timestamp),
                intervention_name=intervention_name,
                treatment_arm_key=treatment_arm_key,
                baseline_arm_key=baseline_arm_key,
                treatment_run_ids=tuple(r.id for r in treatment_runs),
                baseline_run_ids=tuple(r.id for r in baseline_runs),
                predicted_direction=effective_direction,
                pair_by=pair_by,
                group_by=group_by,
                arm_a_n=0, arm_a_mean=None, arm_a_sd=None,
                arm_b_n=0, arm_b_mean=None, arm_b_sd=None,
                effect_size_g=None, se=None, derived_q=None,
                delta_i_population=0.0,
                adequately_powered=False,
                verdict=Verdict.POWER_INSUFFICIENT,
                refutation_class=None,
                per_group=(), pooled=None,
                facts=_aggregate_facts_across_runs(treatment_runs),
                reads_set=frozenset(),
                n_dropped_unpaired=n_dropped,
            )
        facts = _aggregate_facts_across_runs(treatment_runs)
        reads_set = frozenset().union(
            *(f.reads for f in facts),
        ) if facts else frozenset()
        return HypothesisComparisonRow(
            id=str(uuid.uuid4()),
            parent_id=None,
            cycle_id=cycle_id,
            timestamp=_resolved_timestamp(timestamp),
            intervention_name=intervention_name,
            treatment_arm_key=treatment_arm_key,
            baseline_arm_key=baseline_arm_key,
            treatment_run_ids=tuple(r.id for r in treatment_runs),
            baseline_run_ids=tuple(r.id for r in baseline_runs),
            predicted_direction=effective_direction,
            pair_by=pair_by,
            group_by=group_by,
            arm_a_n=gs.n_pairs, arm_a_mean=gs.arm_a_mean,
            arm_a_sd=gs.arm_a_sd,
            arm_b_n=gs.n_pairs, arm_b_mean=gs.arm_b_mean,
            arm_b_sd=gs.arm_b_sd,
            effect_size_g=gs.effect_size_g, se=gs.se,
            derived_q=gs.derived_q,
            delta_i_population=gs.delta_i,
            adequately_powered=gs.adequately_powered,
            verdict=gs.verdict,
            refutation_class=gs.refutation_class,
            per_group=(), pooled=None,
            facts=facts, reads_set=reads_set,
            n_dropped_unpaired=n_dropped,
        )

    # Stratified mode.
    treatment_groups = _partition_runs_by(treatment_runs, group_by)
    baseline_groups = _partition_runs_by(baseline_runs, group_by)
    all_keys = sorted(
        treatment_groups.keys() | baseline_groups.keys(),
        key=lambda k: repr(k),
    )

    per_group: list[GroupStats] = []
    g_se_pairs: list[tuple[float, float]] = []
    n_dropped = 0
    all_a: list[float] = []
    all_b: list[float] = []

    for gkey in all_keys:
        t_g = treatment_groups.get(gkey, [])
        b_g = baseline_groups.get(gkey, [])
        if not t_g or not b_g:
            n_dropped += len(t_g) + len(b_g)
            continue
        gs, dropped = _per_group_stats(
            h, group_value=gkey,
            treatment_runs=t_g,
            baseline_runs=b_g,
            outcome_path=outcome_path,
            pair_by=pair_by, alpha=alpha, power=power,
            predicted_direction=effective_direction,
        )
        n_dropped += dropped
        if gs is None:
            continue
        per_group.append(gs)
        if (gs.effect_size_g is not None
                and gs.se is not None
                and not math.isnan(gs.effect_size_g)
                and not math.isnan(gs.se)):
            g_se_pairs.append((gs.effect_size_g, gs.se))
        if gs.arm_a_mean is not None:
            all_a.extend([gs.arm_a_mean] * gs.n_pairs)
        if gs.arm_b_mean is not None:
            all_b.extend([gs.arm_b_mean] * gs.n_pairs)

    pooled = random_effects_summary(g_se_pairs)
    verdict_p, refutation_p = random_effects_verdict(
        pooled, predicted_direction=effective_direction,
    )

    arm_n = sum(gs.n_pairs for gs in per_group)
    if math.isnan(pooled.pooled_g) or math.isnan(pooled.se_pooled):
        effect_g: float | None = None
        se_top: float | None = None
        derived_q: float | None = None
        delta_i_pop = 0.0
        adequately_powered = False
    else:
        effect_g = float(pooled.pooled_g)
        se_top = float(pooled.se_pooled)
        # Use the pooled n_cells as n for power assessment — the
        # appropriate n for "is the pooled estimate detectable at
        # this number of cells?" rather than total pair count.
        adequately_powered = adequately_powered_paired(
            effect_g, pooled.n_cells, alpha=alpha, power=power,
        )
        q_val = derived_q_from_g_se(effect_g, se_top)
        derived_q = None if math.isnan(q_val) else float(q_val)
        delta_i_pop = (
            delta_i_from_q(q_val) if not math.isnan(q_val) else 0.0
        )

    facts = _aggregate_facts_across_runs(treatment_runs)
    reads_set = frozenset().union(
        *(f.reads for f in facts),
    ) if facts else frozenset()

    return HypothesisComparisonRow(
        id=str(uuid.uuid4()),
        parent_id=None,
        cycle_id=cycle_id,
        timestamp=_resolved_timestamp(timestamp),
        intervention_name=intervention_name,
        treatment_arm_key=treatment_arm_key,
        baseline_arm_key=baseline_arm_key,
        treatment_run_ids=tuple(r.id for r in treatment_runs),
        baseline_run_ids=tuple(r.id for r in baseline_runs),
        predicted_direction=effective_direction,
        pair_by=pair_by,
        group_by=group_by,
        arm_a_n=arm_n,
        arm_a_mean=(
            float(sum(all_a) / arm_n) if arm_n > 0 else None
        ),
        arm_a_sd=None,
        arm_b_n=arm_n,
        arm_b_mean=(
            float(sum(all_b) / arm_n) if arm_n > 0 else None
        ),
        arm_b_sd=None,
        effect_size_g=effect_g,
        se=se_top,
        derived_q=derived_q,
        delta_i_population=delta_i_pop,
        adequately_powered=adequately_powered,
        verdict=verdict_p,
        refutation_class=refutation_p,
        per_group=tuple(per_group),
        pooled=pooled,
        facts=facts,
        reads_set=reads_set,
        n_dropped_unpaired=n_dropped,
    )
