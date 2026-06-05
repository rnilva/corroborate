"""Exploration: crude vs paired DDQN-2010 on Asterix γ=0.999.

Builds one `Panel` from two corpora and four arms:

  crude corpus  `asterix_g0999_ddqn2010`  (single-net `dqn` program)
    crude_vanilla   — arm_key='baseline'                 (15 seeds)
    crude_ddqn2010  — arm_key='bootstrap=…double_greedify_indep'  (15 seeds)
  paired corpus `asterix_g0999_ddqn2010_paired`
    paired_ddqn2010 — sub-corpus 'deep2010', program='paired_dqn' (30 seeds)
    paired_vanilla  — sub-corpus 'vanilla',  program='dqn'        (30 seeds)

The four arms are uniquely identified by `(corpus, arm_key)`; the
crude corpus predates the typed `RunRow.program` column (null after
the diagonal-relaxed union), the paired corpus carries it.

The free-lunch question (PRINCIPLED reproduction of the crude
`double_greedify_indep` result with the two-learner `paired_dqn`
program):

  Does the independent-estimator selector reduce overestimation
  (clipped jensen_gap → 0, unclipped per-state bias → negative)
  like DDQN-2016 WITHOUT harming performance (eval return)?

  FREE LUNCH  — bias eliminated AND eval return ≥ vanilla
  INSEPARABLE — bias eliminated AND eval return << vanilla
                (matches canonical DDQN-2016 cost d ≈ -0.68)

Measurables surfaced (all "considerable" per the request):

  bias / overestimation
    jensen_gap                          clipped max(0, mean(Q-MC))
    mean_per_state_cumulative_bias_late UNCLIPPED per-state bias (signed)
    normalized_bias_redq_late           normalised unclipped bias
    jensen_dormancy_gap                 true-zero vs under-estimating (traces)
  outcome — discounted
    eval_best_burst_mean
    eval_final_mean
    late_window_mean
  outcome — RAW (undiscounted, γ-invariant; needs traces)
    eval_best_burst_raw_mean
    eval_final_raw_mean
    eval_full_auc_raw_mean
  mechanism / structure
    bootstrap_action_mismatch_late
    q_trajectory_autocorr_late
    q_action_grad_overlap_late          (crude only — gradient_probes off
    q_inter_state_grad_overlap_late      in the paired sweep)

Raw-return + dormancy-gap measurables need traces joined. If the
trace parquets are present locally (restored from cloud) the script
joins them and fills the columns via `with_measurables`; otherwise
those rows stay null and are flagged in the printout.

Run from repo root:
    uv run --package corroborate_rl \\
        python3 -m experiments.findings.ddqn_sweeps._paired_2010_exploration
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl

import corroborate.analyses  # pyright: ignore[reportUnusedImport]  # populate registry
import corroborate_rl.dqn.measurables  # pyright: ignore[reportUnusedImport]  # populate measurable registry

from corroborate.data import Panel

_DATA = Path(__file__).resolve().parents[3] / 'experiments' / 'data'

_CRUDE = _DATA / 'asterix_g0999_ddqn2010'
_PAIRED = _DATA / 'asterix_g0999_ddqn2010_paired'

# (corpus_dir, human label) for each arm-bearing corpus.
_CORPORA: tuple[tuple[Path, str], ...] = (
    (_CRUDE, 'crude'),
    (_PAIRED / 'deep2010', 'paired'),
    (_PAIRED / 'vanilla', 'paired'),
)

# Measurables grouped for the printout.
_BIAS = (
    'jensen_gap',
    'mean_per_state_cumulative_bias_late',
    'normalized_bias_redq_late',
    'jensen_dormancy_gap',
)
_OUTCOME_DISCOUNTED = (
    'eval_best_burst_mean',
    'eval_final_mean',
    'late_window_mean',
)
_OUTCOME_RAW = (
    'eval_best_burst_raw_mean',
    'eval_final_raw_mean',
    'eval_full_auc_raw_mean',
)
_STRUCTURE = (
    'bootstrap_action_mismatch_late',
    'q_trajectory_autocorr_late',
    'q_action_grad_overlap_late',
    'q_inter_state_grad_overlap_late',
)
_ALL_MEASURABLES = _BIAS + _OUTCOME_DISCOUNTED + _OUTCOME_RAW + _STRUCTURE
# These can only be filled when traces.parquet is joined.
_NEEDS_TRACES = frozenset(_OUTCOME_RAW + ('jensen_dormancy_gap',))


def _parquet_complete(path: Path) -> bool:
    """A parquet file ends with the 4-byte 'PAR1' magic. Guards
    against a half-written trace download (the 2 GB restore can
    time out at the tail, leaving a size-correct but truncated
    file)."""
    if not path.exists() or path.stat().st_size < 8:
        return False
    with open(path, 'rb') as fh:
        fh.seek(-4, 2)
        return fh.read(4) == b'PAR1'


def _arm_label(corpus: str, arm_key: str) -> str:
    """Human label from the (corpus, arm_key) discriminator."""
    if corpus == 'asterix_g0999_ddqn2010':
        return 'crude_ddqn2010' if 'greedify' in arm_key else 'crude_vanilla'
    if corpus == 'deep2010':
        return 'paired_ddqn2010'
    if corpus == 'vanilla':
        return 'paired_vanilla'
    return f'{corpus}/{arm_key}'


def build_panel() -> Panel:
    """Union the available corpora into one Panel, join traces when
    present, fill the trace-backed measurables, and stamp a clean
    `arm_label` column for stratification."""
    available = [(d, tag) for d, tag in _CORPORA if (d / 'runs.parquet').exists()]
    if not available:
        return Panel.from_dataframe(pl.DataFrame(), stratify_by=('arm_label',))

    # Join traces only for corpora that actually have them locally —
    # from_corpora is all-or-nothing on join_traces, so build per-corpus.
    panels: list[Panel] = []
    for d, _tag in available:
        has_traces = _parquet_complete(d / 'traces.parquet')
        panels.append(Panel.from_corpus(d, join_traces=has_traces))

    cells = pl.concat(
        [p.cells for p in panels if p.cells.height > 0],
        how='diagonal_relaxed',
    )
    sources = tuple(s for p in panels for s in p.sources)

    # Stamp the arm label so all four arms are uniquely stratifiable.
    cells = cells.with_columns(
        pl.struct(['corpus', 'arm_key'])
        .map_elements(
            lambda s: _arm_label(s['corpus'], s['arm_key']),
            return_dtype=pl.Utf8,
        )
        .alias('arm_label'),
    )

    panel = Panel.from_dataframe(
        cells, stratify_by=('arm_label',), sources=sources,
    )
    # Fill trace-backed measurables for arms whose traces were joined.
    return panel.with_measurables(list(_ALL_MEASURABLES))


def _finite_stats(col: pl.Series) -> tuple[float, float, int] | None:
    if col.dtype not in (pl.Float32, pl.Float64, pl.Int32, pl.Int64):
        return None
    vals = col.cast(pl.Float64).drop_nulls()
    vals = vals.filter(vals.is_finite())
    if len(vals) == 0:
        return None
    return float(vals.mean()), float(vals.std()), len(vals)


# Canonical column order so the table is stable run-to-run.
_ARM_ORDER = ('crude_vanilla', 'crude_ddqn2010', 'paired_vanilla', 'paired_ddqn2010')


def print_report(panel: Panel) -> None:
    if panel.cells.height == 0:
        print('Empty panel — no corpora found.')
        return

    present = [a for a in _ARM_ORDER if a in panel.cells['arm_label'].to_list()]
    by_arm = {a: panel.cells.filter(pl.col('arm_label') == a) for a in present}

    print('=' * 92)
    print('Crude vs Paired DDQN-2010  ·  Asterix-MinAtar γ=0.999')
    print('=' * 92)
    hdr = f'{"measurable":38}'
    for a in present:
        hdr += f'  {a[:16]:>16}'
    print(hdr)
    nrow = f'{"n seeds":38}'
    for a in present:
        nrow += f'  {by_arm[a].height:>16}'
    print(nrow)
    print('-' * 92)

    def _section(title: str, names: tuple[str, ...]) -> None:
        print(f'· {title}')
        for m in names:
            cells_present = [by_arm[a] for a in present if m in by_arm[a].columns]
            if not cells_present:
                continue
            stats = {a: _finite_stats(by_arm[a][m]) for a in present if m in by_arm[a].columns}
            if not any(v for v in stats.values()):
                # all-null (e.g. traces not restored)
                flag = '  [needs traces]' if m in _NEEDS_TRACES else '  [n/a]'
                print(f'  {m:36}{flag}')
                continue
            row = f'  {m:36}'
            for a in present:
                s = stats.get(a)
                if s is None:
                    row += f'  {"—":>16}'
                else:
                    row += f'  {s[0]:>+9.3f}±{s[1]:<6.2f}'
            print(row)
        print()

    _section('bias / overestimation', _BIAS)
    _section('outcome — discounted', _OUTCOME_DISCOUNTED)
    _section('outcome — raw (undiscounted)', _OUTCOME_RAW)
    _section('mechanism / structure', _STRUCTURE)

    print('=' * 92)
    _free_lunch(by_arm)


def _free_lunch(by_arm: dict[str, pl.DataFrame]) -> None:
    print('FREE-LUNCH ASSESSMENT')
    print('-' * 92)

    pairs = [
        ('crude_ddqn2010', 'crude_vanilla'),
        ('paired_ddqn2010', 'paired_vanilla'),
        ('paired_ddqn2010', 'crude_vanilla'),  # fallback if paired_vanilla absent
    ]
    seen: set[str] = set()
    for ddqn, van in pairs:
        if ddqn not in by_arm or van not in by_arm:
            continue
        if ddqn in seen:
            continue
        seen.add(ddqn)

        def _mean(arm: str, m: str) -> float:
            df = by_arm[arm]
            if m not in df.columns:
                return float('nan')
            s = _finite_stats(df[m])
            return s[0] if s else float('nan')

        jg_d, jg_v = _mean(ddqn, 'jensen_gap'), _mean(van, 'jensen_gap')
        bias_d = _mean(ddqn, 'mean_per_state_cumulative_bias_late')
        bias_v = _mean(van, 'mean_per_state_cumulative_bias_late')
        ev_d, ev_v = _mean(ddqn, 'eval_best_burst_mean'), _mean(van, 'eval_best_burst_mean')
        raw_d, raw_v = _mean(ddqn, 'eval_best_burst_raw_mean'), _mean(van, 'eval_best_burst_raw_mean')

        # Cohen's d on eval_best_burst_mean (independent-samples).
        d_eval = _cohens_d(by_arm[ddqn], by_arm[van], 'eval_best_burst_mean')

        print(f'{ddqn}  vs  {van}:')
        print(f'  clipped jensen_gap:        {jg_d:8.2f}  vs {jg_v:8.2f}  (Δ {jg_d - jg_v:+.1f})')
        print(f'  UNCLIPPED per-state bias:  {bias_d:8.2f}  vs {bias_v:8.2f}  (Δ {bias_d - bias_v:+.1f})')
        print(f'  eval_best_burst (disc.):   {ev_d:8.2f}  vs {ev_v:8.2f}  (Cohen d = {d_eval:+.3f})')
        if not math.isnan(raw_d):
            print(f'  eval_best_burst (RAW):     {raw_d:8.2f}  vs {raw_v:8.2f}')
        verdict = (
            'FREE LUNCH (bias gone, no harm)'
            if (not math.isnan(d_eval) and d_eval > -0.2)
            else 'INSEPARABLE (eval harm)'
            if not math.isnan(d_eval)
            else 'INDETERMINATE'
        )
        print(f'  → {verdict}')
        print()


def _cohens_d(treat: pl.DataFrame, base: pl.DataFrame, col: str) -> float:
    if col not in treat.columns or col not in base.columns:
        return float('nan')
    t = treat[col].cast(pl.Float64).drop_nulls()
    t = t.filter(t.is_finite())
    b = base[col].cast(pl.Float64).drop_nulls()
    b = b.filter(b.is_finite())
    if len(t) < 2 or len(b) < 2:
        return float('nan')
    nt, nb = len(t), len(b)
    vt, vb = float(t.var()), float(b.var())
    pooled = math.sqrt(((nt - 1) * vt + (nb - 1) * vb) / (nt + nb - 2))
    if pooled == 0:
        return float('nan')
    return (float(t.mean()) - float(b.mean())) / pooled


if __name__ == '__main__':
    panel = build_panel()
    if panel.cells.height:
        labels = panel.cells['arm_label'].value_counts().sort('arm_label')
        print('arms in panel:', dict(zip(labels['arm_label'], labels['count'])))
        print('sources:', [s.corpus for s in panel.sources])
        print()
    print_report(panel)
